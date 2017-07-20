#include "caffe2/core/common.h"

#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"
#include "gtest/gtest.h"

#ifdef __ANDROID__
#include <android/log.h>
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#else
#define ALOGI(...) printf(__VA_ARGS__)
#endif

#ifndef CHECK_NEAR
#define CHECK_NEAR(a,b,d) CAFFE_ENFORCE(fabs((a) - (b)) < d)
#endif
//#define HALF 1
#ifdef HALF
  #define SKIP_SANITY true // basically never passes
  #define TYPE "Half"
#else
  #define SKIP_SANITY false
  #define TYPE ""
#endif

#define DIM 18
#define CHANNEL 128

namespace caffe2 {
void randomBlob(Workspace *ws, std::string name, int D1, int D2, int D3, int D4) {
  CPUContext ctx;
  auto* t = ws->CreateBlob(name)->GetMutable<TensorCPU>();
  if (D1 * D2 * D3 * D4 != 0) {
    t->Resize(D1, D2, D3, D4);
  } else if (D1 * D2 * D3 != 0) {
    t->Resize(D1, D2, D3);
  } else if (D1 * D2 != 0) {
    t->Resize(D1, D2);
  } else if (D1 != 0) {
    t->Resize(D1);
  } else {
    CAFFE_THROW("This is not a valid random blob dimension");
  }
  math::RandGaussian<float, CPUContext>(t->size(), 0, 3, t->mutable_data<float>(), &ctx);
}

void constBlob(Workspace *ws, std::string name, int D1, int D2, int D3, int D4, float val, bool scaled = false) {
  CPUContext ctx;
  auto* t = ws->CreateBlob(name)->GetMutable<TensorCPU>();
  if (D1 * D2 * D3 * D4 != 0) {
    t->Resize(D1, D2, D3, D4);
  } else if (D1 * D2 * D3 != 0) {
    t->Resize(D1, D2, D3);
  } else if (D1 * D2 != 0) {
    t->Resize(D1, D2);
  } else if (D1 != 0) {
    t->Resize(D1);
  } else {
    CAFFE_THROW("This is not a valid const blob dimension");
  }
  if (!scaled) {
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = val;
    }
  } else {
    for (auto i = 0; i < t->size(); ++i) {
      t->mutable_data<float>()[i] = i * val;
    }
  }
}

void runConvBenchAgainstNNPACK(
    int out_channels,
    int in_channels,
    int kern,
    int H,
    int W,
    int ops,
    int iters,
    std::string convName,
    std::string inputName,
    std::string filterName,
    std::string biasName,
    std::string outputName,
    std::function<void(Workspace *, int, int, int, int, int)> start_net = [](Workspace*, int, int, int, int, int){},
    std::function<void(Workspace *)> end_net = [](Workspace*){},
    bool HWC = false
    ) {
  ALOGI("M %d C %d H %d W %d K %d",
        out_channels, in_channels, H, W, kern);
  const float flops = 2.0 * kern * kern * in_channels * out_channels * (H - kern + 1) * (W - kern + 1);
  const float peak_kflops = 400000000.;
  ALOGI("%10s: %fms (400.000 GFLOPS)", "TPeak", flops / peak_kflops);

#define STR_TIME(_str) do {\
  const float ms_per_iter = t.MilliSeconds()/ops/iters;\
  const float flops_impl = flops / ms_per_iter / 1000000.0;\
  ALOGI("%10s: %fms (%.3f GFLOPS)", (_str), ms_per_iter, flops_impl);\
} while(0)


  Timer t;
  Workspace ws;
  randomBlob(&ws, "NNPACK_input", 1, in_channels, H, W);
  randomBlob(&ws, "NNPACK_filter", out_channels, in_channels, kern, kern);
  constBlob(&ws, "NNPACK_bias", out_channels, 0, 0, 0, 0.0);

  NetDef benchNet;
  benchNet.set_name("benchNet");

  start_net(&ws, out_channels, in_channels, kern, H, W);
  for (auto i = 0; i < ops; ++i) {
    {
    auto& op = *(benchNet.add_op());
    op.mutable_device_option()->set_device_type(OPENCL);
    op.set_type(convName);
    op.add_input(inputName );
    op.add_input(filterName);
    op.add_input(biasName  );
    if (HWC) {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NHWC");
    } else {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    op.add_arg()->CopyFrom(MakeArgument("kernel", kern));
    op.add_arg()->CopyFrom(MakeArgument("stride_h", 1));
    op.add_arg()->CopyFrom(MakeArgument("stride_w", 1));
    op.add_arg()->CopyFrom(MakeArgument("pad_t", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_l", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_b", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_r", 0));
    op.add_output(outputName);
    }
  }
  ws.CreateNet(benchNet);
  ws.RunNet("benchNet");
  t.Start();
  for (auto i = 0; i < iters; ++i) {
    ws.RunNet("benchNet");
  }
  STR_TIME(convName.c_str());
  end_net(&ws);

  NetDef refNet;
  refNet.set_name("refNet");
  for (auto i = 0; i < ops; ++i) {
    {
    auto& op = *(refNet.add_op());
    op.set_type("Conv");
    op.set_engine("NNPACK");
    op.add_input("NNPACK_input");
    op.add_input("NNPACK_filter");
    op.add_input("NNPACK_bias");
    op.add_arg()->CopyFrom(MakeArgument("kernel", kern));
    op.add_arg()->CopyFrom(MakeArgument("stride_h", 1));
    op.add_arg()->CopyFrom(MakeArgument("stride_w", 1));
    op.add_arg()->CopyFrom(MakeArgument("pad_t", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_l", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_b", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_r", 0));
    op.add_output("Y_ref");
    }
  }

  ws.RunNetOnce(refNet);
  ws.CreateNet(refNet);
  t.Start();
  for (auto i = 0; i < iters; ++i) {
    ws.RunNet("refNet");
  }
  STR_TIME("NNPACK");
}

std::function<void(Workspace *, int, int, int, int, int)>
createHWCSetupNet(string convName) {
  return [=](Workspace* ws, int out_channels, int in_channels, int kern, int H, int W){
    randomBlob(ws, convName + "_input_cpu", 1, H, W, in_channels              );
    randomBlob(ws, convName + "_filter", out_channels, kern, kern, in_channels);
    constBlob( ws, convName + "_bias", out_channels, 0, 0, 0, 0.0);
    NetDef loadNet;
    auto& op = *(loadNet.add_op());
    op.set_type("CopyToOpenCL" TYPE);
    op.add_input( convName + "_input_cpu");
    op.add_output(convName + "_input");
    ws->RunNetOnce(loadNet);
  };
}

std::function<void(Workspace *, int, int, int, int, int)>
createCHWSetupNet(string convName) {
  return [=](Workspace* ws, int out_channels, int in_channels, int kern, int H, int W){
    randomBlob(ws, convName + "_input_cpu", 1, in_channels, H, W);
    randomBlob(ws, convName + "_filter", out_channels, in_channels, kern, kern);
    constBlob( ws, convName + "_bias", out_channels, 0, 0, 0, 0.0);
    NetDef loadNet;
    auto& op = *(loadNet.add_op());
    op.set_type("CopyToOpenCL" TYPE);
    op.add_input( convName + "_input_cpu");
    op.add_output(convName + "_input");
    ws->RunNetOnce(loadNet);
  };
}

#define ITER(_var, _min, _max, _step) \
  for (auto (_var) = (_min); (_var) < (_max); (_var) += (_step))
#define EXP_ITER(_var, _min, _max, _step) \
  for (auto (_var) = (_min); (_var) < (_max); (_var) *= (_step))

void runConvTest(
    std::string convName,
    bool HWC = false,
    bool skip_sanity = false
) {
  std::function<void(Workspace *, int, int, int, int, int)> setup_func;
  if (HWC) {
    setup_func = createHWCSetupNet(convName);
  } else {
    setup_func = createCHWSetupNet(convName);
  }

  EXP_ITER(out_channels, 64, 2057, 2000) {
  const int in_channels = out_channels;
  //EXP_ITER( in_channels, 4, 257, 2) {
  ITER(kern, 3, 4, 1) {
  EXP_ITER(_width, 64, 1650, 4000) {
  //ITER(height, 10, 23, 8) {
  const int width = _width + 2;
  const int height = width;
    if (skip_sanity) {
      ALOGI("SKIPPING SANITY CHECK\n");
    } else {
      ALOGI("Testing %s (%s) M: %d C: %d K: %d H: %d W: %d...", convName.c_str(),
          HWC ? "NHWC" : "NCHW", out_channels, in_channels, kern, height, width);
    }
    Workspace ws;

    // Create a reference network
    NetDef refNet;
    refNet.set_name("refNet");
    {
    auto& op = *(refNet.add_op());
    op.set_type("Conv");
    if (HWC) {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NHWC");
    } else {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    op.add_input(convName + "_input_cpu");
    op.add_input(convName + "_filter");
    op.add_input(convName + "_bias");
    op.add_arg()->CopyFrom(MakeArgument("kernel", kern));
    op.add_arg()->CopyFrom(MakeArgument("stride_h", 1));
    op.add_arg()->CopyFrom(MakeArgument("stride_w", 1));
    op.add_arg()->CopyFrom(MakeArgument("pad_t", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_l", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_b", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_r", 0));
    op.add_output("ref_output");
    }

    // Create the check network
    NetDef checkNet;
    checkNet.set_name("checkNet");
    {
    auto& op = *(checkNet.add_op());
    op.mutable_device_option()->set_device_type(OPENCL);
    op.set_type(convName);
    if (HWC) {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NHWC");
    } else {
      auto& arg = *(op.add_arg());
      arg.set_name("order");
      arg.set_s("NCHW");
    }
    op.add_input(convName + "_input");
    op.add_input(convName + "_filter");
    op.add_input(convName + "_bias");
    op.add_arg()->CopyFrom(MakeArgument("kernel", kern));
    op.add_arg()->CopyFrom(MakeArgument("stride_h", 1));
    op.add_arg()->CopyFrom(MakeArgument("stride_w", 1));
    op.add_arg()->CopyFrom(MakeArgument("pad_t", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_l", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_b", 0));
    op.add_arg()->CopyFrom(MakeArgument("pad_r", 0));
    op.add_output("check_output");
    }

    NetDef storeNet;
    auto& op = *(storeNet.add_op());
    op.set_type("CopyFromOpenCL" TYPE);
    op.add_input("check_output");
    op.add_output("check_output_cpu");

    // Run the networks
    setup_func(&ws, out_channels, in_channels, kern, height, width);
    ws.CreateNet(checkNet);
    ws.RunNet("checkNet");
	  ws.RunNetOnce(storeNet);
    ws.RunNetOnce(refNet);

    const auto& t1 = ws.GetBlob("ref_output")->Get<TensorCPU>();
    const auto& t2 = ws.GetBlob("check_output_cpu")->Get<TensorCPU>();

    CAFFE_ENFORCE_EQ(t1.dims(), t2.dims());
    const int H = t1.dim32(1);
    const int W = t1.dim32(2);
    const int C = t1.dim32(3);

    for (auto i = 0; i < t1.size() - 0; ++i) {
      const float t1_i = t1.data<float>()[i];
      const float t2_i = t2.data<float>()[i];
      const float max_diff = std::max(0.1 * fabs(t1_i), 0.1);
      if (fabs(t1_i - t2_i) > max_diff && !skip_sanity) {
        ALOGI("(%d %d %d, %d %d %d) Expected %f, but got %f diff by %f",
            i / (W * C),
            (i % (W * C)) / C,
            i % C,
            H, W, C,
            t1_i, t2_i,
            fabs(t1_i - t2_i));
        if (i > 20) {
          CHECK_NEAR(t1_i, t2_i, max_diff);
        }
      }
    }
    ALOGI("passed!\n");
    auto iters = 10;
    auto ops = 10;
    std::string inputName = "Conv_input";
    std::string outputName = "Conv_out";
    runConvBenchAgainstNNPACK(
        out_channels,
        in_channels,
        kern,
        height,
        width,
        ops,
        iters,
        convName,
        "Conv_input",
        "Conv_filter",
        "Conv_bias",
        outputName,
        createHWCSetupNet("Conv"),
        [=](Workspace *ws) {
          NetDef storeNet;
          auto& op = *(storeNet.add_op());
          op.set_type("CopyFromOpenCL" TYPE);
          op.add_input(outputName);
          op.add_output(outputName + "_cpu");
	        ws->RunNetOnce(storeNet);
        }, true);

  sleep(1);
  //} // height
  } // width
  } // kernal
  //} // in channel
  } // out channel
}

TEST(OpenCL, Convolution) {
  runConvTest("Conv" TYPE, true, SKIP_SANITY);
}

} // namespace caffe2

