from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import numpy as np
import math

in_c = 64
out_c = 64
group = 64
kern = 3
spatial = 14
stride = 1

#inp = np.array(range(in_c * spatial * spatial)).astype(np.float32)
#inp = np.array([1.0 for _ in range(in_c * spatial * spatial)]).astype(np.float32)
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
inp = inp.reshape(1,in_c,spatial,spatial)

#k = np.array(range(out_c * in_c / group * kern * kern)).astype(np.float32)
#k = np.array([1.0 for _ in range(out_c * in_c / group * kern * kern)]).astype(np.float32)
k = np.random.rand(out_c, in_c / group, kern, kern).astype(np.float32)
k = k.reshape(out_c, in_c / group, kern, kern)

b = np.zeros(out_c).astype(np.float32)#np.random.rand(16).astype(np.float32)
workspace.FeedBlob('input', inp)
workspace.FeedBlob('kernel', k)
workspace.FeedBlob('bias', b)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
 ['input'],
 ['cl_input'],
 device_option=device_option,
)
op = core.CreateOperator("Conv",
 ['cl_input', 'kernel', 'bias'],
 ['cl_output'],
 kernel=kern,
 order="NCHW",
 group=group,
 stride=stride,
 device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
 ['cl_output'],
 ['output'],
 device_option=device_option,
)
workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("Conv",
  ['input', 'kernel', 'bias'],
  ['ref_output'],
  order='NCHW',
  stride=stride,
  group=group,
  kernel=kern,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > 1 or math.isnan(test_out[i]):
    #print(test_out[i], ref_out[i], i, "ERROR")
    error = True

if error:
  print "ERRORS"
  exit(1)
print("conv success!")

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
 ['input'],
 ['cl_input'],
 device_option=device_option,
)
op = core.CreateOperator("Relu",
 ['cl_input'],
 ['cl_output'],
 kernel=kern,
 order="NCHW",
 group=group,
 stride=stride,
 device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
 ['cl_output'],
 ['output'],
 device_option=device_option,
)
workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("Relu",
  ['input'],
  ['ref_output'],
  order='NCHW',
  stride=stride,
  group=group,
  kernel=kern,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > 1 or math.isnan(test_out[i]):
    #print(test_out[i], ref_out[i], i, "ERROR")
    error = True

if error:
  print "ERRORS"
  exit(1)
print("relu success!")

c = out_c
workspace.ResetWorkspace()
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
#inp = np.array([_ for _ in range(in_c * spatial * spatial)]).astype(np.float32)
#inp = inp.reshape(1, in_c, spatial, spatial)
workspace.FeedBlob('input', inp)
workspace.FeedBlob('scale', np.random.rand(c).astype(np.float32))
workspace.FeedBlob('bias',  np.random.rand(c).astype(np.float32))
workspace.FeedBlob('mean',  np.random.rand(c).astype(np.float32))
workspace.FeedBlob('var',   np.random.rand(c).astype(np.float32))

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
  ['input'],
  ['cl_input'],
 device_option=device_option,
)
op = core.CreateOperator("SpatialBN",
  ['cl_input','scale','bias','mean','var'],
  ['cl_output'],
  kernel=kern,
  order="NCHW",
  group=group,
  stride=stride,
  device_option=device_option,
  is_test=True,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
  ['cl_output'],
  ['output'],
 device_option=device_option,
)

workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("SpatialBN",
  ['input','scale','bias','mean','var'],
  ['ref_output'],
  order='NCHW',
  stride=stride,
  group=group,
  kernel=kern,
  is_test=True,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > max(0.1 * ref_out[i], 0.1) or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    error = True
  if i > 20 and error:
    break

if error:
  print "ERRORS"
  exit(1)
print("spatial bn success!")


workspace.ResetWorkspace()
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
group = 4
workspace.FeedBlob('input', inp)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
  ['input'],
  ['cl_input'],
 device_option=device_option,
)
op = core.CreateOperator("ChannelShuffle",
  ['cl_input'],
  ['cl_output'],
  order="NCHW",
  group=group,
  device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
  ['cl_output'],
  ['output'],
 device_option=device_option,
)

workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("ChannelShuffle",
  ['input'],
  ['ref_output'],
  order='NCHW',
  group=group,
	kernel=1,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > max(0.1 * ref_out[i], 0.1) or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    error = True
  if i > 20 and error:
    break

if error:
  print "ERRORS"
  exit(1)
print("channel shuffle success!")

workspace.ResetWorkspace()
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
inp2 = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
group = 4
workspace.FeedBlob('input', inp)
workspace.FeedBlob('input2', inp2)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
  ['input'],
  ['cl_input'],
 device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyToOpenCL",
  ['input2'],
  ['cl_input2'],
 device_option=device_option,
)
op = core.CreateOperator("Add",
  ['cl_input', 'cl_input2'],
  ['cl_output'],
  order="NCHW",
  device_option=device_option,
)
op_copy3 = core.CreateOperator("CopyFromOpenCL",
  ['cl_output'],
  ['output'],
 device_option=device_option,
)

workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op_copy2)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy3)

ref_op = core.CreateOperator("Add",
  ['input', 'input2'],
  ['ref_output'],
  order='NCHW',
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > max(0.1 * ref_out[i], 0.1) or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    error = True
  if i > 20 and error:
    break

if error:
  print "ERRORS"
  exit(1)
print("add success!")


workspace.ResetWorkspace()
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
inp2 = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)
group = 4
workspace.FeedBlob('input', inp)
workspace.FeedBlob('input2', inp2)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
  ['input'],
  ['cl_input'],
 device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyToOpenCL",
  ['input2'],
  ['cl_input2'],
 device_option=device_option,
)
op = core.CreateOperator("Concat",
  ['cl_input', 'cl_input2'],
  ['cl_output', 'cl_ignore'],
  order="NCHW",
  device_option=device_option,
)
op_copy3 = core.CreateOperator("CopyFromOpenCL",
  ['cl_output'],
  ['output'],
 device_option=device_option,
)

workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op_copy2)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy3)

ref_op = core.CreateOperator("Concat",
  ['input', 'input2'],
  ['ref_output', 'ignore'],
  order='NCHW',
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out = workspace.FetchBlob('ref_output').flatten()
error = False
for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > max(0.1 * ref_out[i], 0.1) or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    error = True
  if i > 20 and error:
    break

if error:
  print "ERRORS"
  exit(1)
print("concat success!")

workspace.ResetWorkspace()
inp = np.random.rand(1, in_c, spatial, spatial).astype(np.float32)

inp = inp.reshape(1,in_c,spatial,spatial)
workspace.FeedBlob('input', inp)

device_option = caffe2_pb2.DeviceOption()
device_option.device_type = caffe2_pb2.OPENCL
op_copy1 = core.CreateOperator("CopyToOpenCL",
  ['input'],
  ['cl_input'],
 device_option=device_option,
)
op = core.CreateOperator("AveragePool",
  ['cl_input'],
  ['cl_output'],
  order="NCHW",
  kernel=3,
  stride=2,
  device_option=device_option,
)
op_copy2 = core.CreateOperator("CopyFromOpenCL",
  ['cl_output'],
  ['output'],
 device_option=device_option,
)

workspace.RunOperatorOnce(op_copy1)
workspace.RunOperatorOnce(op)
workspace.RunOperatorOnce(op_copy2)

ref_op = core.CreateOperator("AveragePool",
  ['input'],
  ['ref_output'],
  order='NCHW',
  kernel=3,
  stride=2,
)
workspace.RunOperatorOnce(ref_op)

test_out = workspace.FetchBlob('output').flatten()
ref_out_ = workspace.FetchBlob('ref_output')
ref_out = ref_out_.flatten()
error = False

for i in range(len(test_out)):# - 1, 0, -1):
  if abs(test_out[i] - ref_out[i]) > max(0.1 * ref_out[i], 0.1) or math.isnan(test_out[i]):
    print(test_out[i], ref_out[i], i, "ERROR")
    error = True
  if i > 20 and error:
    break

if error:
  print "ERRORS"
  exit(1)
print("average pool success!")
