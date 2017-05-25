A Quick Tutorial on Caffe2's C++ API
====================================

*"Blob, Workspace, Operator, Net*
*Purveyors of Aids to Magical Deep-Learners*
*are proud to present*
*THE CAFFE2 C++ TUTORIAL"*

# Preface

This tutorial aims to give you a very quick and simplified overfiew of what Caffe2's C++ API interface. In the next sections, we will go through the very basics of these objects, and present you code snippets that show how one can use such objects to eventually deploy a service.

By design, Caffe2 is composed of 4 very simple objects: **Blob**, **Workspace**, **Operator**, and **Net**. Their relationship can be summarized as follows: Blob stores any time of objects, but usually Tensors; Workspace organizes blobs by their names; Operator takes and produces blobs in a workspace; and finally, Net is a series of Operators that forms a formal task, such as initializing a deep learning model or running a deep learning model.

If you come from the good old Caffe world, you might remember that Caffe uses protobufs - and sometimes people call it config files - to define and store models. Caffe2 also uses protobuf, but nowadays we usually use its Python interface to create and manipulate protobuf objects, instead of directly writing raw protobuf files (similar to the pycaffe extension in Caffe). Thus, in your day to day work, it is possible that you don't really have to work with Caffe2's C++ interface, and Python might be good enough for you. This is the working model of many of our researchers and engineers. However, when you really want to go into the core of the framework, or when you work on platforms such as mobile, understanding the C++ interface might be a very good addition to the Python abstractions.

Let's begin the tutorial with blobs.

# Blob

In Caffe2, Blobs are used to store any type of data. For example, in the most common case, a Blob contains a Tensor, which is Caffe2's multi-dimensional array format. The best way to understand a Blob is that it serves as a typed pointer, like C++17's [`std::any`](http://en.cppreference.com/w/cpp/utility/any) class. For maximal cross-platform support (especially Android and other embedded systems), Caffe2 as of today only uses C++11 features, so we do not use `std::any` directly. We may, in a few years, consider allowing the use of C++17.

Note for our old Caffe friends: Caffe's old Blob definition - a multi-dimensional tensor - now corresponds to Caffe2's `Tensor` class that we will describe shortly. The reason is that, as deep learning gets more and more complex, in some cases we might need Blobs to not just contain a multidimensional Tensor, but some other formats. For example, as we will cover in later tutorials, a blob can contain a pre-packed matrix for optimized inference, or an opaque object such as MKLMemory for Intel MKL integration.

The Blob definition can be found in [caffe2/core/blob.h](https://github.com/caffe2/caffe2/blob/master/caffe2/core/blob.h). Specifically, an empty blob is created with the following simple approach:
```cpp
Blob myblob;
```
This blob will contain nothing to start. Now, let's say we want to initialize this blob to contain an integer, get the pointer to the integer, and set it to 10. We can do:
```cpp
int* myint = myblob.GetMutable<int>();
*myint = 10;
```

After this, we can use IsType to query that the blob does contain an integer:
```cpp
bool isint = myblob.IsType<int>(); // will be true.
bool isfloat = myblob.IsType<float>(); // will be false.
```
And we can use Get() to get const references to the content.
```cpp
const int& myint_const = myblob.Get<int>();
LOG(INFO) << myint_const;  // will be 10.
```

Now, if we call Get() with a wrong type, an exception will be thrown:
```cpp
const float& myfloat = myblob.Get<float>(); // will throw an exception.
```
We can always change the underlying storage type by making yet another `GetMutable()` call, like this:
```cpp
double* mydouble = myblob.GetMutable<double>();
*mydouble = 3.14;
```
Also, if we have a pre-created object, we can use Reset() similar to that of `std` smart pointers to transfer the ownwership  to a blob object:
```cpp
std::vector<int>* pvec = new std::vector<int>();
myblob.Reset(pvec); // this transfers the ownership
bool is_vec = myblob.IsType<std::vector<int>>();
const auto& pvec_const = myblob.Get<std::vector<int>>();
```
Note: the `delete` operator on a pointer to the type must be public.

Simple as that. Got it? If you would rather run the above code instead of reading, the corresponding source code is at [caffe2/contrib/cpptutorial/blob.cc](https://github.com/caffe2/caffe2/blob/master/caffe2/contrib/cpptutorial/blob.cc).

# Tensor: the Most Common Blob Content

Now, despite the fact that Blob can contain anything, in most cases, what gets stored in the blob is a Tensor object - the most common data structure that stores multi-dimensional arrays. The Tensor class is defined as device-specific:

    template <class Context>
    class Tensor;

The Context allows us to abstract away different types of devices, such as a CPU or a CUDA gpu. For now, for the sake of simplicity, let's assume that there is a CPUContext, and we will use Tensor<CPUContext> to show the Tensor interface. For simplicity, we will do

    typedef Tensor<CPUContext> TensorCPU;

Let's start with a Blob and create a TensorCPU object out of it.

    Blob myblob;
    TensorCPU* mytensor = myblob.GetMutable<TensorCPU>();

This tensor is currently empty. There are two key parts that defines a tensor: its shape, and its data type. If you are familiar with Python, this is essentially the shape and dtype of an ndarray. Currently, if we query the 

## What is a Context?
TODO

# More about Blobs
TODO
## Blob type system
TODO
## Blob serialization
TODO

# Workspace
TODO

# Operator
TODO

# Writing your own operator
TODO

# More about Operators
TODO

## Operator Schema
TODO
## Operator Registration
TODO
## Operator Gradients
TODO
# Net
TODO

# More about Nets
## What is Net Type?
## Net in Workspace
