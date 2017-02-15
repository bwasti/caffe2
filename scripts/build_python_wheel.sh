#!/bin/bash
##############################################################################
# This script will build a python wheel file in the folder build/dist
##############################################################################

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
BUILD_ROOT=$CAFFE2_ROOT/build
PYTHON_BUILD_ROOT=$CAFFE2_ROOT/build/caffe2

# Build Caffe2
echo "Building Caffe2"
cd $BUILD_ROOT
cmake .. $(python get_python_libs.py)
make

# Build the wheel from the compiled files
cd $PYTHON_BUILD_ROOT
python ../../scripts/setup.py bdist_wheel

echo
echo
echo "Python wheel built! Located in build/dist/"
