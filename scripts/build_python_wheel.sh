#!/bin/bash
##############################################################################
# This script will build a python wheel file in the folder build/dist
##############################################################################

CAFFE2_ROOT="$( cd "$(dirname "$0")"/.. ; pwd -P)"
BUILD_ROOT=$CAFFE2_ROOT/build

# Build Caffe2
echo "Building Caffe2"
cd $BUILD_ROOT
cmake .. $(python $CAFFE2_ROOT/scripts/get_python_libs.py)
make

# Build the wheel from the compiled files
cd $BUILD_ROOT
python $CAFFE2_ROOT/scripts/setup.py bdist_wheel

echo
echo
echo "Python wheel built! Located in build/dist/"
