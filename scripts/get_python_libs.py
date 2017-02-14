##############################################################################
# Use this script to find your preferred python installation.
##############################################################################
#
# You can use the following to build with your preferred version of python
# if your installation is not being properly detected by CMake.
#
#   mkdir -p build && cd build
#   cmake $(python ../scripts/get_python_libs.py) ..
#   make
#

from distutils import sysconfig
inc = sysconfig.get_python_inc()
print('-DPYTHON_INCLUDE_DIR={inc}'.format(inc=inc))
