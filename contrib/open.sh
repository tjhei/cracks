#!/bin/bash

# This script opens a local shell to run the tests

if test ! -f cracks.cc -o ! -f CMakeLists.txt ; then
  echo "*** This script must be run from the top-level directory."
  exit 1
fi

echo "please run:"
echo ""
echo "mkdir build && cd build && cmake -G Ninja /source && ninja"
echo ""
echo "Then you can run the tests with:"
echo " ctest -j 4 -V"
echo ""

docker run \
       --rm \
       -v `pwd`:/source \
       -it \
       tjhei/dealii:v9.0.1-full-v9.0.1-r5-gcc5

echo "OK, done."
