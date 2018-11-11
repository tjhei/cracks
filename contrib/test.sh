#!/bin/bash

# script to run the CI tests locally using docker

if test ! -f cracks.cc -o ! -f CMakeLists.txt ; then
  echo "*** This script must be run from the top-level directory."
  exit 1
fi

echo "checking indentation ..."
docker run \
       --rm \
       -v `pwd`:/source \
       tjhei/dealii:v8.5.1-full-v8.5.1-r1 \
       bash -c "cd /source && ./contrib/indent && git diff --exit-code" \
    || { echo "Please check indentation!"; exit 1; }
echo "    OK"

echo "checking deal.II v8.5.1 ..."
docker run \
       --rm \
       -v `pwd`:/source \
       tjhei/dealii:v8.5.1-full-v8.5.1-r1 \
       bash -c "mkdir build; cd build; cmake /source && make -j 4 && ./cracks" \
    || { echo "Failed!"; exit 1; }
echo "    OK"

echo "checking deal.II v9.0.0 ..."
docker run \
       --rm \
       -v `pwd`:/source \
       tjhei/dealii:v9.0.1-full-v9.0.1-r3 \
       bash -c "mkdir build; cd build; cmake -D CMAKE_CXX_FLAGS='-Werror' /source && make -j 4 && ./cracks && ctest -j 4" \
    || { echo "Failed!"; exit 1; }
echo "    OK"

echo "OK, done."
