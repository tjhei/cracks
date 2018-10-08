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
       dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease \
       bash -c "cd /source && ./contrib/indent && git diff --exit-code" \
    || { echo "Please check indentation!"; exit 1; }
echo "    OK"

echo "checking deal.II v8.5.1 ..."
docker run \
       --rm \
       -v `pwd`:/source \
       dealii/dealii:v8.5.1-gcc-mpi-fulldepscandi-debugrelease \
       bash -c "mkdir build; cd build; cmake /source && make -j 4 && ./cracks" \
    || { echo "Failed!"; exit 1; }
echo "    OK"

echo "checking deal.II v9.0.0 ..."
docker run \
       --rm \
       -v `pwd`:/source \
       dealii/dealii:v9.0.0-gcc-mpi-fulldepscandi-debugrelease \
       bash -c "mkdir build; cd build; cmake /source && make -j 4 && ./cracks && ctest -j 4" \
    || { echo "Failed!"; exit 1; }
echo "    OK"

echo "OK, done."
