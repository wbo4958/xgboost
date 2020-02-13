#! /bin/bash

cd ../../

mkdir build 2>&1 > /dev/null

cd build

cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DUSE_OPENMP=OFF
make -j4
