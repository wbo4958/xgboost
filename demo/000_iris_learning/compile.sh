#! /bin/bash

cd ../../

mkdir build 2>&1 > /dev/null

cd build

nvidia-smi -q | grep "Product Name" | awk -F: '{print $2}'

cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DUSE_OPENMP=OFF -DCUDA_ARCHITECTURES=75
make -j8
