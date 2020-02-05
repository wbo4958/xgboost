Debug iris
===

**XGBoost** implements a C API originally designed for various language
bindings.  For detailed reference, please check xgboost/c_api.h.  Here is a
demonstration of using the API.

# CMake
If you use **CMake** for your project, you can either install **XGBoost**
somewhere in your system and tell CMake to find it by calling
`find_package(xgboost)`, or put **XGBoost** inside your project's source tree
and call **CMake** command: `add_subdirectory(xgboost)`.  To use
`find_package()`, put the following in your **CMakeLists.txt**:

``` CMake
find_package(xgboost REQUIRED)
add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo xgboost::xgboost)
```

If you want to put XGBoost inside your project (like git submodule), use this
instead:
``` CMake
add_subdirectory(xgboost)
add_executable(api-demo c-api-demo.c)
target_link_libraries(api-demo xgboost)
```

# make
You can start by modifying the makefile in this directory to fit your need.

# How to debug
1. compile xgboost with debug info

``` shell
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DUSE_OPENMP=OFF
```

1. add debug conf for CLION

```
Edit Configurations -> Add new Configuration -> GDB Remote Debug

'target remote' args:    :1234
Symbol file: /home/bobwang/work.d/nvspark/30/dmlc.xgboost/xgboost
Sysroot: /home/bobwang/work.d/nvspark/30/dmlc.xgboost/lib
```

1. run gdbserver command
```
make iris_debug
```

1. press debug button in clion