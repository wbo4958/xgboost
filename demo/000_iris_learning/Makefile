SRC=cudf_xgboost_demo.cpp
TGT=run_demo

cc=g++

CFLAGS ?=-O0 -g -DXGBOOST_USE_CUDF

XGBOOST_ROOT ?=../..
INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include -I/home/bobwang/anaconda3/envs/cudf_dev_30/include
LIB_DIR=-L$(XGBOOST_ROOT)/lib -L/home/bobwang/anaconda3/envs/cudf_dev_30/lib

build: $(TGT)

$(TGT): $(SRC) Makefile
	$(cc) $(CFLAGS) $(INCLUDE_DIR) $(LIB_DIR) -o $(TGT) $(SRC) -lxgboost -lboost_filesystem -lcudf -lNVStrings -lNVCategory -lrmm 

run: $(TGT)
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib:/home/bobwang/anaconda3/envs/cudf_dev_30/lib  ./$(TGT)

debug: $(TGT)
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib:/home/bobwang/anaconda3/envs/cudf_dev_30/lib  gdbserver :1234 ./$(TGT)

remote_debug:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib:/home/bobwang/anaconda3/envs/cudf_dev_30/lib  gdbserver 10.19.183.93:1234 ./$(TGT)

iris_debug:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib gdbserver :1234 ../../xgboost iris.conf

run_iris_debug:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ../../xgboost iris.conf

run_iris_debug_pred:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ../../xgboost iris.conf task=pred model_in=0001.model

run_iris_debug_dump:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ../../xgboost iris.conf task=dump model_in=0001.model name_dump=dump.raw.txt

sketch_iris_debug:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib gdbserver :1234 ../../xgboost iris_sketch.conf

run_sketch_iris_debug:
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ../../xgboost iris_sketch.conf
clean:
	rm -f $(TGT)
