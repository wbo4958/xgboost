/*!
 * Copyright 2019 XGBoost contributors
 *
 * \file c-api-demo.c
 * \brief A simple example of using xgboost C API.
 */

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <cudf/cudf.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <xgboost/data.h>

using namespace std;
using namespace xgboost;
#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  exit(1);                                                              \
}                                                                       \
}

int main(int argc, char **argv) {
//  testCSR();
//  std::string filename = "big.libsvm";
//  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(320, 64, filename);

  // flag to switch to iris or mortgage
  bool isIrisData = true;

  std::vector<std::string> iris_files = {
    "iris.1.column.csv"
  };
  std::vector<std::string> mortgage_files = {
      "/home/bobwang/jupyter/data/mortgage/csv/chunk_benchmark/2010_1.csv",
      "/home/bobwang/jupyter/data/mortgage/csv/chunk_benchmark/2010_2.csv"
  };

  std::vector<std::string> &files = mortgage_files;
  int num_columns = 28;
  if (isIrisData) {
    files = iris_files;
    num_columns = 2;
  }

  DMatrixHandle dhandle;

  int total_rows = 0;
  float missing = 0.0f;
  bool is_first_file = true;

  // this is for mortgage data
  for (auto ptr = files.begin(); ptr < files.end(); ptr++) {
    cudf::csv_read_arg args(cudf::source_info{*ptr});

    for (int i = 0; i < num_columns; i++) {
      args.dtype.push_back("float");
      args.names.push_back(std::to_string(i));
    }
    args.header = -1;

    auto df = cudf::read_csv(args);
    std::cout << "columns:" << df.num_columns() << " rows:" << df.num_rows() << std::endl;

    total_rows += df.num_rows();

    // features
    std::vector<const gdf_column *> feature_cols(df.begin(), df.end() - 1);
    std::vector<const gdf_column *> label_cols(df.end() - 1, df.end());

    df.destroy();
  }

  std::cout << "Total rows: " << total_rows << std::endl;

//  int use_gpu = 1;
//
//  int silent = 1;
//
//  // create the booster
//  BoosterHandle booster;
//  DMatrixHandle eval_dmats[1] = {dhandle};
//  safe_xgboost(XGBoosterCreate(eval_dmats, 1, &booster));
//
//  safe_xgboost(XGBoosterSetParam(booster, "tree_method", use_gpu ? "gpu_hist" : "hist"));
//  if (use_gpu) {
//    // set the number of GPUs and the first GPU to use;
//    // this is not necessary, but provided here as an illustration
//    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "1"));
//    safe_xgboost(XGBoosterSetParam(booster, "gpu_id", "0"));
//    safe_xgboost(XGBoosterSetParam(booster, "predictor", "gpu_predictor"));
//  } else {
//    // avoid evaluating objective and metric on a GPU
//    safe_xgboost(XGBoosterSetParam(booster, "n_gpus", "0"));
//  }
//
//  if (isIrisData) {
//    safe_xgboost(XGBoosterSetParam(booster, "objective", "multi:softprob"));
//    safe_xgboost(XGBoosterSetParam(booster, "num_class", "3"));
//    safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
//    safe_xgboost(XGBoosterSetParam(booster, "max_bin", "50"));
//
//
//  } else {
//    safe_xgboost(XGBoosterSetParam(booster, "objective", "binary:logistic"));
//  }
//
//  safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
//  safe_xgboost(XGBoosterSetParam(booster, "max_depth", "3"));
////  safe_xgboost(XGBoosterSetParam(booster, "verbosity", "3"));
//
//  // train and evaluate for 10 iterations
//  int n_trees = 100;
//  for (int i = 0; i < n_trees; ++i) {
//    safe_xgboost(XGBoosterUpdateOneIter(booster, i, dhandle));
//  }
//
//  // free everything
//  safe_xgboost(XGBoosterFree(booster));
//  safe_xgboost(XGDMatrixFree(dhandle));

  return 0;
}

void testCSR() {
  /**
 * sparse matrix
 * 1 0 2 3 0
 * 4 0 2 3 5
 * 3 1 2 5 0
 */
  float data[] = {1, 2, 3, 4, 2, 3, 5, 3, 1, 2, 5};
  unsigned int colIndex[] = {0, 2, 3, 0, 2, 3, 4, 0, 1, 2, 3};
  size_t rowHeaders[] = {0, 3, 7, 11};
  DMatrixHandle handle;
  XGDMatrixCreateFromCSREx(rowHeaders, colIndex, data, 4, 11, 0, &handle);

}

void CreateBigTestData(const std::string &filename, size_t n_entries) {
  std::ofstream fo(filename.c_str());
  const size_t entries_per_row = 3;
  size_t n_rows = (n_entries + entries_per_row - 1) / entries_per_row;
  std::cout << "CreateBigTestData total rows:" << n_rows << std::endl;
  for (size_t i = 0; i < n_rows; ++i) {
    const char *row = i % 2 == 0 ? " 0:0 1:10 2:20\n" : " 0:0 3:30 4:40\n";
    fo << i << row;
  }
}

std::unique_ptr<DMatrix> CreateSparsePageDMatrix(
    size_t n_entries, size_t page_size, std::string tmp_file) {
  // Create sufficiently large data to make two row pages
  CreateBigTestData(tmp_file, n_entries);
  std::unique_ptr<DMatrix> dmat{DMatrix::Load(
      tmp_file + "#" + tmp_file + ".cache", true, false, "auto", page_size)};

  // Loop over the batches and count the records
  int64_t batch_count = 0;
  int64_t row_count = 0;
  for (const auto &batch : dmat->GetBatches<SparsePage>()) {
    batch_count++;
    row_count += batch.Size();
  }
  std::cout << "batch_cout --- " << batch_count << std::endl;
  std::cout << " row count --- " << row_count << std::endl;
  return dmat;
}
