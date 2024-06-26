if(USE_DMLC_GTEST)
  if(NOT TARGET gtest)
    message(FATAL_ERROR "USE_DMLC_GTEST=ON but dmlc-core didn't bundle gtest")
  endif()
  set(GTEST_LIBRARIES gtest gmock)
else()
  find_package(GTest REQUIRED)
endif()
file(GLOB_RECURSE TEST_SOURCES "*.cc")

if(USE_CUDA)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif()

file(GLOB_RECURSE SYCL_TEST_SOURCES "plugin/test_sycl_*.cc")
if(NOT PLUGIN_SYCL)
  list(REMOVE_ITEM TEST_SOURCES ${SYCL_TEST_SOURCES})
endif()

if(PLUGIN_FEDERATED)
  target_include_directories(testxgboost PRIVATE ${xgboost_SOURCE_DIR}/plugin/federated)
  target_link_libraries(testxgboost PRIVATE federated_client)
else()
  file(GLOB_RECURSE FEDERATED_TEST_SOURCES "plugin/*_federated_*.*")
  list(REMOVE_ITEM TEST_SOURCES ${FEDERATED_TEST_SOURCES})
endif()

target_sources(testxgboost PRIVATE ${TEST_SOURCES} ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc)

if(USE_CUDA AND PLUGIN_RMM)
  find_package(CUDA)
  target_include_directories(testxgboost PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

target_include_directories(testxgboost
  PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
target_link_libraries(testxgboost
  PRIVATE
  ${GTEST_LIBRARIES})

set_output_directory(testxgboost ${xgboost_BINARY_DIR})

# This grouping organises source files nicely in visual studio
auto_source_group("${TEST_SOURCES}")
