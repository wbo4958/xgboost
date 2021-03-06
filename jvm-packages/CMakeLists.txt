find_package(JNI REQUIRED)

list(APPEND JVM_SOURCES
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j.cpp)

if (USE_CUDA)
  list(APPEND JVM_SOURCES
    ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j-spark/src/native/xgboost4j_spark.cpp
    ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native/xgboost4j.cu
    ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j-spark/src/native/xgboost4j_spark_gpu.cu)
endif (USE_CUDA)

add_library(xgboost4j SHARED ${JVM_SOURCES} ${XGBOOST_OBJ_SOURCES})
if (ENABLE_ALL_WARNINGS)
  target_compile_options(xgboost4j PUBLIC -Wall -Wextra)
endif (ENABLE_ALL_WARNINGS)
target_link_libraries(xgboost4j PRIVATE objxgboost)
target_include_directories(xgboost4j
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${JNI_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/jvm-packages/xgboost4j/src/native
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include
  ${PROJECT_SOURCE_DIR}/rabit/include)

if (USE_CUDA)
  target_include_directories(xgboost4j PRIVATE ${xgboost_SOURCE_DIR}/cub/)
  target_compile_options(xgboost4j PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CUDA>>:--std=c++14>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEN_CODE}>)
endif (USE_CUDA)

set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(
  xgboost4j PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(xgboost4j PRIVATE ${JAVA_JVM_LIBRARY})
