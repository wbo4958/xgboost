if(PLUGIN_SYCL)
  set(CMAKE_CXX_COMPILER "icpx")
  add_library(plugin_sycl OBJECT
    ${xgboost_SOURCE_DIR}/plugin/sycl/objective/regression_obj.cc
    ${xgboost_SOURCE_DIR}/plugin/sycl/objective/multiclass_obj.cc
    ${xgboost_SOURCE_DIR}/plugin/sycl/device_manager.cc
    ${xgboost_SOURCE_DIR}/plugin/sycl/predictor/predictor.cc)
  target_include_directories(plugin_sycl
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
    ${xgboost_SOURCE_DIR}/rabit/include)
    target_compile_definitions(plugin_sycl PUBLIC -DXGBOOST_USE_SYCL=1)
    target_link_libraries(plugin_sycl PUBLIC -fsycl)
    set_target_properties(plugin_sycl PROPERTIES
    COMPILE_FLAGS -fsycl
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set_target_properties(plugin_sycl PROPERTIES
    COMPILE_FLAGS "-fsycl -qopenmp")
  endif()
  # Get compilation and link flags of plugin_sycl and propagate to objxgboost
  target_link_libraries(objxgboost PUBLIC plugin_sycl)
  # Add all objects of plugin_sycl to objxgboost
  target_sources(objxgboost INTERFACE $<TARGET_OBJECTS:plugin_sycl>)
endif()

# Add the Federate Learning plugin if enabled.
if(PLUGIN_FEDERATED)
  add_subdirectory(federated)
endif()
