cmake_minimum_required(VERSION 3.15)

set(mlp_source_files
  activation.h
  layer.cc
  layer.h
  mlp.cc
  mlp.h
  neuron.cc
  neuron.h
)
include_directories(${PROJECT_SOURCE_DIR})
add_library(mlp ${mlp_source_files})
target_link_libraries(mlp utils)
