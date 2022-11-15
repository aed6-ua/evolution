cmake_minimum_required(VERSION 3.15)

set(genetic_source_files
  genetic.cc
  genetic.h
  individual.cc
  individual.h
  simulation.cc
  simulation.h
)
include_directories(${PROJECT_SOURCE_DIR})
add_library(genetic ${genetic_source_files})
target_link_libraries(genetic utils)
