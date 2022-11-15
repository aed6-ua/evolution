cmake_minimum_required(VERSION 3.15)

set(utils_source_files
  engine.cc
  engine.h
  point2d.cc
  point2d.h
  utils.cc
  utils.h
)
add_library(utils ${utils_source_files})
