cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(elastx CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(elastx 
  main.cu
  json.hpp
)

set_target_properties(elastx PROPERTIES CUDA_ARCHITECTURES "70;75")
set_property(TARGET elastx PROPERTY CUDA_SEPARABLE_COMPILATION ON)
