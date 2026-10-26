cmake_minimum_required(VERSION 3.24)
project(cuda_data_residency LANGUAGES CXX CUDA)
add_executable(vector_add src/vector_add.cu)
set_target_properties(vector_add PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17)
