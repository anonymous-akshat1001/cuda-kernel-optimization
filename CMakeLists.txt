cmake_minimum_required(VERSION 3.18)
project(cuda_kernel_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_bench src/main.cu)
target_compile_options(cuda_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# Set CMAKE_CUDA_ARCHITECTURES at configure time for a portable, reproducible build.
# Example for a Colab T4: cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
