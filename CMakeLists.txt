cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(cuda_torch_example LANGUAGES CXX)

find_package(CUDAToolkit)
find_package(Torch REQUIRED)
find_package(fmt REQUIRED)

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cuda_torch_example
               src/cuda_torch.cu
               src/cuda_torch.h
               src/helper_string.h
               src/helper_cuda.h
               src/cuda_torch_example.cpp)
target_link_libraries(cuda_torch_example "${TORCH_LIBRARIES}" fmt CUDA::cudart CUDA::cupti)
set_property(TARGET cuda_torch_example PROPERTY LANGUAGE CUDA)
