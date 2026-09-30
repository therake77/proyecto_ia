cmake_minimum_required(VERSION 3.24)

project(cuda_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(cuda_native
    src/bindings.cpp
    src/device_context.cpp
    src/device_array.cpp
    src/elementwise.cu
    src/matmul_tiled.cu
    src/reductions.cu
    src/softmax_ce.cu
)

target_include_directories(cuda_native PRIVATE src)
target_link_libraries(cuda_native PRIVATE CUDA::cudart)

set_target_properties(cuda_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
)

# Free-threaded CPython 3.14: this module does not touch Python objects from
# multiple native threads concurrently, but it also does not declare itself
# GIL-safe (Py_MOD_GIL_NOT_USED). CPython 3.13+ free-threaded builds will
# re-enable the GIL at import time with a RuntimeWarning; that is expected
# and harmless here. See native/cuda_native/README.md.

install(TARGETS cuda_native LIBRARY DESTINATION .)
