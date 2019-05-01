cmake_minimum_required(VERSION 2.6.0)

project(ICP)

set(Pangolin_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Pangolin/build/src" CACHE PATH "Pangolin build directory")
set(Sophus_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Sophus" CACHE PATH "Sophus Include directory")

find_package(Pangolin REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Sophus_INCLUDE_DIR})

file(GLOB srcs *.cpp *.cu *.h *.cuh)
file(GLOB cuda Cuda/*.cu Cuda/*.cpp Cuda/*.cuh Cuda/*.h)
file(GLOB containers Cuda/containers/*.cpp Cuda/containers/*.h Cuda/containers/*.cu Cuda/containers/*.cuh)

set(CUDA_ARCH_BIN "" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include("CudaDetect.cmake")
detect_installed_gpus(CUDA_NVCC_ARCHS)
foreach(NVCC_ARCH IN LISTS CUDA_NVCC_ARCHS)
    list(APPEND CUDA_ARCH_BIN "${NVCC_ARCH} ")
endforeach(NVCC_ARCH)

include("CudaComputeTargetFlags.cmake")
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;--expt-relaxed-constexpr")

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall -std=c++11")

add_executable(ICP 
               ${srcs}
               ${cuda} 
               ${cuda_objs} 
               ${containers}
)

target_link_libraries(ICP
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
)

