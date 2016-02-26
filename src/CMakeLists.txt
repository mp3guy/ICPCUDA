cmake_minimum_required(VERSION 2.6.0)

project(ICP)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread filesystem system date_time chrono REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EIGEN_INCLUDE_DIRS})

file(GLOB srcs *.cpp)
file(GLOB cuda Cuda/*.cu)
file(GLOB containers Cuda/containers/*.cpp)

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall")

add_executable(ICP 
               ${srcs}
               ${cuda} 
               ${cuda_objs} 
               ${containers}
)

target_link_libraries(ICP
                      ${Boost_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${Eigen_LIBRARIES}
                      ${CUDA_LIBRARIES}
)

