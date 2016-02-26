cmake_minimum_required(VERSION 2.6.0)

project(ICPCUDA CXX)


#####################################################################
# Version declaration
# Target file will be named with version suffix
#####################################################################

set(MAJOR_VERSION  1)
set(MINOR_VERSION  0)
set(PATCH_VERSION  0)

set(FULL_VERSION ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION})

set(ICPCUDA_VERSION ${FULL_VERSION})
set(PROJECT_CMAKE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_NAME icpcuda)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#####################################################################
# Find dependencies
#####################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread filesystem system date_time chrono REQUIRED)
find_package(CUDA REQUIRED)




file(GLOB srcs src/*.cpp)
file(GLOB cuda src/Cuda/*.cu)
file(GLOB containers src/Cuda/containers/*.cpp)


set(PUBLIC_HEADERS
    include/icpcuda/ICPOdometry.h
    include/icpcuda/ICPSlowdometry.h
    include/icpcuda/OdometryProvider.h
)

CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(CudaComputeTargetFlags)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -Wall")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(icpcuda
  src/ICPOdometry.cpp
  src/ICPSlowdometry.cpp
  src/Cuda/containers/device_memory.cpp
  src/Cuda/containers/initialization.cpp
  ${cuda}
  ${cuda_objs}
  ${containers}
  ${PUBLIC_HEADERS}
)


target_include_directories(icpcuda INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/icpcuda>
)
target_include_directories(icpcuda PRIVATE 
  $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS};${EIGEN3_INCLUDE_DIR};${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}> 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>
)
target_link_libraries(icpcuda
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)
add_executable(ICP
  src/ICP.cpp
)

target_include_directories(ICP PUBLIC INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include/icpcuda> )
target_include_directories(ICP PRIVATE $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS};${EIGEN3_INCLUDE_DIR};${CUDA_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}> )

target_link_libraries(ICP
  icpcuda
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
)

#####################################################################
# Installation
#####################################################################


set(INCLUDE_INSTALL_DIR include CACHE PATH "Install dir for include headers" )
set(LIB_INSTALL_DIR lib CACHE PATH  "Install dir for libraries" )
set(BIN_INSTALL_DIR bin CACHE PATH  "Install dir for libraries" )
set(SYSCONFIG_INSTALL_DIR /usr/local/ CACHE  PATH  "Install dir for system configuration")
SET(INSTALL_INCLUDE_DIR include/${PROJECT_NAME} CACHE PATH "Install dir for headers")
IF(WIN32 AND NOT CYGWIN)
  SET(DEF_INSTALL_CMAKE_DIR CMake)
ELSE()
  SET(DEF_INSTALL_CMAKE_DIR lib/cmake/${PROJECT_NAME})
ENDIF()
SET(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Install dir for CMake files")

include(GenerateExportHeader)
generate_export_header(${TARGET_NAME})
set_property(TARGET ${TARGET_NAME} PROPERTY VERSION ${ICPCUDA_VERSION})
set_property(TARGET ${TARGET_NAME} PROPERTY SOVERSION ${MAJOR_VERSION})
set_property(TARGET ${TARGET_NAME} PROPERTY
  INTERFACE_${TARGET_NAME}_MAJOR_VERSION ${MAJOR_VERSION})
set_property(TARGET ${TARGET_NAME} APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING ${TARGET_NAME}_MAJOR_VERSION
)

install(TARGETS ${TARGET_NAME} EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${LIB_INSTALL_DIR}
  ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
  RUNTIME DESTINATION ${BIN_INSTALL_DIR}
  INCLUDES DESTINATION ${INCLUDE_INSTALL_DIR}
)

install(EXPORT ${PROJECT_NAME}Targets
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ICPCUDA::
  DESTINATION
    ${INSTALL_CMAKE_DIR}
)


install(
  FILES
    ${PUBLIC_HEADERS}
  DESTINATION
    include
  COMPONENT
    Devel
)


install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}Config.cmake
  DESTINATION
   ${INSTALL_CMAKE_DIR}
  COMPONENT
    Devel
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${PROJECT_CMAKE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKE_DIR}
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}ConfigVersion.cmake
  VERSION ${ICPCUDA_VERSION}
  COMPATIBILITY SameMajorVersion )
  
export(EXPORT ${PROJECT_NAME}Targets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}/${TARGET_NAME}Targets.cmake"
  NAMESPACE ICPCUDA::
)

###################### Add uninstall target ############################
if("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    ADD_CUSTOM_TARGET(uninstall
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake)
endif()

