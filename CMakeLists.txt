cmake_minimum_required( VERSION 3.10 )
project( ComputerVisionToolkit )

if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RUN_TESTS "Run CTest" OFF)
if (RUN_TESTS)
  enable_testing()
endif()

# Include 3rdparty
include_directories( ${CMAKE_SOURCE_DIR}/3rdparty )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/cmake/FindOpenCV.cmake )

# (optional) LibTorch
include( ${CMAKE_SOURCE_DIR}/cmake/FindLibTorch.cmake )

# (optional) OnnxRuntime
include( ${CMAKE_SOURCE_DIR}/cmake/FindOnnxRuntime.cmake )

# Connect lib.cvtoolkit
set( CVTOOLKIT_DIR ${CMAKE_SOURCE_DIR}/lib.cvtoolkit )
set( CVTOOLKIT_INCLUDES ${CVTOOLKIT_DIR}/include )
add_subdirectory( lib.cvtoolkit )

# Connect samples
add_subdirectory( samples )
