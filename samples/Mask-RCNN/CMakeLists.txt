cmake_minimum_required( VERSION 2.8 )
project( mask-rcnn )

find_package( OpenCV REQUIRED )

if(OpenCV_CUDA_VERSION)
    add_definitions( -DHAVE_OPENCV_CUDA )
    message(STATUS "OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
endif()

add_executable( ${PROJECT_NAME} main.cpp )

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 11)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

target_include_directories( ${PROJECT_NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( ${PROJECT_NAME} cvtoolkit )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

message(STATUS "CVTOOLKIT_DIR ${CVTOOLKIT_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
if(OpenCV_CUDA_VERSION)
    message(STATUS "OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
endif()
