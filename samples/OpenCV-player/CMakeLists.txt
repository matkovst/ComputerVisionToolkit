cmake_minimum_required( VERSION 2.8 )
project( opencv-player )

find_package( OpenCV REQUIRED )

add_executable( ${PROJECT_NAME} main.cpp )

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 11)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
#set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

target_include_directories( ${PROJECT_NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( ${PROJECT_NAME} cvtoolkit )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

message(STATUS "CVTOOLKIT_DIR ${CVTOOLKIT_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
