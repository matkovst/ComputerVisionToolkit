cmake_minimum_required( VERSION 3.10 )
project( lib.cvtoolkit.tests )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/cmake/FindOpenCV.cmake )

add_executable( ${PROJECT_NAME} cvtoolkit_unittests.cpp )
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 17)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )
if (${ONNXRUNTIME_FOUND})
    target_include_directories( ${PROJECT_NAME} PUBLIC ${onnxruntime_INCLUDE_DIRS} )
endif()

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} cvtoolkit )
if (UNIX)
    target_link_libraries( ${PROJECT_NAME} pthread )
endif()
if (${TORCH_FOUND})
    target_link_libraries( ${PROJECT_NAME} ${TORCH_LIBRARIES} )
endif()
if (${ONNXRUNTIME_FOUND})
    target_link_libraries( ${PROJECT_NAME} ${onnxruntime_LIBRARIES} )
endif()
