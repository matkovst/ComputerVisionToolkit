cmake_minimum_required( VERSION 3.10 )
project( cvtoolkit )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/cmake/FindOpenCV.cmake )

file(GLOB_RECURSE CVTOOLKIT_INCLUDES "*.hpp")
file(GLOB_RECURSE CVTOOLKIT_SOURCES "*.cpp")

add_library( ${PROJECT_NAME} ${CVTOOLKIT_INCLUDES} ${CVTOOLKIT_SOURCES} )

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 17)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

target_include_directories( ${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include )
target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )
if (${ONNXRUNTIME_FOUND})
    target_include_directories( ${PROJECT_NAME} PUBLIC ${onnxruntime_INCLUDE_DIRS} )
endif()

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
if (UNIX)
    target_link_libraries( ${PROJECT_NAME} pthread )
endif()
if (${TORCH_FOUND})
    target_link_libraries( ${PROJECT_NAME} ${TORCH_LIBRARIES} )
endif()
if (${ONNXRUNTIME_FOUND})
    target_link_libraries( ${PROJECT_NAME} ${onnxruntime_LIBRARIES} )
endif()

# Tests
if (BUILD_TESTS)
    add_subdirectory(lib.cvtoolkit.tests)
endif()

# Install
install( TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/ )

# add_custom_command(TARGET ${PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/lib/include
#     COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/lib/include
# )
