cmake_minimum_required( VERSION 3.10 )
project( optical-flow )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/cmake/FindOpenCV.cmake )

MACRO(add_example NAME)
    add_executable( ${NAME} ${NAME}.cpp )

    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD 17)
    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
    #set(TARGET ${NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

    target_include_directories( ${NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
    target_include_directories( ${NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

    target_link_libraries( ${NAME} cvtoolkit )
    target_link_libraries( ${NAME} ${OpenCV_LIBS} )

    install(TARGETS ${NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
ENDMACRO()

add_example( lk )
add_example( fback )
add_example( dis )
if( ENABLE_OPENCV_CUDA AND OpenCV_CUDA_VERSION )
    add_example( lk_cuda )
endif()
if( HAVE_NVIDIA_HW_OPTFLOW_SUPPORT AND OpenCV_VERSION_MAJOR GREATER_EQUAL 4 AND OpenCV_VERSION_MINOR GREATER_EQUAL 1 AND OpenCV_VERSION_PATCH GREATER_EQUAL 1 )
    add_example( nvidia_hw_optflow_1_0 )
endif()
if( HAVE_NVIDIA_HW_OPTFLOW_SUPPORT AND OpenCV_VERSION_MAJOR GREATER_EQUAL 4 AND OpenCV_VERSION_MINOR GREATER_EQUAL 5 AND OpenCV_VERSION_PATCH GREATER_EQUAL 2 )
    add_example( nvidia_hw_optflow_2_0 )
endif()