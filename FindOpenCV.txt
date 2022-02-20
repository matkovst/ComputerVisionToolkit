include_guard()

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Check OpenCV-CUDA support
if (ENABLE_OPENCV_CUDA)
    if(OpenCV_CUDA_VERSION)
        add_definitions( -DHAVE_OPENCV_CUDA )
        message(STATUS "OpenCV_CUDA_VERSION: ${OpenCV_CUDA_VERSION}")
    endif()

    # Check NVidia Hardware-Accelerated Optical Flow support
    if(OpenCV_CUDA_VERSION)
        if(OpenCV_VERSION_MAJOR GREATER_EQUAL 4 AND OpenCV_VERSION_MINOR GREATER_EQUAL 1 AND OpenCV_VERSION_PATCH GREATER_EQUAL 1)
            add_definitions( -DHAVE_NVIDIA_HW_OPTFLOW_SUPPORT )
            message(STATUS "Enabled NVidia Hardware-Accelerated Optical Flow support")
        endif()
    endif()
endif()