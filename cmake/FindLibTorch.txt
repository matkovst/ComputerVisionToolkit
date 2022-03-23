include_guard()

if (Torch_DIR)

    find_package(Torch)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    
    if(TORCH_FOUND)
        add_definitions( -DTORCH_FOUND )
        include_directories( ${TORCH_INCLUDE_DIRS} )
        message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
        message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
    else()
        message(STATUS "Could not find LibTorch")
    endif()

endif()