cmake_minimum_required( VERSION 3.10 )
project( Monodepth )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/cmake/FindOpenCV.cmake )
include( ${CMAKE_SOURCE_DIR}/cmake/FindLibTorch.cmake )

MACRO(add_example_with_torch NAME)
    add_executable( ${NAME} ${NAME}.cpp )

    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD 17)
    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
    #set(TARGET ${NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

    target_include_directories( ${NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
    target_include_directories( ${NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

    target_link_libraries(${NAME} 
        cvtoolkit
        )
    target_link_libraries(${NAME} 
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        )

    # The following code block is suggested to be used on Windows.
    # According to https://github.com/pytorch/pytorch/issues/25457,
    # the DLLs need to be copied to avoid memory errors.
    if (MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${NAME}
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        $<TARGET_FILE_DIR:${NAME}>)
    endif (MSVC)


    # Install
    install(TARGETS ${NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)

    # if (MSVC)
    #     file(GLOB OPENCV_DLLS "${OpenCV_DIR}/../bin/*.dll")
    #     add_custom_command(TARGET ${NAME}
    #                     POST_BUILD
    #                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #                     ${OPENCV_DLLS}
    #                     ${CMAKE_SOURCE_DIR}/bin)
    #     file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    #     add_custom_command(TARGET ${NAME}
    #                     POST_BUILD
    #                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
    #                     ${TORCH_DLLS}
    #                     ${CMAKE_SOURCE_DIR}/bin)
    # endif (MSVC)

ENDMACRO()

if (${TORCH_FOUND})
    add_example_with_torch( dpt-monodepth )
endif()