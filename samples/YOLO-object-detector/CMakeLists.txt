cmake_minimum_required( VERSION 3.10 )
project( YOLO-object-detector )

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

add_example( yolo-object-detector )

# if (NOT EXISTS "${CMAKE_SOURCE_DIR}/data")
#     add_custom_target(build-time-make-directory ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/data)
# endif()

# if (NOT EXISTS "${CMAKE_SOURCE_DIR}/data/yolov3_default")
#     add_custom_target(build-time-make-directory-2 ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/data/yolov3_default)
#     file(DOWNLOAD https://pjreddie.com/media/files/yolov3.weights ${CMAKE_SOURCE_DIR}/data/yolov3_default/yolo.weights)
# endif()