cmake_minimum_required( VERSION 2.8 )
project( Movement-detector )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/FindOpenCV.txt )

MACRO(add_example NAME)
    add_executable( ${NAME} ${NAME}.cpp )

    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD 14)
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

add_example( simple-motion-detector )