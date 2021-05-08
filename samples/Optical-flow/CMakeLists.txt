cmake_minimum_required( VERSION 2.8 )
project( optical-flow )

find_package( OpenCV REQUIRED )

MACRO(add_example NAME)
    add_executable( ${NAME} ${NAME}.cpp )

    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD 11)
    set(TARGET ${NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
    #set(TARGET ${NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

    target_include_directories( ${NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
    target_include_directories( ${NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

    target_link_libraries( ${NAME} cvtoolkit )
    target_link_libraries( ${NAME} ${OpenCV_LIBS} )
ENDMACRO()

add_example( lk )
add_example( fback )
add_example( dis )

message(STATUS "CVTOOLKIT_DIR ${CVTOOLKIT_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")