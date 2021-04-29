cmake_minimum_required( VERSION 2.8 )
project( ComputerVisionToolkit )

# Connect lib.cvtoolkit
set( CVTOOLKIT_DIR ${CMAKE_SOURCE_DIR}/lib.cvtoolkit )
set( CVTOOLKIT_INCLUDES ${CVTOOLKIT_DIR}/include )
add_subdirectory( lib.cvtoolkit )

# Connect samples
add_subdirectory( samples )
