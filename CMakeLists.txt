cmake_minimum_required( VERSION 3.10 )
project( ComputerVisionToolkit )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/FindOpenCV.txt )

# Connect lib.cvtoolkit
set( CVTOOLKIT_DIR ${CMAKE_SOURCE_DIR}/lib.cvtoolkit )
set( CVTOOLKIT_INCLUDES ${CVTOOLKIT_DIR}/include )
add_subdirectory( lib.cvtoolkit )

# Connect samples
add_subdirectory( samples )
