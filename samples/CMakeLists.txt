cmake_minimum_required( VERSION 3.10 )
project( samples )

# Connect samples
add_subdirectory( OpenCV-player )
add_dependencies( opencv-player cvtoolkit )

add_subdirectory( Mask-RCNN )
add_dependencies( mask-rcnn cvtoolkit )

add_subdirectory( JSON-parser )
add_dependencies( json-parser cvtoolkit )

add_subdirectory( Optical-flow )

add_subdirectory( Background-subtraction )

add_subdirectory( Image-processing )
