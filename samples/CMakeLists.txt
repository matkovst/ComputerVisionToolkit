cmake_minimum_required( VERSION 3.10 )
project( samples )

# Connect samples
add_subdirectory( ___Sample-template___ )

add_subdirectory( OpenCV-player )
add_dependencies( opencv-player cvtoolkit )

add_subdirectory( Mask-RCNN )
add_dependencies( mask-rcnn cvtoolkit )

add_subdirectory( JSON-parser )
add_dependencies( json-parser cvtoolkit )

add_subdirectory( Optical-flow )

add_subdirectory( Background-subtraction )

add_subdirectory( Custom-pipeline )

add_subdirectory( Image-processing )

add_subdirectory( Motion-detector )

add_subdirectory( YOLO-object-detector )

add_subdirectory( Inception-image-classifier )

add_subdirectory( Illumination-estimation )

add_subdirectory( Monodepth )

add_subdirectory( Shadow-removal )

add_subdirectory( EfficientNet )