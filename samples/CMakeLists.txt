cmake_minimum_required( VERSION 2.8 )
project( samples )

# Connect samples
add_subdirectory( OpenCV-player )
add_dependencies( opencv-player cvtoolkit )