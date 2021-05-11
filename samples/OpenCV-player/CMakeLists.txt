cmake_minimum_required( VERSION 3.10 )
project( opencv-player )

# Include OpenCV
include( ${CMAKE_SOURCE_DIR}/FindOpenCV.txt )

add_executable( ${PROJECT_NAME} main.cpp )

set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD 14)
set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
#set(TARGET ${PROJECT_NAME} PROPERTY CMAKE_CXX_EXTENSIONS OFF)

target_include_directories( ${PROJECT_NAME} PUBLIC ${CVTOOLKIT_INCLUDES} )
target_include_directories( ${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} )

target_link_libraries( ${PROJECT_NAME} cvtoolkit )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

install(TARGETS ${PROJECT_NAME}
        CONFIGURATIONS Release
        RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin)
