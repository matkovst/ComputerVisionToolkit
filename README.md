# ComputerVisionToolkit
My personal toolkit for doing CV experiments. It comes with:
- *lib.cvtoolkit* library encapsulating the most common and helpful functions
- a set of *samples* showing how to implement various computer vision tasks

## Requirements
- **C++11** ( or higher ) compiler
- **CMake** >= 2.8
- **OpenCV** >= 3.0

## Build
The whole project can be built with CMake

```bash
export OpenCV_DIR=path/to/opencv
cd build && mkdir build
cmake ..
make
make install
```
