[![GitHub release](https://img.shields.io/github/v/release/matkovst/ComputerVisionToolkit?include_prereleases)](https://github.com/matkovst/ComputerVisionToolkit/releases/tag/v1.0-alpha)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://matkovst.github.io/ComputerVisionToolkit/index.html)

# ComputerVisionToolkit
My personal toolkit for doing CV experiments. It comes with:
- *lib.cvtoolkit* library encapsulating the most common and helpful functions
- a set of *samples* showing how to implement various computer vision tasks

## Requirements
- **C++11** ( or higher ) compiler
- **CMake** >= 3.10
- **OpenCV** >= 3.0
- ( optional ) OpenCV with CUDA support

## Build
The whole project can be built with CMake

```bash
export OpenCV_DIR=path/to/opencv
mkdir build && cd build
cmake ..
make
make install
```

## Run
```bash
cd bin
./opencv-player -?
```
