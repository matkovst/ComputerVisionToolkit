[![GitHub release](https://img.shields.io/github/v/release/matkovst/ComputerVisionToolkit?include_prereleases)](https://github.com/matkovst/ComputerVisionToolkit/releases/tag/v1.0-alpha)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://matkovst.github.io/ComputerVisionToolkit/index.html)

# ComputerVisionToolkit
My personal cross-platform toolkit for doing Image Processing and Intelligent Video Analytics experiments. It comes with:
- *lib.cvtoolkit* library encapsulating the most common and helpful functions
- a set of *samples* showing how to implement various computer vision tasks

## Requirements
- **Windows** / **Linux**
- **C++11** ( or higher ) compiler
- **CMake** >= 3.10
- **OpenCV** >= 3.0:
    - ( optional ) with CUDA support

## Build & Install
The whole project can be built with CMake

```bash
export OpenCV_DIR=path/to/opencv
mkdir build && cd build
cmake ..
make
make install
```

## Run
Choose desired task in *bin/* directory (like *laplacian*) and just execute it.
```bash
cd bin
./laplacian -?
```

