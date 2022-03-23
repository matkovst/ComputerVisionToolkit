[![GitHub release](https://img.shields.io/github/v/release/matkovst/ComputerVisionToolkit?include_prereleases)](https://github.com/matkovst/ComputerVisionToolkit/releases/tag/v1.0-alpha)
[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://matkovst.github.io/ComputerVisionToolkit/index.html)

# ComputerVisionToolkit
My personal cross-platform toolkit for doing Image Processing and Intelligent Video Analytics experiments. It comes with:
- *lib.cvtoolkit* library encapsulating the most common and helpful functions
- a set of *samples* showing how to implement various computer vision tasks

    <details>
    <summary>Samples list</summary>
    <br>
        <ul>
            <li>Background subtraction (exponential forgetting, KNN, MOG2)</li>
            <li>Illumination estimation</li>
            <li>Image processing (color filters, image derivatives, smoothing, histograms)</li>
            <li>Image classification (InceptionV3, EfficientNet)</li>
            <li>Semantic segmentation (Mask R-CNN)</li>
            <li>Object detection (YOLO)</li>
            <li>Monodepth</li>
            <li>Motion detector</li>
            <li>Optical flow</li>
            <li>Shadow removal</li>
        </ul>
    <br><br>
    </details>

## Requirements
- **Linux** / **Windows**
- **C++11** compiler for Linux / **VS 2019** for Windows
- **CMake** >= 3.18
- **OpenCV** >= 3.0:
    - (optional) with CUDA support
- (optional) **LibTorch** >= 1.8.2
- (optional) **ONNX Runtime** >= 1.10.0

## Build & Install
The whole project can be built with CMake

```bash
export OpenCV_DIR=path/to/opencv
# (optional) export ENABLE_OPENCV_CUDA=ON
# (optional) export Torch_DIR=path/to/libtorch

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

