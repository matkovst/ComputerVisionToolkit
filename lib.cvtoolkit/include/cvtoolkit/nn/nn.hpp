/**
 * @file nn_builder.hpp
 *
 * @brief This file contains logic for building arbitrary NNs.
 *
 */

#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "cvtoolkit/image.hpp"
#include "cvtoolkit/nn/utils.hpp"

namespace cvt
{

/*! @brief The base class for all neural networks.
*/
class NeuralNetwork
{
public:

    enum Engine
    {
        OpenCV,
        Torch,
        Onnx
    };

    enum Device
    {
        Cpu,
        Gpu
    };

    struct InitializeData
    {
        std::string modelPath { "" };
        std::string modelConfigPath { "" };
        std::string modelClassesPath { "" };
        cv::Size modelInputSize;
        int engine;
        int device;

        InitializeData(const std::string& modelPath, const std::string& modelConfigPath, 
                        const std::string& modelClassesPath = "", 
                        cv::Size modelInputSize = cv::Size(), 
                        int engine = Engine::OpenCV, 
                        int device = Device::Cpu)
            : modelPath(modelPath)
            , modelConfigPath(modelConfigPath)
            , modelClassesPath(modelClassesPath)
            , modelInputSize(modelInputSize)
            , engine(engine)
            , device(device)
        {}
    };

    NeuralNetwork(const InitializeData& initializeData);

    virtual ~NeuralNetwork() = default;

    /*! @brief Performes nn inference.

        @param images input images.
        @param outs output images.
    */
    virtual void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs) = 0;

    /*! @brief Performes nn inference. Overloaded member function.
    */
    virtual void Infer(const cv::Mat& image, cv::Mat& out);

protected:
    const InitializeData m_initializeData;
};

}