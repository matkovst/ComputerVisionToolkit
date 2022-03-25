/**
 * @file nn_builder.hpp
 *
 * @brief This file contains logic for building arbitrary NNs.
 *
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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

    struct PreprocessData
    {
        cv::Size size;
        int colorConvCode;
        double scale;
        cv::Scalar mean;
        cv::Scalar std;

        PreprocessData(cv::Size size, 
                        int colorConvCode, 
                        double scale, 
                        cv::Scalar mean, 
                        cv::Scalar std)
            : size(size)
            , colorConvCode(colorConvCode)
            , scale(scale)
            , mean(mean)
            , std(std)
        {}

        inline bool doConvertColor() const noexcept
        {
            return (-1 != colorConvCode);
        }

        inline bool doNormalize() const noexcept
        {
            if (0.0 == mean[0] && 0.0 == mean[1] && 0.0 == mean[2] 
                && 1.0 == std[0] && 1.0 == std[1] && 1.0 == std[2])
                return false;
            return true;
        }
    };

    struct PostprocessData
    {
        bool doSoftmax { false };

        PostprocessData(bool doSoftmax)
            : doSoftmax(doSoftmax)
        {}
    };

public:

    NeuralNetwork(const InitializeData& initializeData);

    virtual ~NeuralNetwork() = default;

    /*! @brief Performes nn inference.

        @param images input images.
        @param outs output images.
        @param preprocessData preprocessing rules for input.
    */
    virtual void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs, 
                        const PreprocessData& preprocessData = {cv::Size(), 
                                                                cv::COLOR_BGR2RGB, 
                                                                1.0, 
                                                                cv::Scalar(0.0, 0.0, 0.0), 
                                                                cv::Scalar(1.0, 1.0, 1.0)},
                        const PostprocessData& postprocessData = {false}) = 0;

    /*! @brief Performes nn inference. Overloaded member function.
    */
    virtual void Infer(const cv::Mat& image, cv::Mat& out, 
                        const PreprocessData& preprocessData = {cv::Size(), 
                                                                cv::COLOR_BGR2RGB, 
                                                                1.0, 
                                                                cv::Scalar(0.0, 0.0, 0.0), 
                                                                cv::Scalar(1.0, 1.0, 1.0)},
                        const PostprocessData& postprocessData = {false});

    /*! @brief Indicates whether model loaded successfully.
    */
    virtual inline bool initialized() const noexcept { return m_initialized; };

protected:
    const InitializeData m_initializeData;
    bool m_initialized { false }; // = false if model failed to load
};

}