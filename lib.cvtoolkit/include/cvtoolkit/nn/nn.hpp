/**
 * @file nn.hpp
 *
 * @brief This file contains logic for building arbitrary NNs.
 *
 */

#pragma once

#include <map>
#include <string>
#include <vector>
#include <functional>
#include <filesystem>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/nn/utils.hpp"

namespace fs = std::filesystem;

namespace cvt
{

/*! @brief The base class for all neural networks.
*/
class NeuralNetwork
{

public:

    using Labels = std::map<int, std::string>;

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
        fs::path modelRootDir { "" };
        std::string modelPath { "" };
        std::string modelConfigPath { "" };
        std::string modelClassesPath { "" };
        cv::Size modelInputSize { cv::Size() };
        int engine { -1 };
        int device { -1 };

        InitializeData() = default;

        InitializeData(const fs::path& modelRootDir,
                        const std::string& modelPath, 
                        const std::string& modelConfigPath, 
                        const std::string& modelClassesPath = "", 
                        cv::Size modelInputSize = cv::Size(), 
                        int engine = Engine::OpenCV, 
                        int device = Device::Cpu)
            : modelRootDir(modelRootDir)
            , modelPath(modelPath)
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

    explicit NeuralNetwork(const InitializeData& initializeData);

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
    virtual inline bool initialized() const noexcept { return m_initialized; }

    virtual const std::string& label(size_t id) const noexcept;

protected:
    const InitializeData m_initializeData;
    bool m_initialized { false }; // = false if model failed to load

private:
    Labels m_labels;
    const std::string m_dummyLabel { "" };
};


/*! @brief The interface for loading and holding OpenCV model.
*/
class IOpenCVLoader
{
public:

    IOpenCVLoader() = default;

    virtual ~IOpenCVLoader() = default;

    /*! @brief Load cv::dnn::Net.

        @param modelDataPath path to model directory which contains all necessary files.
        @param device device type.

        @return Whether loading is successful
    */
    bool load(const fs::path& modelDataPath, int device);

protected:
    cv::dnn::Net m_model;
    std::string m_inputName;
    std::string m_outputName;
};

}