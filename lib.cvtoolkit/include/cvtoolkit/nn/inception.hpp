#pragma once

#include <memory>

#include "nn.hpp"

namespace cvt
{

class Inception_OpenCV final : public NeuralNetwork, private IOpenCVLoader
{
public:

    Inception_OpenCV(const InitializeData& initializeData);

    ~Inception_OpenCV() = default;

    void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outputs, 
                const PreprocessData& preprocessData,
                const PostprocessData& postprocessData) override;

private:

    void preprocess(const std::vector<cv::Mat>& images, const PreprocessData& preprocessData);

    void postprocess(const cv::Mat& outLayer, std::vector<cv::Mat>& outputs,
                    const PostprocessData& postprocessData) const;

private:
    cv::Mat m_blob;
};


/** @brief Creates Inception for specified settings
 */
std::shared_ptr<NeuralNetwork> createInception(const NeuralNetwork::InitializeData& initializeData);

}