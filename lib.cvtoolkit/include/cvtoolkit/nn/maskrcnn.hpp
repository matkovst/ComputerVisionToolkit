#pragma once

#include <memory>

#include "nn.hpp"

namespace cvt
{

class MaskRCNN_OpenCV final : public NeuralNetwork, private IOpenCVLoader
{
public:

    MaskRCNN_OpenCV(const Settings& settings);

    ~MaskRCNN_OpenCV() = default;

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


/** @brief Creates Mask R-CNN for specified settings
 */
std::shared_ptr<NeuralNetwork> createMaskRCNN(const NeuralNetwork::Settings& settings);

}