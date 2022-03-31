#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "cvtoolkit/nn/maskrcnn.hpp"

namespace cvt
{

MaskRCNN_OpenCV::MaskRCNN_OpenCV(const Settings& settings)
    : NeuralNetwork(settings)
{
    m_initialized = load(settings.modelRootDir, settings.device);
}

void MaskRCNN_OpenCV::Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outputs, 
                const PreprocessData& preprocessData,
                const PostprocessData& postprocessData)
{

}

void MaskRCNN_OpenCV::preprocess(const std::vector<cv::Mat>& images, 
                    const PreprocessData& preprocessData)
{
    
}

void MaskRCNN_OpenCV::postprocess(const cv::Mat& outLayer, std::vector<cv::Mat>& outputs, 
                    const PostprocessData& postprocessData) const
{
    
}



std::shared_ptr<NeuralNetwork> createMaskRCNN(const NeuralNetwork::Settings& settings)
{
    switch (settings.engine)
    {

    case NeuralNetwork::Engine::OpenCV:
        return std::make_shared<MaskRCNN_OpenCV>(settings);

    default:
        return nullptr;
    }
}

}