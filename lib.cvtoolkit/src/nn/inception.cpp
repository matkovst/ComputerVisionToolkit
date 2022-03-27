#include <cassert>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/nn/inception.hpp"

namespace cvt
{

Inception_OpenCV::Inception_OpenCV(const InitializeData& initializeData)
    : NeuralNetwork(initializeData)
{
    m_initialized = load(initializeData.modelRootDir, initializeData.device);
}

void Inception_OpenCV::Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outputs, 
                const PreprocessData& preprocessData,
                const PostprocessData& postprocessData)
{
    preprocess(images, preprocessData);

    const cv::Mat outLayer = m_model.forward(m_outputName);

    postprocess(outLayer, outputs, postprocessData);
}

void Inception_OpenCV::preprocess(const std::vector<cv::Mat>& images, 
                    const PreprocessData& preprocessData)
{
    cv::dnn::blobFromImages(images, m_blob, preprocessData.scale, 
                        cv::Size(224, 224), cv::Scalar(), true, false);
    m_blob -= 117.0;

    m_model.setInput(m_blob, m_inputName);
}

void Inception_OpenCV::postprocess(const cv::Mat& outLayer, std::vector<cv::Mat>& outputs, 
                    const PostprocessData& postprocessData) const
{
    const auto nImages = outLayer.rows;
    const auto nLabels = outLayer.cols;
    for (int i = 0; i < nImages; ++i)
        outputs.emplace_back(outLayer.row(i));
}



std::shared_ptr<NeuralNetwork> createInception(const NeuralNetwork::InitializeData& initializeData)
{
    switch (initializeData.engine)
    {

    case NeuralNetwork::Engine::OpenCV:
        return std::make_shared<Inception_OpenCV>(initializeData);

    default:
        return nullptr;
    }
}

}