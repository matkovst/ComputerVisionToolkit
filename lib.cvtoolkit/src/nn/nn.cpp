#include "cvtoolkit/nn/nn.hpp"

namespace cvt
{

NeuralNetwork::NeuralNetwork(const InitializeData& initializeData)
    : m_initializeData(initializeData)
{}

void NeuralNetwork::Infer(const cv::Mat& image, cv::Mat& out, 
                            const PreprocessData& preprocessData,
                            const PostprocessData& postprocessData)
{
    if ( !out.empty() )
        out.release(); // decrement the ref counter

    std::vector<cv::Mat> outs;
    Infer({image}, outs, preprocessData, postprocessData);
    out = outs.at(0);
}

}