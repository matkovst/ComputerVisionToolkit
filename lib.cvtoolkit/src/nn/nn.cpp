#include "cvtoolkit/settings.hpp"
#include "cvtoolkit/nn/nn.hpp"

namespace cvt
{

NeuralNetwork::NeuralNetwork(const InitializeData& initializeData)
    : m_initializeData(initializeData)
{
    const auto labelsFile = getFileWithExt(m_initializeData.modelRootDir, ".txt");
    if (labelsFile)
        m_labels = loadLabelsMap(labelsFile.value().string());
}

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

const std::string& NeuralNetwork::label(size_t id) const noexcept
{
    if (m_labels.size() < (id-1))
        return m_dummyLabel;
    return m_labels.at(id);
}



bool IOpenCVLoader::load(const fs::path& modelDataPath, int device)
{
    /* Check TensorFlow */
    const auto pbOpt = getFileWithExt(modelDataPath, ".pb");
    if (pbOpt)
    {
        try
        {
            const auto pbtxtOpt = getFileWithExt(modelDataPath, ".pbtxt");
            m_model = cv::dnn::readNetFromTensorflow(pbOpt.value().string(), 
                            pbtxtOpt ? pbtxtOpt.value().string() : cv::String());

            const auto layerNames = m_model.getLayerNames();
            // if ( !layerNames.empty() )
            //     m_outputName = layerNames.at(layerNames.size() - 1);

            switch (device)
            {
            case NeuralNetwork::Device::Gpu:
#if HAVE_OPENCV_CUDA && CV_MAJOR_VERSION > 3 && CV_MINOR_VERSION > 1
                {
                    m_model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    m_model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                }
                break;
#else
                {
                    m_model.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                    m_model.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
                }
                break;
#endif

            case NeuralNetwork::Device::Cpu:
            default:
                {
                    m_model.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
                    m_model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                }
                break;
            }
        
            return true;
        }
        catch( const cv::Exception& e )
        {
            std::cerr << e.what() << std::endl;
        }
        catch( const std::exception& e )
        {
            std::cerr << e.what() << std::endl;
        }
    }

    return false;
}

}