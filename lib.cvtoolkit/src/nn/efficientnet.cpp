#include <opencv2/imgproc.hpp>
#include "cvtoolkit/nn/efficientnet.hpp"

namespace cvt
{

#ifdef TORCH_FOUND

EfficientNet_Torch::EfficientNet_Torch(const InitializeData& initializeData)
    : NeuralNetwork(initializeData)
{
    /* Check CUDA availability */
    m_device = (NeuralNetwork::Gpu == initializeData.device) ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
    if (torch::kCUDA == m_device)
    {
        if (torch::cuda::is_available())
        {
            m_device = torch::DeviceType::CUDA;
            std::cout << "[EfficientNet_Torch] CUDA is ok! Switching calculations to GPU ..." << std::endl;
        }
        else
        {
            m_device = torch::DeviceType::CPU;
            std::cout << "[EfficientNet_Torch] LibTorch CUDA is not available! Switching calculations to CPU ..." << std::endl;
        }
    }
    else
    {
        std::cout << "[EfficientNet_Torch] CUDA manually disabled! Switching calculations to CPU ..." << std::endl;
    }

    /* Load model */
    if ( !initializeData.modelPath.empty() )
    {
        try 
        {
            cv::TickMeter loadModelTm;
            loadModelTm.start();

            // De-serialize ScriptModule from file
            m_model = torch::jit::load(initializeData.modelPath, m_device);
            m_model.to(m_device);
            m_model.eval();
            m_modelInitialized = true;

            loadModelTm.stop();
            std::cout << "[EfficientNet_Torch] Model loading took " << loadModelTm.getAvgTimeMilli() << "ms" << std::endl;
        }
        catch (const torch::Error& e) 
        {
            std::cerr << "[EfficientNet_Torch] Error loading the model:\n" << e.what();
        }
    }
    else
    {
        std::cout << "[EfficientNet_Torch] Error loading the model: Model path is empty " << std::endl;
    }

    /* Warmup model */
    if (m_modelInitialized)
    {
        try
        {
            cv::TickMeter warmupTm;
            warmupTm.start();

            /* Make input */
            const cv::Size inputSize = (initializeData.modelInputSize.empty()) 
                                    ? cv::Size(224, 224) 
                                    : initializeData.modelInputSize;
            const cv::Mat ones = cv::Mat::zeros(inputSize, CV_8UC3) + cv::Scalar::all(1);
            cvt::Image out;

            Infer(ones, out);

            cv::Mat pred;
            imageToMat(out, pred, true);
                
            warmupTm.stop();

            std::cout << "[EfficientNet_Torch] Model warming up took " 
                      << warmupTm.getAvgTimeMilli() << "ms" << std::endl;
        }
        catch(const torch::Error& e)
        {
            std::cout << "[EfficientNet_Torch] Error warming up the model:\n" << e.what();
        }
    }
}

void EfficientNet_Torch::Infer(const Image& image, Image& out)
{
    Images outs;
    Infer({image}, outs);
    out = outs.at(0);
}

void EfficientNet_Torch::Infer(const Images& images, Images& outs)
{
    if ( !m_modelInitialized )
        return;

    preprocess(images, m_inputs);

    const auto y = m_model.forward(m_inputs);
    if (y.isTuple())
        m_outs = y.toTuple()->elements()[0].toTensor();
    else if (y.isTensor())
        m_outs = y.toTensor().detach().clone();

    postprocess(m_outs, outs);
}

void EfficientNet_Torch::preprocess(const Images& images, std::vector<torch::jit::IValue>& out)
{
    if ( !out.empty() )
        out.clear();

    std::vector<torch::Tensor> inputTensor(images.size());
    for (int i = 0; i < images.size(); ++i)
    {
        const auto& image = images.at(i);
        auto& tensorImage = inputTensor.at(i);

        cv::Mat matImage, matImage32f;
        imageToMat(image, matImage, true);

        // Resize
        if (!m_initializeData.modelInputSize.empty() && m_initializeData.modelInputSize != matImage.size())
            cv::resize(matImage, matImage, m_initializeData.modelInputSize, 0.0, 0.0, cv::INTER_CUBIC);

        // Convert color
        cv::cvtColor(matImage, matImage, cv::COLOR_BGR2RGB);

        // Normalize
        matImage.convertTo(matImage32f, CV_32FC3, 0.003921569);
        matImage32f = (matImage32f - EfficientNet::mean) / EfficientNet::std;

        // Convert to torch::Tensor
        matToTensor(matImage32f, tensorImage);

        tensorImage = tensorImage.permute({2,0,1}); // HWC -> CHW
        tensorImage = tensorImage.toType(torch::kFloat);
        tensorImage.unsqueeze_(0); // CHW -> NCHW
    }

    m_inputTensor = torch::cat(inputTensor);
    out.emplace_back(m_inputTensor.to(m_device));
}

void EfficientNet_Torch::postprocess(const torch::Tensor &in, Images& out)
{
    if ( !out.empty() )
        out.clear();
    
    /* Sotfmax & make output */
    const auto sizes = in.sizes();
    const int nImages = static_cast<int>(sizes[0]);
    const int nLabels = static_cast<int>(sizes[1]);
    for (int i = 0; i < nImages; ++i)
    {
        const auto& batch = in.select(0, i);
        const auto softmaxed = torch::softmax(batch, 0);
        const cv::Mat outMat = cv::Mat(1, nLabels, CV_32F, softmaxed.data_ptr());
        out.emplace_back(outMat.clone());
    }

    // /* Get top K predictions */
    // const int k = 5;
    // const auto topk = torch::topk(in, k);
    // const auto topkIdx = std::get<1>(topk).squeeze(0);
    // std::cout << torch::softmax(in, 1).squeeze(0).index_select(0, topkIdx) << std::endl;
}

#endif

std::shared_ptr<NeuralNetwork> createEfficientNet(const NeuralNetwork::InitializeData& initializeData)
{
    switch (initializeData.engine)
    {
    case NeuralNetwork::Engine::OpenCV: // TODO
        return nullptr;
    case NeuralNetwork::Engine::Torch:
#ifdef TORCH_FOUND
        return std::make_shared<EfficientNet_Torch>(initializeData);
#else
        return nullptr;
#endif
    default:
        return nullptr;
    }
}

}