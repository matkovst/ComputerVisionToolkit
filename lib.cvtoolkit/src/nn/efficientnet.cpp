#include <cassert>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include "cvtoolkit/utils.hpp"
#include "cvtoolkit/nn/efficientnet.hpp"

namespace cvt
{

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    T res = 1;
    for (int i = 0; i < v.size(); ++i)
        res *= v.at(i);
    return res;
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

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
            m_initialized = true;

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
    if (m_initialized)
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

            cv::Mat out;
            NeuralNetwork::Infer(ones, out);
                
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

void EfficientNet_Torch::Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs, 
                                const PreprocessData& preprocessData,
                                const PostprocessData& postprocessData)
{
    if ( !m_initialized )
        return;
    if ( !outs.empty() )
        outs.clear();

    m_inputs.clear();

    preprocess(images, m_inputs);

    const auto y = m_model.forward(m_inputs);
    if (y.isTuple())
        m_outs = y.toTuple()->elements()[0].toTensor();
    else if (y.isTensor())
        m_outs = y.toTensor().detach().clone();

    postprocess(m_outs, outs);
}

void EfficientNet_Torch::preprocess(const std::vector<cv::Mat>& images, std::vector<torch::jit::IValue>& out)
{
    std::vector<torch::Tensor> inputTensor(images.size());
    for (int i = 0; i < images.size(); ++i)
    {
        const auto& image = images.at(i);
        auto& tensorImage = inputTensor.at(i);

        cv::Mat matImage, matImage32f;

        // Resize
        const bool doResize = (!m_initializeData.modelInputSize.empty() 
                                && m_initializeData.modelInputSize != image.size());
        if (doResize)
            cv::resize(matImage, matImage, m_initializeData.modelInputSize, 0.0, 0.0, cv::INTER_CUBIC);
        else
            matImage = image.clone();

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

void EfficientNet_Torch::postprocess(const torch::Tensor &in, std::vector<cv::Mat>& outs)
{
    /* Sotfmax & make output */
    const auto sizes = in.sizes();
    const int nImages = static_cast<int>(sizes[0]);
    const int nLabels = static_cast<int>(sizes[1]);
    for (int i = 0; i < nImages; ++i)
    {
        const auto& batch = in.select(0, i);
        const auto softmaxed = torch::softmax(batch, 0);
        const cv::Mat outMat = cv::Mat(1, nLabels, CV_32F, softmaxed.data_ptr());
        outs.emplace_back(outMat.clone());
    }

    // /* Get top K predictions */
    // const int k = 5;
    // const auto topk = torch::topk(in, k);
    // const auto topkIdx = std::get<1>(topk).squeeze(0);
    // std::cout << torch::softmax(in, 1).squeeze(0).index_select(0, topkIdx) << std::endl;
}

#endif


#ifdef ONNXRUNTIME_FOUND

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}

EfficientNet_Onnx::EfficientNet_Onnx(const InitializeData& initializeData)
    : NeuralNetwork(initializeData)
{
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "EfficientNet_Onnx");
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetInterOpNumThreads(1);

    // Sets graph optimization level
    // Available levels are
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node
    // removals) ORT_ENABLE_EXTENDED -> To enable extended optimizations
    // (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible optimizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    /* Load model */
    if ( !initializeData.modelPath.empty() )
    {
        try 
        {
            cv::TickMeter loadModelTm;
            loadModelTm.start();

            m_session = Ort::Session(env, initializeData.modelPath.c_str(), sessionOptions);

            loadModelTm.stop();
            std::cout << "[EfficientNet_Onnx] Model loading took " << loadModelTm.getAvgTimeMilli() << "ms" << std::endl;


            /* Derive model detailed info */
            
            m_modelInfo.numInputNodes = m_session.GetInputCount();
            m_modelInfo.numOutputNodes = m_session.GetOutputCount();

            m_modelInfo.inputName = m_session.GetInputName(0, m_allocator);
            const Ort::TypeInfo inputTypeInfo = m_session.GetInputTypeInfo(0);
            const auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            m_modelInfo.inputType = inputTensorInfo.GetElementType();
            m_modelInfo.inputDims = inputTensorInfo.GetShape();

            m_modelInfo.outputName = m_session.GetOutputName(0, m_allocator);
            const Ort::TypeInfo outputTypeInfo = m_session.GetOutputTypeInfo(0);
            const auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
            m_modelInfo.outputType = outputTensorInfo.GetElementType();
            m_modelInfo.outputDims = outputTensorInfo.GetShape();

            if (m_modelInfo.inputDims.size() > 2)
                if (-1 != m_modelInfo.inputDims.at(1) && -1 != m_modelInfo.inputDims.at(2))
                    m_inputSize = {
                        static_cast<int>(m_modelInfo.inputDims.at(2)), 
                        static_cast<int>(m_modelInfo.inputDims.at(3))
                        };
            m_inputNames.emplace_back(m_modelInfo.inputName);
            m_outputNames.emplace_back(m_modelInfo.outputName);
            m_initialized = true;

            /* Print model detailed info */

            std::ostringstream infoSs;
            infoSs << std::endl << std::right
                << "[EfficientNet_Onnx] Model info: " << std::endl
                << "\t- Number of Input Nodes = " << m_modelInfo.numInputNodes << std::endl
                << "\t- Input Name = " << m_modelInfo.inputName << std::endl
                << "\t- Input Type = " << m_modelInfo.inputType << std::endl
                << "\t- Input Dimensions = " << m_modelInfo.inputDims << std::endl
                << "\t- Number of Output Nodes = " << m_modelInfo.numOutputNodes << std::endl
                << "\t- Output Name = " << m_modelInfo.outputName << std::endl
                << "\t- Output Type = " << m_modelInfo.outputType << std::endl
                << "\t- Output Dimensions = " << m_modelInfo.outputDims;
            std::cout << infoSs.str() << std::endl;
        }
        catch (const std::exception& e) 
        {
            std::cerr << "[EfficientNet_Onnx] Error loading the model:\n" << e.what();
        }
    }
    else
    {
        std::cout << "[EfficientNet_Onnx] Error loading the model: Model path is empty " << std::endl;
    }

    /* Warmup model */
    if (m_initialized)
    {
        try
        {
            cv::TickMeter warmupTm;
            warmupTm.start();

            /* Make input */
            const cv::Mat ones = cv::Mat::zeros(m_inputSize, CV_8UC3) + cv::Scalar::all(1);

            cv::Mat out;
            NeuralNetwork::Infer(ones, out);
                
            warmupTm.stop();

            std::cout << "[EfficientNet_Onnx] Model warming up took " 
                      << warmupTm.getAvgTimeMilli() << "ms" << std::endl;
        }
        catch(const std::exception& e)
        {
            std::cout << "[EfficientNet_Onnx] Error warming up the model:\n" << e.what();
        }
    }
}

void EfficientNet_Onnx::Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs, 
                                const PreprocessData& preprocessData,
                                const PostprocessData& postprocessData)
{
    if ( !m_initialized )
        return;
    if ( !outs.empty() )
        outs.clear();

    m_inputData.clear();
    m_outputData.clear();

    preprocess(images, m_inputData, m_outputData, preprocessData);

    m_session.Run(Ort::RunOptions{nullptr}, m_inputNames.data(),
                m_inputData.tensor.data(), 1, m_outputNames.data(),
                m_outputData.tensor.data(), 1);

    postprocess(m_outputData, outs, images.size(), postprocessData);
}

void EfficientNet_Onnx::preprocess(const std::vector<cv::Mat>& images, OrtData& inputData, OrtData& outputData, 
                                    const PreprocessData& preprocessData)
{
    assert(("[EfficientNet_Onnx][preprocess] Got 0 images.", !images.empty()));

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                                    OrtAllocatorType::OrtArenaAllocator, 
                                    OrtMemType::OrtMemTypeDefault);

    const size_t nImages = images.size();
    std::vector<int64_t> input0Dims, output0Dims;
    std::copy(m_modelInfo.inputDims.begin(), m_modelInfo.inputDims.end(),
              std::back_inserter(input0Dims));
    input0Dims[0] = nImages;
    std::copy(m_modelInfo.outputDims.begin(), m_modelInfo.outputDims.end(),
              std::back_inserter(output0Dims));
    output0Dims[0] = nImages;

    const size_t inputTensorSize = vectorProduct(input0Dims);
    const size_t outputTensorSize = vectorProduct(output0Dims);
    inputData.tensorValues.resize(inputTensorSize);
    outputData.tensorValues.resize(outputTensorSize);

    /* Pre-process input images */
    std::vector<cv::Mat> preprocessedImages(nImages);
    cv::Mat blobs;
    for (int i = 0; i < nImages; ++i)
    {
        const auto& image = images.at(i);
        auto& preprocessedImage = preprocessedImages.at(i);
        cv::Mat matImage, blob;

        // Resize
        const bool doResize = (!m_inputSize.empty() && m_inputSize != image.size());
        if (doResize)
            cv::resize(image, matImage, m_inputSize, 0.0, 0.0, cv::INTER_CUBIC);
        else
            matImage = image.clone();

        // Convert color
        if (preprocessData.doConvertColor())
            cv::cvtColor(matImage, matImage, preprocessData.colorConvCode);

        // Scale
        matImage.convertTo(preprocessedImage, CV_32FC3, preprocessData.scale);
        
        // Normalize
        if (preprocessData.doNormalize())
            preprocessedImage = (preprocessedImage - preprocessData.mean) / preprocessData.std;
    }

    // HWC -> NCHW
    cv::dnn::blobFromImages(preprocessedImages, blobs);

    /* Allocate model input */
    inputData.tensorValues.assign(blobs.begin<float>(), blobs.end<float>());
    inputData.tensor.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputData.tensorValues.data(), inputTensorSize, 
        input0Dims.data(), input0Dims.size()));

    /* Allocate model output */
    outputData.tensor.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputData.tensorValues.data(), outputTensorSize,
        output0Dims.data(), output0Dims.size()));
}

void EfficientNet_Onnx::postprocess(const OrtData& outputData, std::vector<cv::Mat>& outputs, 
                                    int nImages, 
                                    const PostprocessData& postprocessData) const
{
    assert(("[EfficientNet_Onnx][postprocess] Inference result tensor is empty.",
                !outputData.empty()));

    const int nLabels = m_modelInfo.outputDims.at(1);
    outputs.resize(nImages);
    for (int i = 0; i < nImages; ++i)
    {
        auto& iOutput = outputs.at(i);
        const cv::Mat outMat = cv::Mat(1, nLabels, CV_32F, (void*)outputData.tensorValues.data());

        if (postprocessData.doSoftmax)
        {
            cv::exp(outMat, iOutput);
            cv::divide(iOutput, cv::sum(iOutput)[0], iOutput);
        }
        else
        {
            iOutput = outMat.clone();
        }
    }
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

    case NeuralNetwork::Engine::Onnx:
#ifdef ONNXRUNTIME_FOUND
        return std::make_shared<EfficientNet_Onnx>(initializeData);
#else
        return nullptr;
#endif

    default:
        return nullptr;
    }
}

}