#pragma once

#include <memory>

#ifdef TORCH_FOUND
#include <torch/script.h>
#include <torch/cuda.h>
#endif
#include "nn.hpp"

namespace cvt
{

namespace EfficientNet
{

static const cv::Scalar mean = {0.485, 0.456, 0.406};
static const cv::Scalar std = {0.229, 0.224, 0.225};

}

#ifdef TORCH_FOUND

class EfficientNet_Torch final : public NeuralNetwork
{
public:

    EfficientNet_Torch(const InitializeData& initializeData);

    ~EfficientNet_Torch() = default;

    void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs) override;

private:

    void preprocess(const std::vector<cv::Mat>& images, std::vector<torch::jit::IValue>& out);

    void postprocess(const torch::Tensor &in, std::vector<cv::Mat>& outs);

private:
    torch::DeviceType m_device { torch::DeviceType::CPU };
    torch::jit::script::Module m_model;
    bool m_initialized { false };
    torch::Tensor m_inputTensor;
    std::vector<torch::jit::IValue> m_inputs;
    torch::Tensor m_outs;
};

static void matToTensor(const cv::Mat& in, torch::Tensor& out)
{
    const bool isChar = (in.type() & 0xF) < 2;
    const std::vector<int64_t> dims = {in.rows, in.cols, in.channels()};
    out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat).clone();
}

#endif


#ifdef ONNXRUNTIME_FOUND

#include <onnxruntime_cxx_api.h>

class EfficientNet_Onnx final : public NeuralNetwork
{
public:

    EfficientNet_Onnx(const InitializeData& initializeData);

    ~EfficientNet_Onnx() = default;

    void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs) override;

private:

    typedef struct
    {
        std::vector<Ort::Value> tensors;
        std::vector<float> tensorsValues;

        void clear()
        {
            tensors.clear();
            tensorsValues.clear();
        }
    } OrtData;

    void preprocess(const std::vector<cv::Mat>& images, OrtData& inputData, OrtData& outputData);

    void postprocess(const OrtData& inputData, std::vector<cv::Mat>& outputs) const;

private:
    Ort::Session m_session { nullptr };
    Ort::AllocatorWithDefaultOptions m_allocator;
    std::vector<const char*> m_inputNames, m_outputNames;
    cv::Size m_inputSize;
    OrtData m_inputData, m_outputData;
    bool m_initialized { false };
    
    struct
    {
        size_t numInputNodes;
        const char* inputName;
        ONNXTensorElementDataType inputType;
        std::vector<int64_t> inputDims;
        size_t numOutputNodes;
        const char* outputName;
        ONNXTensorElementDataType outputType;
        std::vector<int64_t> outputDims;
    } m_modelInfo;
};

#endif


/** @brief Creates EfficientNet for specified settings
 */
std::shared_ptr<NeuralNetwork> createEfficientNet(const NeuralNetwork::InitializeData& initializeData);

}