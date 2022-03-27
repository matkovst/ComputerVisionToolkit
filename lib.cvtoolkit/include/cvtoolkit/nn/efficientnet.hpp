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

    void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs, 
                const PreprocessData& preprocessData,
                const PostprocessData& postprocessData) override;

private:

    void preprocess(const std::vector<cv::Mat>& images, std::vector<torch::jit::IValue>& out);

    void postprocess(const torch::Tensor &in, std::vector<cv::Mat>& outs);

private:
    torch::DeviceType m_device { torch::DeviceType::CPU };
    torch::jit::script::Module m_model;
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

/**
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model has a symbolic first dimension (i.e. -1x3x224x224)
 *
 *
 * NOTE: Some onnx models may not have a symbolic first dimension. To prepare the onnx model, see the python code snippet below.
 * =============  Python Example  ======================
 * import onnx
 * model = onnx.load_model('model.onnx')
 * model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
 * onnx.save_model(model, 'model-symbolic.onnx')
 * 
 */
class EfficientNet_Onnx final : public NeuralNetwork
{
public:

    EfficientNet_Onnx(const InitializeData& initializeData);

    ~EfficientNet_Onnx() = default;

    void Infer(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& outs, 
                const PreprocessData& preprocessData,
                const PostprocessData& postprocessData) override;

private:

    typedef struct
    {
        std::vector<Ort::Value> tensor; // a vector of 1 element
        std::vector<float> tensorValues;

        void clear()
        {
            tensor.clear();
            tensorValues.clear();
        }

        bool empty() const noexcept
        {
            return tensorValues.empty();
        }
    } OrtData;

    void preprocess(const std::vector<cv::Mat>& images, OrtData& inputData, OrtData& outputData, 
                    const PreprocessData& preprocessData);

    void postprocess(const OrtData& outputData, std::vector<cv::Mat>& outputs, int nImages,
                    const PostprocessData& postprocessData) const;

private:
    Ort::Session m_session { nullptr };
    Ort::AllocatorWithDefaultOptions m_allocator;
    std::vector<const char*> m_inputNames, m_outputNames;
    cv::Size m_inputSize;
    OrtData m_inputData, m_outputData;
    
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