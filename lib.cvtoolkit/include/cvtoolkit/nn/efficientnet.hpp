#pragma once

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

    void Infer(const Image& image, Image& out) override;

    void Infer(const Images& images, Images& outs) override;

private:

    void preprocess(const Images& images, std::vector<torch::jit::IValue>& out);

    void postprocess(const torch::Tensor &in, Images& out);

private:
    torch::DeviceType m_device { torch::DeviceType::CPU };
    torch::jit::script::Module m_model;
    bool m_modelInitialized { false };
    torch::Tensor m_inputTensor;
    std::vector<torch::jit::IValue> m_inputs;
    torch::Tensor m_outs;
};

static void imageToTensor(const cvt::Image& in, torch::Tensor& out)
{
    const bool isChar = (in.type & 0xF) < 2;
    const std::vector<int64_t> dims = {in.h, in.w, in.c};
    out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat);
}

static void matToTensor(const cv::Mat& in, torch::Tensor& out)
{
    const bool isChar = (in.type() & 0xF) < 2;
    const std::vector<int64_t> dims = {in.rows, in.cols, in.channels()};
    out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat).clone();
}

#endif

/** @brief Creates EfficientNet for specified settings
 */
std::shared_ptr<NeuralNetwork> createEfficientNet(const NeuralNetwork::InitializeData& initializeData);

}