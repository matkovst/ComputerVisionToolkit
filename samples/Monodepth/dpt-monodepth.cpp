#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/settings.hpp>
#include <cvtoolkit/utils.hpp>

#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>


const static std::string SampleName = "dpt-monodepth";
const static std::string TitleName = "Dpt-monodepth";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @json j        |        | path to json }"
        ;

class DPTMonodepthSettings final : public cvt::JsonSettings
{
public:
    DPTMonodepthSettings(const std::string& jPath, const std::string& nodeName)
        : JsonSettings(jPath, nodeName)
    {
        if ( m_jNodeSettings.empty() )
        {
            std::cerr << "[DPTMonodepthSettings] Could not find " << nodeName << " section" << std::endl;
            return;
        }

        if ( !m_jNodeSettings["model-path"].empty() )
            m_modelPath = static_cast<std::string>(m_jNodeSettings["model-path"]);

        if ( !m_jNodeSettings["model-input-height"].empty() )
            m_modelInputHeight = static_cast<int>(m_jNodeSettings["model-input-height"]);

        m_modelInputSize = {m_modelInputWidth, m_modelInputHeight};
    }

    ~DPTMonodepthSettings() = default;

    std::string summary() const noexcept
    {
        std::ostringstream oss;
        oss << std::endl << std::right
            << "\tSPECIFIC SETTINGS: " << std::endl
            << "\t\t- modelPath = " << modelPath();

        return JsonSettings::summary() + oss.str();
    }


    const std::string& modelPath() const noexcept
    {
        return m_modelPath;
    }

    int modelInputWidth() const noexcept
    {
        return m_modelInputWidth;
    }

    int modelInputHeight() const noexcept
    {
        return m_modelInputHeight;
    }

    cv::Size modelInputSize() const noexcept
    {
        return m_modelInputSize;
    }

private:
    std::string m_modelPath { "" };
    int m_modelInputWidth { 672 };
    int m_modelInputHeight { 384 };
    cv::Size m_modelInputSize;
};



class DPTMonodepth final
{
public:
    DPTMonodepth(const std::shared_ptr<DPTMonodepthSettings>& settings)
        :m_settings(settings)
    {
        /* Check CUDA availability */
        m_device = (m_settings->gpu()) ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
        if (torch::kCUDA == m_device)
        {
            if (torch::cuda::is_available())
            {
                m_device = torch::DeviceType::CUDA;
                std::cout << "[DPTMonodepth] CUDA is ok! Switching calculations to GPU ..." << std::endl;
            }
            else
            {
                m_device = torch::DeviceType::CPU;
                std::cout << "[DPTMonodepth] LibTorch CUDA is not available! Switching calculations to CPU ..." << std::endl;
            }
        }
        else
        {
            std::cout << "[DPTMonodepth] CUDA manually disabled! Switching calculations to CPU ..." << std::endl;
        }

        /* Load model */
        if ( !m_settings->modelPath().empty() )
        {
            try 
            {
                cv::TickMeter loadModelTm;
                loadModelTm.start();

                // De-serialize ScriptModule from file
                m_model = torch::jit::load(m_settings->modelPath(), m_device);
                // model.to(device);
                // model.eval();
                m_modelInitialized = true;

                loadModelTm.stop();
                std::cout << "[DPTMonodepth] Model loading took " << loadModelTm.getAvgTimeMilli() << "ms" << std::endl;
            }
            catch (const torch::Error& e) 
            {
                std::cerr << "[DPTMonodepth] Error loading the model:\n" << e.what();
            }
        }
        else
        {
            std::cout << "[DPTMonodepth] Error loading the model: Model path is empty " << std::endl;
        }

        /* Warmup model */
        if (m_modelInitialized)
        {
            try
            {
                cv::TickMeter warmupTm;
                warmupTm.start();

                const cv::Mat ones = cv::Mat::ones(m_settings->modelInputSize(), CV_8UC3);
                cv::Mat pred;
                Infer(ones, pred);
                
                warmupTm.stop();

                if (m_settings->modelInputSize() == pred.size())
                    std::cout << "[DPTMonodepth] Warmup successful. Sum = " 
                              << cv::sum(pred)[0] << std::endl;
                else
                    std::cout << "[DPTMonodepth] Warmup yielded wrong result. Sum = " 
                              << cv::sum(pred)[0] << std::endl;
                std::cout << "[DPTMonodepth] Model warming up took " 
                          << warmupTm.getAvgTimeMilli() << "ms" << std::endl;
            }
            catch(const torch::Error& e)
            {
                std::cout << "[DPTMonodepth] Error warming up the model:\n" << e.what();
            }
        }
    }

    ~DPTMonodepth() = default;

    void Infer(const cv::Mat& frame, cv::Mat& out, bool swapRB = true)
    {
        if ( !m_modelInitialized )
            return;

        preprocess(frame, m_inputs, m_device, swapRB);

        const auto y = m_model.forward(m_inputs);
        if (y.isTuple())
            m_outs = y.toTuple()->elements()[0].toTensor();
        else if (y.isTensor())
            m_outs = y.toTensor().detach().clone();

        postprocess(m_outs, out);
    }

    bool initialize() const noexcept
    {
        return m_modelInitialized;
    }

    /* Convert absolute depth to normalized */
    static void normalize(const cv::Mat& in, cv::Mat& out, int dtype = -1)
    {
        // double minDepth = 0.0;
        // double maxDepth = std::numeric_limits<double>().max();
        // cv::minMaxIdx(out, &minDepth, &maxDepth);
        // const float minMaxRange = (static_cast<float>(maxDepth) - static_cast<float>(minDepth));
        // if (std::numeric_limits<float>().epsilon() < minMaxRange)
        // {
        //     cv::normalize(out, out, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);
        // }

        if (-1 == dtype)
            dtype = in.type();
        cv::normalize(in, out, 0.0, 255.0, cv::NORM_MINMAX, dtype);
    }

private:
    const std::shared_ptr<DPTMonodepthSettings>& m_settings;
    torch::DeviceType m_device { torch::DeviceType::CPU };
    torch::jit::script::Module m_model;
    bool m_modelInitialized { false };
    cv::Mat m_inputMat;
    cv::Mat m_inputMat32f;
    torch::Tensor m_inputTensor;
    std::vector<torch::jit::IValue> m_inputs;
    torch::Tensor m_outs;

    /* Convert cv::Mat to torch::Tensor */
    void matToTensor(const cv::Mat& in, torch::Tensor& out)
    {
        const bool isChar = (in.type() & 0xF) < 2;
        const std::vector<int64_t> dims = {in.rows, in.cols, in.channels()};
        out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat);
    }

    /* Make model input. */
    void preprocess(const cv::Mat& in, std::vector<torch::jit::IValue>& out, 
                    torch::DeviceType device = torch::DeviceType::CPU, bool swapRB = true)
    {
        if ( !out.empty() )
            out.clear();

        // Resize
        if (in.size() != m_settings->modelInputSize())
            cv::resize(in, m_inputMat, m_settings->modelInputSize(), 0.0, 0.0, cv::INTER_CUBIC);
        else
            m_inputMat = in.clone();

        // Convert color
        if (swapRB)
            cv::cvtColor(m_inputMat, m_inputMat, cv::COLOR_BGR2RGB);

        // Normalize
        const cv::Scalar mean(0.5f, 0.5f, 0.5f);
        const cv::Scalar std(0.5f, 0.5f, 0.5f);
        cv::subtract(m_inputMat, mean, m_inputMat32f, cv::noArray(), CV_32FC3);
        cv::divide(m_inputMat32f, std, m_inputMat32f, CV_32FC3);

        // Convert to torch::Tensor
        matToTensor(m_inputMat32f, m_inputTensor);

        m_inputTensor = m_inputTensor.permute({2,0,1}); // HWC -> CHW
        m_inputTensor = m_inputTensor.toType(torch::kFloat);
        m_inputTensor /= 255.0;
        m_inputTensor.unsqueeze_(0); // CHW -> NCHW

        out.emplace_back(m_inputTensor.to(device));
    }

    void postprocess(const torch::Tensor &in, cv::Mat& out)
    {
        if ( !out.empty() )
            out.release();
    
        const auto sizes = in.sizes();
        const int batchSize = static_cast<int>(sizes[0]);
        const int h = static_cast<int>(sizes[1]);
        const int w = static_cast<int>(sizes[2]);
        out = cv::Mat(h, w, CV_32F, in.data_ptr());

        cv::multiply(out, 1000.0f, out);
    }
};



int main(int argc, char** argv)
{
    std::cout << ">>> Program started. Have fun!" << std::endl;
    /* Parse command-line args */
    cv::CommandLineParser parser(argc, argv, argKeys);
    parser.about(TitleName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    /* Make Settings */
    const std::string jsonPath = parser.get<std::string>("@json");
    std::shared_ptr<DPTMonodepthSettings> jSettings = std::make_shared<DPTMonodepthSettings>(jsonPath, SampleName);
    std::cout << "[" << TitleName << "]" << jSettings->summary() << std::endl;

    /* Open stream */
    std::shared_ptr<cvt::OpenCVPlayer> player = std::make_shared<cvt::OpenCVPlayer>(jSettings->input(), 
                                                                                    jSettings->inputSize());
    const cv::Mat& frame0 = player->frame0();
    if (frame0.empty())
    {
        std::cerr << ">>> ERROR: The first frame is empty" << std::endl;
        return -1;
    }

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(TitleName, player, metrics);
    const cv::Size imSize = frame0.size();
    const double fps = player->fps();

    /* Task-specific declarations */
    cv::Mat absoluteDepthMap;

    /* Initialize model */
    std::shared_ptr<DPTMonodepth> model = std::make_shared<DPTMonodepth>(jSettings);
    if ( !model )
    {
        std::cerr << ">>> [ERROR] Could not create DPTMonodepth" << std::endl;
        return -1;
    }
    else if ( !model->initialize() )
    {
        std::cerr << ">>> [ERROR] DPTMonodepth created but DPT model has not been initialized" << std::endl;
        return -1;
    }

    /* Main loop */
    cv::Mat frame, out;
    cv::Mat outDetailed = cv::Mat::zeros(frame0.size(), CV_8UC3);
    cv::Mat outDepth = cv::Mat::zeros(frame0.size(), CV_8UC3);
    bool loop = true;
    const bool isImage = (player->getInputType(jSettings->input()) == cvt::OpenCVPlayer::InputType::IMAGE);
    while ( loop )
    {
        /* Controls */
        int action = gui.listenKeyboard();
        if ( action == gui.CONTINUE ) continue;
        if ( action == gui.CLOSE ) 
        {
            loop = false;
        }

        /* Capturing */
        *player >> frame;
        if (frame.empty())
            break;

        /* Computer vision magic */
        {
            auto m = metrics->measure();

            model->Infer(frame, absoluteDepthMap, true);

            // Get appropriate result
            DPTMonodepth::normalize(absoluteDepthMap, outDepth, CV_8U);
            // double minDepth, maxDepth;
            // cv::minMaxIdx(absoluteDepthMap, &minDepth, &maxDepth);
            // absoluteDepthMap = static_cast<float>(maxDepth) - absoluteDepthMap;
            cv::resize(absoluteDepthMap, absoluteDepthMap, frame.size(), 0.0, 0.0, cv::INTER_CUBIC);
            cv::resize(outDepth, outDepth, frame.size(), 0.0, 0.0, cv::INTER_CUBIC);
        }

        /* Display info */
        if ( jSettings->record() || jSettings->display() )
        {
            if ( frame.rows >= 1080 && frame.cols >= 1920 )
            {
                cv::resize(frame, outDetailed, cv::Size(), 0.5, 0.5);
                cv::resize(frame, outDepth, cv::Size(), 0.5, 0.5);
            }
            else
            {
                outDetailed = frame.clone();
            }

            if ( jSettings->display() )
            {
                if ( outDetailed.channels() == 1 )
                {
                    cv::cvtColor(outDetailed, outDetailed, cv::COLOR_GRAY2BGR);
                }
                if ( outDepth.channels() == 1 )
                {
                    cv::cvtColor(outDepth, outDepth, cv::COLOR_GRAY2BGR);
                }
                
                /* Draw details */
                const cv::Scalar checkpointColor(0, 0, 255);
                const int crossRadius = 10;
                const int totalCheckpoints = 16;
                const int nCheckpoints = sqrt(totalCheckpoints);
                for (int h = 1; h < (nCheckpoints + 1); ++h)
                {
                    const int py = h * (outDepth.rows / (nCheckpoints + 1));
                    for (int w = 1; w < (nCheckpoints + 1); ++w)
                    {
                        const int px = w * (outDepth.cols / (nCheckpoints + 1));
                        const cv::Point checkpoint(px, py);
                        // cv::circle(outDepth, checkpoint, 4, checkpointColor, -1);
                        cv::line(outDepth, checkpoint - cv::Point(crossRadius, 0), 
                                checkpoint + cv::Point(crossRadius, 0), checkpointColor, 1);
                        cv::line(outDepth, checkpoint - cv::Point(0, crossRadius), 
                                checkpoint + cv::Point(0, crossRadius), checkpointColor, 1);
                        cv::line(outDetailed, checkpoint - cv::Point(crossRadius, 0), 
                                checkpoint + cv::Point(crossRadius, 0), cv::Scalar(0, 255, 0), 1);
                        cv::line(outDetailed, checkpoint - cv::Point(0, crossRadius), 
                                checkpoint + cv::Point(0, crossRadius), cv::Scalar(0, 255, 0), 1);

                        const int z = static_cast<int>(absoluteDepthMap.at<float>(checkpoint));
                        cv::putText(outDepth, "(" + std::to_string(checkpoint.x) + "," + 
                                    std::to_string(checkpoint.y) + ") ", checkpoint - cv::Point(20, 40), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, checkpointColor, 1);
                        cv::putText(outDepth, std::to_string(z), checkpoint - cv::Point(20, 20), 
                                    cv::FONT_HERSHEY_SIMPLEX, 0.5, checkpointColor, 1);
                    }
                }

                cvt::hstack2images(outDetailed, outDepth, out);
                gui.imshow(out, jSettings->record());
            }

            if ( jSettings->record() )
            {
                *player << out;
            }
        }

        /* In the case of image finish looping and just wait */
        if ( isImage )
        {
            cv::waitKey();
            break;
        }
    }
    
    std::cout << ">>> Inference metrics: " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
