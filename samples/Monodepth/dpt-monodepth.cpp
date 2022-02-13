// #include <iostream>
// #include <iomanip>
// #include <sstream>

// #include <opencv2/core.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/dnn.hpp>
// #include <opencv2/highgui.hpp>

// #include <torch/script.h> // One-stop header.
// #include <torch/cuda.h>


// const cv::Size INPUT_SIZE = {640, 640};
// const double CONF_THRESH = 0.25;
// const double NMS_THRESH = 0.45;
// const torch::DeviceType DEFAULT_DEVICE_TYPE = torch::DeviceType::CPU; // или "torch::DeviceType::CUDA"
// const cv::Scalar FIRE_COLOR = cv::Scalar(0, 255, 0); // зеленый, для большей контрастности
// const cv::Scalar CONF_BOX_COLOR = cv::Scalar(0, 0, 0);
// const cv::Scalar CONF_COLOR = cv::Scalar(0, 255, 0);
// const std::string WIN_NAME = "LibTorch test";
// const cv::String ARG_KEYS =
//         "{ help usage ?   |        | print help }"
//         "{ @input i       |  0     | input stream }"
//         "{ @model m       |        | path to .torchscript model }"
//         "{ gpu g          | false  | use GPU }"
//         ;

// /* "Очищенный" выход модели */
// struct InferOut
// {
//     cv::Rect xyxy;
//     std::string label;
//     double conf;

//     InferOut(cv::Rect r, std::string l, double c)
//         : xyxy(r)
//         , label(l)
//         , conf(c)
//     {}
// };

// /* Конвертируем cv::Mat в torch::Tensor */
// void matToTensor(const cv::Mat& in, torch::Tensor& out)
// {
//     const bool isChar = (in.type() & 0xF) < 2;
//     const std::vector<int64_t> dims = {in.rows, in.cols, in.channels()};
//     out = torch::from_blob(in.data, dims, isChar ? torch::kByte : torch::kFloat);
// }

// /* Делаем вход для модели из тензора. ВНИМАНИЕ! Исходный тензор меняется. */
// void preprocess(torch::Tensor& in, std::vector<torch::jit::IValue>& out, torch::DeviceType device = DEFAULT_DEVICE_TYPE)
// {
//     in = in.permute({2,0,1}); // HWC -> CHW
//     in = in.toType(torch::kFloat);
//     in /= 255.0;
//     in.unsqueeze_(0); // CHW -> NCHW

//     out.emplace_back(in.to(device));
// }

// /*
// Извлекаем результат модели из "сырого" выхода.

// Выход имеет следующую структуру: (N, B, C + 5), где
//     N - количество входных кадров в батче (в нашем случае один)
//     B - количество bounding box, всегда фиксировано для определенного разрешения.
//         В нашем случае, для разрешения 640х640 оно равно 25200
//     C - количество классов (в нашем случае один - fire)
// Итого, выход будет иметь размерность (1, 25200, 6).

// */
// void postprocess(const torch::Tensor& in, std::vector<InferOut>& out, 
//                     double confThresh = 0.25, double nmsThresh = 0.45, cv::Size targetSize = cv::Size())
// {
//     std::vector<cv::Rect> localBoxes;
//     std::vector<float> localConfidences;

//     const int batchIdx = 0; // NOTE: Обрабатываем только первый кадр
//     const int boxLen = in.sizes().at(1);
//     for (int64_t boxIdx = 0; boxIdx < boxLen; ++boxIdx)
//     {
//         const auto boxInfo = in[batchIdx][boxIdx];
//         const int cx = boxInfo[0].item<int>();
//         const int cy = boxInfo[1].item<int>();
//         const int w = boxInfo[2].item<int>();
//         const int h = boxInfo[3].item<int>();
//         const double classConf = boxInfo[4].item<double>();
//         const double objConf = boxInfo[5].item<double>();

//         if (classConf < confThresh)
//             continue;

//         const double fullConf = classConf * objConf;

//         const int x1 = cx - w / 2;
//         const int y1 = cy - h / 2;
//         const int x2 = cx + w / 2;
//         const int y2 = cy + h / 2;
//         const cv::Rect xyxy(x1, y1, w, h);

//         localBoxes.emplace_back(xyxy);
//         localConfidences.emplace_back(fullConf);
//     }

//     // Применяем NMS
//     std::vector<int> nmsIndices;
//     cv::dnn::NMSBoxes(localBoxes, localConfidences, confThresh, nmsThresh, nmsIndices);
//     for (size_t i = 0; i < nmsIndices.size(); ++i)
//     {
//         size_t idx = nmsIndices[i];
//         out.emplace_back( localBoxes[idx], "fire", localConfidences[idx] );
//     }

//     // Скалируем результат, если нужно
//     if (targetSize.empty() || targetSize == INPUT_SIZE)
//         return;
//     const double scalex = targetSize.width / static_cast<double>(INPUT_SIZE.width);
//     const double scaley = targetSize.height / static_cast<double>(INPUT_SIZE.height);
//     for (auto& o : out)
//     {
//         o.xyxy.x = static_cast<int>(o.xyxy.x * scalex);
//         o.xyxy.y = static_cast<int>(o.xyxy.y * scaley);
//         o.xyxy.width = static_cast<int>(o.xyxy.width * scalex);
//         o.xyxy.height = static_cast<int>(o.xyxy.height * scaley);
//     }
// }

// int main(int argc, char** argv)
// {
//     std::cout << ">>> Program started" << std::endl;

//     // Парсим входные параметры и "захватываем" видеопоток
//     cv::VideoCapture cap;
//     cv::CommandLineParser parser(argc, argv, ARG_KEYS);
//     const std::string input = parser.get<std::string>("@input");
//     if( input.size() == 1 && isdigit(input[0]) )
//         cap.open(input[0] - '0'); // webcam
//     else
//         cap.open(input); // видео-файл или живой поток
//     if( !cap.isOpened() )
//     {
//         std::cout << ">>> Could not initialize capturing...\n";
//         return 0;
//     }

//     // Проверяем доступность CUDA
//     const bool gpu = parser.get<bool>("gpu");
//     torch::DeviceType device = DEFAULT_DEVICE_TYPE;
//     if (gpu)
//     {
//         if (torch::cuda::is_available())
//         {
//             device = torch::DeviceType::CUDA;
//             std::cout << ">>> CUDA is ok! Switching calculations to GPU ..." << std::endl;
//         }
//         else
//         {
//             std::cout << ">>> LibTorch CUDA is not available! Switching calculations to CPU ..." << std::endl;
//         }
//     }
//     else
//     {
//         std::cout << ">>> CUDA manually disabled! Switching calculations to CPU ..." << std::endl;
//     }

//     // Загружаем модель
//     cv::TickMeter loadModelTm;
//     loadModelTm.start();
//     const std::string modelPath = parser.get<std::string>("@model");
//     torch::jit::script::Module model;
//     try 
//     {
//         // Де-сериализуем ScriptModule из файла
//         model = torch::jit::load(modelPath, device);
//     }
//     catch (const torch::Error& e) 
//     {
//         std::cerr << ">>> Error loading the model:\n" << e.what();
//         return -1;
//     }
//     loadModelTm.stop();
//     std::cout << ">>> Model loading took " << loadModelTm.getAvgTimeMilli() << "ms" << std::endl;

//     // Прогреваем и проверяем модель
//     cv::TickMeter warmupTm;
//     warmupTm.start();
//     {
//         // Создаем вектор входов
//         std::vector<torch::jit::IValue> inputs;
//         inputs.emplace_back( torch::ones({1, 3, 384, 672}).to(device) );

//         // Запускаем модель и преобразуем ее выход в тензор
//         const auto y = model.forward(inputs);
//         const torch::Tensor pred = y.toTuple()->elements()[0].toTensor();
//         std::cout << ">>> Warmup successful: " << (25200 == pred.sizes().at(1)) << std::endl;
//         std::cout << ">>> Warmup sum: " << torch::sum(pred).item<int>() << std::endl;
//     }
//     warmupTm.stop();
//     std::cout << ">>> Model warming up took " << warmupTm.getAvgTimeMilli() << "ms" << std::endl;

//     // Инициализируем главные объекты
//     cv::Mat frame; // кадр, полученный из видеопотока в оригинальном разрешении и формате BGR
//     cv::Mat resizedRGBFrame; // кадр, отмасштабированный и сконвертированный в RGB
//     torch::Tensor inputTensor; // кадр в формате torch::Tensor
//     std::vector<torch::jit::IValue> inputs; // Вход для модели
//     cv::TickMeter inferenceTm; // замер времени для инференса
//     cv::Mat detailedFrame; // для отображения, оригинальный кадр + разметка

//     // Запускаем основной цикл обработки
//     cv::namedWindow(WIN_NAME, 1);
//     char cKey = 0;
//     while (27 != cKey)
//     {
//         inputs.clear();

//         cap >> frame;
//         if (frame.empty())
//             break;

//         // /* DEBUG */ frame = cv::imread("/home/stas/Projects/MyPrototypes/fire-detection-from-images/yolov5/data/images/zidane.jpg");
//         detailedFrame = frame.clone();

//         inferenceTm.start();

//         // (Pre-process) Создаем вход для модели
//         cv::resize(frame, resizedRGBFrame, INPUT_SIZE, 0.0, 0.0, cv::INTER_LINEAR);
//         cv::cvtColor(resizedRGBFrame, resizedRGBFrame, cv::COLOR_BGR2RGB);
//         matToTensor(resizedRGBFrame, inputTensor);
//         preprocess(inputTensor, inputs, device);

//         // (Inrefence) Запускаем модель
//         const auto y = model.forward(inputs);
//         const torch::Tensor pred = y.toTuple()->elements()[0].toTensor();

//         // (Post-process) Обрабатываем результат
//         std::vector<InferOut> inferOuts;
//         postprocess(pred, inferOuts, CONF_THRESH, NMS_THRESH, frame.size());

//         inferenceTm.stop();

//         // Отображаем результат
//         const double lineWidth = std::max(
//             round((detailedFrame.rows + detailedFrame.cols + detailedFrame.channels()) / 2 * 0.003),
//             2.0
//             );
//         const double fontThk = std::max( lineWidth - 1, 1.0 );
//         for (const auto& inferOut : inferOuts) // рисуем разметку
//         {
//             cv::rectangle(detailedFrame, inferOut.xyxy, FIRE_COLOR, lineWidth);

//             std::stringstream confStream;
//             confStream << std::fixed << std::setprecision(2) << inferOut.conf;
//             const std::string label = inferOut.label + " " + confStream.str();
//             const cv::Size textSize = cv::getTextSize(label, 0, lineWidth / 3, fontThk, nullptr);

//             cv::Point pt1 = inferOut.xyxy.tl();
//             cv::Point pt2(0, 0);
//             const bool outside = (inferOut.xyxy.y - textSize.height - 3) >= 0; // текст выходит за bbox
//             pt2.x = pt1.x + textSize.width;
//             pt2.y = (outside) ? pt1.y - textSize.height - 3 : pt1.y + textSize.height + 3;

//             cv::rectangle(detailedFrame, pt1, pt2, CONF_BOX_COLOR, -1);
//             pt1 = (outside) ? pt1 - cv::Point2i(0, 2) : pt1 + cv::Point2i(0, textSize.height + 2);
//             cv::putText(detailedFrame, label, pt1, 0, lineWidth / 3, CONF_COLOR, fontThk);
//         }
//         cv::imshow( WIN_NAME, detailedFrame );

//         cKey = static_cast<char>(cv::waitKey(25));
//     }
//     std::cout << ">>> Inference (including pre- and post-processing) took " 
//               << inferenceTm.getAvgTimeMilli() << "ms on the average" << std::endl;
//     std::cout << ">>> Inference (including pre- and post-processing) yielded " 
//               << inferenceTm.getFPS() << " FPS on the average" << std::endl;
 
//     cap.release();
//     cv::destroyAllWindows();
    
//     std::cout << ">>> Program finished successfully" << std::endl;
//     return 0;
// }



#include <signal.h>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/json.hpp>
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

private:
    std::string m_modelPath { "" };
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

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(TitleName, player, metrics);
    const cv::Size imSize = player->frame0().size();
    const double fps = player->fps();

    /* Task-specific declarations */
    cv::Mat resizedRGBFrame; // scaled RGB
    torch::Tensor inputTensor; // scaled RGB torch::Tensor
    std::vector<torch::jit::IValue> inputs; // Model input
    cv::TickMeter inferenceTm;

    /* Check CUDA availability */
    torch::DeviceType device = (jSettings->gpu()) ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
    if (torch::kCUDA == device)
    {
        if (torch::cuda::is_available())
        {
            device = torch::DeviceType::CUDA;
            std::cout << ">>> CUDA is ok! Switching calculations to GPU ..." << std::endl;
        }
        else
        {
            device = torch::DeviceType::CPU;
            std::cout << ">>> LibTorch CUDA is not available! Switching calculations to CPU ..." << std::endl;
        }
    }
    else
    {
        std::cout << ">>> CUDA manually disabled! Switching calculations to CPU ..." << std::endl;
    }

    /* Load model */
    torch::jit::script::Module model;
    if ( !jSettings->modelPath().empty() )
    {
        cv::TickMeter loadModelTm;
        loadModelTm.start();
        try 
        {
            // De-serialize ScriptModule from file
            model = torch::jit::load(jSettings->modelPath(), device);
            model.to(device);
            model.eval();
        }
        catch (const torch::Error& e) 
        {
            std::cerr << ">>> Error loading the model:\n" << e.what();
            return -1;
        }
        loadModelTm.stop();
        std::cout << ">>> Model loading took " << loadModelTm.getAvgTimeMilli() << "ms" << std::endl;
    }
    else
    {
        std::cout << ">>> Error loading the model: Model path is empty " << std::endl;
    }

    /* Warmup model */
    try
    {
        cv::TickMeter warmupTm;
        warmupTm.start();
        
        // Make input
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back( torch::ones({1, 3, 384, 672}).to(device) );

        // Run model
        const auto y = model.forward(inputs);
        const torch::Tensor pred = y.toTuple()->elements()[0].toTensor();
        std::cout << ">>> Warmup successful: " << (25200 == pred.sizes().at(1)) << std::endl;
        std::cout << ">>> Warmup sum: " << torch::sum(pred).item<int>() << std::endl;

        warmupTm.stop();
        std::cout << ">>> Model warming up took " << warmupTm.getAvgTimeMilli() << "ms" << std::endl;
    }
    catch(const torch::Error& e)
    {
        std::cout << ">>> Error warming up the model:\n" << e.what();
    }

    /* Main loop */
    cv::Mat frame, out;
    cv::Mat outDetailed = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
    cv::Mat outDepth = cv::Mat::zeros(player->frame0().size(), CV_8UC3);
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
        if ( frame.empty() ) 
        {
            break;
        }

        /* Computer vision magic */
        {
            auto m = metrics->measure();
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
                // outDepth = grad.clone();
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
