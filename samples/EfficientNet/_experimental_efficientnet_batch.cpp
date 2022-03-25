
#ifdef ONNXRUNTIME_FOUND

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int main(int argc, char *argv[])
{
    std::string model_path = argv[1];
    int model_width = std::stoi(argv[2]);
    int model_height = std::stoi(argv[3]);
    std::string video_path = argv[4];

    // Initialize ONNX Runtime and extract params
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnxruntime");
    Ort::SessionOptions session_options;

    Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);

    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    if (num_input_nodes != 1)
        throw std::invalid_argument("Model has more than 1 input");

    std::vector<std::vector<int64_t>> input_dims;
    std::vector<int64_t> input_node_dims =
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Input shape: ";
    for (auto dim : input_node_dims)
      std::cout << dim << " ";
    std::cout << std::endl;
    if (input_node_dims[0] != -1)
        throw std::invalid_argument("Model was not exported with dynamic shape");

    input_node_dims[0] = 2;  // Batch size
    input_node_dims[1] = 3;  // Num channels
    input_node_dims[2] = model_width;
    input_node_dims[3] = model_height;

    input_dims.push_back(input_node_dims);

    std::vector<const char *> input_names;
    Ort::AllocatorWithDefaultOptions allocator;
    input_names.push_back(session.GetInputName(0, allocator));

    std::vector<const char *> output_names;
    for (size_t i = 0; i < num_output_nodes; i++)
        output_names.push_back(session.GetOutputName(i, allocator));

    size_t input_node_size =
        input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];

    std::vector<size_t> input_sizes;
    input_sizes.push_back(input_node_size);

    // Start capturing
    cv::VideoCapture cap(video_path);
    cv::VideoWriter output_video;

    int fourcc = cv::VideoWriter::fourcc('F', 'M', 'P', '4');
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::Size video_size =
        cv::Size((int) cap.get(cv::CAP_PROP_FRAME_WIDTH), (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    int half_image_px = static_cast<int>(video_size.width / 2);

    output_video.open("result.avi", fourcc, fps, video_size, true);

    while (1)
    {
        cv::Mat bgr_image;
        cap.read(bgr_image);

        if (bgr_image.empty())
            break;

        // Preprocess image
        cv::Mat rgb_image;
        cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);

        std::vector<cv::Mat> crops;

        cv::Rect2f left_crop(0, 0, half_image_px, video_size.height);
        cv::Rect2f right_crop(half_image_px, 0, half_image_px, video_size.height);

        crops.push_back(rgb_image(left_crop));
        crops.push_back(rgb_image(right_crop));

        // Run inference
        cv::Mat input_blob;
        cv::dnn::blobFromImages(crops,
                                input_blob,
                                1.0,
                                cv::Size(input_node_dims[2], input_node_dims[3]));

        std::vector<float> input_tensor_values(input_node_size);
        input_tensor_values.assign(input_blob.begin<float>(), input_blob.end<float>());

        std::vector<Ort::Value> input_tensor;
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                       OrtMemType::OrtMemTypeDefault);
        input_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info,
                                                                  input_tensor_values.data(),
                                                                  input_tensor_values.size(),
                                                                  input_node_dims.data(),
                                                                  input_node_dims.size()));
        std::vector<Ort::Value> output_tensors;

        for (size_t i = 0; i < num_output_nodes; i++)
            output_tensors.emplace_back(nullptr);

        session.Run(Ort::RunOptions {nullptr},
                    input_names.data(),
                    input_tensor.data(),
                    num_input_nodes,
                    output_names.data(),
                    output_tensors.data(),
                    num_output_nodes);

        std::vector<cv::Mat> predictions;
        for (size_t i = 0; i < output_tensors.size(); i++)
        {
            cv::Mat out;
            auto shape = output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
            int num_elements = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementCount();

            std::vector<int> dims(std::begin(shape), std::end(shape));

            auto type = output_tensors[i].GetTensorTypeAndShapeInfo().GetElementType();
            switch (type)
            {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    out.create(int(shape.size()), dims.data(), CV_32F);
                    std::memcpy(out.data,
                                output_tensors[i].GetTensorMutableData<float>(),
                                num_elements * sizeof(float));
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    out.create(int(shape.size()), dims.data(), CV_32S);
                    std::memcpy(out.data,
                                output_tensors[i].GetTensorMutableData<int>(),
                                num_elements * sizeof(int));
                    break;
                default:
                    throw std::runtime_error("invalid data type " + std::to_string(type));
                    break;
            }

            predictions.push_back(out);
        }

        // Postprocess predictions
        // Predictions[0] = dets = [Batch x Num Boxes x 5]
        // Predictions[1] = labels = [Batch x Num Boxes]
        auto dets = predictions.at(0);
        auto labels = predictions.at(1);

        int batch_size = dets.size[0];
        int num_boxes = dets.size[1];

        std::vector<std::vector<cv::Rect2f>> results;
        for (int i = 0; i < batch_size; i++)
        {
            std::vector<cv::Rect2f> detections;
            for (int j = 0; j < num_boxes; j++)
            {
                float score = dets.at<float>(i, j, 4);

                if (score < 0.5)
                    continue;

                float min_col = dets.at<float>(i, j, 0) / input_node_dims[2];
                float min_row = dets.at<float>(i, j, 1) / input_node_dims[3];
                float max_col = dets.at<float>(i, j, 2) / input_node_dims[2];
                float max_row = dets.at<float>(i, j, 3) / input_node_dims[3];

                detections.push_back(
                    cv::Rect2f(min_col, min_row, max_col - min_col, max_row - min_row));
            }

            results.push_back(detections);
        }

        std::vector<cv::Rect2f> merged_results;
        for (int i = 0; i < 2; i++)
        {
            auto detections = results.at(i);

            float col_offset = 0.5 * i;
            float width_factor = 0.5;

            for (auto detection : detections)
            {
                auto x = detection.x * width_factor + col_offset;
                auto y = detection.y;
                auto width = detection.width * width_factor;
                auto height = detection.height;

                merged_results.push_back(cv::Rect2f(x, y, width, height));
            }
        }

        // Paint and write frame
        for (auto detection : merged_results)
        {
            cv::Scalar color = cv::Scalar(177, 13, 242);
            int min_col = detection.x * bgr_image.cols;
            int min_row = detection.y * bgr_image.rows;
            int width = detection.width * bgr_image.cols;
            int height = detection.height * bgr_image.rows;

            cv::Rect2f bbox_rect(min_col, min_row, width, height);
            cv::rectangle(bgr_image, bbox_rect, color, 3);
        }

        output_video << bgr_image;
    }

    output_video.release();
}

#endif