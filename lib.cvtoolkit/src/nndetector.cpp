#include <cvtoolkit/nndetector.hpp>

#include <fstream>
#include <sstream>

namespace cvt
{

MaskRCNNObjectDetector::MaskRCNNObjectDetector( const std::string& cfgPath, const std::string& modelPath, const std::string& cocoPath, 
                                                int backend, int target )
{
    try
    {
        m_net = cv::dnn::readNetFromTensorflow( modelPath, cfgPath );
        m_net.setPreferableBackend( backend );
        m_net.setPreferableTarget( target );

        readObjectClasses( cocoPath );

        m_outNames.reserve(2);
        m_outNames.emplace_back("detection_out_final");
        m_outNames.emplace_back("detection_masks");
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

void MaskRCNNObjectDetector::Infer( const cv::Mat& frame, InferOuts& out, float confThreshold, const ObjectClasses& acceptedClasses )
{
    preprocess( frame );

    std::vector<cv::Mat> maskRCNNOuts;
    m_net.forward( maskRCNNOuts, m_outNames );

    postprocess( frame, maskRCNNOuts, out, confThreshold, acceptedClasses );
}

void MaskRCNNObjectDetector::Filter( const InferOuts& in, InferOuts& out, const ObjectClasses& acceptedClasses )
{
    std::copy_if(in.begin(), in.end(), std::back_inserter(out), 
        [acceptedClasses](InferOut iOut)
        {
            return acceptedClasses.find(iOut.classId) != acceptedClasses.end();
        }
    );
}

void MaskRCNNObjectDetector::readObjectClasses( const std::string& classPath )
{
    std::ifstream ifs(classPath.c_str());
    if ( !ifs.is_open() )
    {
        std::cout << "coco.names file " << classPath << " not found" << std::endl;
    }

    int lineId = 0;
    std::string line;
    while ( std::getline(ifs, line) ) 
    {
        m_objectClasses[lineId] = line;
        ++lineId;
    }
}

inline void MaskRCNNObjectDetector::preprocess( const cv::Mat& frame )
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    cv::dnn::blobFromImage(frame, blob, 1.0, frame.size(), cv::Scalar(), true, false);

    m_net.setInput(blob, "");
}

void MaskRCNNObjectDetector::postprocess( const cv::Mat& frame, const std::vector<cv::Mat>& outs, 
                InferOuts& inferOuts, float confThreshold, const ObjectClasses& acceptedClasses )
{
    cv::Mat outDetections = outs[0];
    cv::Mat outMasks = outs[1];
        
    // Output size of masks is NxCxHxW where
    // N - number of detected boxes
    // C - number of classes (excluding background)
    // HxW - segmentation shape
    const int numDetections = outDetections.size[2];
    const int numClasses = outMasks.size[1];
        
    outDetections = outDetections.reshape(1, static_cast<int>(outDetections.total() / 7));
    for ( int i = 0; i < numDetections; ++i )
    {
        float score = outDetections.at<float>(i, 2);
        if ( score < confThreshold )
        {
            continue;
        }

        int classId = static_cast<int>(outDetections.at<float>(i, 1));
        if ( !acceptedClasses.empty() && acceptedClasses.find(classId) == acceptedClasses.end() )
        {
            continue;
        }
        
        // Extract the bounding box
        int left = static_cast<int>(frame.cols * outDetections.at<float>(i, 3));
        int top = static_cast<int>(frame.rows * outDetections.at<float>(i, 4));
        int right = static_cast<int>(frame.cols * outDetections.at<float>(i, 5));
        int bottom = static_cast<int>(frame.rows * outDetections.at<float>(i, 6));
                
        // left = cv::max(0, cv::min(left, frame.cols - 1));
        // top = cv::max(0, cv::min(top, frame.rows - 1));
        // right = cv::max(0, cv::min(right, frame.cols - 1));
        // bottom = cv::max(0, cv::min(bottom, frame.rows - 1));
        // cv::Rect box = cv::Rect(left, top, right - left + 1, bottom - top + 1);

        /* Cast coords to frame size (if needed) */
        cv::Rect box = cv::Rect(left, top, right - left, bottom - top) & cv::Rect(cv::Point(0, 0), cv::Point(frame.cols, frame.rows));
                
        // Extract the mask for the object
        cv::Mat objectMask(outMasks.size[2], outMasks.size[3],CV_32F, outMasks.ptr<float>(i, classId));
            
        InferOut iOut = { classId, getClassName(classId), score, box, objectMask };
        inferOuts.emplace_back( iOut );
                
            // // Draw bounding box, colorize and show the mask on the image
            // drawBox(frame, classId, score, box, objectMask);
    }
}

inline std::string MaskRCNNObjectDetector::getClassName( int classId )
{
    if ( m_objectClasses.empty() )
    {
        return "";
    }
    return m_objectClasses[classId];
}


YOLOObjectNNDetector::YOLOObjectNNDetector( const std::string& cfgPath, const std::string& modelPath, const std::string& classNamesPath, 
                                                int backend, int target )
{
    try
    {
        m_net = cv::dnn::readNetFromDarknet( cfgPath, modelPath );
        m_net.setPreferableBackend( backend );
        m_net.setPreferableTarget( target );

        readObjectClasses( classNamesPath );

        m_outNames = m_net.getUnconnectedOutLayersNames();
        m_outLayers = m_net.getUnconnectedOutLayers();
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

void YOLOObjectNNDetector::Infer( const cv::Mat& frame, InferOuts& out, float confThreshold, const ObjectClasses& acceptedClasses )
{
    preprocess( frame );

    std::vector<cv::Mat> outLayers;
    m_net.forward( outLayers, m_outNames );

    postprocess( frame, outLayers, out, confThreshold, acceptedClasses );
}

void YOLOObjectNNDetector::Filter( const InferOuts& in, InferOuts& out, const ObjectClasses& acceptedClasses )
{
    std::copy_if(in.begin(), in.end(), std::back_inserter(out), 
        [acceptedClasses](InferOut iOut)
        {
            return acceptedClasses.find(iOut.classId) != acceptedClasses.end();
        }
    );
}

const ObjectClasses& YOLOObjectNNDetector::yoloObjectClasses() const noexcept
{
    return m_objectClasses;
}

void YOLOObjectNNDetector::readObjectClasses( const std::string& classPath )
{
    std::ifstream ifs(classPath.c_str());
    if ( !ifs.is_open() )
    {
        std::cout << ">>> [YOLOObjectNNDetector] Class names file " << classPath << " not found" << std::endl;
    }

    int lineId = 0;
    std::string line;
    while ( std::getline(ifs, line) ) 
    {
        m_objectClasses[lineId] = line;
        ++lineId;
    }
}

inline void YOLOObjectNNDetector::preprocess( const cv::Mat& frame )
{
    static cv::Mat blob;
    // Create a 4D blob from a frame.
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);

    m_net.setInput(blob);
}

void YOLOObjectNNDetector::postprocess( const cv::Mat& frame, const std::vector<cv::Mat>& outs, 
                InferOuts& inferOuts, float confThreshold, const ObjectClasses& acceptedClasses )
{
    std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Network produces output blob with a shape NxC where N is a number of
        // detected objects and C is a number of classes + 4 where the first 4
        // numbers are [center_x, center_y, width, height]
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence < confThreshold) continue;

            const int classId = classIdPoint.x;
            if ( !acceptedClasses.empty() && acceptedClasses.find(classId) == acceptedClasses.end() )
            {
                continue;
            }
            
            // Extract the bounding box
            int centerX = (int)(data[0] * frame.cols);
            int centerY = (int)(data[1] * frame.rows);
            int width = (int)(data[2] * frame.cols);
            int height = (int)(data[3] * frame.rows);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
                
            /* Cast coords to frame size (if needed) */
            cv::Rect box = cv::Rect(left, top, width, height) & cv::Rect(cv::Point(0, 0), cv::Point(frame.cols, frame.rows));

            classIds.push_back(classId);
            confidences.push_back((float)confidence);
            boxes.push_back(box);
        }
    }

    /* NMS */
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, DEFAULT_NMS_THRESH, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        InferOut iOut = { classIds[idx], getClassName(classIds[idx]), confidences[idx], boxes[idx], cv::Mat() };
        inferOuts.emplace_back( iOut );
    }
}

inline std::string YOLOObjectNNDetector::getClassName( int classId )
{
    if ( m_objectClasses.empty() )
    {
        return "";
    }
    return m_objectClasses[classId];
}

}
