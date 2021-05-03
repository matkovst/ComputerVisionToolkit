#pragma once

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

}
