#pragma once

#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>

#include "types.hpp"


namespace cvt
{

static const float DEFAULT_CONF = 0.25f;

using ObjectClasses = std::map<int, std::string>;


/*! @brief The base class for all neural network detectors.
*/
class NNDetector
{
public:
    NNDetector() {}

    virtual ~NNDetector() = default;

protected:
    cv::dnn::Net m_net;
};


/*! @brief The base class for all neural network object detectors.

    The derived classes inherited from this class may solve detection problems as well as semantic segmentation problems.
*/
class ObjectNNDetector : public NNDetector
{
public:
    ObjectNNDetector() {}
    
    virtual ~ObjectNNDetector() = default;

    /*! @brief Performes model inference.

        @param frame input frame
        @param out output structure
        @param confThreshold minimum allowed object confidence
        @param acceptedClasses the list of accepted classes required for filtration. 
        If none, no filtration is performed.
    */
    virtual void Infer( const cv::Mat& frame, InferOuts& out, 
                        float confThreshold = DEFAULT_CONF, const ObjectClasses& acceptedClasses = ObjectClasses() ) = 0;


    /*! @brief Performes model output filtration.

        @param in raw model output
        @param out filtered model output
        @param acceptedClasses the list of accepted classes required for filtration.
    */
    virtual void Filter( const InferOuts& in, InferOuts& out, const ObjectClasses& acceptedClasses ) = 0;

protected:
    ObjectClasses m_objectClasses;

    /*! @brief Reads class names from file.

        @param classPath path to the class names file
    */
    virtual void readObjectClasses( const std::string& classPath ) = 0;
};


/*! @brief The class implements Mask R-CNN alhorithm.
*/
class MaskRCNNObjectDetector final : public ObjectNNDetector
{
public:
    MaskRCNNObjectDetector( const std::string& cfgPath, const std::string& modelPath, const std::string& cocoPath, 
                            int backend = cv::dnn::DNN_BACKEND_DEFAULT, int target = cv::dnn::DNN_TARGET_CPU );

    ~MaskRCNNObjectDetector() = default;

    void Infer( const cv::Mat& frame, InferOuts& out, 
                float confThreshold = DEFAULT_CONF, const ObjectClasses& acceptedClasses = ObjectClasses() ) override;

    void Filter( const InferOuts& in, InferOuts& out, const ObjectClasses& acceptedClasses ) override;

protected:
    void readObjectClasses( const std::string& classPath ) override;

private:
    std::vector<cv::String> m_outNames;

    inline void preprocess( const cv::Mat& frame );

    void postprocess( const cv::Mat& frame, const std::vector<cv::Mat>& outs, 
                InferOuts& inferOuts, float confThreshold = 0.25f, const ObjectClasses& acceptedClasses = ObjectClasses() );

    inline std::string getClassName( int classId );
};

}