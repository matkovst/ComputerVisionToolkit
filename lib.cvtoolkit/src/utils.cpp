#include <cvtoolkit/utils.hpp>

#include <fstream>

namespace cvt
{

int sqArea( Area area )
{
    int xMin = std::numeric_limits<int>::max();
    int yMin = std::numeric_limits<int>::max();
    int xMax = 0;
    int yMax = 0;
    for ( const auto& point : area )
    {
        if ( point.x > xMax ) xMax = point.x;
        if ( point.x < xMin ) xMin = point.x;
        if ( point.y > yMax ) yMax = point.y;
        if ( point.y < yMin ) yMin = point.y;
    }

    const int w = xMax - xMin;
    const int h = yMax - yMin;
    const cv::Point offset(-xMin, -yMin);
    cv::Mat rect = cv::Mat::zeros(h, w, CV_8U);
    Areas areas{area};
    cv::drawContours(rect, areas, -1, cv::Scalar(255), -1, 8, cv::noArray(), 2147483647, offset);
    const double sum = cv::sum(rect)[0] / 255.0;

    return static_cast<int>(sum);
}

int totalSqArea( Areas areas )
{
    int totalArea = 0;
    for (const auto& area : areas)
    {
        totalArea += sqArea(area);
    }
    return totalArea;
}

bool isFlowCorrect(cv::Point2f u) 
{
    return (!cvIsNaN(u.x) && fabs(u.x) < 1e9) && (!cvIsNaN(u.y) && fabs(u.y) < 1e9);
}

void drawMotionField(const cv::Mat_<cv::Point2f>& optflow, cv::Mat& out, int stride)
{
    if ( out.empty() )
    {
        out = cv::Mat::zeros(optflow.size(), CV_8UC3);
    }

    for ( int y = 0; y < optflow.rows; y += stride ) 
    {
        for ( int x = 0; x < optflow.cols; x += stride ) 
        {
            cv::Point2f uu = -optflow(y, x);
            if ( !isFlowCorrect(uu) )
            {
                continue;
            }
            
            cv::Point2i p1(x, y);
            cv::Point2i p2(x + int(uu.x), y + int(uu.y));

            cv::Vec2f vv = cv::Vec2f(uu.x, uu.y);
            double mag = cv::norm(vv);
            unsigned int H = 255 - static_cast<int>(255 - mag) * 280/ 255;    
            unsigned int hi = (H/60) % 6;
            float S=1.f;
            float V=1.f ;
            float f = H/60.f - H/60;
            float p = V * (1 - S);
            float q = V * (1 - f * S);
            float t = V * (1 - (1 - f) * S);
            cv::Point3f res;
            if( hi == 0 ) //R = V,  G = t,  B = p
                res = cv::Point3f( p, t, V );
            if( hi == 1 ) // R = q, G = V,  B = p
                res = cv::Point3f( p, V, q );
            if( hi == 2 ) // R = p, G = V,  B = t
                res = cv::Point3f( t, V, p );
            if( hi == 3 ) // R = p, G = q,  B = V
                res = cv::Point3f( V, q, p );
            if( hi == 4 ) // R = t, G = p,  B = V
                res = cv::Point3f( V, p, t );
            if( hi == 5 ) // R = V, G = p,  B = q
                res = cv::Point3f( q, p, V );
            int b = int(std::max(0.f, std::min (res.x, 1.f)) * 255.f);
            int g = int(std::max(0.f, std::min (res.y, 1.f)) * 255.f);
            int r = int(std::max(0.f, std::min (res.z, 1.f)) * 255.f);
            cv::Scalar color(b, g, r);

            cv::arrowedLine(out, p1, p2, color, 1);
        }
    }
}

void drawInferOut( cv::Mat& frame, const InferOut& inferOut, cv::Scalar color, bool drawObjectMask, bool drawLabel )
{
    const int thickness = 2;
    if ( color.isReal() )
    {
        /// Experimental ///
        // uchar colorUchar = static_cast<uchar>(255 * inferOut.classId);
        // uchar data[3] = { colorUchar, colorUchar, colorUchar };
        // cv::Mat colorId = cv::Mat(1, 1, CV_8UC3, data);

        // cv::Mat colorMat;
        // cv::applyColorMap(colorId, colorMat, cv::COLORMAP_JET);

        // cv::Vec3b & pixel = colorMat.at<cv::Vec3b>(0, 0);
        // color = cv::Scalar(pixel[0], pixel[1], pixel[2]);

        static cv::RNG rng(12345);
        color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    cv::rectangle(frame, inferOut.location, color, thickness);

    if ( drawLabel )
    {
        std::string className = ( inferOut.className != "" ) ? inferOut.className + ": " : "";
        std::string label = className + cv::format("%.2f", inferOut.confidence);

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);

        cv::putText(frame, label, inferOut.location.tl() - cv::Point(0, 16), cv::FONT_HERSHEY_PLAIN, 1, color, thickness);
    }

    if ( drawObjectMask )
    {
        /* Resize the mask, threshold, color and apply it on the image */
        cv::Mat tmpObjectMask = inferOut.objectMask.clone();
        cv::resize(tmpObjectMask, tmpObjectMask, cv::Size(inferOut.location.width, inferOut.location.height));
        cv::Mat mask = (tmpObjectMask > 0.2);
        cv::Mat coloredRoi = (0.3 * color + 0.7 * frame(inferOut.location));
        coloredRoi.convertTo(coloredRoi, CV_8UC3);

        /* Draw the contours on the image */
        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        mask.convertTo(mask, CV_8U);
        cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(coloredRoi, contours, -1, color, thickness, cv::LINE_8, hierarchy, 100);
        coloredRoi.copyTo(frame(inferOut.location), mask);
    }
}

void drawInferOuts( cv::Mat& frame, const InferOuts& inferOuts, cv::Scalar color, bool drawObjectMask, bool drawLabels )
{
    for ( const auto& inferOut : inferOuts )
    {
        drawInferOut( frame, inferOut, color, drawObjectMask, drawLabels );
    }
}

void drawAreaMask( cv::Mat& frame, const Areas& areas, double opacity )
{
    const int frameArea = frame.size().area();
    const int areasArea = totalSqArea(areas);
    if ( frameArea == areasArea ) return; // if so, does not need to draw red lines in the corners of frame

    cv::Mat areaMask = cv::Mat::zeros(frame.size(), frame.type());
    cv::drawContours(areaMask, areas, -1, cv::Scalar(255, 127, 0), -1);
    cv::addWeighted(frame, opacity, areaMask, 1.0 - opacity , 0.0, frame);
    cv::drawContours(frame, areas, -1, cv::Scalar(255, 127, 0), 1);
}

void drawAreaMaskNeg( cv::Mat& frame, const Areas& areas, double opacity )
{
    const int frameArea = frame.size().area();
    const int areasArea = totalSqArea(areas);
    if ( frameArea == areasArea ) return; // if so, does not need to draw red lines in the corners of frame

    cv::Mat areaMask = cv::Mat(frame.size(), frame.type(), cv::Scalar(0, 0, 255));
    cv::drawContours(areaMask, areas, -1, cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(frame, opacity, areaMask, 1.0 - opacity , 0.0, frame);
    cv::drawContours(frame, areas, -1, cv::Scalar(0, 0, 255), 1);
}

void hstack2images( const cv::Mat& l, const cv::Mat& r, cv::Mat& out )
{
    CV_Assert( l.size() == r.size() );
    CV_Assert( l.type() == CV_8UC3 && r.type() == CV_8UC3 );

    const int width = l.cols;
    const int height = l.rows;

    out = cv::Mat::zeros(height, 2 * width, CV_8UC3);
	cv::Mat MatSub = out.colRange(0, width);
	l.copyTo(MatSub);
	MatSub = out.colRange(width, 2 * width);
	r.copyTo(MatSub);
}

void stack4images( const cv::Mat& lt, const cv::Mat& rt, const cv::Mat& lb, const cv::Mat& rb, cv::Mat& out )
{
    CV_Assert( 4 * lt.rows == (lt.rows + rt.rows + lb.rows + rb.rows) );
    CV_Assert( 4 * lt.cols == (lt.cols + rt.cols + lb.cols + rb.cols) );
    CV_Assert( lt.type() == CV_8UC3 && rt.type() == CV_8UC3 && lb.type() == CV_8UC3 && rb.type() == CV_8UC3 );

    const int width = lt.cols;
    const int height = lt.rows;
    cv::Mat MatSub;

    cv::Mat Top(height, 2 * width, CV_8UC3, cv::Scalar::all(0));
	MatSub = Top.colRange(0, width);
	lt.copyTo(MatSub);
	MatSub = Top.colRange(width, 2 * width);
	rt.copyTo(MatSub);

    cv::Mat Bottom(height, 2 * width, CV_8UC3, cv::Scalar::all(0));
	MatSub = Bottom.colRange(0, width);
	lb.copyTo(MatSub);
	MatSub = Bottom.colRange(width, 2 * width);
	rb.copyTo(MatSub);

    out = cv::Mat::zeros(2 * height, 2 * width, CV_8UC3);
	MatSub = out.rowRange(0, height);
	Top.copyTo(MatSub);
	MatSub = out.rowRange(height, 2 * height);
	Bottom.copyTo(MatSub);
}

cv::Size parseResolution(const std::string& resol)
{
    size_t pos = resol.find("x");
    const int w = std::stoi(resol.substr(0, pos));
    const int h = std::stoi(resol.substr(pos + 1));
    return cv::Size(w, h);
}

json makeJsonObject(const std::string& jPath)
{
    json j;

    if ( jPath.empty() )
    {
        std::cout << ">>> JSON path must not be empty" << std::endl;
        return j;
    }

    std::ifstream i(jPath.c_str());
    if ( !i.good() )
    {
        std::cout << ">>> Could not read JSON file. Possibly file does not exist" << std::endl;
        return j;
    }

    i >> j;
    if ( j.empty() )
    {
        std::cerr << ">>> Could not create JSON object from file" << std::endl;
        return j;
    }
    
    return j;
}

}