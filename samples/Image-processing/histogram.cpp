#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/fast_math.hpp> // for cvRound
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/cvgui.hpp>
#include <cvtoolkit/utils.hpp>

#include <json.hpp>
using json = nlohmann::json;


const static std::string WinName = "Calc Histogram";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        "{ @json j        |        | path to json }"
        ;

static void calcHistBGR(const cv::Mat& in, cv::Mat& out, int histSize)
{
    float range[] = { 0, 256 };
    const float* histRanges[] = { range, range, range };
    int histSizes[] = {histSize, histSize};
    int channels[] = {0, 1, 2};
    cv::calcHist( &in, 1, channels, cv::Mat(), out, 1, histSizes, histRanges );
}


int main(int argc, char** argv)
{
    /* Parse command-line args */
    cv::CommandLineParser parser(argc, argv, argKeys);
    parser.about(WinName);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const std::string input = parser.get<std::string>("@input");
    const double scaleFactor = parser.get<double>("resize");
    const bool doResize = (scaleFactor != 1.0);
    const bool record = parser.get<bool>("record");
    const std::string jsonPath = parser.get<std::string>("@json");
    
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    /* Open stream */
    std::shared_ptr<cvt::OpenCVPlayer> player = std::make_shared<cvt::OpenCVPlayer>(input, scaleFactor);

    /* Create GUI */
    auto metrics = std::make_shared<cvt::MetricMaster>();
    cvt::GUI gui(WinName, player, metrics);

    std::cout << ">>> Input: " << input << std::endl;
    std::cout << ">>> Resolution: " << player->frame0().size() << std::endl;
    std::cout << ">>> Record: " << std::boolalpha << record << std::endl;
    std::cout << ">>> JSON file: " << jsonPath << std::endl;

    /* Parse JSON */
    int histSize = 256;
    cv::Point pt1(200, 200);
    cv::Point pt2(600, 200);
    double lineBufAlpha = 1.0 / 500.0;
    double decisionThresh = 0.5;
    if ( !jsonPath.empty() ) 
    {
        std::ifstream i(jsonPath.c_str());
        if ( i.good() )
        {
            json j;
            i >> j;
            if ( !j.empty() )
            {
                histSize = ( j["histSize"].empty() ) ? histSize : static_cast<int>(j["histSize"]);
                lineBufAlpha = ( j["lineBufAlpha"].empty() ) ? lineBufAlpha : static_cast<double>(j["lineBufAlpha"]);
                decisionThresh = ( j["decisionThresh"].empty() ) ? decisionThresh : static_cast<double>(j["decisionThresh"]);
                const int pt1x = ( j["pt1x"].empty() ) ? 200 : static_cast<int>(j["pt1x"]);
                const int pt1y = ( j["pt1y"].empty() ) ? 200 : static_cast<int>(j["pt1y"]);
                const int pt2x = ( j["pt2x"].empty() ) ? 600 : static_cast<int>(j["pt2x"]);
                const int pt2y = ( j["pt2y"].empty() ) ? 200 : static_cast<int>(j["pt2y"]);
                pt1 = cv::Point(pt1x, pt1y);
                pt2 = cv::Point(pt2x, pt2y);
            }
            else std::cerr << ">>> Could not create JSON object from file" << std::endl;
        }
        else std::cout << ">>> Could not read JSON file. Possibly file does not exist" << std::endl;
    }
    else std::cout << ">>> JSON path must not be empty" << std::endl;

    /* Sample-specific declarations */
    const double lineLength = cv::norm(pt1 - pt2);
    const int hist_w = lineLength * 1.5;
    const int hist_h = lineLength;
    const int bin_w = cvRound( (double) hist_w / histSize );
    const double lineBufAlpha0 = 1.0 / 25.0; // = 1 sec
    cv::LineIterator lineit0(player->frame0(), pt1, pt2);
    cv::Mat lineBuf(cv::Size(1, lineit0.count), player->frame0().type());
    cv::Mat lineBuf_32F(cv::Size(1, lineit0.count), CV_32FC3);
    cv::Mat lineBufMean_32F = cv::Mat::zeros(cv::Size(1, lineit0.count), CV_32FC3);
    cv::Mat b_hist, g_hist, r_hist, bgr_hist;
    double bhattDist = 1.0;

    /* Main loop */
    bool loop = true;
    cv::Mat frame, frameHSV, out, histOut;
    cv::Mat HSVMask, frameHSVMasked;
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

            /* Prepare frame */
            // frameHSVMasked.release();
            // cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
            // const cv::Scalar YELLOW_LOWER = cv::Scalar(17, 55, 55);
            // const cv::Scalar YELLOW_UPPER = cv::Scalar(40, 255, 255);
            // const cv::Scalar BLUE_LOWER = cv::Scalar(90, 55, 55);
            // const cv::Scalar BLUE_UPPER = cv::Scalar(130, 255, 255);
            // cv::inRange(frameHSV, YELLOW_LOWER, YELLOW_UPPER, HSVMask);
            // cv::bitwise_and(frameHSV, frameHSV, frameHSVMasked, HSVMask);
            // cv::inRange(frameHSVMasked, BLUE_LOWER, BLUE_UPPER, HSVMask);
            // cv::bitwise_and(frameHSVMasked, frameHSVMasked, frameHSVMasked, HSVMask);

            /* Observe line */
            cv::LineIterator lineit(frame, pt1, pt2);
            for(int i = 0; i < lineit.count; i++, ++lineit)
            {
                const cv::Vec3b pixel = *(const cv::Vec3b*)*lineit;
                // lineBuf.at<cv::Vec3b>(0, i) = pixel; // same as cv::Mat(..., vec.data())
                lineBuf_32F.at<cv::Vec3f>(0, i) = cv::Vec3f(pixel); // same as cv::Mat(..., vec.data())
            }

            /* Compute mean */
            const double alpha = ( player->frameNum() > 3 * player->fps() ) ? lineBufAlpha : lineBufAlpha0;
            cv::accumulateWeighted(lineBuf_32F, lineBufMean_32F, alpha);

            /* Produce BGR histogram */
            cv::MatND lineHist_32F, lineHistMean_32F;
            calcHistBGR(lineBuf_32F, lineHist_32F, histSize);
            calcHistBGR(lineBufMean_32F, lineHistMean_32F, histSize);
            cv::normalize(lineHist_32F, lineHist_32F, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat() );
            cv::normalize(lineHistMean_32F, lineHistMean_32F, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat() );

            /* Draw hist image */
            cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            for( int i = 1; i < histSize; i++ )
            {
                cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(lineHist_32F.at<float>(i-1)) ),
                    cv::Point( bin_w*(i), hist_h - cvRound(lineHist_32F.at<float>(i)) ),
                    cv::Scalar( 0, 255, 0), 2, 8, 0  );
                cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(lineHistMean_32F.at<float>(i-1)) ),
                    cv::Point( bin_w*(i), hist_h - cvRound(lineHistMean_32F.at<float>(i)) ),
                    cv::Scalar( 255, 0, 0), 2, 8, 0  );
            }

            /* Produce BGR histogram separately */
            // std::vector<cv::Mat> bgr_planes;
            // cv::split( lineBufMean_32F, bgr_planes );
            // float range[] = { 0, 256 }; //the upper boundary is exclusive
            // const float* histRange[] = { range };

            // cv::calcHist( &bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, histRange );
            // cv::calcHist( &bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, histRange );
            // cv::calcHist( &bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, histRange );
            // cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 0,0,0) );
            // cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
            // cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
            // cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

            // for( int i = 1; i < histSize; i++ )
            // {
            //     cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ),
            //         cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
            //         cv::Scalar( 255, 0, 0), 2, 8, 0  );
            //     cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ),
            //         cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
            //         cv::Scalar( 0, 255, 0), 2, 8, 0  );
            //     cv::line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ),
            //         cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
            //         cv::Scalar( 0, 0, 255), 2, 8, 0  );
            // }

            /* Draw hist info */
            {
                const int ymargin = 22;
                int offset = ymargin;
                for( int compare_method = 0; compare_method < 4; compare_method++ )
                {
                    double comp = cv::compareHist( lineHist_32F, lineHistMean_32F, compare_method );
                    const cv::String text = "Compare hist " + std::to_string(compare_method) + ": " + std::to_string(comp);
                    cv::putText(histImage, text, cv::Point(0, offset), cv::FONT_HERSHEY_PLAIN, 1.2,
                                cv::Scalar(0, 255, 0), 1, 8);
                    offset += ymargin;

                    if (compare_method == cv::HISTCMP_BHATTACHARYYA)
                    {
                        bhattDist = comp;
                    }
                }
            }
            histOut = histImage;
        }

        /* Display info */
        cv::resize(frame, out, cv::Size(0, 0), 0.5, 0.5, cv::INTER_NEAREST);
        const cv::Scalar color = ( bhattDist > decisionThresh ) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::line(out, pt1, pt2, color, 2);
        if ( record )
        {
            *player << out;
        }
        if ( out.channels() == 1 )
        {
            cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        }
        gui.imshow(out, record);
        cv::imshow("Hist", histOut);
    }
    
    std::cout << ">>> " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
