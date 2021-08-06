#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cvtoolkit/cvplayer.hpp>
#include <cvtoolkit/cvgui.hpp>


const static std::string WinName = "LK Optical Flow (CUDA)";

const cv::String argKeys =
        "{ help usage ?   |        | print help }"
        "{ @input i       |  0     | input stream }"
        "{ resize r       |  1.0   | resize scale factor }"
        "{ record e       |  false | do record }"
        ;


static void downloadVector(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
    d_mat.download(mat);
}

static void downloadVector(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec)
{
    vec.resize(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
    d_mat.download(mat);
}

static void uploadVector(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat)
{
    if ( !d_mat.empty() ) d_mat.release();
    d_mat.create(1, vec.size(), CV_32FC2);
    cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
    d_mat.upload(mat);
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

    std::string input = parser.get<std::string>("@input");
    double scaleFactor = parser.get<double>("resize");
    bool doResize = (scaleFactor != 1.0);
    bool record = parser.get<bool>("record");
    
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

    // Create some random colors
    std::vector<cv::Scalar> colors;
    cv::RNG rng;
    for ( int i = 0; i < 100; ++i )
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(cv::Scalar(r, g, b));
    }

    /* Take first frame and find corners in it */
    cv::Mat prevGray;
    cv::cvtColor(player->frame0(), prevGray, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat d_prevGray(prevGray);
    cv::cuda::GpuMat d_gray;
    cv::cuda::GpuMat d_p0, d_p1;
    cv::cuda::GpuMat d_status;

    const int maxCorners = 100;
    const double qualityLevel = 0.3;
    const double minDistance = 7;
    const int blockSize = 7;
    cv::Ptr<cv::cuda::CornersDetector> cornersDetector = 
        cv::cuda::createGoodFeaturesToTrackDetector(d_prevGray.type(), maxCorners, qualityLevel, minDistance, blockSize);
    cornersDetector->detect(d_prevGray, d_p0);

    // Create a mask image for drawing purposes
    cv::Mat mask = cv::Mat::zeros(player->frame0().size(), player->frame0().type());

    /* Create LK object */
    const cv::Size winSize(15, 15);
    const int maxLevel = 2;
    std::vector<uchar> status;
    std::vector<float> err;
    const int iters = 10;
    cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), iters, 0.03);
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = 
        cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel, iters);

    /* Main loop */
    bool loop = true;
    cv::Mat frame, gray, out;
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

        /* Pre-process */
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        /* Computer vision magic */
        {
            auto m = metrics->measure();

            d_gray.upload(gray);
            d_pyrLK_sparse->calc(d_prevGray, d_gray, d_p0, d_p1, d_status);

            /* Swap prev and next points (not so easy to perform with CUDA) */
            const int oldSize = d_status.cols;
            const int newSize = cv::cuda::countNonZero(d_status);
            if (newSize == 0)
            {
                std::cout << ">>> WARNING: Could not find any feature point" << std::endl;
                continue;
            }
            std::vector<cv::Point2f> d_p0_vec(d_p0.cols);
            downloadVector(d_p0, d_p0_vec);

            std::vector<cv::Point2f> d_p1_vec(d_p1.cols);
            downloadVector(d_p1, d_p1_vec);

            std::vector<uchar> d_status_vec(d_status.cols);
            downloadVector(d_status, d_status_vec);
                
            std::vector<cv::Point2f> d_p0_vec_reduced;
            d_p0_vec_reduced.reserve(newSize);
            for ( size_t i = 0; i < d_p1_vec.size(); ++i )
            {
                if ( d_status_vec[i] > 0 ) 
                {
                    d_p0_vec_reduced.push_back(d_p1_vec[i]);
                    // draw the tracks
                    cv::line(mask, d_p1_vec[i], d_p0_vec[i], colors[i], 2);
                    cv::circle(frame, d_p1_vec[i], 5, colors[i], -1);
                }
            }
            uploadVector(d_p0_vec_reduced, d_p0);
        }

        cv::cuda::swap( d_gray, d_prevGray );
    
        /* Display info */
        cv::add(frame, mask, out);

        if ( record )
        {
            *player << out;
        }
        if ( out.channels() == 1 )
        {
            cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        }
        gui.imshow(out, record);
    }
    
    std::cout << ">>> " << metrics->summary() << std::endl;
    std::cout << ">>> Program successfully finished" << std::endl;
    return 0;
}
