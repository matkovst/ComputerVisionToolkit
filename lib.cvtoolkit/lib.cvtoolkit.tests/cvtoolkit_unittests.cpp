#define BOOST_TEST_MODULE LibCvtoolkit
#include <boost/test/included/unit_test.hpp>

#include "cvtoolkit/logger.hpp"
#include "cvtoolkit/settings.hpp"
#include "cvtoolkit/nn/nn.hpp"
#include "cvtoolkit/nn/inception.hpp"

namespace fs = std::filesystem;

BOOST_AUTO_TEST_CASE(test_spdlog_logger)
{
    try
    {
        auto logger = cvt::createLogger("test_spdlog_logger", spdlog::level::debug);
        BOOST_CHECK_NO_THROW(logger->debug("debug"));
        BOOST_CHECK_NO_THROW(logger->info("info"));
        BOOST_CHECK_NO_THROW(logger->error("error"));
        BOOST_CHECK_NO_THROW(logger->warn("warning"));
    }
    catch(const std::exception& e)
    {
        BOOST_FAIL("Error creating logger");
        exit(1);
    }
}

BOOST_AUTO_TEST_CASE(test_settings)
{
    class MySettings final : public cvt::JsonSettings, public cvt::JsonModelSettings
    {
    public:
        MySettings(const fs::path& jPath, const std::string& nodeName)
            : JsonSettings(jPath, nodeName)
            , JsonModelSettings(jPath, nodeName)
        {}

        ~MySettings() = default;

        std::string summary() const noexcept
        {
            return JsonSettings::summary() + JsonModelSettings::summary();
        }
    };

    /* Try testing wrong path */
    auto mySettings = std::make_shared<MySettings>("path-666", "node-666");
    BOOST_REQUIRE(nullptr != mySettings && !mySettings->initialized());
    mySettings.reset();

    /* Try testing true settings */
    // fs::path mySettingsPath = fs::path("..") / ".." / ".." / "lib.cvtoolkit" 
    //                             / "lib.cvtoolkit.tests" / "cvtoolkit_unittests.json";
    // BOOST_CHECK_MESSAGE(!fs::exists(mySettingsPath), 
    //                     "To continue test you must create cvtoolkit_unittests.json");

    // json jSettings;
    // jSettings["my-settings"] = {};
    // mySettings = std::make_shared<MySettings>(jSettings, "my-settings");
    // BOOST_REQUIRE(nullptr != mySettings && mySettings->initialized());
}

BOOST_AUTO_TEST_CASE(test_nn_module)
{
    using namespace cvt;

    try
    {
        /* Try feeding empty data */
        NeuralNetwork::InitializeData data;
        auto model = createInception(data);
        BOOST_REQUIRE(nullptr == model);

        /* Try feeding wrong engine */
        data.engine = 999;
        model = createInception(data);
        BOOST_REQUIRE(nullptr == model);

        /* Try feeding wrong path */
        data.engine = NeuralNetwork::Engine::OpenCV;
        data.modelRootDir = "";
        model = createInception(data);
        BOOST_REQUIRE(nullptr != model && !model->initialized());
        model.reset();

        /* Try testing right model */
        fs::path inceptionRootDir = fs::path("..") / ".." / ".." / "data" / "inception5h";
        BOOST_CHECK_MESSAGE(fs::exists(inceptionRootDir), 
                            "To continue test you must download inception5h");

        inceptionRootDir.swap(data.modelRootDir);
        model = createInception(data);
        BOOST_REQUIRE(nullptr != model && model->initialized());
            
        const cv::Mat modelInput = cv::Mat::ones(100, 100, CV_8UC3);
        cv::Mat modelOutput;
        model->Infer(modelInput, modelOutput);
        BOOST_REQUIRE(!modelOutput.empty() 
                        && modelOutput.type() == CV_32F 
                        && 1 == modelOutput.rows);

        using Mats = std::vector<cv::Mat>;
        const Mats modelInputs = {modelInput, modelInput};
        Mats modelOutputs;
        model->Infer(modelInputs, modelOutputs);
        BOOST_REQUIRE(!modelOutputs.empty() 
                        && modelOutputs.at(0).type() == CV_32F 
                        && modelInputs.size() == modelOutputs.size());
    }
    catch(const std::exception& e)
    {
        const std::string msg = std::string("Error while running test_nn_module: ") + e.what();
        BOOST_FAIL(msg.c_str());
        exit(1);
    }
}