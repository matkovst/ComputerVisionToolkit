#define BOOST_TEST_MODULE LibCvtoolkit
#include <boost/test/included/unit_test.hpp>

#include "cvtoolkit/logger.hpp"
// #include "cvtoolkit/nn/nn.hpp"
// #include "cvtoolkit/nn/efficientnet.hpp"

// namespace fs = std::filesystem;

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

// BOOST_AUTO_TEST_CASE(test_nn_module)
// {
//     try
//     {
//         /* Try feeding empty data */
//         cvt::NeuralNetwork::InitializeData data;
//         auto model = cvt::createEfficientNet(data);
//         BOOST_REQUIRE_EQUAL(model, nullptr);
//     }
//     catch(const std::exception& e)
//     {
//         BOOST_FAIL("Error while running test_nn_module: " << e.what());
//         exit(1);
//     }
// }