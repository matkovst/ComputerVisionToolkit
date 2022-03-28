#define BOOST_TEST_MODULE LibCvtoolkit
#include <boost/test/included/unit_test.hpp>

#include "cvtoolkit/logger.hpp"

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