#include "cvtoolkit/logger.hpp"

#include <cstdio>
#include <iostream>
#include "spdlog/sinks/stdout_color_sinks.h" // or "../stdout_sinks.h" if no color needed
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"

namespace cvt
{

LoggerPtr createLogger(const std::string& name, spdlog::level::level_enum logLevel)
{
    const auto existingLogger = spdlog::get(name);
    if (existingLogger != nullptr)
        return existingLogger;

    LoggerPtr logger;
    try
    {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(spdlog::level::debug);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S:%e] [%n] [%^%l%$] %v");

        logger.reset(new spdlog::logger(name, {console_sink}));

        logger->set_level(logLevel);

        spdlog::register_logger(logger);
        logger->flush_on(spdlog::level::level_enum::warn);

    }
    catch (const spdlog::spdlog_ex& e)
    {
        std::cout << "Logger \"name\" initialization faild" << e.what() << std::endl;
    }

    return logger;
}

}