#pragma once

#include "spdlog/spdlog.h"

#include "spdlog/spdlog.h"
#include <memory>
#include <string>

namespace cvt
{

using LoggerPtr = std::shared_ptr<spdlog::logger>;

LoggerPtr createLogger(const std::string& name, 
                        spdlog::level::level_enum logLevel = spdlog::level::info);

}