#pragma once

#include <iostream>
#include <sstream>
#include <memory>
#include <chrono>

namespace cvt
{

/*! @brief The class for taking readings of a code scope.

    Its usage looks like
    @code{.cpp}
        auto metrics = std::make_shared<cvt::MetricMaster>();
        {
            auto m = metrics->measure();

            // do smth
        }
        std::cout << metrics->summary() << std::endl;
    @endcode
*/
class MetricMaster final : public std::enable_shared_from_this<MetricMaster>
{

    class MetricAcolyte final
    {
    public:
        ~MetricAcolyte()
        {
            m_master.lock()->finish();
        }

        friend class MetricMaster;

    private:
        std::weak_ptr<MetricMaster> m_master;

        explicit MetricAcolyte( std::weak_ptr<MetricMaster> master )
            : m_master(master)
        {
            m_master.lock()->start();
        }
    };

public:
    MetricMaster() {}

    ~MetricMaster() = default;

    MetricAcolyte measure()
    {
        return MetricAcolyte( weak_from_this() );
    }

    void start()
    {
        m_startTime = std::chrono::high_resolution_clock::now();
    }

    void finish()
    {
        ++m_calls;

        auto elapsed = std::chrono::high_resolution_clock::now() - m_startTime;
        m_currTimeMS = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

        m_totalTimeMS += m_currTimeMS;
        
        m_avgTimeMS = ( m_calls > 0 ) ? m_totalTimeMS / m_calls : 0;
    }

    int totalCalls() const noexcept
    {
        return m_calls;
    }

    long long totalTime() const noexcept
    {
        return m_totalTimeMS;
    }

    long long currentTime() const noexcept
    {
        return m_currTimeMS;
    }

    long long avgTime() const noexcept
    {
        return m_avgTimeMS;
    }

    std::string summary() const noexcept
    {
        std::stringstream ss;
        ss << "Total frames: " << totalCalls() 
            << ", total time: " << totalTime()
            << ", average time: " << avgTime();

        return ss.str();
    }

private:
    int m_calls { 0 };
    long long m_totalTimeMS { 0 };
    long long m_currTimeMS { 0 };
    long long m_avgTimeMS { 0 };
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

}