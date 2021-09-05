#pragma once

#include <cmath>
#include <iostream>

namespace cvt
{

/*! @brief The class helps to obtain Poisson estimations.

    @note The basic time span is 1 second.
*/
class PoissonEstimator final
{
public:

    /*! @brief Constructor.

        @param initialEvents Initial number of occured events.
        @param initialSeconds Initial number of seconds initialEvents occured within.
    */
    PoissonEstimator(int initialEvents = 1, std::int64_t initialSeconds = 60)
        : m_totalEvents(initialEvents)
        , m_totalSeconds(initialSeconds)
        , m_lambda(static_cast<double>(initialEvents) / initialSeconds)
    {
    }

    ~PoissonEstimator() = default;


    /*! @brief Predicts probability of occuring next event within N seconds.

        @param seconds desired span.

        @return event probability.
    */
    double predict(std::int64_t seconds)
    {
        double noEvents = Poisson(0, seconds);
        return 1 - noEvents;
    }

    void observeEvent(std::int64_t timestamp)
    {
        if ( m_firstObservedEvent == -1 )
        {
            m_firstObservedEvent = timestamp;
            m_totalEvents++;
        }
        else // Update lambda
        {
            m_totalEvents++;
            m_totalSeconds += timestamp - m_firstObservedEvent;
            m_lambda = static_cast<double>(m_totalEvents) / m_totalSeconds;
        }
    }

private:
    int m_totalEvents { 1 };
    std::int64_t m_totalSeconds { 60 };
    double m_lambda { 1.0 / 60 };
    std::int64_t m_firstObservedEvent { -1 };

    double Poisson(int k, int span = 1)
    {
        const double spannedLambda = span * m_lambda;
        return (pow(spannedLambda, k) / fact(k)) * exp(-spannedLambda);
    }

    int fact(int n)
    {
        return (n == 0) || (n == 1) ? 1 : n * fact(n - 1);
    }
};

}