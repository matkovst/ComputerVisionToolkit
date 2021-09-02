#pragma once

#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "types.hpp"
#include "metrics.hpp"

namespace cvt
{

/*! @brief The class implements event trigger logic.

    The initial trigger state is OFF. After getting onCount-number values it switches to ABOUT_TO_ON and right after that to ON.
    If offCount-number values have not been got, it turns to ABOUT_TO_OFF and right after that to OFF.

                  ______ON______
                /               \
        ABOUT_TO_ON        ABOUT_TO_OFF
   ____OFF____/                   \____OFF____
*/
class EventTrigger final
{
public:

    enum State
    {
        ABOUT_TO_OFF    = -1,
        OFF             = 0,
        ABOUT_TO_ON     = 1,
        ON              = 2
    };

    EventTrigger(int onCount = 1, int offCount = 1)
        : m_onCount(onCount)
        , m_offCount(offCount)
    {
    }

    ~EventTrigger() = default;

    /*! @brief Updates trigger.

        @param i current observed binary variable

        @return trigger state
    */
    int update(bool v)
    {
        bool state = count(v);
        int triggerState = State::OFF;
        if ( state )
        {
            if ( m_state )
            {
                triggerState = State::ON;
            }
            else
            {
                triggerState = State::ABOUT_TO_ON;
                m_counter += m_offCount;
            }
        }
        else
        {
            if ( m_state )
            {
                triggerState = State::ABOUT_TO_OFF;
            }
            else
            {
                triggerState = State::OFF;
            }
        }
        m_state = state;
        return triggerState;
    }

    int state() const noexcept
    {
        return m_state;
    }

    int operator()(bool i)
    {
        return update(i);
    }

private:
    int m_onCount { 1 };
    int m_offCount { 1 };
    int m_counter { 0 };
    bool m_state { false };

    bool count(bool v)
    {
        m_counter += v ? v : -1;
        if (m_counter < 0) m_counter = 0;
        if (m_counter > m_onCount + m_offCount) m_counter = (m_onCount + m_offCount);
        return (m_counter >= m_onCount);
    }

};

}