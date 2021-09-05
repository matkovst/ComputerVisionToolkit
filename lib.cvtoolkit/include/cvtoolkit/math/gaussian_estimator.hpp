#pragma once

#include <cmath>
#include <iostream>

namespace cvt
{

/*! @brief The class helps to obtain Gaussian estimations.
*/
class GaussianEstimator final
{
public:

    const double InvSqrt2pi = 0.3989422804014327;

    /*! @brief Constructor.

        @param alpha Update rate for running formula.
    */
    GaussianEstimator(double alpha = 1.0 / 100)
        : m_alpha(alpha)
    {
    }


    /*! @brief Destructor.
    */
    ~GaussianEstimator() = default;


    /*! @brief Observes new "true" point.

        @param x "true" point.
    */
    void observe(double x)
    {
        m_mu = (1 - m_alpha) * m_mu + (m_alpha * x);
        m_sigma2 = (1 - m_alpha) * m_sigma2 + m_alpha * (x - m_mu) * (x - m_mu);
    }


    /*! @brief Calculates Gaussian PDF for a given point.

        @param x given point.
    */
    double pdf(double x)
    {
        double d = (x - m_mu) / sqrt(m_sigma2);
        return InvSqrt2pi / sqrt(m_sigma2) * exp(-0.5 * x * x);
    }


    /*! @brief Calculates standardized Euclidean distance between a point and Gaussian.

        @param x given point.
    */
    double distance(double x)
    {
        return sqrt(pow(x - m_mu, 2) / m_sigma2);
    }


    /*! @brief Returns mean.
    */
    double mean() const noexcept
    {
        return m_mu;
    }


    /*! @brief Returns stdev2.
    */
    double stdev2() const noexcept
    {
        return m_sigma2;
    }

private:
    double m_alpha { 1.0 / 100 };
    double m_mu { 0.0 };
    double m_sigma2 { 1.0 };
};

}