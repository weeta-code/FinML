#ifndef FINML_OPTIONS_IMPLIED_VOLATILITY_H
#define FINML_OPTIONS_IMPLIED_VOLATILITY_H

#include "finml/options/pricing.h"
#include <memory>
#include <functional>

namespace finml {
namespace options {

// IV calculator 
class ImpliedVolatility {
private:
    std::shared_ptr<PricingModel> model; // Pricing model to use
    double precision; // Precision for convergence
    size_t max_iterations; // Maximum number of iterations

public:
    // Constructor
    explicit ImpliedVolatility(
        std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>(),
        double precision = 0.0001,
        size_t max_iterations = 100
    );

    // IV calculator using bisection
    double calculateBisection(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        OptionType type
    ) const;

    // IV calculator using Newton-Raphson
    double calculateNewtonRaphson(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        OptionType type
    ) const;

    // IV calculator using Brent's method
    double calculateBrent(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        OptionType type
    ) const;

    
    double calculate(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        OptionType type
    ) const;

    // IV surface calculator
    std::vector<std::vector<double>> calculateSurface(
        const std::vector<std::vector<double>>& market_prices,
        double S,
        const std::vector<double>& strikes,
        double r,
        const std::vector<double>& maturities,
        OptionType type
    ) const;
};

} // namespace options
} // namespace finml

#endif // FINML_OPTIONS_IMPLIED_VOLATILITY_H 