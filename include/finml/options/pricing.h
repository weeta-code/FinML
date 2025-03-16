 #ifndef FINML_OPTIONS_PRICING_H
#define FINML_OPTIONS_PRICING_H

#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <random>
#include <functional>

namespace finml {
namespace options {

// Option type enumeration
enum class OptionType {
    CALL,
    PUT
};

// Option style enumeration
enum class OptionStyle {
    EUROPEAN,
    AMERICAN,
    ASIAN,
    BARRIER
};

// Option pricing model ADT
class PricingModel {
public:
    virtual ~PricingModel() = default;
    
    virtual double price(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
    
    // Greeks calculations
    virtual double delta(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
    virtual double gamma(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
    virtual double theta(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
    virtual double vega(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
    virtual double rho(double S, double K, double r, double sigma, double T, OptionType type) const = 0;
};

// Black Scholes model
class BlackScholes : public PricingModel {
private:
    // Cumulative distribution function
    double normalCDF(double x) const;
    
    // Probability density function
    double normalPDF(double x) const;
    
    // d1 parameter
    double d1(double S, double K, double r, double sigma, double T) const;
    // d2 parameter
    double d2(double S, double K, double r, double sigma, double T) const;

public:
    // Calculate option price
    double price(double S, double K, double r, double sigma, double T, OptionType type) const override;
    
    // Greeks calculations
    double delta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double gamma(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double theta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double vega(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double rho(double S, double K, double r, double sigma, double T, OptionType type) const override;
};

// Binomial tree model
class BinomialTree : public PricingModel {
private:
    size_t steps; // Number of time steps in the binomial tree

public:
    // Constructor
    explicit BinomialTree(size_t steps = 100);
    
    double price(double S, double K, double r, double sigma, double T, OptionType type, OptionStyle style = OptionStyle::EUROPEAN) const;
    double price(double S, double K, double r, double sigma, double T, OptionType type) const override;
    
    double delta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double gamma(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double theta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double vega(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double rho(double S, double K, double r, double sigma, double T, OptionType type) const override;
};

// Monte Carlo model 
class MonteCarlo : public PricingModel {
private:
    size_t num_simulations; // Number of Monte Carlo simulations
    size_t num_steps; // Number of time steps in each simulation

public:
    // Constructor
    explicit MonteCarlo(size_t num_simulations = 10000, size_t num_steps = 100);
    
    double price(
        double S, 
        double K, 
        double r, 
        double sigma, 
        double T, 
        OptionType type, 
        OptionStyle style = OptionStyle::EUROPEAN,
        std::function<double(double, double, OptionType)> payoff_func = nullptr
    ) const;
    double price(double S, double K, double r, double sigma, double T, OptionType type) const override;
    
    // Greeks
    double delta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double gamma(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double theta(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double vega(double S, double K, double r, double sigma, double T, OptionType type) const override;
    double rho(double S, double K, double r, double sigma, double T, OptionType type) const override;
};

} // namespace options
} // namespace finml

#endif // FINML_OPTIONS_PRICING_H