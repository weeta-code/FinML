#ifndef FINML_OPTIONS_STRATEGIES_H
#define FINML_OPTIONS_STRATEGIES_H

#include "finml/options/pricing.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace finml {
namespace options {

// Option Contract struct
struct OptionContract {
    double K;           // Strike price
    double T;           // Time to maturity in years
    OptionType type;    // Option type (CALL or PUT)
    OptionStyle style;  // Option style (EUROPEAN, AMERICAN, etc.)
    int quantity;       // Number of contracts (positive for long, negative for short)
    double premium;     // Premium paid/received per contract
    
    // Constructor
    OptionContract(
        double K,
        double T,
        OptionType type,
        OptionStyle style = OptionStyle::EUROPEAN,
        int quantity = 1,
        double premium = 0.0
    );
};

// Analyzing options strategies 
class OptionsStrategy {
private:
    std::string name;                           // Strategy name
    std::vector<OptionContract> contracts;      // Option contracts in the strategy
    std::shared_ptr<PricingModel> model;        // Pricing model to use
    double underlying_position;                 // Position in the underlying asset (shares)
    
public:
    // Constructor
    explicit OptionsStrategy(
        const std::string& name = "Custom Strategy",
        std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
    );
    
    // Add an option contract to the strategy
    void addContract(const OptionContract& contract);
    
    // Add a position in the underlying asset
    void addUnderlyingPosition(double quantity);
    
    // Calculate payoff at expiration
    double payoff(double S_T) const;
    double value(double S, double r, double sigma, double t = 0.0) const;
    
    // Greeks
    double delta(double S, double r, double sigma, double t = 0.0) const;
    double gamma(double S, double r, double sigma, double t = 0.0) const;
    double theta(double S, double r, double sigma, double t = 0.0) const;
    double vega(double S, double r, double sigma, double t = 0.0) const;
    double rho(double S, double r, double sigma, double t = 0.0) const;
 
    // PnL function
    double profitLoss(double S_T) const;
    
    // Break-even points
    std::vector<double> breakEvenPoints() const;
    
    double maxProfit(double S_min = 0.0, double S_max = 1000.0) const;
    double maxLoss(double S_min = 0.0, double S_max = 1000.0) const;

    std::string getName() const;
    const std::vector<OptionContract>& getContracts() const;
    double getUnderlyingPosition() const;
};


OptionsStrategy createLongCall(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createLongPut(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);


OptionsStrategy createBullCallSpread(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createBearPutSpread(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);


OptionsStrategy createStraddle(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createStrangle(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createButterflySpread(
    double S,
    double K1,
    double K2,
    double K3,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createIronCondor(
    double S,
    double K1,
    double K2,
    double K3,
    double K4,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createCoveredCall(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

OptionsStrategy createProtectivePut(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model = std::make_shared<BlackScholes>()
);

} // namespace options
} // namespace finml

#endif // FINML_OPTIONS_STRATEGIES_H 