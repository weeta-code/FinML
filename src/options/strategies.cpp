#include "finml/options/strategies.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <numeric>

namespace finml {
namespace options {

// OptionContract implementation
OptionContract::OptionContract(
    double K,
    double T,
    OptionType type,
    OptionStyle style,
    int quantity,
    double premium
) : K(K), T(T), type(type), style(style), quantity(quantity), premium(premium) {}

// OptionsStrategy implementation
OptionsStrategy::OptionsStrategy(
    const std::string& name,
    std::shared_ptr<PricingModel> model
) : name(name), model(model), underlying_position(0.0) {}

void OptionsStrategy::addContract(const OptionContract& contract) {
    contracts.push_back(contract);
}

void OptionsStrategy::addUnderlyingPosition(double quantity) {
    underlying_position += quantity;
}

double OptionsStrategy::payoff(double S_T) const {
    double total_payoff = 0.0;
    
    // Calculate payoff from option contracts
    for (const auto& contract : contracts) {
        double option_payoff = 0.0;
        
        if (contract.type == OptionType::CALL) {
            option_payoff = std::max(S_T - contract.K, 0.0);
        } else {
            option_payoff = std::max(contract.K - S_T, 0.0);
        }
        
        total_payoff += contract.quantity * option_payoff;
    }
    
    // Add payoff from underlying position
    total_payoff += underlying_position * S_T;
    
    return total_payoff;
}

double OptionsStrategy::value(double S, double r, double sigma, double t) const {
    double total_value = 0.0;
    
    // Calculate value from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option value
        double option_value = model->price(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_value += contract.quantity * option_value;
    }
    
    // Add value from underlying position
    total_value += underlying_position * S;
    
    return total_value;
}

double OptionsStrategy::delta(double S, double r, double sigma, double t) const {
    double total_delta = 0.0;
    
    // Calculate delta from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option delta
        double option_delta = model->delta(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_delta += contract.quantity * option_delta;
    }
    
    // Add delta from underlying position
    total_delta += underlying_position;
    
    return total_delta;
}

double OptionsStrategy::gamma(double S, double r, double sigma, double t) const {
    double total_gamma = 0.0;
    
    // Calculate gamma from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option gamma
        double option_gamma = model->gamma(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_gamma += contract.quantity * option_gamma;
    }
    
    // Underlying position has zero gamma
    
    return total_gamma;
}

double OptionsStrategy::theta(double S, double r, double sigma, double t) const {
    double total_theta = 0.0;
    
    // Calculate theta from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option theta
        double option_theta = model->theta(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_theta += contract.quantity * option_theta;
    }
    
    // Underlying position has zero theta
    
    return total_theta;
}

double OptionsStrategy::vega(double S, double r, double sigma, double t) const {
    double total_vega = 0.0;
    
    // Calculate vega from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option vega
        double option_vega = model->vega(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_vega += contract.quantity * option_vega;
    }
    
    // Underlying position has zero vega
    
    return total_vega;
}

double OptionsStrategy::rho(double S, double r, double sigma, double t) const {
    double total_rho = 0.0;
    
    // Calculate rho from option contracts
    for (const auto& contract : contracts) {
        // Adjust time to maturity
        double T_adjusted = std::max(contract.T - t, 0.0);
        
        // Calculate option rho
        double option_rho = model->rho(S, contract.K, r, sigma, T_adjusted, contract.type);
        
        total_rho += contract.quantity * option_rho;
    }
    
    // Underlying position has zero rho
    
    return total_rho;
}

double OptionsStrategy::profitLoss(double S_T) const {
    double total_pl = payoff(S_T);
    
    // Subtract initial premiums
    for (const auto& contract : contracts) {
        total_pl -= contract.quantity * contract.premium;
    }
    
    // Subtract cost of underlying position (simplified)
    // In a real implementation, we would need to know the initial price of the underlying
    
    return total_pl;
}

std::vector<double> OptionsStrategy::breakEvenPoints() const {
    std::vector<double> break_even_points;
    
    // Collect all strike prices as potential break-even points
    std::vector<double> potential_points;
    for (const auto& contract : contracts) {
        potential_points.push_back(contract.K);
    }
    
    // Add some points in between strikes
    std::sort(potential_points.begin(), potential_points.end());
    for (size_t i = 1; i < potential_points.size(); ++i) {
        potential_points.push_back((potential_points[i-1] + potential_points[i]) / 2.0);
    }
    
    // Add points below lowest strike and above highest strike
    if (!potential_points.empty()) {
        potential_points.push_back(potential_points.front() * 0.5);
        potential_points.push_back(potential_points.back() * 1.5);
    } else {
        // Default range if no strikes
        potential_points = {50.0, 100.0, 150.0, 200.0};
    }
    
    // Sort all potential points
    std::sort(potential_points.begin(), potential_points.end());
    
    // Find break-even points using bisection method
    for (size_t i = 1; i < potential_points.size(); ++i) {
        double S_low = potential_points[i-1];
        double S_high = potential_points[i];
        double pl_low = profitLoss(S_low);
        double pl_high = profitLoss(S_high);
        
        // Check if there's a sign change (break-even point)
        if (pl_low * pl_high <= 0.0 && pl_low != pl_high) {
            // Bisection method to find the break-even point
            for (int j = 0; j < 20; ++j) { // 20 iterations should be enough for good precision
                double S_mid = (S_low + S_high) / 2.0;
                double pl_mid = profitLoss(S_mid);
                
                if (std::abs(pl_mid) < 0.01) { // Close enough to zero
                    break_even_points.push_back(S_mid);
                    break;
                }
                
                if (pl_mid * pl_low <= 0.0) {
                    S_high = S_mid;
                    pl_high = pl_mid;
                } else {
                    S_low = S_mid;
                    pl_low = pl_mid;
                }
            }
        }
    }
    
    return break_even_points;
}

double OptionsStrategy::maxProfit(double S_min, double S_max) const {
    // Sample a range of prices to find maximum profit
    const int num_samples = 1000;
    double max_profit = -std::numeric_limits<double>::infinity();
    
    for (int i = 0; i <= num_samples; ++i) {
        double S = S_min + (S_max - S_min) * i / num_samples;
        double pl = profitLoss(S);
        max_profit = std::max(max_profit, pl);
    }
    
    return max_profit;
}

double OptionsStrategy::maxLoss(double S_min, double S_max) const {
    // Sample a range of prices to find maximum loss
    const int num_samples = 1000;
    double max_loss = std::numeric_limits<double>::infinity();
    
    for (int i = 0; i <= num_samples; ++i) {
        double S = S_min + (S_max - S_min) * i / num_samples;
        double pl = profitLoss(S);
        max_loss = std::min(max_loss, pl);
    }
    
    return -max_loss; // Return as a positive number
}

std::string OptionsStrategy::getName() const {
    return name;
}

const std::vector<OptionContract>& OptionsStrategy::getContracts() const {
    return contracts;
}

double OptionsStrategy::getUnderlyingPosition() const {
    return underlying_position;
}

// Strategy factory functions
OptionsStrategy createLongCall(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    OptionsStrategy strategy("Long Call", model);
    
    // Calculate premium
    double premium = model->price(S, K, r, sigma, T, OptionType::CALL);
    
    // Add long call
    OptionContract call(K, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, premium);
    strategy.addContract(call);
    
    return strategy;
}

OptionsStrategy createLongPut(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    OptionsStrategy strategy("Long Put", model);
    
    // Calculate premium
    double premium = model->price(S, K, r, sigma, T, OptionType::PUT);
    
    // Add long put
    OptionContract put(K, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, premium);
    strategy.addContract(put);
    
    return strategy;
}

OptionsStrategy createBullCallSpread(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    if (K1 >= K2) {
        throw std::invalid_argument("K1 must be less than K2 for a bull call spread");
    }
    
    OptionsStrategy strategy("Bull Call Spread", model);
    
    // Calculate premiums
    double premium1 = model->price(S, K1, r, sigma, T, OptionType::CALL);
    double premium2 = model->price(S, K2, r, sigma, T, OptionType::CALL);
    
    // Add long call at lower strike
    OptionContract call1(K1, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, premium1);
    strategy.addContract(call1);
    
    // Add short call at higher strike
    OptionContract call2(K2, T, OptionType::CALL, OptionStyle::EUROPEAN, -1, premium2);
    strategy.addContract(call2);
    
    return strategy;
}

OptionsStrategy createBearPutSpread(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    if (K1 >= K2) {
        throw std::invalid_argument("K1 must be less than K2 for a bear put spread");
    }
    
    OptionsStrategy strategy("Bear Put Spread", model);
    
    // Calculate premiums
    double premium1 = model->price(S, K1, r, sigma, T, OptionType::PUT);
    double premium2 = model->price(S, K2, r, sigma, T, OptionType::PUT);
    
    // Add short put at lower strike
    OptionContract put1(K1, T, OptionType::PUT, OptionStyle::EUROPEAN, -1, premium1);
    strategy.addContract(put1);
    
    // Add long put at higher strike
    OptionContract put2(K2, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, premium2);
    strategy.addContract(put2);
    
    return strategy;
}

OptionsStrategy createStraddle(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    OptionsStrategy strategy("Straddle", model);
    
    // Calculate premiums
    double call_premium = model->price(S, K, r, sigma, T, OptionType::CALL);
    double put_premium = model->price(S, K, r, sigma, T, OptionType::PUT);
    
    // Add long call
    OptionContract call(K, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, call_premium);
    strategy.addContract(call);
    
    // Add long put
    OptionContract put(K, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, put_premium);
    strategy.addContract(put);
    
    return strategy;
}

OptionsStrategy createStrangle(
    double S,
    double K1,
    double K2,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    if (K1 >= K2) {
        throw std::invalid_argument("K1 must be less than K2 for a strangle");
    }
    
    OptionsStrategy strategy("Strangle", model);
    
    // Calculate premiums
    double put_premium = model->price(S, K1, r, sigma, T, OptionType::PUT);
    double call_premium = model->price(S, K2, r, sigma, T, OptionType::CALL);
    
    // Add long put at lower strike
    OptionContract put(K1, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, put_premium);
    strategy.addContract(put);
    
    // Add long call at higher strike
    OptionContract call(K2, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, call_premium);
    strategy.addContract(call);
    
    return strategy;
}

OptionsStrategy createButterflySpread(
    double S,
    double K1,
    double K2,
    double K3,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    if (K1 >= K2 || K2 >= K3) {
        throw std::invalid_argument("K1 < K2 < K3 must be satisfied for a butterfly spread");
    }
    
    // Check if K2 is equidistant from K1 and K3
    if (std::abs((K3 - K2) - (K2 - K1)) > 0.01) {
        throw std::invalid_argument("K2 should be approximately equidistant from K1 and K3 for a balanced butterfly spread");
    }
    
    OptionsStrategy strategy("Butterfly Spread", model);
    
    // Calculate premiums
    double premium1 = model->price(S, K1, r, sigma, T, OptionType::CALL);
    double premium2 = model->price(S, K2, r, sigma, T, OptionType::CALL);
    double premium3 = model->price(S, K3, r, sigma, T, OptionType::CALL);
    
    // Add long call at lowest strike
    OptionContract call1(K1, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, premium1);
    strategy.addContract(call1);
    
    // Add short calls at middle strike
    OptionContract call2(K2, T, OptionType::CALL, OptionStyle::EUROPEAN, -2, premium2);
    strategy.addContract(call2);
    
    // Add long call at highest strike
    OptionContract call3(K3, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, premium3);
    strategy.addContract(call3);
    
    return strategy;
}

OptionsStrategy createIronCondor(
    double S,
    double K1,
    double K2,
    double K3,
    double K4,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    if (K1 >= K2 || K2 >= K3 || K3 >= K4) {
        throw std::invalid_argument("K1 < K2 < K3 < K4 must be satisfied for an iron condor");
    }
    
    OptionsStrategy strategy("Iron Condor", model);
    
    // Calculate premiums
    double put_premium1 = model->price(S, K1, r, sigma, T, OptionType::PUT);
    double put_premium2 = model->price(S, K2, r, sigma, T, OptionType::PUT);
    double call_premium3 = model->price(S, K3, r, sigma, T, OptionType::CALL);
    double call_premium4 = model->price(S, K4, r, sigma, T, OptionType::CALL);
    
    // Add long put at lowest strike
    OptionContract put1(K1, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, put_premium1);
    strategy.addContract(put1);
    
    // Add short put at lower middle strike
    OptionContract put2(K2, T, OptionType::PUT, OptionStyle::EUROPEAN, -1, put_premium2);
    strategy.addContract(put2);
    
    // Add short call at upper middle strike
    OptionContract call3(K3, T, OptionType::CALL, OptionStyle::EUROPEAN, -1, call_premium3);
    strategy.addContract(call3);
    
    // Add long call at highest strike
    OptionContract call4(K4, T, OptionType::CALL, OptionStyle::EUROPEAN, 1, call_premium4);
    strategy.addContract(call4);
    
    return strategy;
}

OptionsStrategy createCoveredCall(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    OptionsStrategy strategy("Covered Call", model);
    
    // Calculate premium
    double premium = model->price(S, K, r, sigma, T, OptionType::CALL);
    
    // Add short call
    OptionContract call(K, T, OptionType::CALL, OptionStyle::EUROPEAN, -1, premium);
    strategy.addContract(call);
    
    // Add long position in the underlying
    strategy.addUnderlyingPosition(1.0);
    
    return strategy;
}

OptionsStrategy createProtectivePut(
    double S,
    double K,
    double T,
    double r,
    double sigma,
    std::shared_ptr<PricingModel> model
) {
    OptionsStrategy strategy("Protective Put", model);
    
    // Calculate premium
    double premium = model->price(S, K, r, sigma, T, OptionType::PUT);
    
    // Add long put
    OptionContract put(K, T, OptionType::PUT, OptionStyle::EUROPEAN, 1, premium);
    strategy.addContract(put);
    
    // Add long position in the underlying
    strategy.addUnderlyingPosition(1.0);
    
    return strategy;
}

} // namespace options
} // namespace finml 