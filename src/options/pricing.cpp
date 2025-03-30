#include "finml/options/pricing.h"
#include "finml/core/matrix.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace finml {
namespace options {

// Utility class for normal distribution functions
class NormalDistribution {
public:
    static double cdf(double x) {
        return 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
    }
    
    static double pdf(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }
};

// Helper functions for Greeks calculations
namespace {
    // Helper for expiration payoff
    double getExpirationPayoff(double S, double K, OptionType type) {
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }
    
    // Helper for expiration delta
    double getExpirationDelta(double S, double K, OptionType type) {
        if (type == OptionType::CALL) {
            return S > K ? 1.0 : 0.0;
        } else {
            return S < K ? -1.0 : 0.0;
        }
    }
    
    // Helper for zero Greeks at expiration
    double getZeroGreeks() {
        return 0.0;
    }
}

// Black-Scholes Model Implementation
double BlackScholes::d1(double S, double K, double r, double sigma, double T) const {
    if (T <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument("Time to maturity and volatility must be positive");
    }
    return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}

double BlackScholes::d2(double S, double K, double r, double sigma, double T) const {
    return d1(S, K, r, sigma, T) - sigma * std::sqrt(T);
}

double BlackScholes::price(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return getExpirationPayoff(S, K, type);
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return S * NormalDistribution::cdf(d1_val) - K * std::exp(-r * T) * NormalDistribution::cdf(d2_val);
    } else {
        return K * std::exp(-r * T) * NormalDistribution::cdf(-d2_val) - S * NormalDistribution::cdf(-d1_val);
    }
}

double BlackScholes::delta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return getExpirationDelta(S, K, type);
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return NormalDistribution::cdf(d1_val);
    } else {
        return NormalDistribution::cdf(d1_val) - 1.0;
    }
}

double BlackScholes::gamma(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0 || sigma <= 0.0) {
        return getZeroGreeks();
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    return NormalDistribution::pdf(d1_val) / (S * sigma * std::sqrt(T));
}

double BlackScholes::theta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0 || sigma <= 0.0) {
        return getZeroGreeks();
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return -S * NormalDistribution::pdf(d1_val) * sigma / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * NormalDistribution::cdf(d2_val);
    } else {
        return -S * NormalDistribution::pdf(d1_val) * sigma / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * NormalDistribution::cdf(-d2_val);
    }
}

double BlackScholes::vega(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return getZeroGreeks();
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    return S * std::sqrt(T) * NormalDistribution::pdf(d1_val) / 100.0; // Divided by 100 to get the change per 1% change in volatility
}

double BlackScholes::rho(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return getZeroGreeks();
    }
    
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return K * T * std::exp(-r * T) * NormalDistribution::cdf(d2_val) / 100.0; // Divided by 100 to get the change per 1% change in interest rate
    } else {
        return -K * T * std::exp(-r * T) * NormalDistribution::cdf(-d2_val) / 100.0;
    }
}

// Binomial Tree Model Implementation
BinomialTree::BinomialTree(size_t steps) : steps(steps) {}

double BinomialTree::price(double S, double K, double r, double sigma, double T, OptionType type, OptionStyle style) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }
    
    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp(r * dt) - d) / (u - d);
    double discount = std::exp(-r * dt);
    
    // Initialize the asset prices at maturity (final nodes)
    std::vector<double> prices(steps + 1);
    for (size_t i = 0; i <= steps; ++i) {
        prices[i] = S * std::pow(u, steps - i) * std::pow(d, i);
    }
    
    // Initialize option values at maturity
    std::vector<double> option_values(steps + 1);
    for (size_t i = 0; i <= steps; ++i) {
        if (type == OptionType::CALL) {
            option_values[i] = std::max(prices[i] - K, 0.0);
        } else {
            option_values[i] = std::max(K - prices[i], 0.0);
        }
    }
    
    // Work backwards through the tree
    for (int j = steps - 1; j >= 0; --j) {
        for (size_t i = 0; i <= j; ++i) {
            // Calculate the underlying asset price at this node
            double price = S * std::pow(u, j - i) * std::pow(d, i);
            
            // Calculate the option value at this node
            double option_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1]);
            
            // For American options, check if early exercise is optimal
            if (style == OptionStyle::AMERICAN) {
                double exercise_value = 0.0;
                if (type == OptionType::CALL) {
                    exercise_value = std::max(price - K, 0.0);
                } else {
                    exercise_value = std::max(K - price, 0.0);
                }
                option_value = std::max(option_value, exercise_value);
            }
            
            option_values[i] = option_value;
        }
    }
    
    return option_values[0];
}

double BinomialTree::price(double S, double K, double r, double sigma, double T, OptionType type) const {
    return price(S, K, r, sigma, T, type, OptionStyle::EUROPEAN);
}

double BinomialTree::delta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return S > K ? 1.0 : 0.0;
        } else {
            return S < K ? -1.0 : 0.0;
        }
    }
    
    // Calculate prices at first two nodes
    double dt = T / steps;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    
    double price_up = price(S * u, K, r, sigma, T - dt, type, OptionStyle::AMERICAN);
    double price_down = price(S * d, K, r, sigma, T - dt, type, OptionStyle::AMERICAN);
    
    return (price_up - price_down) / (S * u - S * d);
}

double BinomialTree::gamma(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01 * S; // Small change in stock price
    double delta_plus = delta(S + h, K, r, sigma, T, type);
    double delta_minus = delta(S - h, K, r, sigma, T, type);
    
    return (delta_plus - delta_minus) / (2 * h);
}

double BinomialTree::theta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in time (in years)
    double price_now = price(S, K, r, sigma, T, type, OptionStyle::AMERICAN);
    double price_later = price(S, K, r, sigma, T - h, type, OptionStyle::AMERICAN);
    
    return (price_later - price_now) / h;
}

double BinomialTree::vega(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in volatility
    double price_high = price(S, K, r, sigma + h, T, type, OptionStyle::AMERICAN);
    double price_low = price(S, K, r, sigma - h, T, type, OptionStyle::AMERICAN);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in volatility
}

double BinomialTree::rho(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    double h = 0.01; // Small change in interest rate
    double price_high = price(S, K, r + h, sigma, T, type, OptionStyle::AMERICAN);
    double price_low = price(S, K, r - h, sigma, T, type, OptionStyle::AMERICAN);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in interest rate
}

// Monte Carlo Model Implementation
MonteCarlo::MonteCarlo(size_t num_paths, size_t num_steps) 
    : num_paths(num_paths), num_steps(num_steps) {}

double MonteCarlo::price(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }
    
    double dt = T / num_steps;
    double sum_payoffs = 0.0;
    
    // Use matrix random number generator
    auto& engine = core::getRandomEngine();
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (size_t path = 0; path < num_paths; ++path) {
        double St = S;
        
        // Simulate price path
        for (size_t step = 0; step < num_steps; ++step) {
            double z = normal(engine);
            St *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * z);
        }
        
        // Calculate payoff
        double payoff = 0.0;
        if (type == OptionType::CALL) {
            payoff = std::max(St - K, 0.0);
        } else {
            payoff = std::max(K - St, 0.0);
        }
        
        sum_payoffs += payoff;
    }
    
    return std::exp(-r * T) * (sum_payoffs / num_paths);
}

double MonteCarlo::delta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return S > K ? 1.0 : 0.0;
        } else {
            return S < K ? -1.0 : 0.0;
        }
    }
    
    double h = 0.01 * S; // Small change in stock price
    double price_plus = price(S + h, K, r, sigma, T, type);
    double price_minus = price(S - h, K, r, sigma, T, type);
    
    return (price_plus - price_minus) / (2 * h);
}

double MonteCarlo::gamma(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01 * S; // Small change in stock price
    double price_plus = price(S + h, K, r, sigma, T, type);
    double price_center = price(S, K, r, sigma, T, type);
    double price_minus = price(S - h, K, r, sigma, T, type);
    
    return (price_plus - 2 * price_center + price_minus) / (h * h);
}

double MonteCarlo::theta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in time (in years)
    double price_now = price(S, K, r, sigma, T, type);
    double price_later = price(S, K, r, sigma, T - h, type);
    
    return (price_later - price_now) / h;
}

double MonteCarlo::vega(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in volatility
    double price_high = price(S, K, r, sigma + h, T, type);
    double price_low = price(S, K, r, sigma - h, T, type);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in volatility
}

double MonteCarlo::rho(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in interest rate
    double price_high = price(S, K, r + h, sigma, T, type);
    double price_low = price(S, K, r - h, sigma, T, type);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in interest rate
}

} // namespace options
} // namespace finml 