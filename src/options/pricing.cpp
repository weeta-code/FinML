#include "finml/options/pricing.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace finml {
namespace options {

// Black-Scholes Model Implementation
double BlackScholes::normalCDF(double x) const {
    return 0.5 * (1 + std::erf(x / std::sqrt(2.0)));
}

double BlackScholes::normalPDF(double x) const {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

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
        // At expiration
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return S * normalCDF(d1_val) - K * std::exp(-r * T) * normalCDF(d2_val);
    } else {
        return K * std::exp(-r * T) * normalCDF(-d2_val) - S * normalCDF(-d1_val);
    }
}

double BlackScholes::delta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return S > K ? 1.0 : 0.0;
        } else {
            return S < K ? -1.0 : 0.0;
        }
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return normalCDF(d1_val);
    } else {
        return normalCDF(d1_val) - 1.0;
    }
}

double BlackScholes::gamma(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0; // Gamma is zero at expiration or with zero volatility
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    return normalPDF(d1_val) / (S * sigma * std::sqrt(T));
}

double BlackScholes::theta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return -S * normalPDF(d1_val) * sigma / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * normalCDF(d2_val);
    } else {
        return -S * normalPDF(d1_val) * sigma / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * normalCDF(-d2_val);
    }
}

double BlackScholes::vega(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double d1_val = d1(S, K, r, sigma, T);
    return S * std::sqrt(T) * normalPDF(d1_val) / 100.0; // Divided by 100 to get the change per 1% change in volatility
}

double BlackScholes::rho(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double d2_val = d2(S, K, r, sigma, T);
    
    if (type == OptionType::CALL) {
        return K * T * std::exp(-r * T) * normalCDF(d2_val) / 100.0; // Divided by 100 to get the change per 1% change in interest rate
    } else {
        return -K * T * std::exp(-r * T) * normalCDF(-d2_val) / 100.0;
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
MonteCarlo::MonteCarlo(size_t num_simulations, size_t num_steps)
    : num_simulations(num_simulations), num_steps(num_steps) {}

double MonteCarlo::price(
    double S, 
    double K, 
    double r, 
    double sigma, 
    double T, 
    OptionType type, 
    OptionStyle style,
    std::function<double(double, double, OptionType)> payoff_func
) const {
    if (T <= 0.0) {
        // At expiration
        if (type == OptionType::CALL) {
            return std::max(S - K, 0.0);
        } else {
            return std::max(K - S, 0.0);
        }
    }
    
    // Default payoff function for European options
    if (!payoff_func) {
        payoff_func = [](double S_T, double K, OptionType type) {
            if (type == OptionType::CALL) {
                return std::max(S_T - K, 0.0);
            } else {
                return std::max(K - S_T, 0.0);
            }
        };
    }
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> normal(0.0, 1.0);
    
    double dt = T / num_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * std::sqrt(dt);
    
    double sum_payoffs = 0.0;
    
    for (size_t i = 0; i < num_simulations; ++i) {
        double S_t = S;
        
        // For Asian options, we need to track the average price
        double sum_prices = 0.0;
        
        // For barrier options, we need to track if the barrier was hit
        bool barrier_hit = false;
        
        // Simulate the price path
        for (size_t j = 0; j < num_steps; ++j) {
            double z = normal(gen);
            S_t = S_t * std::exp(drift + vol * z);
            
            // For Asian options, accumulate prices
            if (style == OptionStyle::ASIAN) {
                sum_prices += S_t;
            }
            
            // For American options, check for early exercise
            if (style == OptionStyle::AMERICAN) {
                double exercise_value = payoff_func(S_t, K, type);
                double continuation_value = 0.0; // This would require nested Monte Carlo, simplified here
                
                // If early exercise is optimal, add the payoff and break
                if (exercise_value > continuation_value) {
                    sum_payoffs += exercise_value * std::exp(-r * j * dt);
                    break;
                }
            }
            
            // For barrier options, check if barrier is hit (simplified)
            if (style == OptionStyle::BARRIER) {
                // Example: Down-and-out barrier at 0.8*S
                if (S_t < 0.8 * S) {
                    barrier_hit = true;
                    break;
                }
            }
        }
        
        // Calculate the payoff based on the option style
        double payoff = 0.0;
        
        if (style == OptionStyle::EUROPEAN || 
            (style == OptionStyle::AMERICAN && S_t == S_t) || // If we didn't break early
            (style == OptionStyle::BARRIER && !barrier_hit)) {
            payoff = payoff_func(S_t, K, type);
        } else if (style == OptionStyle::ASIAN) {
            double avg_price = sum_prices / num_steps;
            payoff = payoff_func(avg_price, K, type);
        }
        
        sum_payoffs += payoff * std::exp(-r * T);
    }
    
    return sum_payoffs / num_simulations;
}

double MonteCarlo::price(double S, double K, double r, double sigma, double T, OptionType type) const {
    return price(S, K, r, sigma, T, type, OptionStyle::EUROPEAN, nullptr);
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
    double price_plus = price(S + h, K, r, sigma, T, type, OptionStyle::AMERICAN);
    double price_minus = price(S - h, K, r, sigma, T, type, OptionStyle::AMERICAN);
    
    return (price_plus - price_minus) / (2 * h);
}

double MonteCarlo::gamma(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01 * S; // Small change in stock price
    double price_plus = price(S + h, K, r, sigma, T, type, OptionStyle::AMERICAN);
    double price_center = price(S, K, r, sigma, T, type, OptionStyle::AMERICAN);
    double price_minus = price(S - h, K, r, sigma, T, type, OptionStyle::AMERICAN);
    
    return (price_plus - 2 * price_center + price_minus) / (h * h);
}

double MonteCarlo::theta(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in time (in years)
    double price_now = price(S, K, r, sigma, T, type, OptionStyle::AMERICAN);
    double price_later = price(S, K, r, sigma, T - h, type, OptionStyle::AMERICAN);
    
    return (price_later - price_now) / h;
}

double MonteCarlo::vega(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in volatility
    double price_high = price(S, K, r, sigma + h, T, type, OptionStyle::AMERICAN);
    double price_low = price(S, K, r, sigma - h, T, type, OptionStyle::AMERICAN);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in volatility
}

double MonteCarlo::rho(double S, double K, double r, double sigma, double T, OptionType type) const {
    if (T <= 0.0) {
        return 0.0;
    }
    
    double h = 0.01; // Small change in interest rate
    double price_high = price(S, K, r + h, sigma, T, type, OptionStyle::AMERICAN);
    double price_low = price(S, K, r - h, sigma, T, type, OptionStyle::AMERICAN);
    
    return (price_high - price_low) / (2 * h) / 100.0; // Divided by 100 to get the change per 1% change in interest rate
}

} // namespace options
} // namespace finml 