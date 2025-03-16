#include "finml/options/implied_volatility.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace finml {
namespace options {

ImpliedVolatility::ImpliedVolatility(
    std::shared_ptr<PricingModel> model,
    double precision,
    size_t max_iterations
) : model(model), precision(precision), max_iterations(max_iterations) {}

double ImpliedVolatility::calculateBisection(
    double market_price,
    double S,
    double K,
    double r,
    double T,
    OptionType type
) const {
    // Check for valid inputs
    if (market_price <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 0.0) {
        throw std::invalid_argument("Market price, stock price, strike price, and time to maturity must be positive");
    }
    
    // Initial volatility bounds
    double sigma_low = 0.001;  // 0.1%
    double sigma_high = 5.0;   // 500%
    
    // Calculate option prices at bounds
    double price_low = model->price(S, K, r, sigma_low, T, type);
    double price_high = model->price(S, K, r, sigma_high, T, type);
    
    // Check if market price is within bounds
    if (market_price <= price_low) {
        return sigma_low;
    }
    if (market_price >= price_high) {
        return sigma_high;
    }
    
    // Bisection method
    double sigma_mid = 0.0;
    double price_mid = 0.0;
    
    for (size_t i = 0; i < max_iterations; ++i) {
        sigma_mid = (sigma_low + sigma_high) / 2.0;
        price_mid = model->price(S, K, r, sigma_mid, T, type);
        
        // Check for convergence
        if (std::abs(price_mid - market_price) < precision) {
            return sigma_mid;
        }
        
        // Update bounds
        if (price_mid < market_price) {
            sigma_low = sigma_mid;
        } else {
            sigma_high = sigma_mid;
        }
    }
    
    return sigma_mid;
}

double ImpliedVolatility::calculateNewtonRaphson(
    double market_price,
    double S,
    double K,
    double r,
    double T,
    OptionType type
) const {
    // Check for valid inputs
    if (market_price <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 0.0) {
        throw std::invalid_argument("Market price, stock price, strike price, and time to maturity must be positive");
    }
    
    // Initial volatility guess
    double sigma = 0.2;  // 20%
    
    // Newton-Raphson method
    for (size_t i = 0; i < max_iterations; ++i) {
        double price = model->price(S, K, r, sigma, T, type);
        double vega = model->vega(S, K, r, sigma, T, type) * 100.0; // Convert from per 1% to per 1.0
        
        // Check for convergence
        if (std::abs(price - market_price) < precision) {
            return sigma;
        }
        
        // Avoid division by zero
        if (std::abs(vega) < 1e-10) {
            break;
        }
        
        // Update volatility
        double delta_sigma = (market_price - price) / vega;
        sigma += delta_sigma;
        
        // Ensure volatility stays within reasonable bounds
        sigma = std::max(0.001, std::min(5.0, sigma));
        
        // Check for small updates
        if (std::abs(delta_sigma) < precision) {
            return sigma;
        }
    }
    
    // If Newton-Raphson fails, fall back to bisection
    return calculateBisection(market_price, S, K, r, T, type);
}

double ImpliedVolatility::calculateBrent(
    double market_price,
    double S,
    double K,
    double r,
    double T,
    OptionType type
) const {
    // Check for valid inputs
    if (market_price <= 0.0 || S <= 0.0 || K <= 0.0 || T <= 0.0) {
        throw std::invalid_argument("Market price, stock price, strike price, and time to maturity must be positive");
    }
    
    // Initial volatility bounds
    double a = 0.001;  // 0.1%
    double b = 5.0;    // 500%
    
    // Calculate option prices at bounds
    double fa = model->price(S, K, r, a, T, type) - market_price;
    double fb = model->price(S, K, r, b, T, type) - market_price;
    
    // Check if market price is within bounds
    if (fa * fb >= 0.0) {
        if (std::abs(fa) < std::abs(fb)) {
            return a;
        } else {
            return b;
        }
    }
    
    // Ensure a < b
    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    
    double c = a;
    double fc = fa;
    double d = 0.0;
    double s = 0.0;
    double fs = 0.0;
    bool mflag = true;
    
    // Brent's method
    for (size_t i = 0; i < max_iterations; ++i) {
        // Check if we need to use bisection
        if (fa * fc > 0.0 || std::abs(fc) >= std::abs(fa)) {
            c = a;
            fc = fa;
            d = b - a;
            mflag = true;
        }
        
        if (std::abs(fc) < std::abs(fa)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        
        // Convergence check
        double tol = 2.0 * precision * std::abs(b) + 0.5 * precision;
        double m = 0.5 * (c - b);
        
        if (std::abs(m) <= tol || fb == 0.0) {
            return b;
        }
        
        // Try interpolation
        if (mflag && std::abs(d) > tol) {
            s = fb / fa;
            
            if (a == c) {
                // Linear interpolation
                s = b - s * (b - a);
            } else {
                // Inverse quadratic interpolation
                double q = fa / fc;
                double r = fb / fc;
                s = b - r * (b - c) * (r - q) / ((q - r) * (fa - fb));
            }
            
            // Check if s is within bounds
            if (s < 0.25 * (3.0 * b + c) || s > b) {
                s = 0.5 * (b + c);
                mflag = false;
            } else {
                mflag = true;
            }
        } else {
            // Use bisection
            s = 0.5 * (b + c);
            mflag = false;
        }
        
        // Calculate function value at s
        fs = model->price(S, K, r, s, T, type) - market_price;
        
        // Update bounds
        d = a;
        a = b;
        fa = fb;
        
        if (fa * fs < 0.0) {
            b = s;
            fb = fs;
        } else {
            c = s;
            fc = fs;
        }
    }
    
    return b;
}

double ImpliedVolatility::calculate(
    double market_price,
    double S,
    double K,
    double r,
    double T,
    OptionType type
) const {
    // Try Newton-Raphson first (faster convergence)
    try {
        return calculateNewtonRaphson(market_price, S, K, r, T, type);
    } catch (const std::exception&) {
        // If Newton-Raphson fails, try Brent's method (more robust)
        try {
            return calculateBrent(market_price, S, K, r, T, type);
        } catch (const std::exception&) {
            // If Brent's method fails, fall back to bisection (most robust)
            return calculateBisection(market_price, S, K, r, T, type);
        }
    }
}

std::vector<std::vector<double>> ImpliedVolatility::calculateSurface(
    const std::vector<std::vector<double>>& market_prices,
    double S,
    const std::vector<double>& strikes,
    double r,
    const std::vector<double>& maturities,
    OptionType type
) const {
    // Check dimensions
    if (market_prices.size() != maturities.size()) {
        throw std::invalid_argument("Number of rows in market_prices must match number of maturities");
    }
    
    for (size_t i = 0; i < market_prices.size(); ++i) {
        if (market_prices[i].size() != strikes.size()) {
            throw std::invalid_argument("Number of columns in market_prices must match number of strikes");
        }
    }
    
    // Initialize implied volatility surface
    std::vector<std::vector<double>> iv_surface(maturities.size(), std::vector<double>(strikes.size(), 0.0));
    
    // Calculate implied volatility for each point
    for (size_t i = 0; i < maturities.size(); ++i) {
        for (size_t j = 0; j < strikes.size(); ++j) {
            try {
                iv_surface[i][j] = calculate(market_prices[i][j], S, strikes[j], r, maturities[i], type);
            } catch (const std::exception&) {
                // If calculation fails, set to NaN
                iv_surface[i][j] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    
    return iv_surface;
}

} // namespace options
} // namespace finml 