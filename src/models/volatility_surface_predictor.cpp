#include "finml/models/volatility_surface_predictor.h"
#include "finml/optim/loss.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace finml {
namespace models {

// Constructor
VolatilitySurfacePredictor::VolatilitySurfacePredictor(
    int strike_dims,
    int maturity_dims,
    int time_window,
    int forecast_horizon,
    int hidden_size,
    int cnn_filters,
    int cnn_kernel_size,
    double learning_rate,
    double dropout_rate,
    int seed
) : strike_dims(strike_dims),
    maturity_dims(maturity_dims),
    time_window(time_window),
    forecast_horizon(forecast_horizon),
    learning_rate(learning_rate),
    dropout_rate(dropout_rate),
    random_engine(seed) {
    
    // Input size for CNN: strike_dims * maturity_dims * 2 (call and put IVs)
    int input_channels = 2; // Call and put IV surfaces
    int input_features = strike_dims * maturity_dims;
    
    // Create CNN for spatial feature extraction
    spatial_features_extractor = std::make_unique<nn::Conv1D>(
        input_channels, 
        cnn_filters, 
        cnn_kernel_size, 
        1,  // stride
        cnn_kernel_size / 2, // padding (to maintain dimension)
        true, // use bias
        "SpatialFeatureExtractor"
    );
    
    // CNN output size
    int cnn_output_size = cnn_filters * input_features;
    
    // Create LSTM for temporal prediction
    temporal_predictor = std::make_unique<nn::LSTM>(
        cnn_output_size,
        hidden_size,
        true, // use bias
        "TemporalPredictor"
    );
    
    // Final output layer to project to volatility surface shape
    output_layer = std::make_unique<nn::Linear>(
        hidden_size,
        strike_dims * maturity_dims * 2, // 2 for call and put IVs
        true, // use bias
        "OutputProjection"
    );
    
    // Create decoder network (sequential of layers)
    decoder = std::make_unique<nn::Sequential>("SurfaceDecoder");
    // Add temporal_predictor and output_layer to decoder
    
    // Initialize SSVI model
    // Typical SSVI function: w(k, t, params) = theta * (1 + rho * phi(k) + sqrt((phi(k) + rho)^2 + (1-rho^2)))
    // where phi(k) is a function of log-moneyness
    ssvi_model = [](double k, double t, const SSVIParameters& params) -> double {
        // k is log-moneyness, t is time to maturity
        double phi_k = params.a * k + params.b * k * k;
        double w = params.theta * t * (1 + params.rho * phi_k + 
                  std::sqrt(std::pow(phi_k + params.rho, 2.0) + params.eta * params.eta * (1 - params.rho * params.rho)));
        return w; // total variance w = sigma^2 * t
    };
    
    // Initialize default SSVI parameters
    current_ssvi_params = {
        0.1,   // theta: overall level
        -0.7,  // rho: correlation parameter (-1 <= rho <= 1)
        0.3,   // eta: ATM volatility of vol
        0.1,   // lambda: density of the wings
        0.5,   // a: left wing parameter
        0.5    // b: right wing parameter
    };
    
    std::cout << "Initialized VolatilitySurfacePredictor with:"
              << " strike_dims=" << strike_dims
              << " maturity_dims=" << maturity_dims
              << " time_window=" << time_window
              << " forecast_horizon=" << forecast_horizon
              << " cnn_filters=" << cnn_filters
              << " hidden_size=" << hidden_size
              << std::endl;
}

// Convert volatility surface to matrix format for model input
core::Matrix VolatilitySurfacePredictor::surfaceToMatrix(const VolatilitySurface& surface) {
    // Create matrix with size (2, strike_dims * maturity_dims)
    // First channel is call volatilities, second channel is put volatilities
    core::Matrix result(2, strike_dims * maturity_dims);
    
    // Fill in the matrices
    for (int m = 0; m < maturity_dims; ++m) {
        for (int s = 0; s < strike_dims; ++s) {
            int index = m * strike_dims + s;
            
            // Ensure we don't access out of bounds
            if (m < surface.call_volatilities.size() && s < surface.call_volatilities[m].size()) {
                result.at(0, index) = core::Value::create(surface.call_volatilities[m][s]);
            } else {
                // Use default value if out of bounds
                result.at(0, index) = core::Value::create(0.2f); // Default IV of 20%
            }
            
            if (m < surface.put_volatilities.size() && s < surface.put_volatilities[m].size()) {
                result.at(1, index) = core::Value::create(surface.put_volatilities[m][s]);
            } else {
                // Use default value if out of bounds
                result.at(1, index) = core::Value::create(0.2f); // Default IV of 20%
            }
        }
    }
    
    return result;
}

// Convert matrix from model output back to volatility surface
VolatilitySurface VolatilitySurfacePredictor::matrixToSurface(
    const core::Matrix& matrix, 
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    double spot_price
) {
    VolatilitySurface result;
    result.strikes = strikes;
    result.maturities = maturities;
    result.spot_price = spot_price;
    result.timestamp = static_cast<double>(std::time(nullptr));
    
    // Resize the volatility arrays
    result.call_volatilities.resize(maturities.size());
    result.put_volatilities.resize(maturities.size());
    
    for (size_t m = 0; m < maturities.size(); ++m) {
        result.call_volatilities[m].resize(strikes.size());
        result.put_volatilities[m].resize(strikes.size());
        
        for (size_t s = 0; s < strikes.size(); ++s) {
            int index = m * strikes.size() + s;
            
            if (index < matrix.numCols()) {
                if (0 < matrix.numRows()) {
                    // Get call IV
                    result.call_volatilities[m][s] = matrix.at(0, index)->data;
                }
                
                if (1 < matrix.numRows()) {
                    // Get put IV
                    result.put_volatilities[m][s] = matrix.at(1, index)->data;
                }
            }
        }
    }
    
    return result;
}

// Train the model on historical volatility surfaces
void VolatilitySurfacePredictor::train(
    const std::vector<VolatilitySurface>& historical_surfaces,
    int epochs,
    int batch_size,
    double validation_split
) {
    if (historical_surfaces.size() < time_window + forecast_horizon) {
        throw std::invalid_argument("Not enough historical data for training");
    }
    
    std::cout << "Training model on " << historical_surfaces.size() << " historical surfaces..." << std::endl;
    
    // TODO: Implement full training functionality
    // This would involve:
    // 1. Creating sequences of surfaces
    // 2. Forward pass through CNN+LSTM
    // 3. Computing loss
    // 4. Backward pass
    // 5. Updating weights
    
    std::cout << "Training completed (dummy implementation)" << std::endl;
}

// Predict future volatility surfaces
std::vector<VolatilitySurface> VolatilitySurfacePredictor::predictSurfaces(
    const std::vector<VolatilitySurface>& recent_surfaces,
    int num_steps
) {
    if (recent_surfaces.size() < time_window) {
        throw std::invalid_argument("Not enough recent data for prediction");
    }
    
    std::cout << "Predicting " << num_steps << " future volatility surfaces..." << std::endl;
    
    // Use the most recent strikes, maturities, and spot price for output
    const auto& last_surface = recent_surfaces.back();
    std::vector<double> strikes = last_surface.strikes;
    std::vector<double> maturities = last_surface.maturities;
    double spot_price = last_surface.spot_price;
    
    // Generate a no-arbitrage surface for demonstration
    // In a real implementation, this would use the CNN-LSTM prediction
    std::vector<VolatilitySurface> predicted_surfaces;
    
    for (int step = 0; step < num_steps; ++step) {
        // Use SSVI parameters to generate an arbitrage-free surface
        // In real implementation, would adjust parameters based on model prediction
        SSVIParameters params = current_ssvi_params;
        
        // Slight random adjustments for demo purposes
        std::uniform_real_distribution<double> dist(-0.01, 0.01);
        params.theta += dist(random_engine);
        params.rho = std::clamp(params.rho + dist(random_engine), -0.99, 0.99);
        params.eta += dist(random_engine);
        
        VolatilitySurface predicted_surface = generateArbitrageFreeSSVISurface(
            params, strikes, maturities, spot_price);
        predicted_surface.timestamp = last_surface.timestamp + (step + 1) * (24 * 60 * 60); // Add days
        
        predicted_surfaces.push_back(predicted_surface);
    }
    
    std::cout << "Prediction completed (using SSVI parameterization)" << std::endl;
    return predicted_surfaces;
}

// Generate a volatility surface using SSVI parameterization
VolatilitySurface VolatilitySurfacePredictor::generateArbitrageFreeSSVISurface(
    const SSVIParameters& params,
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    double spot_price
) {
    VolatilitySurface surface;
    surface.strikes = strikes;
    surface.maturities = maturities;
    surface.spot_price = spot_price;
    surface.timestamp = static_cast<double>(std::time(nullptr));
    
    // Resize the volatility arrays
    surface.call_volatilities.resize(maturities.size());
    surface.put_volatilities.resize(maturities.size());
    
    for (size_t m = 0; m < maturities.size(); ++m) {
        double t = maturities[m];
        surface.call_volatilities[m].resize(strikes.size());
        surface.put_volatilities[m].resize(strikes.size());
        
        for (size_t s = 0; s < strikes.size(); ++s) {
            double k = strikes[s];
            // Calculate log-moneyness
            double log_moneyness = std::log(k / spot_price);
            
            // Calculate total variance using SSVI model
            double w = ssvi_model(log_moneyness, t, params);
            
            // Convert total variance to volatility
            double iv = std::sqrt(w / t);
            
            // Use same IV for both call and put (assuming put-call parity holds)
            surface.call_volatilities[m][s] = iv;
            surface.put_volatilities[m][s] = iv;
        }
    }
    
    return surface;
}

// Detect calendar spread arbitrage opportunities
std::vector<VolatilityArbitrageOpportunity> VolatilitySurfacePredictor::detectCalendarArbitrage(
    const VolatilitySurface& surface
) {
    std::vector<VolatilityArbitrageOpportunity> opportunities;
    
    // Calendar arbitrage exists when volatility does not increase with maturity
    // for the same strike price (i.e., vega risk cannot decrease with time)
    for (size_t s = 0; s < surface.strikes.size(); ++s) {
        double strike = surface.strikes[s];
        
        for (size_t m = 0; m < surface.maturities.size() - 1; ++m) {
            double t1 = surface.maturities[m];
            double t2 = surface.maturities[m + 1];
            
            // Check call options
            double iv1_call = surface.call_volatilities[m][s];
            double iv2_call = surface.call_volatilities[m + 1][s];
            
            // Calculate total variance (w = sigma^2 * t)
            double w1_call = iv1_call * iv1_call * t1;
            double w2_call = iv2_call * iv2_call * t2;
            
            // Calendar arbitrage condition: w2 < w1 (total variance should be non-decreasing with time)
            if (w2_call < w1_call) {
                VolatilityArbitrageOpportunity opportunity;
                opportunity.type = ArbitrageType::CALENDAR_SPREAD;
                opportunity.magnitude = w1_call - w2_call;
                
                // Record the involved points
                VolatilitySurfacePoint point1 = {strike, t1, iv1_call, options::OptionType::CALL};
                VolatilitySurfacePoint point2 = {strike, t2, iv2_call, options::OptionType::CALL};
                opportunity.involved_points = {point1, point2};
                
                opportunity.description = "Calendar arbitrage on CALL options: Strike=" + std::to_string(strike) +
                                         ", T1=" + std::to_string(t1) + ", T2=" + std::to_string(t2) +
                                         ", IV1=" + std::to_string(iv1_call) + ", IV2=" + std::to_string(iv2_call);
                
                opportunity.timestamp = surface.timestamp;
                opportunities.push_back(opportunity);
            }
            
            // Check put options
            double iv1_put = surface.put_volatilities[m][s];
            double iv2_put = surface.put_volatilities[m + 1][s];
            
            // Calculate total variance
            double w1_put = iv1_put * iv1_put * t1;
            double w2_put = iv2_put * iv2_put * t2;
            
            // Calendar arbitrage condition
            if (w2_put < w1_put) {
                VolatilityArbitrageOpportunity opportunity;
                opportunity.type = ArbitrageType::CALENDAR_SPREAD;
                opportunity.magnitude = w1_put - w2_put;
                
                // Record the involved points
                VolatilitySurfacePoint point1 = {strike, t1, iv1_put, options::OptionType::PUT};
                VolatilitySurfacePoint point2 = {strike, t2, iv2_put, options::OptionType::PUT};
                opportunity.involved_points = {point1, point2};
                
                opportunity.description = "Calendar arbitrage on PUT options: Strike=" + std::to_string(strike) +
                                         ", T1=" + std::to_string(t1) + ", T2=" + std::to_string(t2) +
                                         ", IV1=" + std::to_string(iv1_put) + ", IV2=" + std::to_string(iv2_put);
                
                opportunity.timestamp = surface.timestamp;
                opportunities.push_back(opportunity);
            }
        }
    }
    
    return opportunities;
}

// Detect butterfly arbitrage opportunities
std::vector<VolatilityArbitrageOpportunity> VolatilitySurfacePredictor::detectButterflyArbitrage(
    const VolatilitySurface& surface
) {
    std::vector<VolatilityArbitrageOpportunity> opportunities;
    
    // Butterfly arbitrage exists when the volatility smile is not convex
    // i.e., the second derivative of IV with respect to strike is negative
    for (size_t m = 0; m < surface.maturities.size(); ++m) {
        double t = surface.maturities[m];
        
        for (size_t s = 0; s < surface.strikes.size() - 2; ++s) {
            double k1 = surface.strikes[s];
            double k2 = surface.strikes[s + 1];
            double k3 = surface.strikes[s + 2];
            
            // Check convexity for call options
            double iv1_call = surface.call_volatilities[m][s];
            double iv2_call = surface.call_volatilities[m][s + 1];
            double iv3_call = surface.call_volatilities[m][s + 2];
            
            // Calculate total variance
            double w1_call = iv1_call * iv1_call * t;
            double w2_call = iv2_call * iv2_call * t;
            double w3_call = iv3_call * iv3_call * t;
            
            // Convert to normalized variance (d^2w/dk^2 must be positive)
            double log_k1 = std::log(k1 / surface.spot_price);
            double log_k2 = std::log(k2 / surface.spot_price);
            double log_k3 = std::log(k3 / surface.spot_price);
            
            // Calculate discrete second derivative
            double first_diff1 = (w2_call - w1_call) / (log_k2 - log_k1);
            double first_diff2 = (w3_call - w2_call) / (log_k3 - log_k2);
            double second_diff = (first_diff2 - first_diff1) / (0.5 * (log_k3 - log_k1));
            
            // Butterfly arbitrage condition: second derivative is negative
            if (second_diff < 0) {
                VolatilityArbitrageOpportunity opportunity;
                opportunity.type = ArbitrageType::BUTTERFLY;
                opportunity.magnitude = std::abs(second_diff);
                
                // Record the involved points
                VolatilitySurfacePoint point1 = {k1, t, iv1_call, options::OptionType::CALL};
                VolatilitySurfacePoint point2 = {k2, t, iv2_call, options::OptionType::CALL};
                VolatilitySurfacePoint point3 = {k3, t, iv3_call, options::OptionType::CALL};
                opportunity.involved_points = {point1, point2, point3};
                
                opportunity.description = "Butterfly arbitrage on CALL options: Maturity=" + std::to_string(t) +
                                         ", K1=" + std::to_string(k1) + ", K2=" + std::to_string(k2) + 
                                         ", K3=" + std::to_string(k3);
                
                opportunity.timestamp = surface.timestamp;
                opportunities.push_back(opportunity);
            }
            
            // Check convexity for put options
            double iv1_put = surface.put_volatilities[m][s];
            double iv2_put = surface.put_volatilities[m][s + 1];
            double iv3_put = surface.put_volatilities[m][s + 2];
            
            // Calculate total variance
            double w1_put = iv1_put * iv1_put * t;
            double w2_put = iv2_put * iv2_put * t;
            double w3_put = iv3_put * iv3_put * t;
            
            // Calculate discrete second derivative
            first_diff1 = (w2_put - w1_put) / (log_k2 - log_k1);
            first_diff2 = (w3_put - w2_put) / (log_k3 - log_k2);
            second_diff = (first_diff2 - first_diff1) / (0.5 * (log_k3 - log_k1));
            
            // Butterfly arbitrage condition
            if (second_diff < 0) {
                VolatilityArbitrageOpportunity opportunity;
                opportunity.type = ArbitrageType::BUTTERFLY;
                opportunity.magnitude = std::abs(second_diff);
                
                // Record the involved points
                VolatilitySurfacePoint point1 = {k1, t, iv1_put, options::OptionType::PUT};
                VolatilitySurfacePoint point2 = {k2, t, iv2_put, options::OptionType::PUT};
                VolatilitySurfacePoint point3 = {k3, t, iv3_put, options::OptionType::PUT};
                opportunity.involved_points = {point1, point2, point3};
                
                opportunity.description = "Butterfly arbitrage on PUT options: Maturity=" + std::to_string(t) +
                                         ", K1=" + std::to_string(k1) + ", K2=" + std::to_string(k2) + 
                                         ", K3=" + std::to_string(k3);
                
                opportunity.timestamp = surface.timestamp;
                opportunities.push_back(opportunity);
            }
        }
    }
    
    return opportunities;
}

// Detect vertical spread arbitrage
std::vector<VolatilityArbitrageOpportunity> VolatilitySurfacePredictor::detectVerticalArbitrage(
    const VolatilitySurface& surface
) {
    // Vertical spread arbitrage is not directly visible in IV space
    // It requires computing option prices and checking for violations
    // This is a simplified implementation
    return {};
}

// Detect call-put parity arbitrage
std::vector<VolatilityArbitrageOpportunity> VolatilitySurfacePredictor::detectCallPutParityArbitrage(
    const VolatilitySurface& surface
) {
    // Call-put parity arbitrage is not directly visible in IV space
    // It requires computing option prices and checking for violations
    // This is a simplified implementation
    return {};
}

// Main method to detect all types of arbitrage opportunities
std::vector<VolatilityArbitrageOpportunity> VolatilitySurfacePredictor::detectArbitrageOpportunities(
    const VolatilitySurface& surface,
    double threshold
) {
    std::vector<VolatilityArbitrageOpportunity> opportunities;
    
    // Detect calendar spread arbitrage
    auto calendar_arb = detectCalendarArbitrage(surface);
    opportunities.insert(opportunities.end(), calendar_arb.begin(), calendar_arb.end());
    
    // Detect butterfly arbitrage
    auto butterfly_arb = detectButterflyArbitrage(surface);
    opportunities.insert(opportunities.end(), butterfly_arb.begin(), butterfly_arb.end());
    
    // Filter out opportunities with magnitude below threshold
    opportunities.erase(
        std::remove_if(opportunities.begin(), opportunities.end(),
                      [threshold](const VolatilityArbitrageOpportunity& opp) {
                          return opp.magnitude < threshold;
                      }),
        opportunities.end()
    );
    
    return opportunities;
}

// Create volatility surface from market prices
VolatilitySurface VolatilitySurfacePredictor::createVolatilitySurfaceFromMarketPrices(
    const std::vector<std::vector<double>>& call_prices,
    const std::vector<std::vector<double>>& put_prices,
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    double spot_price,
    double risk_free_rate
) {
    VolatilitySurface surface;
    surface.strikes = strikes;
    surface.maturities = maturities;
    surface.spot_price = spot_price;
    surface.timestamp = static_cast<double>(std::time(nullptr));
    
    // Create implied volatility calculator
    options::ImpliedVolatility iv_calculator;
    
    // Resize volatility arrays
    surface.call_volatilities.resize(maturities.size());
    surface.put_volatilities.resize(maturities.size());
    
    for (size_t m = 0; m < maturities.size(); ++m) {
        surface.call_volatilities[m].resize(strikes.size());
        surface.put_volatilities[m].resize(strikes.size());
        
        for (size_t s = 0; s < strikes.size(); ++s) {
            // Calculate call IV
            if (m < call_prices.size() && s < call_prices[m].size()) {
                try {
                    surface.call_volatilities[m][s] = iv_calculator.calculate(
                        call_prices[m][s],
                        spot_price,
                        strikes[s],
                        risk_free_rate,
                        maturities[m],
                        options::OptionType::CALL
                    );
                } catch (const std::exception& e) {
                    // Use a default value if calculation fails
                    std::cerr << "Call IV calculation failed: " << e.what() << std::endl;
                    surface.call_volatilities[m][s] = 0.2; // 20% default
                }
            } else {
                surface.call_volatilities[m][s] = 0.2; // Default
            }
            
            // Calculate put IV
            if (m < put_prices.size() && s < put_prices[m].size()) {
                try {
                    surface.put_volatilities[m][s] = iv_calculator.calculate(
                        put_prices[m][s],
                        spot_price,
                        strikes[s],
                        risk_free_rate,
                        maturities[m],
                        options::OptionType::PUT
                    );
                } catch (const std::exception& e) {
                    // Use a default value if calculation fails
                    std::cerr << "Put IV calculation failed: " << e.what() << std::endl;
                    surface.put_volatilities[m][s] = 0.2; // 20% default
                }
            } else {
                surface.put_volatilities[m][s] = 0.2; // Default
            }
        }
    }
    
    return surface;
}

// Generate synthetic volatility surface for testing
VolatilitySurface VolatilitySurfacePredictor::generateSyntheticSurface(
    const std::vector<double>& strikes,
    const std::vector<double>& maturities,
    double spot_price,
    bool include_arbitrage_opportunities,
    double arbitrage_magnitude
) {
    // Start with an arbitrage-free SSVI surface
    VolatilitySurface surface = generateArbitrageFreeSSVISurface(
        current_ssvi_params, strikes, maturities, spot_price);
    
    if (include_arbitrage_opportunities) {
        // Add calendar arbitrage - violate the total variance monotonicity
        if (!maturities.empty() && maturities.size() > 1) {
            size_t m1 = maturities.size() / 3;
            size_t m2 = m1 + 1;
            
            if (m1 < surface.call_volatilities.size() && m2 < surface.call_volatilities.size()) {
                // Pick a random strike to introduce arbitrage
                std::uniform_int_distribution<size_t> strike_dist(0, strikes.size() - 1);
                size_t s = strike_dist(random_engine);
                
                if (s < surface.call_volatilities[m1].size() && s < surface.call_volatilities[m2].size()) {
                    // Increase IV for shorter maturity to create calendar arbitrage
                    double t1 = maturities[m1];
                    double t2 = maturities[m2];
                    
                    // Current total variance
                    double iv1 = surface.call_volatilities[m1][s];
                    double iv2 = surface.call_volatilities[m2][s];
                    double w1 = iv1 * iv1 * t1;
                    double w2 = iv2 * iv2 * t2;
                    
                    // Increase w1 to be greater than w2
                    double new_w1 = w2 + arbitrage_magnitude;
                    double new_iv1 = std::sqrt(new_w1 / t1);
                    
                    // Update the IV
                    surface.call_volatilities[m1][s] = new_iv1;
                    surface.put_volatilities[m1][s] = new_iv1; // Keep put-call parity
                    
                    std::cout << "Added calendar arbitrage at strike=" << strikes[s]
                              << ", t1=" << t1 << ", t2=" << t2
                              << ", new_iv1=" << new_iv1 << ", iv2=" << iv2 << std::endl;
                }
            }
        }
        
        // Add butterfly arbitrage - violate convexity
        if (!strikes.empty() && strikes.size() > 2) {
            size_t s1 = strikes.size() / 3;
            size_t s2 = s1 + 1;
            size_t s3 = s2 + 1;
            
            std::uniform_int_distribution<size_t> maturity_dist(0, maturities.size() - 1);
            size_t m = maturity_dist(random_engine);
            
            if (m < surface.call_volatilities.size() && 
                s1 < surface.call_volatilities[m].size() &&
                s2 < surface.call_volatilities[m].size() &&
                s3 < surface.call_volatilities[m].size()) {
                
                double t = maturities[m];
                double k1 = strikes[s1];
                double k2 = strikes[s2];
                double k3 = strikes[s3];
                
                // Current total variance
                double iv1 = surface.call_volatilities[m][s1];
                double iv2 = surface.call_volatilities[m][s2];
                double iv3 = surface.call_volatilities[m][s3];
                
                double w1 = iv1 * iv1 * t;
                double w2 = iv2 * iv2 * t;
                double w3 = iv3 * iv3 * t;
                
                // Make w2 higher to break convexity
                double expected_w2 = w1 + (w3 - w1) * (k2 - k1) / (k3 - k1);
                double new_w2 = expected_w2 + arbitrage_magnitude;
                double new_iv2 = std::sqrt(new_w2 / t);
                
                // Update the IV
                surface.call_volatilities[m][s2] = new_iv2;
                surface.put_volatilities[m][s2] = new_iv2; // Keep put-call parity
                
                std::cout << "Added butterfly arbitrage at t=" << t
                          << ", k1=" << k1 << ", k2=" << k2 << ", k3=" << k3
                          << ", new_iv2=" << new_iv2 << std::endl;
            }
        }
    }
    
    return surface;
}

// Fit SSVI parameters to a volatility surface (simplified implementation)
SSVIParameters VolatilitySurfacePredictor::fitSSVIParameters(const VolatilitySurface& surface) {
    // This would normally use optimization to fit SSVI params to the surface
    // For simplicity, return the current parameters with small adjustments
    SSVIParameters params = current_ssvi_params;
    
    // Make small random adjustments
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    params.theta += dist(random_engine);
    params.rho = std::clamp(params.rho + dist(random_engine), -0.99, 0.99);
    params.eta += dist(random_engine);
    
    return params;
}

// Calculate no-arbitrage bounds for a volatility surface
std::pair<VolatilitySurface, VolatilitySurface> VolatilitySurfacePredictor::calculateNoArbitrageBounds(
    const VolatilitySurface& surface
) {
    // This is a complex calculation in practice
    // For simplicity, create surfaces with slightly higher/lower IVs
    VolatilitySurface lower_bound = surface;
    VolatilitySurface upper_bound = surface;
    
    // Adjust IVs slightly for the bounds
    for (size_t m = 0; m < surface.maturities.size(); ++m) {
        for (size_t s = 0; s < surface.strikes.size(); ++s) {
            double iv_call = surface.call_volatilities[m][s];
            double iv_put = surface.put_volatilities[m][s];
            
            // Create bounds
            lower_bound.call_volatilities[m][s] = iv_call * 0.95;
            lower_bound.put_volatilities[m][s] = iv_put * 0.95;
            
            upper_bound.call_volatilities[m][s] = iv_call * 1.05;
            upper_bound.put_volatilities[m][s] = iv_put * 1.05;
        }
    }
    
    return {lower_bound, upper_bound};
}

// Save model weights (placeholder implementation)
bool VolatilitySurfacePredictor::saveModel(const std::string& filename) {
    std::cout << "Model saving not implemented yet. Would save to: " << filename << std::endl;
    return false;
}

// Load model weights (placeholder implementation)
bool VolatilitySurfacePredictor::loadModel(const std::string& filename) {
    std::cout << "Model loading not implemented yet. Would load from: " << filename << std::endl;
    return false;
}

} // namespace models
} // namespace finml 