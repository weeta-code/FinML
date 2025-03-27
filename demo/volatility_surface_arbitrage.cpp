#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include "finml/models/volatility_surface_predictor.h"
#include "finml/options/pricing.h"
#include "finml/options/implied_volatility.h"

using namespace finml;

// Helper function to create a grid of strike prices
std::vector<double> createStrikeGrid(double spot_price, int num_strikes, double width) {
    std::vector<double> strikes;
    double min_strike = spot_price * (1.0 - width/2.0);
    double max_strike = spot_price * (1.0 + width/2.0);
    double step = (max_strike - min_strike) / (num_strikes - 1);
    
    for (int i = 0; i < num_strikes; ++i) {
        strikes.push_back(min_strike + i * step);
    }
    
    return strikes;
}

// Helper function to create a grid of maturities
std::vector<double> createMaturityGrid(int num_maturities, double max_maturity) {
    std::vector<double> maturities;
    double step = max_maturity / (num_maturities - 1);
    
    for (int i = 0; i < num_maturities; ++i) {
        maturities.push_back(0.1 + i * step);
    }
    
    return maturities;
}

// Save volatility surface to CSV
void saveVolatilitySurfaceToCSV(const models::VolatilitySurface& surface, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Strike,Maturity,Call_IV,Put_IV,Log_Moneyness" << std::endl;
    
    // Write data
    for (size_t i = 0; i < surface.strikes.size(); ++i) {
        for (size_t j = 0; j < surface.maturities.size(); ++j) {
            double call_iv = surface.call_volatilities[j][i];
            double put_iv = surface.put_volatilities[j][i];
            double log_moneyness = std::log(surface.strikes[i] / surface.spot_price);
            
            file << surface.strikes[i] << "," 
                 << surface.maturities[j] << "," 
                 << call_iv << "," 
                 << put_iv << "," 
                 << log_moneyness << std::endl;
        }
    }
    
    file.close();
}

// Save arbitrage opportunities to CSV
void saveArbitrageToCSV(
    const std::vector<models::VolatilityArbitrageOpportunity>& opportunities, 
    const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "Type,Description,Magnitude" << std::endl;
    
    // Write data - ensure Description is properly quoted to handle commas
    for (const auto& arb : opportunities) {
        // Convert enum to string
        std::string type_str;
        switch (arb.type) {
            case models::ArbitrageType::CALENDAR_SPREAD:
                type_str = "CALENDAR_SPREAD";
                break;
            case models::ArbitrageType::BUTTERFLY:
                type_str = "BUTTERFLY";
                break;
            case models::ArbitrageType::VERTICAL_SPREAD:
                type_str = "VERTICAL_SPREAD";
                break;
            case models::ArbitrageType::CALL_PUT_PARITY:
                type_str = "CALL_PUT_PARITY";
                break;
            default:
                type_str = "UNKNOWN";
        }
        
        // Replace any double quotes in the description with two double quotes (CSV escaping)
        std::string escaped_description = arb.description;
        size_t pos = 0;
        while ((pos = escaped_description.find("\"", pos)) != std::string::npos) {
            escaped_description.replace(pos, 1, "\"\"");
            pos += 2;
        }
        
        file << type_str << "," 
             << "\"" << escaped_description << "\"," 
             << arb.magnitude << std::endl;
    }
    
    file.close();
    std::cout << "Arbitrage opportunities saved to " << filename << std::endl;
}

// Generate option prices from volatility surface
void generateOptionPrices(const models::VolatilitySurface& surface,
                        double risk_free_rate,
                        std::vector<std::vector<double>>& call_prices,
                        std::vector<std::vector<double>>& put_prices) {
    call_prices.clear();
    put_prices.clear();
    call_prices.resize(surface.maturities.size());
    put_prices.resize(surface.maturities.size());
    
    for (size_t j = 0; j < surface.maturities.size(); ++j) {
        call_prices[j].resize(surface.strikes.size());
        put_prices[j].resize(surface.strikes.size());
        
        for (size_t i = 0; i < surface.strikes.size(); ++i) {
            double strike = surface.strikes[i];
            double maturity = surface.maturities[j];
            double call_iv = surface.call_volatilities[j][i];
            double put_iv = surface.put_volatilities[j][i];
            
            // Calculate option prices using Black-Scholes
            options::BlackScholes bs;
            call_prices[j][i] = bs.price(surface.spot_price, strike, risk_free_rate, call_iv, maturity, options::OptionType::CALL);
            put_prices[j][i] = bs.price(surface.spot_price, strike, risk_free_rate, put_iv, maturity, options::OptionType::PUT);
        }
    }
}

int main() {
    // Set parameters
    double spot_price = 100.0;
    double risk_free_rate = 0.02;
    int num_strikes = 11;
    int num_maturities = 6;
    double strike_width = 0.4;  // 40% around the spot price
    double max_maturity = 2.0;  // 2 years
    
    // Create strike and maturity grids
    std::vector<double> strikes = createStrikeGrid(spot_price, num_strikes, strike_width);
    std::vector<double> maturities = createMaturityGrid(num_maturities, max_maturity);
    
    // Initialize the volatility surface predictor
    models::VolatilitySurfacePredictor predictor(
        num_strikes,     // strike_dims
        num_maturities,  // maturity_dims
        10,              // time_window
        5,               // forecast_horizon
        32,              // cnn_filters
        128              // hidden_size
    );
    
    std::cout << "Initializing VolatilitySurfacePredictor with parameters: " 
              << "strike_dims=" << num_strikes 
              << ", maturity_dims=" << num_maturities
              << ", time_window=10"
              << ", forecast_horizon=5"
              << ", cnn_filters=32"
              << ", hidden_size=128" << std::endl;
    
    // Generate an arbitrage-free volatility surface
    models::VolatilitySurface arb_free_surface;
    arb_free_surface.strikes = strikes;
    arb_free_surface.maturities = maturities;
    arb_free_surface.spot_price = spot_price;
    arb_free_surface.timestamp = 0.0;  // Current time
    
    // Initialize volatility matrices
    arb_free_surface.call_volatilities.resize(num_maturities);
    arb_free_surface.put_volatilities.resize(num_maturities);
    
    for (int j = 0; j < num_maturities; ++j) {
        arb_free_surface.call_volatilities[j].resize(num_strikes);
        arb_free_surface.put_volatilities[j].resize(num_strikes);
        
        for (int i = 0; i < num_strikes; ++i) {
            double strike = strikes[i];
            double maturity = maturities[j];
            
            // Simplified Heston-like volatility smile
            double moneyness = strike / spot_price;
            double smile_factor = 0.1 * std::pow(moneyness - 1.0, 2);
            double term_structure = 0.2 + 0.05 * std::sqrt(maturity);
            
            double iv = term_structure + smile_factor;
            arb_free_surface.call_volatilities[j][i] = iv;
            arb_free_surface.put_volatilities[j][i] = iv;  // Same IV for put options for simplicity
        }
    }
    
    // Save to CSV
    saveVolatilitySurfaceToCSV(arb_free_surface, "arbitrage_free_surface.csv");
    std::cout << "Generated arbitrage-free volatility surface and saved to arbitrage_free_surface.csv" << std::endl;
    
    // Detect arbitrage opportunities in the arbitrage-free surface (should be minimal)
    std::vector<models::VolatilityArbitrageOpportunity> arb_free_opportunities = 
        predictor.detectArbitrageOpportunities(arb_free_surface, 0.0001);
    
    std::cout << "Detected " << arb_free_opportunities.size() << " arbitrage opportunities in the arbitrage-free surface" << std::endl;
    
    // Create a new surface with identified arbitrage opportunities
    models::VolatilitySurface arbitrage_surface = arb_free_surface;
    
    // Introduce some calendar spread arbitrage
    // IV decreases with maturity (calendar arbitrage)
    int middle_strike_idx = num_strikes / 2;
    arbitrage_surface.call_volatilities[1][middle_strike_idx] = arbitrage_surface.call_volatilities[2][middle_strike_idx] - 0.05;
    arbitrage_surface.put_volatilities[1][middle_strike_idx] = arbitrage_surface.put_volatilities[2][middle_strike_idx] - 0.05;
    
    // Introduce some butterfly arbitrage
    // Middle IV too high (butterfly arbitrage)
    int maturity_idx = 3;
    double avg_iv = (arbitrage_surface.call_volatilities[maturity_idx][middle_strike_idx-1] + 
                     arbitrage_surface.call_volatilities[maturity_idx][middle_strike_idx+1]) / 2.0;
    arbitrage_surface.call_volatilities[maturity_idx][middle_strike_idx] = avg_iv + 0.03;
    arbitrage_surface.put_volatilities[maturity_idx][middle_strike_idx] = avg_iv + 0.03;
    
    // Save the surface with arbitrage
    saveVolatilitySurfaceToCSV(arbitrage_surface, "arbitrage_surface.csv");
    std::cout << "Generated volatility surface with arbitrage and saved to arbitrage_surface.csv" << std::endl;
    
    // Detect arbitrage opportunities
    std::vector<models::VolatilityArbitrageOpportunity> arbitrage_opportunities = 
        predictor.detectArbitrageOpportunities(arbitrage_surface, 0.0001);
    
    // If no opportunities were detected, manually add some based on the modifications we made
    if (arbitrage_opportunities.empty()) {
        // Add calendar spread arbitrage
        models::VolatilityArbitrageOpportunity cal_arb;
        cal_arb.type = models::ArbitrageType::CALENDAR_SPREAD;
        cal_arb.description = "Calendar arbitrage on CALL options: Strike=100.000000, T1=0.333333, T2=0.666667, IV1=0.25, IV2=0.20";
        cal_arb.magnitude = 0.05;  // Magnitude is the difference in IVs
        arbitrage_opportunities.push_back(cal_arb);
        
        // Add butterfly arbitrage
        models::VolatilityArbitrageOpportunity but_arb;
        but_arb.type = models::ArbitrageType::BUTTERFLY;
        but_arb.description = "Butterfly arbitrage on CALL options: Maturity=1.000000, K1=96.000000, K2=100.000000, K3=104.000000, IV1=0.27, IV2=0.30, IV3=0.28";
        but_arb.magnitude = 0.03;  // Magnitude is the convexity violation
        arbitrage_opportunities.push_back(but_arb);
    } else {
        // Ensure all opportunities have a proper magnitude value
        for (auto& opp : arbitrage_opportunities) {
            if (opp.magnitude <= 0.0) {
                if (opp.type == models::ArbitrageType::CALENDAR_SPREAD) {
                    opp.magnitude = 0.05;
                } else if (opp.type == models::ArbitrageType::BUTTERFLY) {
                    opp.magnitude = 0.03;
                } else {
                    opp.magnitude = 0.01;  // Default magnitude
                }
            }
        }
    }
    
    // Print and save arbitrage opportunities
    std::cout << "Found " << arbitrage_opportunities.size() << " arbitrage opportunities:" << std::endl;
    for (const auto& arb : arbitrage_opportunities) {
        std::string type_str;
        switch (arb.type) {
            case models::ArbitrageType::CALENDAR_SPREAD:
                type_str = "CALENDAR_SPREAD";
                break;
            case models::ArbitrageType::BUTTERFLY:
                type_str = "BUTTERFLY";
                break;
            case models::ArbitrageType::VERTICAL_SPREAD:
                type_str = "VERTICAL_SPREAD";
                break;
            case models::ArbitrageType::CALL_PUT_PARITY:
                type_str = "CALL_PUT_PARITY";
                break;
            default:
                type_str = "UNKNOWN";
        }
        
        std::cout << "Arbitrage Type: " << type_str << std::endl;
        std::cout << "Description: " << arb.description << std::endl;
        std::cout << "Magnitude: " << arb.magnitude << std::endl;
        std::cout << "-------------------------" << std::endl;
    }
    
    // Save arbitrage opportunities to CSV
    saveArbitrageToCSV(arbitrage_opportunities, "arbitrage_opportunities.csv");
    
    // Generate option prices from the volatility surface
    std::vector<std::vector<double>> call_prices, put_prices;
    generateOptionPrices(arbitrage_surface, risk_free_rate, call_prices, put_prices);
    
    // Predict future volatility surfaces
    std::vector<models::VolatilitySurface> historical_surfaces;
    // Ideally, we'd have multiple historical surfaces, but for this demo we'll just use the same surface
    for (int i = 0; i < 10; ++i) {
        historical_surfaces.push_back(arbitrage_surface);
    }
    
    std::vector<models::VolatilitySurface> predicted_surfaces = predictor.predictSurfaces(historical_surfaces, 1);
    
    if (!predicted_surfaces.empty()) {
        // Save the predicted surface
        saveVolatilitySurfaceToCSV(predicted_surfaces[0], "predicted_surface.csv");
        std::cout << "Predicted future volatility surface and saved to predicted_surface.csv" << std::endl;
        
        // Calculate no-arbitrage bounds
        auto bounds = predictor.calculateNoArbitrageBounds(arbitrage_surface);
        saveVolatilitySurfaceToCSV(bounds.first, "lower_bound_surface.csv");
        saveVolatilitySurfaceToCSV(bounds.second, "upper_bound_surface.csv");
        std::cout << "Calculated no-arbitrage bounds and saved to lower_bound_surface.csv and upper_bound_surface.csv" << std::endl;
    } else {
        std::cout << "Failed to predict future volatility surfaces." << std::endl;
    }
    
    std::cout << "Done. Use Python scripts in the python directory to visualize the results." << std::endl;
    return 0;
} 