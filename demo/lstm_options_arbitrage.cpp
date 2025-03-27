#include "finml/models/lstm_stock_predictor.h"
#include "finml/options/pricing.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <map>

using namespace finml;

// Structure to represent an option contract
struct OptionContract {
    double strike;
    double market_price;
    double days_to_expiry;
    options::OptionType type;
    std::string symbol;
};

// Function to generate synthetic options data for demonstration
std::vector<OptionContract> generateSyntheticOptions(
    double current_price, 
    double current_volatility,
    const std::string& symbol, 
    bool include_arbitrage_opportunities = true) {
    
    std::vector<OptionContract> contracts;
    std::vector<double> strikes = {
        current_price * 0.90, current_price * 0.95, current_price * 0.975,
        current_price, 
        current_price * 1.025, current_price * 1.05, current_price * 1.10
    };
    
    std::vector<int> days_to_expiry = {7, 14, 30, 60, 90};
    
    options::BlackScholes bs_model;
    double risk_free_rate = 0.03; // 3% risk-free rate
    
    // Random generator for introducing noise and arbitrage opportunities
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> noise_dist(-0.1, 0.1);
    std::uniform_real_distribution<double> arb_dist(0.08, 0.20); // Larger price differences
    std::uniform_int_distribution<int> arb_opportunity_dist(0, 9); // 10% of options will have arbitrage
    
    for (double strike : strikes) {
        for (int days : days_to_expiry) {
            // Add call option
            double T = days / 365.0;
            double call_price = bs_model.price(
                current_price, strike, risk_free_rate, 
                current_volatility, T, options::OptionType::CALL);
            
            // Add noise to price
            call_price *= (1.0 + noise_dist(rng));
            
            // Potentially introduce an arbitrage opportunity
            bool create_arb = include_arbitrage_opportunities && arb_opportunity_dist(rng) == 0;
            if (create_arb) {
                call_price *= (1.0 + arb_dist(rng)); // Inflate price significantly
            }
            
            std::string call_symbol = symbol + std::to_string(days) + "C" + 
                                     std::to_string(static_cast<int>(strike * 100));
            
            contracts.push_back({
                strike, call_price, static_cast<double>(days), 
                options::OptionType::CALL, call_symbol
            });
            
            // Add put option
            double put_price = bs_model.price(
                current_price, strike, risk_free_rate, 
                current_volatility, T, options::OptionType::PUT);
            
            // Add noise to price
            put_price *= (1.0 + noise_dist(rng));
            
            // Potentially introduce an arbitrage opportunity
            create_arb = include_arbitrage_opportunities && arb_opportunity_dist(rng) == 0;
            if (create_arb) {
                put_price *= (1.0 + arb_dist(rng)); // Inflate price significantly
            }
            
            std::string put_symbol = symbol + std::to_string(days) + "P" + 
                                    std::to_string(static_cast<int>(strike * 100));
            
            contracts.push_back({
                strike, put_price, static_cast<double>(days), 
                options::OptionType::PUT, put_symbol
            });
        }
    }
    
    return contracts;
}

// Structure to store arbitrage opportunities
struct ArbitrageOpportunity {
    std::string symbol;
    double market_price;
    double model_price;
    double price_diff_pct;
    std::string action;
    std::string option_type;
    double strike;
    double days_to_expiry;
};

// Function to find arbitrage opportunities
std::vector<ArbitrageOpportunity> findArbitrageOpportunities(
    const std::vector<double>& predicted_prices,
    const std::vector<double>& predicted_volatility,
    const std::vector<OptionContract>& option_contracts,
    double risk_free_rate,
    double threshold_pct = 5.0) {
    
    // Create the pricing models
    options::BlackScholes bs_model;
    options::BinomialTree binomial_model(100);  // 100 steps
    
    std::cout << "Looking for arbitrage opportunities..." << std::endl;
    
    // Current stock price
    double current_price = predicted_prices[0];
    std::vector<ArbitrageOpportunity> opportunities;
    
    for (const auto& contract : option_contracts) {
        // Convert days to expiry to years
        double T = contract.days_to_expiry / 365.0;
        
        // Calculate theoretical prices using different models
        double bs_price = bs_model.price(
            current_price, contract.strike, risk_free_rate, 
            predicted_volatility[0], T, contract.type);
            
        double binomial_price = binomial_model.price(
            current_price, contract.strike, risk_free_rate, 
            predicted_volatility[0], T, contract.type);
        
        // Average model price
        double avg_model_price = (bs_price + binomial_price) / 2.0;
        
        // Calculate price difference (arbitrage potential)
        double price_diff = contract.market_price - avg_model_price;
        double price_diff_pct = std::abs(price_diff) / contract.market_price * 100.0;
        
        // If the difference is significant (e.g., > threshold%), it might be an arbitrage opportunity
        if (price_diff_pct > threshold_pct) {
            std::string action = price_diff > 0 ? "SELL" : "BUY";
            std::string option_type = (contract.type == options::OptionType::CALL) ? "CALL" : "PUT";
            
            ArbitrageOpportunity opportunity = {
                contract.symbol,
                contract.market_price,
                avg_model_price,
                price_diff_pct,
                action,
                option_type,
                contract.strike,
                contract.days_to_expiry
            };
            
            opportunities.push_back(opportunity);
            
            std::cout << "Potential arbitrage on " << contract.symbol 
                      << " " << option_type
                      << " (Strike: " << contract.strike 
                      << ", Expiry: " << contract.days_to_expiry << " days)" << std::endl;
                      
            std::cout << "  Market price: $" << std::fixed << std::setprecision(2) << contract.market_price << std::endl;
            std::cout << "  BS model price: $" << std::fixed << std::setprecision(2) << bs_price << std::endl;
            std::cout << "  Binomial model price: $" << std::fixed << std::setprecision(2) << binomial_price << std::endl;
            std::cout << "  Difference: $" << std::fixed << std::setprecision(2) << std::abs(price_diff) 
                      << " (" << std::fixed << std::setprecision(1) << price_diff_pct << "%)" << std::endl;
            std::cout << "  Recommended action: " << action << std::endl;
            std::cout << std::endl;
        }
    }
    
    return opportunities;
}

// Function to generate synthetic stock price data
std::vector<std::vector<double>> generateStockData(int num_days, double initial_price, double volatility) {
    std::vector<std::vector<double>> stock_data;
    
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::normal_distribution<double> return_dist(0.0001, volatility / std::sqrt(252.0));
    
    double current_price = initial_price;
    double high, low, open, close, volume;
    
    for (int i = 0; i < num_days; ++i) {
        double daily_return = return_dist(rng);
        open = current_price;
        close = open * std::exp(daily_return);
        
        // Generate realistic high, low, and volume
        high = std::max(open, close) * (1.0 + std::abs(return_dist(rng)) * 0.5);
        low = std::min(open, close) * (1.0 - std::abs(return_dist(rng)) * 0.5);
        volume = 1000000 * (1.0 + return_dist(rng) * 5.0);
        
        stock_data.push_back({open, high, low, close, volume});
        current_price = close;
    }
    
    return stock_data;
}

// Function to prepare training data for LSTM
void prepareTrainingData(
    const std::vector<std::vector<double>>& stock_data,
    int sequence_length,
    std::vector<std::vector<double>>& X_train,
    std::vector<std::vector<double>>& y_train) {
    
    // Calculate log returns and volatility
    std::vector<double> log_returns;
    for (size_t i = 1; i < stock_data.size(); ++i) {
        double prev_close = stock_data[i-1][3];
        double curr_close = stock_data[i][3];
        log_returns.push_back(std::log(curr_close / prev_close));
    }
    
    // Calculate volatility (20-day rolling standard deviation of returns)
    const int vol_window = 20;
    std::vector<double> volatility(log_returns.size(), 0.0);
    
    for (size_t i = vol_window - 1; i < log_returns.size(); ++i) {
        double sum_squared = 0.0;
        double mean = 0.0;
        
        for (int j = 0; j < vol_window; ++j) {
            mean += log_returns[i - j];
        }
        mean /= vol_window;
        
        for (int j = 0; j < vol_window; ++j) {
            double diff = log_returns[i - j] - mean;
            sum_squared += diff * diff;
        }
        
        volatility[i] = std::sqrt(sum_squared / (vol_window - 1)) * std::sqrt(252.0); // Annualized
    }
    
    // Create sequences for LSTM
    for (size_t i = vol_window; i < stock_data.size() - 1; ++i) {
        if (i < sequence_length) continue;
        
        std::vector<double> sequence;
        for (int j = 0; j < sequence_length; ++j) {
            double normalized_close = stock_data[i - j][3] / stock_data[i - sequence_length][3];
            double normalized_volume = stock_data[i - j][4] / 1000000.0;
            sequence.push_back(normalized_close);
            sequence.push_back(normalized_volume);
            sequence.push_back(volatility[i - j]);
        }
        
        X_train.push_back(sequence);
        y_train.push_back({stock_data[i+1][3] / stock_data[i][3], volatility[i]});
    }
}

// Function to save arbitrage opportunities to CSV
void saveArbitrageToCSV(const std::vector<ArbitrageOpportunity>& opportunities, const std::string& filename) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    
    // Write header
    file << "Symbol,Market Price,Model Price,Difference (%),Action,Option Type,Strike,Days to Expiry\n";
    
    // Write data
    for (const auto& opp : opportunities) {
        file << opp.symbol << ","
             << opp.market_price << ","
             << opp.model_price << ","
             << opp.price_diff_pct << ","
             << opp.action << ","
             << opp.option_type << ","
             << opp.strike << ","
             << opp.days_to_expiry << "\n";
    }
    
    file.close();
    std::cout << "Arbitrage opportunities saved to " << filename << std::endl;
}

int main() {
    // Setup parameters
    double initial_stock_price = 150.0;
    double initial_volatility = 0.25;
    double risk_free_rate = 0.03;
    std::string stock_symbol = "AAPL";
    int num_days = 500;
    int sequence_length = 20;
    
    std::cout << "Generating synthetic stock data..." << std::endl;
    auto stock_data = generateStockData(num_days, initial_stock_price, initial_volatility);
    double current_price = stock_data.back()[3]; // Last close price
    
    std::cout << "Preparing training data..." << std::endl;
    std::vector<std::vector<double>> X_train;
    std::vector<std::vector<double>> y_train;
    prepareTrainingData(stock_data, sequence_length, X_train, y_train);
    
    std::cout << "Generated " << X_train.size() << " training sequences" << std::endl;
    
    // Extract price and volatility targets
    std::vector<std::vector<double>> price_targets, volatility_targets;
    for (const auto& target : y_train) {
        price_targets.push_back({target[0]});
        volatility_targets.push_back({target[1]});
    }
    
    // Create and train LSTM models
    std::cout << "Training price prediction LSTM..." << std::endl;
    models::LSTMStockPredictor price_predictor(
        sequence_length * 3, // 3 features per time step: price, volume, volatility
        64,                  // Hidden size
        sequence_length,     // Sequence length
        1,                   // Output size (price return)
        42                   // Random seed
    );
    
    price_predictor.train(
        X_train,
        price_targets,
        10,     // Epochs
        0.001,  // Learning rate
        0.2,    // Dropout rate
        32      // Batch size
    );
    
    std::cout << "Training volatility prediction LSTM..." << std::endl;
    models::LSTMStockPredictor volatility_predictor(
        sequence_length * 3, // 3 features per time step: price, volume, volatility
        64,                  // Hidden size
        sequence_length,     // Sequence length
        1,                   // Output size (volatility)
        43                   // Different random seed
    );
    
    volatility_predictor.train(
        X_train,
        volatility_targets,
        10,     // Epochs
        0.001,  // Learning rate
        0.2,    // Dropout rate
        32      // Batch size
    );
    
    // Prepare last sequence for prediction
    std::vector<double> last_sequence;
    for (int j = 0; j < sequence_length; ++j) {
        double normalized_close = stock_data[stock_data.size() - 1 - j][3] / stock_data[stock_data.size() - sequence_length - 1][3];
        double normalized_volume = stock_data[stock_data.size() - 1 - j][4] / 1000000.0;
        double vol = (j < volatility_targets.size()) ? volatility_targets[volatility_targets.size() - 1 - j][0] : 0.2;
        last_sequence.push_back(normalized_close);
        last_sequence.push_back(normalized_volume);
        last_sequence.push_back(vol);
    }
    
    // Make predictions
    std::cout << "Making predictions..." << std::endl;
    std::vector<double> predicted_returns = price_predictor.predictNextDays(last_sequence, 30);
    std::vector<double> predicted_volatility = volatility_predictor.predictNextDays(last_sequence, 30);
    
    // Convert returns to prices
    std::vector<double> predicted_prices;
    double last_price = current_price;
    predicted_prices.push_back(last_price);
    
    for (double ret : predicted_returns) {
        last_price *= ret;
        predicted_prices.push_back(last_price);
    }
    
    // Print predictions
    std::cout << "Predicted prices for next 30 days:" << std::endl;
    for (size_t i = 0; i < predicted_prices.size() && i < 10; ++i) {
        std::cout << "Day " << i << ": $" << std::fixed << std::setprecision(2) << predicted_prices[i];
        if (i < predicted_volatility.size()) {
            std::cout << " (Volatility: " << std::fixed << std::setprecision(1) << predicted_volatility[i] * 100.0 << "%)";
        }
        std::cout << std::endl;
    }
    
    // Generate synthetic options data
    std::cout << "\nGenerating synthetic options data..." << std::endl;
    auto option_contracts = generateSyntheticOptions(current_price, predicted_volatility[0], stock_symbol);
    std::cout << "Generated " << option_contracts.size() << " option contracts" << std::endl;
    
    // Find arbitrage opportunities
    auto opportunities = findArbitrageOpportunities(predicted_prices, predicted_volatility, option_contracts, risk_free_rate);
    
    // Save results to CSV for visualization in Colab
    saveArbitrageToCSV(opportunities, "arbitrage_opportunities.csv");
    
    // Save predicted prices and volatility for visualization
    std::ofstream price_file("predicted_prices.csv");
    price_file << "Day,Price,Volatility\n";
    for (size_t i = 0; i < predicted_prices.size(); ++i) {
        price_file << i << "," << predicted_prices[i] << ",";
        if (i < predicted_volatility.size()) {
            price_file << predicted_volatility[i] << "\n";
        } else {
            price_file << "\n";
        }
    }
    price_file.close();
    
    std::cout << "Analysis complete! Check arbitrage_opportunities.csv and predicted_prices.csv for results." << std::endl;
    return 0;
} 