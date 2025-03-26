#include "finml/models/lstm_stock_predictor.h"
#include "finml/models/stock_pattern_cnn.h"
#include "finml/core/matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

// Structure to represent a labeled chart pattern
struct LabeledPattern {
    finml::core::Matrix image;
    bool is_bullish;
    std::string pattern_name;
};

// Hybrid trading model that combines LSTM predictions with CNN pattern recognition
class HybridTradingModel {
private:
    finml::models::LSTMStockPredictor lstm_model_;
    finml::models::StockPatternCNN cnn_model_;
    
    // Tunable parameters
    double lstm_weight_ = 0.7;
    double cnn_weight_ = 0.3;
    
public:
    HybridTradingModel(int input_size, int hidden_size, int sequence_length, int output_size,
                      int img_channels, int img_height, int img_width) 
        : lstm_model_(input_size, hidden_size, sequence_length, output_size, 42),
          cnn_model_(img_channels, img_height, img_width) {
        
        std::cout << "Hybrid Trading Model initialized with:" << std::endl
                  << "  LSTM parameters: input_size=" << input_size 
                  << ", hidden_size=" << hidden_size
                  << ", sequence_length=" << sequence_length
                  << ", output_size=" << output_size << std::endl
                  << "  CNN parameters: channels=" << img_channels
                  << ", height=" << img_height
                  << ", width=" << img_width << std::endl;
    }
    
    // Train the LSTM model
    void trainLSTM(const std::vector<std::vector<double>>& sequences,
                  const std::vector<std::vector<double>>& targets,
                  int epochs, double learning_rate) {
        
        std::cout << "Training LSTM model with " << sequences.size() << " sequences..." << std::endl;
        lstm_model_.train(sequences, targets, epochs, learning_rate, 0.2, 32, 0.2, 5);
    }
    
    // Generate a trading signal based on LSTM prediction and CNN pattern recognition
    std::pair<std::string, double> generateSignal(const std::vector<double>& sequence, 
                                                 const finml::core::Matrix& chart_pattern) {
        
        // Get LSTM prediction for next day
        std::vector<double> lstm_prediction = lstm_model_.predictNextDays(sequence, 1);
        double predicted_price_change = lstm_prediction[0] - sequence.back();
        bool lstm_bullish = predicted_price_change > 0;
        double lstm_confidence = std::abs(predicted_price_change) / sequence.back();
        
        // Normalize to a 0-1 scale with a cap at 5% price change
        lstm_confidence = std::min(lstm_confidence * 20.0, 1.0);
        
        // Get CNN pattern recognition signal
        auto [cnn_bullish, cnn_confidence] = cnn_model_.predict(chart_pattern);
        
        // Combine signals
        double bullish_score = (lstm_bullish ? lstm_weight_ * lstm_confidence : -lstm_weight_ * lstm_confidence) +
                              (cnn_bullish ? cnn_weight_ * cnn_confidence : -cnn_weight_ * cnn_confidence);
        
        std::string signal = "NEUTRAL";
        double confidence = 0.0;
        
        if (bullish_score > 0.15) {
            signal = "BUY";
            confidence = bullish_score;
        } else if (bullish_score < -0.15) {
            signal = "SELL";
            confidence = -bullish_score;
        } else {
            signal = "HOLD";
            confidence = 1.0 - std::abs(bullish_score) / 0.15;
        }
        
        std::cout << "Generated trading signal: " << signal
                  << " (confidence: " << confidence << ")" << std::endl
                  << "  LSTM contribution: " << (lstm_bullish ? "BULLISH" : "BEARISH")
                  << " with confidence " << lstm_confidence << std::endl
                  << "  CNN contribution: " << (cnn_bullish ? "BULLISH" : "BEARISH")
                  << " with confidence " << cnn_confidence << std::endl;
        
        return {signal, confidence};
    }
    
    // Run a backtest on historical data
    void runBacktest(const std::vector<std::vector<double>>& price_history,
                    const std::vector<finml::core::Matrix>& chart_patterns) {
        
        if (price_history.size() != chart_patterns.size() + 1) {
            std::cerr << "Error: Mismatched data sizes for backtesting" << std::endl;
            return;
        }
        
        double initial_balance = 10000.0;
        double balance = initial_balance;
        double shares = 0.0;
        
        std::cout << "\nRunning backtest with initial balance: $" << initial_balance << std::endl;
        
        for (size_t i = 0; i < chart_patterns.size(); ++i) {
            // Extract sequence for LSTM
            std::vector<double> sequence(price_history[i].begin(), price_history[i].end());
            
            // Generate trading signal
            auto [signal, confidence] = generateSignal(sequence, chart_patterns[i]);
            
            // Execute signal
            double current_price = price_history[i+1].back();
            double next_price = price_history[i+1].back();
            
            if (signal == "BUY" && balance > 0) {
                double amount = balance * confidence;
                shares += amount / current_price;
                balance -= amount;
                std::cout << "Day " << i << ": BUY " << (amount / current_price) 
                          << " shares at $" << current_price << std::endl;
            } else if (signal == "SELL" && shares > 0) {
                double amount = shares * confidence;
                balance += amount * current_price;
                shares -= amount;
                std::cout << "Day " << i << ": SELL " << amount 
                          << " shares at $" << current_price << std::endl;
            } else {
                std::cout << "Day " << i << ": HOLD position" << std::endl;
            }
            
            // Print daily summary
            double total_value = balance + shares * next_price;
            std::cout << "  Portfolio value: $" << total_value 
                      << " (Cash: $" << balance << ", Shares: " << shares 
                      << " worth $" << (shares * next_price) << ")" << std::endl;
        }
        
        // Final portfolio value
        double final_price = price_history.back().back();
        double final_value = balance + shares * final_price;
        double return_pct = (final_value - initial_balance) / initial_balance * 100.0;
        
        std::cout << "\nBacktest Results:" << std::endl;
        std::cout << "  Initial Balance: $" << initial_balance << std::endl;
        std::cout << "  Final Balance: $" << final_value << std::endl;
        std::cout << "  Return: " << return_pct << "%" << std::endl;
        std::cout << "  Final Position: $" << balance << " cash, " 
                  << shares << " shares worth $" << (shares * final_price) << std::endl;
    }
};

// Function to read CSV stock data
std::vector<std::vector<double>> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        
        // Skip date
        std::getline(ss, value, ',');
        
        // Read OHLCV values
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception& e) {
                row.push_back(0.0);
            }
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    return data;
}

// Function to generate synthetic chart patterns for CNN training
std::vector<LabeledPattern> generateSyntheticPatterns(int num_patterns, int height, int width) {
    std::vector<LabeledPattern> patterns;
    std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::cout << "Generating " << num_patterns << " synthetic chart patterns..." << std::endl;
    
    for (int i = 0; i < num_patterns; ++i) {
        finml::core::Matrix image(height, width);
        
        // Create a random pattern
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                double pixel_value = dist(rng);
                image.at(r, c) = finml::core::Value::create(pixel_value);
            }
        }
        
        // Randomly assign bullish or bearish
        bool is_bullish = dist(rng) > 0.5;
        
        // Create pattern
        LabeledPattern pattern;
        pattern.image = image;
        pattern.is_bullish = is_bullish;
        pattern.pattern_name = is_bullish ? "Bullish Pattern " + std::to_string(i) : "Bearish Pattern " + std::to_string(i);
        
        patterns.push_back(pattern);
    }
    
    return patterns;
}

// Prepare training data for LSTM model
void prepareTrainingData(const std::vector<std::vector<double>>& stock_data, 
                        int sequence_length,
                        std::vector<std::vector<double>>& sequences,
                        std::vector<std::vector<double>>& targets) {
    
    // Assuming stock_data has rows with [Open, High, Low, Close, Volume]
    // We'll use Close prices for simplicity
    
    // Extract close prices
    std::vector<double> close_prices;
    for (const auto& row : stock_data) {
        close_prices.push_back(row[3]); // Close price is at index 3
    }
    
    // Normalize prices
    double max_price = *std::max_element(close_prices.begin(), close_prices.end());
    for (auto& price : close_prices) {
        price /= max_price;
    }
    
    // Create sequences and targets
    for (size_t i = 0; i <= close_prices.size() - sequence_length - 1; ++i) {
        std::vector<double> sequence(close_prices.begin() + i, 
                                    close_prices.begin() + i + sequence_length);
        std::vector<double> target = {close_prices[i + sequence_length]};
        
        sequences.push_back(sequence);
        targets.push_back(target);
    }
    
    std::cout << "Prepared " << sequences.size() << " training sequences" << std::endl;
}

int main() {
    try {
        // Read stock data
        std::cout << "Reading stock data..." << std::endl;
        auto stock_data = readCSV("data/aapl_2013_2023.csv");
        
        if (stock_data.empty()) {
            std::cerr << "Error: No data loaded from CSV file" << std::endl;
            return 1;
        }
        
        std::cout << "Loaded " << stock_data.size() << " days of stock data" << std::endl;
        
        // LSTM model parameters
        const int input_size = 1;
        const int hidden_size = 64;
        const int sequence_length = 20;
        const int output_size = 1;
        
        // CNN model parameters
        const int img_channels = 1;
        const int img_height = 28;
        const int img_width = 28;
        
        // Prepare training data for LSTM
        std::vector<std::vector<double>> sequences;
        std::vector<std::vector<double>> targets;
        prepareTrainingData(stock_data, sequence_length, sequences, targets);
        
        // Generate synthetic patterns for CNN
        auto patterns = generateSyntheticPatterns(100, img_height, img_width);
        
        // Create hybrid model
        HybridTradingModel model(input_size, hidden_size, sequence_length, output_size,
                              img_channels, img_height, img_width);
        
        // Train LSTM model
        std::cout << "\nTraining LSTM model..." << std::endl;
        model.trainLSTM(sequences, targets, 50, 0.01);
        
        std::cout << "\nTraining complete. Running backtest..." << std::endl;
        
        // Prepare data for backtesting (using a subset of the data)
        const size_t test_size = std::min(static_cast<size_t>(30), stock_data.size() - sequence_length);
        std::vector<std::vector<double>> test_data(stock_data.end() - test_size - sequence_length, stock_data.end());
        
        // Create synthetic chart patterns for testing
        std::vector<finml::core::Matrix> test_patterns;
        for (size_t i = 0; i < test_size; ++i) {
            test_patterns.push_back(patterns[i % patterns.size()].image);
        }
        
        // Run backtest
        model.runBacktest(test_data, test_patterns);
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 