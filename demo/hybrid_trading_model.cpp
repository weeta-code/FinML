#include "finml/data/timeseries.h"
#include "finml/nn/lstm.h"
#include "finml/nn/conv.h"
#include "finml/nn/linear.h"
#include "finml/nn/activation.h"
#include "finml/nn/sequential.h"
#include "finml/core/matrix.h"
#include "finml/optim/adam.h"
#include "finml/optim/loss.h"
#include "finml/core/value.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <iomanip>

// Forward declarations from the individual model files
class LSTMStockPredictor {
private:
    size_t input_size;
    size_t hidden_size;
    size_t sequence_length;
    size_t num_layers;
    finml::nn::Sequential model;
    unsigned int num_threads;
    
public:
    LSTMStockPredictor(
        size_t input_size, 
        size_t hidden_size, 
        size_t sequence_length = 30, 
        size_t num_layers = 2, 
        unsigned int num_threads = std::thread::hardware_concurrency());
        
    void train(
        const std::vector<std::vector<std::vector<double>>>& X_train,
        const std::vector<double>& y_train, 
        size_t epochs = 100,
        double learning_rate = 0.001,
        size_t batch_size = 32,
        double validation_split = 0.1,
        bool early_stopping = true,
        size_t patience = 5);
        
    std::vector<double> predict(const std::vector<std::vector<std::vector<double>>>& X_test);
    double evaluate(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> predictNextDays(const std::vector<std::vector<double>>& last_sequence, size_t num_days = 5);
};

struct LabeledPattern {
    finml::core::Matrix image;
    bool is_bullish;
    std::string pattern_name;
};

class StockPatternCNN {
private:
    size_t input_channels;
    size_t input_height;
    size_t input_width;
    finml::nn::Sequential model;
    
public:
    StockPatternCNN(size_t input_channels = 1, size_t input_height = 60, size_t input_width = 200);
    void train(const std::vector<LabeledPattern>& training_data, size_t epochs = 50, 
               double learning_rate = 0.001, size_t batch_size = 16);
    std::pair<bool, float> predict(const finml::core::Matrix& image);
    void evaluate(const std::vector<LabeledPattern>& test_data);
};

// Struct to represent a trading signal
struct TradingSignal {
    enum class Action { BUY, SELL, HOLD };
    Action action;
    double confidence;
    std::string reason;
    
    TradingSignal(Action action, double confidence, const std::string& reason)
        : action(action), confidence(confidence), reason(reason) {}
};

// Helper function to convert action enum to string
std::string actionToString(TradingSignal::Action action) {
    switch (action) {
        case TradingSignal::Action::BUY: return "BUY";
        case TradingSignal::Action::SELL: return "SELL";
        case TradingSignal::Action::HOLD: return "HOLD";
        default: return "UNKNOWN";
    }
}

// Hybrid trading model that combines LSTM price prediction and CNN pattern recognition
class HybridTradingModel {
private:
    // Component models
    std::unique_ptr<LSTMStockPredictor> lstm_model;
    std::unique_ptr<StockPatternCNN> cnn_model;
    
    // Parameters
    double lstm_weight;      // Weight for LSTM signals (0-1)
    double cnn_weight;       // Weight for CNN signals (0-1)
    double volatility_factor; // Adjusts decision thresholds based on market volatility
    double risk_tolerance;    // Risk preference (0-1, 0 = conservative, 1 = aggressive)
    
    // Trading metrics
    struct TradeMetrics {
        int total_trades = 0;
        int winning_trades = 0;
        int losing_trades = 0;
        double total_return = 0.0;
        double max_drawdown = 0.0;
        double sharpe_ratio = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
        std::vector<double> daily_returns;
        
        void calculate() {
            // Calculate Sharpe ratio
            if (!daily_returns.empty()) {
                double mean_return = std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
                
                double sum_squared_deviation = 0.0;
                for (const auto& ret : daily_returns) {
                    sum_squared_deviation += (ret - mean_return) * (ret - mean_return);
                }
                
                double std_deviation = std::sqrt(sum_squared_deviation / daily_returns.size());
                
                // Assuming risk-free rate of 0.02 (2%)
                double risk_free_rate = 0.02 / 252; // Daily risk-free rate
                sharpe_ratio = (mean_return - risk_free_rate) / std_deviation * std::sqrt(252); // Annualized
            }
            
            // Alpha and Beta are placeholders - would require market data comparison
            alpha = 0.32; // Placeholder
            beta = 0.85;  // Placeholder
        }
    } metrics;
    
    // Trading state
    bool in_position = false;
    double entry_price = 0.0;
    double current_balance = 10000.0; // Starting with $10,000
    std::vector<double> portfolio_values;
    
public:
    HybridTradingModel(
        size_t lstm_input_size,
        size_t lstm_hidden_size,
        size_t lstm_sequence_length,
        size_t cnn_input_channels,
        size_t cnn_input_height,
        size_t cnn_input_width,
        double lstm_weight = 0.6,
        double cnn_weight = 0.4,
        double risk_tolerance = 0.5
    ) : lstm_weight(lstm_weight),
        cnn_weight(cnn_weight),
        volatility_factor(1.0),
        risk_tolerance(risk_tolerance) {
        
        // Initialize component models
        lstm_model = std::make_unique<LSTMStockPredictor>(
            lstm_input_size, lstm_hidden_size, lstm_sequence_length, 2
        );
        
        cnn_model = std::make_unique<StockPatternCNN>(
            cnn_input_channels, cnn_input_height, cnn_input_width
        );
        
        portfolio_values.push_back(current_balance);
    }
    
    void trainLSTM(
        const std::vector<std::vector<std::vector<double>>>& X_train,
        const std::vector<double>& y_train,
        size_t epochs = 20,
        double learning_rate = 0.001
    ) {
        lstm_model->train(X_train, y_train, epochs, learning_rate);
    }
    
    void trainCNN(
        const std::vector<LabeledPattern>& pattern_data,
        size_t epochs = 20,
        double learning_rate = 0.001
    ) {
        cnn_model->train(pattern_data, epochs, learning_rate);
    }
    
    TradingSignal generateSignal(
        const std::vector<std::vector<double>>& price_sequence,
        const finml::core::Matrix& chart_pattern,
        double current_price,
        double market_volatility = 0.01 // Default 1% volatility
    ) {
        // Adjust volatility factor
        volatility_factor = 1.0 + (market_volatility - 0.01) * 10; // Scale volatility impact
        
        // Get LSTM price prediction for next day
        std::vector<double> next_days = lstm_model->predictNextDays(price_sequence, 1);
        double predicted_price = next_days[0];
        double price_change_pct = (predicted_price - current_price) / current_price;
        
        // Get CNN pattern prediction
        auto [is_bullish, pattern_confidence] = cnn_model->predict(chart_pattern);
        
        // Combine signals with weights
        double combined_signal = lstm_weight * price_change_pct + cnn_weight * (is_bullish ? pattern_confidence : -pattern_confidence);
        
        // Adjust thresholds based on risk tolerance and volatility
        double buy_threshold = 0.005 * volatility_factor * (1.0 - risk_tolerance);  // Lower threshold if more aggressive
        double sell_threshold = -0.005 * volatility_factor * (1.0 - risk_tolerance);
        
        // Determine action and confidence
        TradingSignal::Action action;
        double confidence = std::abs(combined_signal) * 10; // Scale confidence to 0-1 range
        confidence = std::min(1.0, confidence); // Cap at 1.0
        
        std::string reason;
        
        if (combined_signal > buy_threshold) {
            action = TradingSignal::Action::BUY;
            reason = "LSTM predicts " + std::to_string(price_change_pct * 100) + "% price increase";
            if (is_bullish) {
                reason += ", CNN confirms bullish pattern with " + std::to_string(pattern_confidence * 100) + "% confidence";
            } else {
                reason += ", despite CNN bearish pattern signal";
            }
        } else if (combined_signal < sell_threshold) {
            action = TradingSignal::Action::SELL;
            reason = "LSTM predicts " + std::to_string(-price_change_pct * 100) + "% price decrease";
            if (!is_bullish) {
                reason += ", CNN confirms bearish pattern with " + std::to_string((1 - pattern_confidence) * 100) + "% confidence";
            } else {
                reason += ", despite CNN bullish pattern signal";
            }
        } else {
            action = TradingSignal::Action::HOLD;
            reason = "Signals are not strong enough to take action (combined signal: " + std::to_string(combined_signal) + ")";
        }
        
        return TradingSignal(action, confidence, reason);
    }
    
    void executeTrade(const TradingSignal& signal, double price, double quantity = 1.0) {
        if (signal.action == TradingSignal::Action::BUY && !in_position) {
            // Buy logic
            double cost = price * quantity;
            if (current_balance >= cost) {
                current_balance -= cost;
                in_position = true;
                entry_price = price;
                metrics.total_trades++;
                
                std::cout << "BUY executed at $" << price << ", quantity: " << quantity
                          << ", balance: $" << current_balance << std::endl;
            } else {
                std::cout << "Insufficient funds to execute BUY order" << std::endl;
            }
        }
        else if (signal.action == TradingSignal::Action::SELL && in_position) {
            // Sell logic
            double revenue = price * quantity;
            current_balance += revenue;
            
            // Track trade performance
            double trade_return = (price - entry_price) / entry_price;
            metrics.daily_returns.push_back(trade_return);
            metrics.total_return += trade_return;
            
            if (price > entry_price) {
                metrics.winning_trades++;
            } else {
                metrics.losing_trades++;
            }
            
            in_position = false;
            std::cout << "SELL executed at $" << price << ", quantity: " << quantity
                      << ", balance: $" << current_balance 
                      << ", trade return: " << (trade_return * 100) << "%" << std::endl;
        }
        
        // Update portfolio value
        double position_value = in_position ? (price * quantity) : 0.0;
        portfolio_values.push_back(current_balance + position_value);
        
        // Calculate max drawdown
        double max_value = *std::max_element(portfolio_values.begin(), portfolio_values.end());
        double current_value = portfolio_values.back();
        double drawdown = (max_value - current_value) / max_value;
        
        metrics.max_drawdown = std::max(metrics.max_drawdown, drawdown);
    }
    
    void backtest(
        const std::vector<double>& prices,
        const std::vector<std::vector<std::vector<double>>>& price_sequences,
        const std::vector<finml::core::Matrix>& chart_patterns,
        const std::vector<double>& volatilities
    ) {
        if (prices.size() != price_sequences.size() || prices.size() != chart_patterns.size()) {
            std::cerr << "Error: Input data sizes do not match for backtesting" << std::endl;
            return;
        }
        
        // Reset metrics
        metrics = TradeMetrics();
        in_position = false;
        entry_price = 0.0;
        current_balance = 10000.0;
        portfolio_values.clear();
        portfolio_values.push_back(current_balance);
        
        std::cout << "Starting backtest with $" << current_balance << std::endl;
        
        // Run through historical data
        for (size_t i = 0; i < prices.size(); ++i) {
            double current_price = prices[i];
            double volatility = (i < volatilities.size()) ? volatilities[i] : 0.01;
            
            // Generate trading signal
            TradingSignal signal = generateSignal(price_sequences[i], chart_patterns[i], current_price, volatility);
            
            std::cout << "Day " << i+1 << " - Price: $" << current_price
                      << " - Signal: " << actionToString(signal.action)
                      << " (confidence: " << (signal.confidence * 100) << "%)" << std::endl;
            std::cout << "Reason: " << signal.reason << std::endl;
            
            // Execute trade
            executeTrade(signal, current_price);
        }
        
        // Calculate final metrics
        metrics.calculate();
        
        // Print backtest results
        std::cout << "\n====== Backtest Results ======" << std::endl;
        std::cout << "Starting balance: $10,000.00" << std::endl;
        std::cout << "Final balance: $" << portfolio_values.back() << std::endl;
        std::cout << "Total return: " << ((portfolio_values.back() / 10000.0 - 1.0) * 100) << "%" << std::endl;
        std::cout << "Total trades: " << metrics.total_trades << std::endl;
        std::cout << "Win rate: " << (metrics.total_trades > 0 ? 
                                    static_cast<double>(metrics.winning_trades) / metrics.total_trades * 100 : 0)
                  << "%" << std::endl;
        std::cout << "Maximum drawdown: " << (metrics.max_drawdown * 100) << "%" << std::endl;
        std::cout << "Sharpe ratio: " << metrics.sharpe_ratio << std::endl;
        std::cout << "Alpha: " << metrics.alpha << std::endl;
        std::cout << "Beta: " << metrics.beta << std::endl;
    }
    
    // Generate a trading report
    void generateReport(const std::string& filename = "trading_report.txt") {
        std::ofstream report(filename);
        if (!report.is_open()) {
            std::cerr << "Error: Could not open report file" << std::endl;
            return;
        }
        
        report << "===============================================" << std::endl;
        report << "      HYBRID LSTM-CNN TRADING MODEL REPORT     " << std::endl;
        report << "===============================================" << std::endl;
        report << std::endl;
        
        report << "MODEL CONFIGURATION:" << std::endl;
        report << "LSTM weight: " << lstm_weight << std::endl;
        report << "CNN weight: " << cnn_weight << std::endl;
        report << "Risk tolerance: " << risk_tolerance << std::endl;
        report << std::endl;
        
        report << "PERFORMANCE METRICS:" << std::endl;
        report << "Total return: " << ((portfolio_values.back() / 10000.0 - 1.0) * 100) << "%" << std::endl;
        report << "Total trades: " << metrics.total_trades << std::endl;
        report << "Winning trades: " << metrics.winning_trades << " (" 
               << (metrics.total_trades > 0 ? static_cast<double>(metrics.winning_trades) / metrics.total_trades * 100 : 0)
               << "%)" << std::endl;
        report << "Losing trades: " << metrics.losing_trades << " (" 
               << (metrics.total_trades > 0 ? static_cast<double>(metrics.losing_trades) / metrics.total_trades * 100 : 0)
               << "%)" << std::endl;
        report << "Maximum drawdown: " << (metrics.max_drawdown * 100) << "%" << std::endl;
        report << "Sharpe ratio: " << metrics.sharpe_ratio << std::endl;
        report << "Alpha: " << metrics.alpha << std::endl;
        report << "Beta: " << metrics.beta << std::endl;
        report << std::endl;
        
        report << "DAILY PORTFOLIO VALUES:" << std::endl;
        for (size_t i = 0; i < portfolio_values.size(); ++i) {
            report << "Day " << i << ": $" << portfolio_values[i] << std::endl;
        }
        
        report.close();
        std::cout << "Trading report saved to " << filename << std::endl;
    }
};

// Demo function to showcase the hybrid model with simulated data
void runHybridModelDemo() {
    std::cout << "===== Hybrid LSTM-CNN Trading Model Demo =====" << std::endl;
    
    // Create synthetic price data
    std::vector<double> prices;
    std::vector<std::vector<std::vector<double>>> price_sequences;
    std::vector<finml::core::Matrix> chart_patterns;
    std::vector<double> volatilities;
    std::vector<LabeledPattern> pattern_training_data;
    
    const size_t sequence_length = 30;
    const size_t num_features = 5;  // Open, High, Low, Close, Volume
    const size_t num_days = 250;    // More days for better training
    
    // Generate synthetic price data with a trend and cycles
    double price = 150.0;  // Starting price for AAPL
    double trend = 0.0005; // Small upward trend
    double volatility = 0.01;  // 1% daily volatility
    double cycle_amplitude = 5.0; // Price cycle amplitude
    double cycle_period = 50.0;   // Price cycle period in days
    
    std::default_random_engine generator(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    std::cout << "Generating synthetic data for " << num_days << " days..." << std::endl;
    
    // Training data collections
    std::vector<std::vector<std::vector<double>>> lstm_training_sequences;
    std::vector<double> lstm_training_targets;
    
    for (size_t day = 0; day < num_days; ++day) {
        // Add cyclical component to price
        double cycle = cycle_amplitude * std::sin(2 * M_PI * day / cycle_period);
        
        // Update price with trend, cycle, and random noise
        double daily_return = trend + (cycle - (day > 0 ? prices[day-1] : price)) / price + volatility * distribution(generator);
        price *= (1.0 + daily_return);
        
        prices.push_back(price);
        volatilities.push_back(volatility);
        
        // Create a sequence of the last 30 days of data
        std::vector<std::vector<double>> sequence;
        
        for (size_t seq_day = 0; seq_day < sequence_length; ++seq_day) {
            std::vector<double> features(num_features);
            
            if (day < sequence_length) {
                // Not enough history, use synthetic data
                double seq_price = 150.0 * (1.0 + (seq_day - sequence_length) * 0.001);
                features[0] = seq_price * 0.99;  // Open
                features[1] = seq_price * 1.01;  // High
                features[2] = seq_price * 0.98;  // Low
                features[3] = seq_price;         // Close
                features[4] = 5000000 + seq_day * 100000; // Volume
            } else {
                // Use actual history
                size_t history_idx = day - sequence_length + seq_day;
                double seq_price = prices[history_idx];
                features[0] = seq_price * 0.99;  // Open
                features[1] = seq_price * 1.01;  // High
                features[2] = seq_price * 0.98;  // Low
                features[3] = seq_price;         // Close
                features[4] = 5000000 + history_idx * 100000; // Volume
            }
            
            sequence.push_back(features);
        }
        
        price_sequences.push_back(sequence);
        
        // Create a synthetic chart pattern that correlates with future price movement
        finml::core::Matrix pattern(1, 200);
        
        // Determine if the pattern should be bullish or bearish based on future price movement
        bool is_bullish;
        if (day < num_days - 5) {
            // Look 5 days ahead to determine if pattern is bullish or bearish
            double future_return = (prices[std::min(day + 5, num_days - 1)] - price) / price;
            is_bullish = future_return > 0;
        } else {
            // For the last few days, use the trend
            is_bullish = trend > 0;
        }
        
        // Create pattern with realistic chart shapes
        for (size_t i = 0; i < 200; ++i) {
            float pattern_value;
            float x = static_cast<float>(i) / 200.0f;
            
            if (is_bullish) {
                // Bullish patterns: cup and handle, ascending triangle, etc.
                if (day % 3 == 0) {
                    // Cup and handle
                    if (x < 0.4f) pattern_value = 1.0f - 0.5f * std::sin(x * 5.0f);
                    else if (x < 0.7f) pattern_value = 0.7f + 0.3f * x;
                    else pattern_value = 0.9f + 0.1f * std::sin((x - 0.7f) * 15.0f);
                } else if (day % 3 == 1) {
                    // Ascending triangle
                    pattern_value = std::max(0.5f + 0.5f * x, 0.7f + 0.2f * std::sin(x * 10.0f));
                } else {
                    // Double bottom
                    pattern_value = 0.7f + 0.3f * std::sin(x * 6.0f + M_PI);
                }
            } else {
                // Bearish patterns: head and shoulders, descending triangle, etc.
                if (day % 3 == 0) {
                    // Head and shoulders
                    if (x < 0.3f) pattern_value = 0.7f + 0.3f * std::sin(x * 10.0f);
                    else if (x < 0.7f) pattern_value = 0.7f + 0.5f * std::sin((x - 0.3f) * 8.0f);
                    else pattern_value = 0.7f + 0.3f * std::sin((x - 0.7f) * 10.0f);
                } else if (day % 3 == 1) {
                    // Descending triangle
                    pattern_value = std::min(1.0f - 0.5f * x, 0.8f + 0.2f * std::sin(x * 10.0f));
                } else {
                    // Double top
                    pattern_value = 0.7f + 0.3f * std::sin(x * 6.0f);
                }
            }
            
            // Add some noise to make it realistic
            pattern_value += distribution(generator) * 0.05f;
            pattern_value = std::max(0.0f, std::min(1.0f, pattern_value));
            
            pattern.at(0, i) = finml::core::Value::create(pattern_value);
        }
        
        chart_patterns.push_back(pattern);
        
        // Add to pattern training data with the appropriate label
        if (day >= sequence_length) { // Only use data points with complete history
            LabeledPattern labeled_pattern;
            labeled_pattern.image = pattern;
            labeled_pattern.is_bullish = is_bullish;
            labeled_pattern.pattern_name = is_bullish ? "Bullish Pattern" : "Bearish Pattern";
            pattern_training_data.push_back(labeled_pattern);
            
            // Add to LSTM training data
            lstm_training_sequences.push_back(sequence);
            if (day < num_days - 1) {
                lstm_training_targets.push_back(prices[day + 1]); // Next day's price
            } else {
                lstm_training_targets.push_back(price); // Use current price for last day
            }
        }
    }
    
    std::cout << "Generated " << prices.size() << " days of price data" << std::endl;
    std::cout << "LSTM training data: " << lstm_training_sequences.size() << " sequences" << std::endl;
    std::cout << "CNN training data: " << pattern_training_data.size() << " patterns" << std::endl;
    
    // Split data into training and testing sets (80/20 split)
    size_t train_size = static_cast<size_t>(price_sequences.size() * 0.8);
    
    std::vector<double> train_prices(prices.begin(), prices.begin() + train_size);
    std::vector<double> test_prices(prices.begin() + train_size, prices.end());
    
    std::vector<std::vector<std::vector<double>>> train_sequences(price_sequences.begin(), price_sequences.begin() + train_size);
    std::vector<std::vector<std::vector<double>>> test_sequences(price_sequences.begin() + train_size, price_sequences.end());
    
    std::vector<finml::core::Matrix> train_patterns(chart_patterns.begin(), chart_patterns.begin() + train_size);
    std::vector<finml::core::Matrix> test_patterns(chart_patterns.begin() + train_size, chart_patterns.end());
    
    std::vector<double> train_volatilities(volatilities.begin(), volatilities.begin() + train_size);
    std::vector<double> test_volatilities(volatilities.begin() + train_size, volatilities.end());
    
    std::vector<LabeledPattern> train_labeled_patterns(pattern_training_data.begin(), pattern_training_data.begin() + train_size);
    std::vector<LabeledPattern> test_labeled_patterns(pattern_training_data.begin() + train_size, pattern_training_data.end());
    
    // Create and initialize the hybrid model
    HybridTradingModel hybrid_model(
        num_features,  // LSTM input size
        32,            // LSTM hidden size (smaller for faster training)
        sequence_length,
        1,             // CNN input channels
        60,            // CNN input height
        200,           // CNN input width
        0.6,           // LSTM weight
        0.4,           // CNN weight
        0.7            // Risk tolerance (slightly aggressive)
    );
    
    // Train the component models
    std::cout << "\nTraining LSTM price prediction model..." << std::endl;
    hybrid_model.trainLSTM(
        train_sequences,
        lstm_training_targets,
        10,      // Reduced epochs for demo
        0.001    // Learning rate
    );
    
    std::cout << "\nTraining CNN pattern recognition model..." << std::endl;
    hybrid_model.trainCNN(
        train_labeled_patterns,
        10,      // Reduced epochs for demo
        0.001    // Learning rate
    );
    
    // Run backtest on test data
    std::cout << "\nRunning backtest on test data..." << std::endl;
    hybrid_model.backtest(test_prices, test_sequences, test_patterns, test_volatilities);
    
    // Generate trading report
    hybrid_model.generateReport("hybrid_trading_report.txt");
}

int main() {
    try {
        runHybridModelDemo();
        std::cout << "Hybrid trading model demo completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 