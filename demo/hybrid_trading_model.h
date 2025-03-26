#ifndef HYBRID_TRADING_MODEL_H
#define HYBRID_TRADING_MODEL_H

#include <finml/core/matrix.h>
#include <finml/models/lstm_stock_predictor.h>
#include <finml/models/stock_pattern_cnn.h>
#include <finml/optimization/loss.h>
#include <string>
#include <vector>
#include <memory>
#include <fstream>

// Structure to represent chart pattern with its label
struct LabeledPattern {
    finml::core::Matrix image;  // The visual pattern
    bool is_bullish;           // True if pattern indicates bullish movement
    std::string pattern_name;   // Name of the pattern (e.g., "Head and Shoulders")
};

// Simple structure to represent a trade
struct Trade {
    int day;
    bool is_buy;
    double price;
    double quantity;
    std::string reason;
};

// Structure to represent trading performance
struct TradingPerformance {
    double initial_capital;
    double final_capital;
    double total_return;
    double annualized_return;
    double sharpe_ratio;
    double max_drawdown;
    int total_trades;
    int winning_trades;
    int losing_trades;
    double win_rate;
    double profit_factor;
};

// The Hybrid Trading Model class that combines LSTM and CNN models
class HybridTradingModel {
public:
    // Constructor
    HybridTradingModel(
        size_t lstm_input_size,
        size_t lstm_hidden_size,
        size_t sequence_length,
        size_t cnn_input_channels,
        size_t cnn_input_height,
        size_t cnn_input_width,
        double lstm_weight,
        double cnn_weight,
        double risk_tolerance
    ) : lstm_input_size_(lstm_input_size),
        lstm_hidden_size_(lstm_hidden_size),
        sequence_length_(sequence_length),
        cnn_input_channels_(cnn_input_channels),
        cnn_input_height_(cnn_input_height),
        cnn_input_width_(cnn_input_width),
        lstm_weight_(lstm_weight),
        cnn_weight_(cnn_weight),
        risk_tolerance_(risk_tolerance) {
        
        // Initialize the LSTM model for price prediction
        lstm_model_ = std::make_unique<LSTMStockPredictor>(
            lstm_input_size,    // Input features (OHLCV)
            lstm_hidden_size,   // Hidden size
            sequence_length,    // Sequence length
            1                   // Output size (price prediction)
        );
        
        // Initialize the CNN model for pattern recognition
        cnn_model_ = std::make_unique<StockPatternCNN>(
            cnn_input_channels,  // Input channels (grayscale)
            cnn_input_height,    // Input height
            cnn_input_width      // Input width
        );
    }
    
    // Train the LSTM model
    void trainLSTM(
        const std::vector<std::vector<std::vector<double>>>& sequences,
        const std::vector<double>& targets,
        size_t epochs,
        double learning_rate
    ) {
        if (sequences.empty() || targets.empty()) {
            throw std::runtime_error("Empty training data provided to LSTM");
        }
        
        if (sequences.size() != targets.size()) {
            throw std::runtime_error("Mismatch between number of sequences and targets");
        }
        
        std::cout << "Training LSTM model with " << sequences.size() << " sequences..." << std::endl;
        
        // Train the LSTM model
        lstm_model_->train(
            sequences,
            targets,
            epochs,
            learning_rate,
            16,             // Batch size
            0.0,            // No dropout
            false,          // No early stopping
            1               // Validation frequency
        );
        
        std::cout << "LSTM training completed!" << std::endl;
    }
    
    // Train the CNN model
    void trainCNN(
        const std::vector<LabeledPattern>& patterns,
        size_t epochs,
        double learning_rate
    ) {
        if (patterns.empty()) {
            throw std::runtime_error("Empty training data provided to CNN");
        }
        
        std::cout << "Training CNN model with " << patterns.size() << " patterns..." << std::endl;
        
        // Train the CNN model
        cnn_model_->train(
            patterns,
            epochs,
            learning_rate,
            16  // Batch size
        );
        
        std::cout << "CNN training completed!" << std::endl;
    }
    
    // Generate a trading signal from model predictions
    double generateSignal(
        const std::vector<std::vector<double>>& sequence,
        const finml::core::Matrix& pattern,
        double current_price,
        double volatility
    ) {
        // Get price prediction from LSTM
        std::vector<double> predictions = lstm_model_->predictNextDays(sequence, 1);
        double predicted_price = predictions[0];
        double predicted_return = (predicted_price - current_price) / current_price;
        
        // Get pattern prediction from CNN
        bool pattern_is_bullish;
        double pattern_confidence;
        cnn_model_->predict(pattern, pattern_is_bullish, pattern_confidence);
        
        // Combine predictions to make a trading decision
        double lstm_signal = lstm_weight_ * predicted_return / volatility;
        double cnn_signal = cnn_weight_ * (pattern_is_bullish ? pattern_confidence : -pattern_confidence);
        double combined_signal = lstm_signal + cnn_signal;
        
        // Adjust signal based on risk tolerance
        return combined_signal * risk_tolerance_;
    }
    
    // Run a backtest on historical data
    void backtest(
        const std::vector<double>& prices,
        const std::vector<std::vector<std::vector<double>>>& sequences,
        const std::vector<finml::core::Matrix>& patterns,
        const std::vector<double>& volatilities
    ) {
        if (prices.empty() || sequences.empty() || patterns.empty() || volatilities.empty()) {
            throw std::runtime_error("Empty data provided for backtesting");
        }
        
        if (prices.size() != sequences.size() || prices.size() != patterns.size() || prices.size() != volatilities.size()) {
            throw std::runtime_error("Data size mismatch in backtest input");
        }
        
        std::cout << "Running backtest on " << prices.size() << " days of data..." << std::endl;
        
        // Initialize trading state
        double cash = 10000.0;  // Starting with $10,000
        double shares = 0.0;
        std::vector<Trade> trades;
        std::vector<double> portfolio_values;
        double max_portfolio_value = cash;
        
        // Loop through each day in the test set
        for (size_t day = 0; day < prices.size(); ++day) {
            // Skip the first day as we need previous data for comparison
            if (day == 0) {
                portfolio_values.push_back(cash);
                continue;
            }
            
            // Get current price
            double current_price = prices[day];
            double previous_price = prices[day - 1];
            
            // Get current sequence and pattern
            const auto& sequence = sequences[day];
            const auto& pattern = patterns[day];
            double volatility = volatilities[day];
            
            // Generate trading signal
            double risk_adjusted_signal = generateSignal(sequence, pattern, current_price, volatility);
            
            // Get predictions for reporting
            std::vector<double> predictions = lstm_model_->predictNextDays(sequence, 1);
            double predicted_price = predictions[0];
            double predicted_return = (predicted_price - current_price) / current_price;
            
            bool pattern_is_bullish;
            double pattern_confidence;
            cnn_model_->predict(pattern, pattern_is_bullish, pattern_confidence);
            
            // Decision making
            double portfolio_value = cash + shares * current_price;
            portfolio_values.push_back(portfolio_value);
            max_portfolio_value = std::max(max_portfolio_value, portfolio_value);
            
            std::string decision_reason;
            
            if (risk_adjusted_signal > 0.1 && cash > 0) {
                // Buy signal
                double amount_to_invest = cash * std::min(1.0, risk_adjusted_signal);
                double shares_to_buy = amount_to_invest / current_price;
                
                cash -= shares_to_buy * current_price;
                shares += shares_to_buy;
                
                decision_reason = "Buy: LSTM predicts " + std::to_string(predicted_return * 100) + 
                                 "% return, CNN indicates " + (pattern_is_bullish ? "bullish" : "bearish") +
                                 " pattern with " + std::to_string(pattern_confidence * 100) + "% confidence";
                
                Trade trade;
                trade.day = day;
                trade.is_buy = true;
                trade.price = current_price;
                trade.quantity = shares_to_buy;
                trade.reason = decision_reason;
                trades.push_back(trade);
                
            } else if (risk_adjusted_signal < -0.1 && shares > 0) {
                // Sell signal
                double shares_to_sell = shares * std::min(1.0, -risk_adjusted_signal);
                
                cash += shares_to_sell * current_price;
                shares -= shares_to_sell;
                
                decision_reason = "Sell: LSTM predicts " + std::to_string(predicted_return * 100) + 
                                 "% return, CNN indicates " + (pattern_is_bullish ? "bullish" : "bearish") +
                                 " pattern with " + std::to_string(pattern_confidence * 100) + "% confidence";
                
                Trade trade;
                trade.day = day;
                trade.is_buy = false;
                trade.price = current_price;
                trade.quantity = shares_to_sell;
                trade.reason = decision_reason;
                trades.push_back(trade);
            }
        }
        
        // Calculate performance metrics
        performance_.initial_capital = 10000.0;
        performance_.final_capital = cash + shares * prices.back();
        performance_.total_return = (performance_.final_capital / performance_.initial_capital) - 1.0;
        performance_.annualized_return = std::pow(1.0 + performance_.total_return, 252.0 / prices.size()) - 1.0;
        
        // Calculate max drawdown
        double max_drawdown = 0.0;
        for (size_t i = 0; i < portfolio_values.size(); ++i) {
            double drawdown = (max_portfolio_value - portfolio_values[i]) / max_portfolio_value;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
        performance_.max_drawdown = max_drawdown;
        
        // Calculate trade statistics
        performance_.total_trades = trades.size();
        performance_.winning_trades = 0;
        performance_.losing_trades = 0;
        double total_profit = 0.0;
        double total_loss = 0.0;
        
        for (size_t i = 0; i < trades.size(); ++i) {
            if (i + 1 < trades.size() && trades[i].is_buy && !trades[i+1].is_buy) {
                double profit = (trades[i+1].price - trades[i].price) * trades[i].quantity;
                if (profit > 0) {
                    performance_.winning_trades++;
                    total_profit += profit;
                } else {
                    performance_.losing_trades++;
                    total_loss -= profit;  // Make loss positive
                }
            }
        }
        
        performance_.win_rate = performance_.total_trades > 0 ? 
            static_cast<double>(performance_.winning_trades) / performance_.total_trades : 0.0;
            
        performance_.profit_factor = total_loss > 0 ? total_profit / total_loss : 0.0;
        
        // Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        double sum_returns = 0.0;
        double sum_squared_returns = 0.0;
        
        for (size_t i = 1; i < portfolio_values.size(); ++i) {
            double daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1.0;
            sum_returns += daily_return;
            sum_squared_returns += daily_return * daily_return;
        }
        
        double avg_return = sum_returns / (portfolio_values.size() - 1);
        double std_dev = std::sqrt(sum_squared_returns / (portfolio_values.size() - 1) - avg_return * avg_return);
        performance_.sharpe_ratio = std_dev > 0 ? (avg_return / std_dev) * std::sqrt(252.0) : 0.0;
        
        // Store trades and portfolio values for reporting
        trades_ = trades;
        portfolio_values_ = portfolio_values;
        
        std::cout << "Backtest completed!" << std::endl;
        std::cout << "Final portfolio value: $" << performance_.final_capital << std::endl;
        std::cout << "Total return: " << performance_.total_return * 100 << "%" << std::endl;
        std::cout << "Annualized return: " << performance_.annualized_return * 100 << "%" << std::endl;
        std::cout << "Sharpe ratio: " << performance_.sharpe_ratio << std::endl;
        std::cout << "Maximum drawdown: " << performance_.max_drawdown * 100 << "%" << std::endl;
        std::cout << "Total trades: " << performance_.total_trades << std::endl;
        std::cout << "Win rate: " << performance_.win_rate * 100 << "%" << std::endl;
    }
    
    // Generate a trading report
    void generateReport(const std::string& filename) {
        std::ofstream report(filename);
        
        if (!report.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        
        report << "=======================================================" << std::endl;
        report << "        HYBRID TRADING MODEL PERFORMANCE REPORT        " << std::endl;
        report << "=======================================================" << std::endl;
        report << std::endl;
        
        report << "MODEL CONFIGURATION:" << std::endl;
        report << "---------------------" << std::endl;
        report << "LSTM Input Size: " << lstm_input_size_ << std::endl;
        report << "LSTM Hidden Size: " << lstm_hidden_size_ << std::endl;
        report << "Sequence Length: " << sequence_length_ << std::endl;
        report << "CNN Input Dimensions: " << cnn_input_channels_ << "x" 
               << cnn_input_height_ << "x" << cnn_input_width_ << std::endl;
        report << "LSTM Weight: " << lstm_weight_ << std::endl;
        report << "CNN Weight: " << cnn_weight_ << std::endl;
        report << "Risk Tolerance: " << risk_tolerance_ << std::endl;
        report << std::endl;
        
        report << "PERFORMANCE METRICS:" << std::endl;
        report << "--------------------" << std::endl;
        report << "Initial Capital: $" << performance_.initial_capital << std::endl;
        report << "Final Capital: $" << performance_.final_capital << std::endl;
        report << "Total Return: " << (performance_.total_return * 100) << "%" << std::endl;
        report << "Annualized Return: " << (performance_.annualized_return * 100) << "%" << std::endl;
        report << "Sharpe Ratio: " << performance_.sharpe_ratio << std::endl;
        report << "Maximum Drawdown: " << (performance_.max_drawdown * 100) << "%" << std::endl;
        report << std::endl;
        
        report << "TRADE STATISTICS:" << std::endl;
        report << "-----------------" << std::endl;
        report << "Total Trades: " << performance_.total_trades << std::endl;
        report << "Winning Trades: " << performance_.winning_trades << std::endl;
        report << "Losing Trades: " << performance_.losing_trades << std::endl;
        report << "Win Rate: " << (performance_.win_rate * 100) << "%" << std::endl;
        report << "Profit Factor: " << performance_.profit_factor << std::endl;
        report << std::endl;
        
        report << "TRADE JOURNAL:" << std::endl;
        report << "--------------" << std::endl;
        for (const auto& trade : trades_) {
            report << "Day " << trade.day << ": " 
                   << (trade.is_buy ? "BUY" : "SELL") << " " 
                   << trade.quantity << " shares at $" << trade.price 
                   << " - " << trade.reason << std::endl;
        }
        report << std::endl;
        
        report << "PORTFOLIO VALUES:" << std::endl;
        report << "-----------------" << std::endl;
        for (size_t i = 0; i < portfolio_values_.size(); ++i) {
            report << "Day " << i << ": $" << portfolio_values_[i] << std::endl;
        }
        
        report.close();
        std::cout << "Trading report generated: " << filename << std::endl;
    }
    
private:
    // LSTM model parameters
    size_t lstm_input_size_;
    size_t lstm_hidden_size_;
    size_t sequence_length_;
    
    // CNN model parameters
    size_t cnn_input_channels_;
    size_t cnn_input_height_;
    size_t cnn_input_width_;
    
    // Hybrid model parameters
    double lstm_weight_;
    double cnn_weight_;
    double risk_tolerance_;
    
    // Component models
    std::unique_ptr<LSTMStockPredictor> lstm_model_;
    std::unique_ptr<StockPatternCNN> cnn_model_;
    
    // Backtest results
    std::vector<Trade> trades_;
    std::vector<double> portfolio_values_;
    TradingPerformance performance_;
};

#endif // HYBRID_TRADING_MODEL_H 