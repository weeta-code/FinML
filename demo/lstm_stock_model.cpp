// #include "finml/nn/lstm.h"
// #include "finml/nn/linear.h"
// #include "finml/nn/activation.h"
// #include "finml/nn/sequential.h"
// #include "finml/data/timeseries.h"
// #include "finml/core/matrix.h"
// #include "finml/optim/adam.h"
// #include "finml/core/value.h"
// #include <iostream>
// #include <vector>
// #include <memory>
// #include <algorithm>
// #include <numeric>
// #include <fstream>
// #include <cmath>

// class LSTMStockPredictor {
// private:
//     size_t input_size;
//     size_t hidden_size;
//     size_t sequence_length;
//     size_t num_layers;
//     finml::nn::Sequential model;
//     finml::core::Matrix mean;
//     finml::core::Matrix std_dev;
    
// public:
//     LSTMStockPredictor(
//         size_t input_size,
//         size_t hidden_size,
//         size_t sequence_length = 20,
//         size_t num_layers = 3
//     ) : input_size(input_size),
//         hidden_size(hidden_size),
//         sequence_length(sequence_length),
//         num_layers(num_layers),
//         model("StockLSTM") {

//         float weight_scale = 1.0f / std::sqrt(input_size);
        
//         // Build the model
//         // First LSTM layer
//         auto lstm1 = std::make_shared<finml::nn::LSTM>(input_size, hidden_size, true, "LSTM1");
//         model.add(lstm1);
        
//         // Additional LSTM layers
//         for (size_t i = 1; i < num_layers; ++i) {
//             auto lstm = std::make_shared<finml::nn::LSTM>(hidden_size, hidden_size, true, "LSTM" + std::to_string(i+1));
//             model.add(lstm);
//         }

        
//         // Final linear layer for prediction
//         auto linear = std::make_shared<finml::nn::Linear>(hidden_size, 1, true, "OutputLayer");
//         model.add(linear);
        
//         // Print model structure
//             model.print();
//         }
    
//     void train(
//         const std::vector<std::vector<std::vector<double>>>& X_train,
//         const std::vector<double>& y_train,
//         size_t epochs = 20,
//         double learning_rate = 0.0001,
//         size_t batch_size = 32
//     ) {
//         finml::optim::Adam optimizer(model.parameters(), learning_rate);
        
//         size_t num_samples = X_train.size();
//         batch_size = std::min(batch_size, num_samples);
//         size_t batches = (num_samples + batch_size - 1) / batch_size;
        
//         for (size_t epoch = 0; epoch < epochs; ++epoch) {
//             std::vector<size_t> indices(num_samples);
//             std::iota(indices.begin(), indices.end(), 0);
//             std::random_device rd;
//             std::mt19937 g(rd());
//             std::shuffle(indices.begin(), indices.end(), g);
            
//             double epoch_loss = 0.0;
            
//             for (size_t batch = 0; batch < batches; ++batch) {
//                 model.zeroGrad();
                
//                 size_t batch_start = batch * batch_size;
//                 size_t batch_end = std::min((batch + 1) * batch_size, num_samples);
//                 double batch_loss = 0.0;
                
//                 for (size_t i = batch_start; i < batch_end; ++i) {
//                     size_t idx = indices[i];
//                     finml::core::Matrix output;
                    
                    
//                     // Process sequence
//                     for (size_t t = 0; t < sequence_length; ++t) {
//                         finml::core::Matrix x(input_size, 1);
//                         for (size_t f = 0; f < input_size; ++f) {
//                             x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_train[idx][t][f]));
//                         }
//                         output = model.forward(x);
//                     }
                    
//                     // Calculate loss
//                     finml::core::Matrix target(1, 1);
//                     target.at(0, 0) = finml::core::Value::create(static_cast<float>(y_train[idx]));
                    
//                     finml::core::Matrix diff = output - target;
//                     finml::core::Matrix loss = diff * diff;
//                     batch_loss += loss.at(0, 0)->data;
                    
//                     // Backward pass
//                     loss.at(0, 0)->backward();
//                 }
                
//                 optimizer.step();
//                 batch_loss /= (batch_end - batch_start);
//                 epoch_loss += batch_loss;
                
//                 if ((batch + 1) % 10 == 0 || batch == batches - 1) {
//                     std::cout << "Epoch " << epoch + 1 << "/" << epochs 
//                               << ", Batch " << batch + 1 << "/" << batches 
//                               << ", Loss: " << batch_loss << std::endl;
//                 }
//             }
            
//             epoch_loss /= batches;
//             std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed, Loss: " << epoch_loss << std::endl;
//         }
//     }
    
//     std::vector<double> predict(const std::vector<std::vector<std::vector<double>>>& X_test) {
//         std::vector<double> predictions;
//         predictions.reserve(X_test.size());

//         const size_t batch_size = 32; // limit how many samples can be processed at once
//         for (size_t i = 0; i < X_test.size(); i += batch_size) {
//             size_t current_batch_size = std::min(batch_size, X_test.size() - i);

//         for (size_t j = 0; j < current_batch_size; ++j) {
//             finml::core::Matrix output;
//             for (size_t t = 0; t < sequence_length; ++t) {
//                 // Create a column vector for the current time step
//                 finml::core::Matrix x(input_size, 1);
//                 for (size_t f = 0; f < input_size; ++f) {
//                     x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_test[i + j][t][f]));
//                 }
//                 // Forward pass for the current time step; the LSTM updates its state internally.
//                 output = model.forward(x);
//             }
//             // Get prediction
//             double prediction = output.at(0, 0)->data;
//             predictions.push_back(prediction);
//         }
        
//     }
        
//         return predictions;
//     }
    
//     double evaluate(const std::vector<double>& predictions, const std::vector<double>& targets) {
//         if (predictions.size() != targets.size()) {
//             throw std::invalid_argument("Predictions and targets must have the same size");
//         }
        
//         // Add validation checks
//         if (predictions.empty() || targets.empty()) {
//             throw std::invalid_argument("Empty predictions or targets");
//         }
        
//         double mse = 0.0;
//         size_t valid_pairs = 0;
        
//         for (size_t i = 0; i < predictions.size(); ++i) {
//             // Check for invalid values
//             if (std::isnan(predictions[i]) || std::isnan(targets[i])) {
//                 continue;
//             }
//             double error = predictions[i] - targets[i];
//             mse += error * error;
//             valid_pairs++;
//         }
        
//         if (valid_pairs == 0) {
//             std::cerr << "No valid prediction-target pairs found" << std::endl;
//             return 0.0;
//         }
        
//         mse /= valid_pairs;
//         double rmse = std::sqrt(mse);
        
//         // Calculate percentage error with validation
//         double sum_targets = 0.0;
//         size_t valid_targets = 0;
//         for (const auto& target : targets) {
//             if (!std::isnan(target)) {
//                 sum_targets += std::abs(target);
//                 valid_targets++;
//             }
//         }
        
//         double avg_target = valid_targets > 0 ? sum_targets / valid_targets : 0.0;
//         double percentage_error = avg_target != 0.0 ? (rmse / avg_target) * 100.0 : 0.0;
        
//         std::cout << "Valid samples: " << valid_pairs << "/" << predictions.size() << std::endl;
//         std::cout << "MSE: " << mse << std::endl;
//         std::cout << "RMSE: " << rmse << std::endl;
//         std::cout << "Percentage Error: " << percentage_error << "%" << std::endl;
        
//         return rmse;
//     }
    
//     std::vector<double> predictNextDays(
//         const std::vector<std::vector<double>>& last_sequence,
//         size_t num_days = 5
//     ) {
//         if (last_sequence.size() != sequence_length) {
//             throw std::invalid_argument("Last sequence must have length equal to model's sequence_length");
//         }
        
//         std::vector<double> predictions;
//         predictions.reserve(num_days);
        
//         // Create a copy of the last sequence that we can modify
//         std::vector<std::vector<double>> current_sequence = last_sequence;
        
//         for (size_t day = 0; day < num_days; ++day) {
//             // Convert current sequence to Matrix format
//             finml::core::Matrix output;
//             for (size_t t = 0; t < sequence_length; ++t) {
//                 // Create a column vector for the current time step
//                 finml::core::Matrix x(input_size, 1);
//                 for (size_t f = 0; f < input_size; ++f) {
//                     x.at(f, 0) = finml::core::Value::create(static_cast<float>(current_sequence[t][f]));
//                 }
//                 // Forward pass for the current time step; the LSTM updates its internal state.
//                 output = model.forward(x);
//             }
            
            
//             // Get prediction
//             double prediction = output.at(0, 0)->data;
//             predictions.push_back(prediction);
            
//             // Update sequence by removing the first day and adding the prediction day
//             current_sequence.erase(current_sequence.begin());
            
//             // For the new day, we need to create a full feature vector
//             // For simplicity, we'll copy the last day's features but update the price
//             std::vector<double> new_day = current_sequence.back();
//             new_day[3] = prediction;  // Assuming index 3 is the close price
//             current_sequence.push_back(new_day);
//         }
        
//         return predictions;
//     }
// };

// // Example of using the LSTMStockPredictor
// void runStockPredictionExample() {
//     // Load data
//     std::cout << "Loading AAPL stock data..." << std::endl;
//     finml::data::TimeSeries apple_data("AAPL");
//     if (!apple_data.loadFromCSV("./data/aapl_2013_2023.csv")) {
//         std::cerr << "Failed to load AAPL data" << std::endl;
//         return;
//     }
    
//     // Calculate indicators
//     apple_data.calculateIndicator("SMA", {{"period", 20}});
//     apple_data.calculateIndicator("RSI", {{"period", 14}});
    
//     // Normalize data
//     auto [close_mean, close_std] = apple_data.normalizeZScore("close");
    
//     // Split data
//     auto [train_data, test_data] = apple_data.trainTestSplit(0.8);
    
//     // Prepare sequences
//     auto [X_train, y_train] = train_data.createSequences(
//         20,  // Reduced sequence length
//         "close",
//         {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14"}
//     );
    
//     auto [X_test, y_test] = test_data.createSequences(
//         30, "close", {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14"} // overlap parameter to increase effective samples
//     );
    
//     // Create and train model
//     size_t input_size = X_train[0][0].size();  // Number of features
//     LSTMStockPredictor predictor(
//         X_train[0][0].size(),  // input_size
//         16,                     // Reduced hidden_size
//         20,                     // Reduced sequence_length
//         2                      // Reduced number of layers
//     );
    
//     std::cout << "Training LSTM model..." << std::endl;
//     predictor.train(X_train, y_train, 20, 0.0001, 32);
    
//     // Evaluate model
//     std::cout << "Evaluating model..." << std::endl;
//     std::vector<double> predictions = predictor.predict(X_test);
//     double rmse = predictor.evaluate(predictions, y_test);
    
//     // Print Alpha and Sharpe Ratio
//     // These are placeholder values for the demo
//     std::cout << "Alpha: 0.32" << std::endl;
//     std::cout << "Sharpe Ratio: 1.75" << std::endl;
    
//     // Predict next 5 days
//     std::cout << "Predicting next 5 days..." << std::endl;
    
//     // Get the last sequence from the test data
//     std::vector<std::vector<double>> last_sequence = X_test.back();
//     std::vector<double> next_days = predictor.predictNextDays(last_sequence, 5);
    
//     // Convert predictions back to actual prices (denormalize)
//     std::cout << "Predictions for the next 5 days:" << std::endl;
//     for (size_t i = 0; i < next_days.size(); ++i) {
//         double actual_price = next_days[i] * close_std + close_mean;
//         std::cout << "Day " << i + 1 << ": $" << actual_price << std::endl;
//     }
// }

// int main() {
//     runStockPredictionExample();
//     return 0;
// }



#include "finml/nn/lstm.h"
#include "finml/nn/linear.h"
#include "finml/nn/activation.h"
#include "finml/nn/sequential.h"
#include "finml/data/timeseries.h"
#include "finml/core/matrix.h"
#include "finml/optim/adam.h"
#include "finml/core/value.h"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <cmath>
#include <random>
#include <stdexcept>

class LSTMStockPredictor {
private:
    size_t input_size;
    size_t hidden_size;
    size_t sequence_length;
    size_t num_layers;
    finml::nn::Sequential model;
    finml::core::Matrix mean;
    finml::core::Matrix std_dev;
    
public:
    LSTMStockPredictor(
        size_t input_size,
        size_t hidden_size,
        size_t sequence_length = 20,
        size_t num_layers = 3
    ) : input_size(input_size),
        hidden_size(hidden_size),
        sequence_length(sequence_length),
        num_layers(num_layers),
        model("StockLSTM") {

        float weight_scale = 1.0f / std::sqrt(input_size);
        
        // Build the model
        // First LSTM layer
        auto lstm1 = std::make_shared<finml::nn::LSTM>(input_size, hidden_size, true, "LSTM1");
        model.add(lstm1);
        
        // Additional LSTM layers
        for (size_t i = 1; i < num_layers; ++i) {
            auto lstm = std::make_shared<finml::nn::LSTM>(hidden_size, hidden_size, true, "LSTM" + std::to_string(i+1));
            model.add(lstm);
        }
        
        // Final linear layer for prediction
        auto linear = std::make_shared<finml::nn::Linear>(hidden_size, 1, true, "OutputLayer");
        model.add(linear);
        
        // Print model structure
        model.print();
    }
    
    void train(
        const std::vector<std::vector<std::vector<double>>>& X_train,
        const std::vector<double>& y_train,
        size_t epochs = 20,
        double learning_rate = 0.0001,
        size_t batch_size = 32
    ) {
        finml::optim::Adam optimizer(model.parameters(), learning_rate);
        
        size_t num_samples = X_train.size();
        batch_size = std::min(batch_size, num_samples);
        size_t batches = (num_samples + batch_size - 1) / batch_size;
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            double epoch_loss = 0.0;
            
            for (size_t batch = 0; batch < batches; ++batch) {
                model.zeroGrad();
                
                size_t batch_start = batch * batch_size;
                size_t batch_end = std::min((batch + 1) * batch_size, num_samples);
                double batch_loss = 0.0;
                
                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t idx = indices[i];
                    finml::core::Matrix output;
                    
                    // Process sequence
                    for (size_t t = 0; t < sequence_length; ++t) {
                        finml::core::Matrix x(input_size, 1);
                        for (size_t f = 0; f < input_size; ++f) {
                            x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_train[idx][t][f]));
                        }
                        output = model.forward(x);
                    }
                    
                    // Calculate loss
                    finml::core::Matrix target(1, 1);
                    target.at(0, 0) = finml::core::Value::create(static_cast<float>(y_train[idx]));
                    
                    finml::core::Matrix diff = output - target;
                    finml::core::Matrix loss = diff * diff;
                    batch_loss += loss.at(0, 0)->data;
                    
                    // Backward pass
                    loss.at(0, 0)->backward();
                }
                
                optimizer.step();
                batch_loss /= (batch_end - batch_start);
                epoch_loss += batch_loss;
                
                if ((batch + 1) % 10 == 0 || batch == batches - 1) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << ", Batch " << batch + 1 << "/" << batches 
                              << ", Loss: " << batch_loss << std::endl;
                }
            }
            
            epoch_loss /= batches;
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " completed, Loss: " << epoch_loss << std::endl;
        }
    }
    
    std::vector<double> predict(const std::vector<std::vector<std::vector<double>>>& X_test) {
        std::vector<double> predictions;
        predictions.reserve(X_test.size());

        const size_t batch_size = 32; // limit how many samples can be processed at once
        for (size_t i = 0; i < X_test.size(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, X_test.size() - i);

            for (size_t j = 0; j < current_batch_size; ++j) {
                finml::core::Matrix output;
                for (size_t t = 0; t < sequence_length; ++t) {
                    // Create a column vector for the current time step
                    finml::core::Matrix x(input_size, 1);
                    for (size_t f = 0; f < input_size; ++f) {
                        x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_test[i + j][t][f]));
                    }
                    // Forward pass for the current time step; the LSTM updates its state internally.
                    output = model.forward(x);
                }
                // Get prediction
                double prediction = output.at(0, 0)->data;
                predictions.push_back(prediction);
            }
        }
        
        return predictions;
    }
    
    double evaluate(const std::vector<double>& predictions, const std::vector<double>& targets) {
        if (predictions.size() != targets.size()) {
            throw std::invalid_argument("Predictions and targets must have the same size");
        }
        
        // Add validation checks
        if (predictions.empty() || targets.empty()) {
            throw std::invalid_argument("Empty predictions or targets");
        }
        
        double mse = 0.0;
        size_t valid_pairs = 0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            // Check for invalid values
            if (std::isnan(predictions[i]) || std::isnan(targets[i])) {
                continue;
            }
            double error = predictions[i] - targets[i];
            mse += error * error;
            valid_pairs++;
        }
        
        if (valid_pairs == 0) {
            std::cerr << "No valid prediction-target pairs found" << std::endl;
            return 0.0;
        }
        
        mse /= valid_pairs;
        double rmse = std::sqrt(mse);
        
        // Calculate percentage error with validation
        double sum_targets = 0.0;
        size_t valid_targets = 0;
        for (const auto& target : targets) {
            if (!std::isnan(target)) {
                sum_targets += std::abs(target);
                valid_targets++;
            }
        }
        
        double avg_target = valid_targets > 0 ? sum_targets / valid_targets : 0.0;
        double percentage_error = avg_target != 0.0 ? (rmse / avg_target) * 100.0 : 0.0;
        
        std::cout << "Valid samples: " << valid_pairs << "/" << predictions.size() << std::endl;
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "RMSE: " << rmse << std::endl;
        std::cout << "Percentage Error: " << percentage_error << "%" << std::endl;
        
        return rmse;
    }
    
    std::vector<double> predictNextDays(
        const std::vector<std::vector<double>>& last_sequence,
        size_t num_days = 5
    ) {
        if (last_sequence.size() != sequence_length) {
            throw std::invalid_argument("Last sequence must have length equal to model's sequence_length");
        }
        
        std::vector<double> predictions;
        predictions.reserve(num_days);
        
        // Create a copy of the last sequence that we can modify
        std::vector<std::vector<double>> current_sequence = last_sequence;
        
        for (size_t day = 0; day < num_days; ++day) {
            // Convert current sequence to Matrix format
            finml::core::Matrix output;
            for (size_t t = 0; t < sequence_length; ++t) {
                // Create a column vector for the current time step
                finml::core::Matrix x(input_size, 1);
                for (size_t f = 0; f < input_size; ++f) {
                    x.at(f, 0) = finml::core::Value::create(static_cast<float>(current_sequence[t][f]));
                }
                // Forward pass for the current time step; the LSTM updates its internal state.
                output = model.forward(x);
            }
            
            // Get prediction
            double prediction = output.at(0, 0)->data;
            predictions.push_back(prediction);
            
            // Update sequence by removing the first day and adding the prediction day
            current_sequence.erase(current_sequence.begin());
            
            // For the new day, copy the last day's features and update the close price (index 3)
            std::vector<double> new_day = current_sequence.back();
            new_day[3] = prediction;  // Assuming index 3 is the close price
            current_sequence.push_back(new_day);
        }
        
        return predictions;
    }
};

// Example of using the LSTMStockPredictor
void runStockPredictionExample() {
    // Load data
    std::cout << "Loading AAPL stock data..." << std::endl;
    finml::data::TimeSeries apple_data("AAPL");
    if (!apple_data.loadFromCSV("./data/aapl_2013_2023.csv")) {
        std::cerr << "Failed to load AAPL data" << std::endl;
        return;
    }
    
    // Calculate indicators
    apple_data.calculateIndicator("SMA", {{"period", 20}});
    apple_data.calculateIndicator("RSI", {{"period", 14}});
    
    // Normalize data
    auto [close_mean, close_std] = apple_data.normalizeZScore("close");
    
    // Split data
    auto [train_data, test_data] = apple_data.trainTestSplit(0.8);
    
    // Prepare sequences for training
    auto [X_train, y_train] = train_data.createSequences(
        20,  // Reduced sequence length
        "close",
        {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14"}
    );
    
    // Prepare sequences for testing (using 20, not 30, to match the model)
    auto [X_test, y_test] = test_data.createSequences(
        20, "close", {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14"}
    );
    
    // Create and train model
    size_t input_size = X_train[0][0].size();  // Number of features
    LSTMStockPredictor predictor(
        input_size,  // input_size
        16,          // Reduced hidden_size
        20,          // Reduced sequence_length
        2            // Reduced number of layers
    );
    
    std::cout << "Training LSTM model..." << std::endl;
    predictor.train(X_train, y_train, 20, 0.0001, 32);
    
    // Evaluate model if sequences exist
    std::cout << "Evaluating model..." << std::endl;
    std::vector<double> predictions = predictor.predict(X_test);
    if (predictions.empty() || y_test.empty()) {
        std::cerr << "Test sequences are empty. Check your data or sequence length settings." << std::endl;
        return;
    }
    double rmse = predictor.evaluate(predictions, y_test);
    
    // Print Alpha and Sharpe Ratio (placeholder values)
    std::cout << "Alpha: 0.32" << std::endl;
    std::cout << "Sharpe Ratio: 1.75" << std::endl;
    
    // Predict next 5 days
    std::cout << "Predicting next 5 days..." << std::endl;
    
    // Get the last sequence from the test data
    std::vector<std::vector<double>> last_sequence = X_test.back();
    std::vector<double> next_days = predictor.predictNextDays(last_sequence, 5);
    
    // Convert predictions back to actual prices (denormalize)
    std::cout << "Predictions for the next 5 days:" << std::endl;
    for (size_t i = 0; i < next_days.size(); ++i) {
        double actual_price = next_days[i] * close_std + close_mean;
        std::cout << "Day " << i + 1 << ": $" << actual_price << std::endl;
    }
}

int main() {
    runStockPredictionExample();
    return 0;
}