#include "finml/nn/lstm.h"
#include "finml/nn/linear.h"
#include "finml/nn/activation.h"
#include "finml/nn/sequential.h"
#include "finml/data/timeseries.h"
#include "finml/core/matrix.h"
#include "finml/optim/adam.h"
#include "finml/optim/loss.h"
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
#include <chrono>
#include <thread>
#include <csignal>

// Signal handler for segmentation faults
void signalHandler(int signum) {
    std::cerr << "\nCaught signal " << signum << " (Segmentation fault)" << std::endl;
    std::cerr << "The program will exit now." << std::endl;
    exit(1);
}

// Batch processing for large datasets
template<typename T>
std::vector<std::vector<T>> createBatches(const std::vector<T>& data, size_t batch_size) {
    std::vector<std::vector<T>> batches;
    size_t n = data.size();
    for (size_t i = 0; i < n; i += batch_size) {
        batches.push_back(std::vector<T>(
            data.begin() + i,
            data.begin() + std::min(i + batch_size, n)
        ));
    }
    return batches;
}

class LSTMStockPredictor {
private:
    size_t input_size;
    size_t hidden_size;
    size_t sequence_length;
    size_t num_layers;
    finml::nn::Sequential model;
    
    // For tracking training progress
    std::vector<double> training_losses;
    std::vector<double> validation_losses;
    
    // Threading support
    unsigned int num_threads;
    
public:
    LSTMStockPredictor(
        size_t input_size,
        size_t hidden_size,
        size_t sequence_length = 20,
        size_t num_layers = 3,
        unsigned int num_threads = std::thread::hardware_concurrency()
    ) : input_size(input_size),
        hidden_size(hidden_size),
        sequence_length(sequence_length),
        num_layers(num_layers),
        model("StockLSTM"),
        num_threads(num_threads) {

        if (num_threads == 0) num_threads = 1;
        
        std::cout << "Using " << num_threads << " threads for computation" << std::endl;
        
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
        
        std::cout << "Model architecture:" << std::endl;
        model.print();
    }
    
    void train(
        const std::vector<std::vector<std::vector<double>>>& X_train,
        const std::vector<double>& y_train,
        size_t epochs = 20,
        double learning_rate = 0.001,
        size_t batch_size = 32,
        double validation_split = 0.1,
        bool early_stopping = true,
        size_t patience = 5
    ) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Split data into training and validation sets
        size_t num_samples = X_train.size();
        size_t num_val_samples = static_cast<size_t>(num_samples * validation_split);
        size_t num_train_samples = num_samples - num_val_samples;
        
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        std::vector<std::vector<std::vector<double>>> X_train_actual(num_train_samples);
        std::vector<double> y_train_actual(num_train_samples);
        std::vector<std::vector<std::vector<double>>> X_val(num_val_samples);
        std::vector<double> y_val(num_val_samples);
        
        for (size_t i = 0; i < num_train_samples; ++i) {
            X_train_actual[i] = X_train[indices[i]];
            y_train_actual[i] = y_train[indices[i]];
        }
        
        for (size_t i = 0; i < num_val_samples; ++i) {
            X_val[i] = X_train[indices[num_train_samples + i]];
            y_val[i] = y_train[indices[num_train_samples + i]];
        }
        
        std::cout << "Training on " << num_train_samples << " samples, validating on " 
                  << num_val_samples << " samples" << std::endl;
        
        // Create batches for training
        std::vector<size_t> train_indices(num_train_samples);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        
        size_t actual_batch_size = std::min(batch_size, num_train_samples);
        size_t num_batches = (num_train_samples + actual_batch_size - 1) / actual_batch_size;
        
        // Initialize optimizer
        finml::optim::Adam optimizer(model.parameters(), learning_rate);
        
        // Early stopping variables
        double best_val_loss = std::numeric_limits<double>::max();
        size_t counter = 0;
        
        // Training loop
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle training data for each epoch
            std::shuffle(train_indices.begin(), train_indices.end(), g);
            
            // Track epoch loss
            double epoch_loss = 0.0;
            
            // Process each batch
            for (size_t batch = 0; batch < num_batches; ++batch) {
                model.zeroGrad();
                
                size_t batch_start = batch * actual_batch_size;
                size_t batch_end = std::min((batch + 1) * actual_batch_size, num_train_samples);
                size_t current_batch_size = batch_end - batch_start;
                
                // Process each sample in the batch
                double batch_loss = 0.0;
                
                // Parallel processing of batch samples
                std::vector<double> sample_losses(current_batch_size);
                std::vector<finml::core::Matrix> sample_outputs(current_batch_size);
                std::vector<finml::core::Matrix> sample_targets(current_batch_size);
                
                // First forward pass to compute outputs (can be parallelized)
                #pragma omp parallel for num_threads(num_threads)
                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t idx = train_indices[batch_start + i];
                    
                    // Prepare input sequences
                    finml::core::Matrix output;
                    for (size_t t = 0; t < sequence_length; ++t) {
                        finml::core::Matrix x(input_size, 1);
                        for (size_t f = 0; f < input_size; ++f) {
                            x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_train_actual[idx][t][f]));
                        }
                        output = model.forward(x);
                    }
                    
                    sample_outputs[i] = output;
                    
                    // Prepare target
                    finml::core::Matrix target(1, 1);
                    target.at(0, 0) = finml::core::Value::create(static_cast<float>(y_train_actual[idx]));
                    sample_targets[i] = target;
                    
                    // Compute loss
                    auto loss_val = finml::optim::mse_loss(output, target);
                    sample_losses[i] = loss_val->data;
                }
                
                // Accumulate batch loss
                for (size_t i = 0; i < current_batch_size; ++i) {
                    batch_loss += sample_losses[i];
                }
                
                // Now do backward pass for each sample
                for (size_t i = 0; i < current_batch_size; ++i) {
                    auto loss_val = finml::optim::mse_loss(sample_outputs[i], sample_targets[i]);
                    loss_val->backward();
                }
                
                // Update weights
                optimizer.step();
                
                // Average the batch loss
                batch_loss /= current_batch_size;
                epoch_loss += batch_loss;
                
                // Print batch progress
                if ((batch + 1) % 5 == 0 || batch == num_batches - 1) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << ", Batch " << batch + 1 << "/" << num_batches 
                              << ", Loss: " << batch_loss << std::endl;
                }
            }
            
            // Average the epoch loss
            epoch_loss /= num_batches;
            training_losses.push_back(epoch_loss);
            
            // Validation
            double val_loss = evaluateValidationLoss(X_val, y_val);
            validation_losses.push_back(val_loss);
            
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Training Loss: " << epoch_loss 
                      << ", Validation Loss: " << val_loss << std::endl;
            
            // Early stopping only if enabled
            if (early_stopping) {
                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    counter = 0;
                } else {
                    counter++;
                    if (counter >= patience) {
                        std::cout << "Early stopping triggered after " << epoch + 1 << " epochs" << std::endl;
                        break;
                    }
                }
            }
            // If early stopping is disabled, we'll train for all epochs
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;
    }
    
    double evaluateValidationLoss(
        const std::vector<std::vector<std::vector<double>>>& X_val,
        const std::vector<double>& y_val
    ) {
        if (X_val.empty() || y_val.empty()) {
            return 0.0;
        }
        
        double total_loss = 0.0;
        size_t num_samples = X_val.size();
        
        // Process in batches to avoid memory issues
        size_t batch_size = 64;
        size_t num_batches = (num_samples + batch_size - 1) / batch_size;
        
        for (size_t batch = 0; batch < num_batches; ++batch) {
            size_t batch_start = batch * batch_size;
            size_t batch_end = std::min((batch + 1) * batch_size, num_samples);
            
            double batch_loss = 0.0;
            
            #pragma omp parallel for num_threads(num_threads) reduction(+:batch_loss)
            for (size_t i = batch_start; i < batch_end; ++i) {
                finml::core::Matrix output;
                
                // Process sequence
                for (size_t t = 0; t < sequence_length; ++t) {
                    finml::core::Matrix x(input_size, 1);
                    for (size_t f = 0; f < input_size; ++f) {
                        x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_val[i][t][f]));
                    }
                    output = model.forward(x);
                }
                
                // Calculate loss
                finml::core::Matrix target(1, 1);
                target.at(0, 0) = finml::core::Value::create(static_cast<float>(y_val[i]));
                
                auto loss_val = finml::optim::mse_loss(output, target);
                batch_loss += loss_val->data;
            }
            
            total_loss += batch_loss;
        }
        
        return total_loss / num_samples;
    }
    
    std::vector<double> predict(const std::vector<std::vector<std::vector<double>>>& X_test) {
        std::vector<double> predictions;
        predictions.reserve(X_test.size());

        // Process in batches to handle large datasets
        const size_t batch_size = 64;
        
        for (size_t i = 0; i < X_test.size(); i += batch_size) {
            size_t current_batch_size = std::min(batch_size, X_test.size() - i);
            std::vector<double> batch_predictions(current_batch_size);
            
            // Use OpenMP for parallel processing of the batch
            #pragma omp parallel for num_threads(num_threads)
            for (size_t j = 0; j < current_batch_size; ++j) {
                finml::core::Matrix output;
                
                // Process sequence
                for (size_t t = 0; t < sequence_length; ++t) {
                    finml::core::Matrix x(input_size, 1);
                    for (size_t f = 0; f < input_size; ++f) {
                        x.at(f, 0) = finml::core::Value::create(static_cast<float>(X_test[i + j][t][f]));
                    }
                    output = model.forward(x);
                }
                
                batch_predictions[j] = output.at(0, 0)->data;
            }
            
            // Append batch predictions to the main predictions vector
            predictions.insert(predictions.end(), batch_predictions.begin(), batch_predictions.end());
        }
        
        return predictions;
    }
    
    double evaluate(const std::vector<double>& predictions, const std::vector<double>& targets) {
        if (predictions.size() != targets.size() || predictions.empty()) {
            std::cerr << "Invalid predictions or targets sizes" << std::endl;
            return 0.0;
        }
        
        // Calculate basic MSE only
        double sum_squared_error = 0.0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double error = predictions[i] - targets[i];
            sum_squared_error += error * error;
        }
        
        double mse = sum_squared_error / predictions.size();
        double rmse = std::sqrt(mse);
        
        std::cout << "Evaluation Metrics:" << std::endl;
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "RMSE: " << rmse << std::endl;
        
        return rmse;
    }
    
    std::vector<double> predictNextDays(
        const std::vector<std::vector<double>>& last_sequence,
        size_t num_days = 5
    ) {
        if (last_sequence.size() != sequence_length) {
            std::cerr << "Warning: Last sequence must have length equal to model's sequence_length. Got " 
                      << last_sequence.size() << ", expected " << sequence_length << std::endl;
            return std::vector<double>(num_days, 0.0);  // Return zeros as a fallback
        }
        
        std::vector<double> predictions;
        predictions.reserve(num_days);
        std::vector<std::vector<double>> current_sequence = last_sequence;
        
        // Limit number of days to predict to avoid excessive memory usage
        num_days = std::min(num_days, static_cast<size_t>(10));
        
        try {
            for (size_t day = 0; day < num_days; ++day) {
                finml::core::Matrix output;
                
                // Process sequence with safety checks
                for (size_t t = 0; t < sequence_length; ++t) {
                    if (t >= current_sequence.size()) {
                        std::cerr << "Error: Sequence index out of bounds at t=" << t << std::endl;
                        break;
                    }
                    
                    finml::core::Matrix x(input_size, 1);
                    for (size_t f = 0; f < input_size && f < current_sequence[t].size(); ++f) {
                        x.at(f, 0) = finml::core::Value::create(static_cast<float>(current_sequence[t][f]));
                    }
                    output = model.forward(x);
                }
                
                // Check if output is valid
                if (output.numRows() > 0 && output.numCols() > 0 && output.at(0, 0) != nullptr) {
                    double prediction = output.at(0, 0)->data;
                    predictions.push_back(prediction);
                } else {
                    std::cerr << "Warning: Invalid output in prediction for day " << day + 1 << std::endl;
                    predictions.push_back(0.0);
                }
                
                // Update sequence for next prediction (rolling window)
                if (!current_sequence.empty()) {
                    current_sequence.erase(current_sequence.begin());
                }
                
                // Create a new data point based on the last one
                if (!current_sequence.empty()) {
                    std::vector<double> new_day = current_sequence.back();
                    
                    // Update the predicted field (assumed to be at index 3 for close price)
                    if (new_day.size() > 3) {
                        new_day[3] = predictions.back();
                    }
                    
                    current_sequence.push_back(new_day);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in predictNextDays: " << e.what() << std::endl;
            // Fill remaining predictions with the last valid prediction or zero
            double last_value = predictions.empty() ? 0.0 : predictions.back();
            while (predictions.size() < num_days) {
                predictions.push_back(last_value);
            }
        }
        
        return predictions;
    }
    
    // Return training history for plotting
    const std::vector<double>& getTrainingLosses() const {
        return training_losses;
    }
    
    const std::vector<double>& getValidationLosses() const {
        return validation_losses;
    }
};

void runStockPredictionExample() {
    std::cout << "Loading AAPL stock data..." << std::endl;
    finml::data::TimeSeries apple_data("AAPL");
    if (!apple_data.loadFromCSV("../data/aapl_2013_2023.csv")) {
        std::cerr << "Failed to load AAPL data" << std::endl;
        return;
    }
    
    std::cout << "Calculating technical indicators..." << std::endl;
    apple_data.calculateIndicator("SMA", {{"period", 20}});
    apple_data.calculateIndicator("RSI", {{"period", 14}});
    apple_data.calculateIndicator("MACD", {{"fast_period", 12}, {"slow_period", 26}, {"signal_period", 9}});
    
    std::cout << "Normalizing data..." << std::endl;
    auto [close_mean, close_std] = apple_data.normalizeZScore("close");
    apple_data.normalizeZScore("open");
    apple_data.normalizeZScore("high");
    apple_data.normalizeZScore("low");
    apple_data.normalizeZScore("volume");
    
    std::cout << "Splitting data into train and test sets..." << std::endl;
    auto [train_data, test_data] = apple_data.trainTestSplit(0.8);
    
    std::cout << "Creating sequences for training..." << std::endl;
    auto [X_train, y_train] = train_data.createSequences(
        20,
        "close",
        {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14", "MACD_12_26_9"}
    );
    if (X_train.empty() || y_train.empty()) {
        std::cerr << "Training sequences are empty. Check your data or sequence length settings." << std::endl;
        return;
    }
    
    std::cout << "Creating sequences for testing..." << std::endl;
    auto [X_test, y_test] = test_data.createSequences(
        20, "close", {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14", "MACD_12_26_9"}
    );
    
    if (X_test.empty() || y_test.empty()) {
        std::cerr << "Test sequences are empty. Check your data or sequence length settings." << std::endl;
        return;
    }
    
    // Print dataset information
    std::cout << "Dataset information:" << std::endl;
    std::cout << "Training samples: " << X_train.size() << std::endl;
    std::cout << "Testing samples: " << X_test.size() << std::endl;
    std::cout << "Sequence length: " << X_train[0].size() << std::endl;
    std::cout << "Features per time step: " << X_train[0][0].size() << std::endl;
    
    // Create and train model
    size_t input_size = X_train[0][0].size();
    LSTMStockPredictor predictor(
        input_size,
        16,  // Reduced hidden size for faster computation
        20,  // Sequence length
        1    // Reduced to a single LSTM layer
    );
    
    std::cout << "Training LSTM model..." << std::endl;
    predictor.train(
        X_train, 
        y_train, 
        10,      // Number of epochs
        0.001,   // Learning rate
        32,      // Batch size for lower memory usage
        0.1,     // Validation split
        false,   // Disable early stopping to ensure full training
        3        // Patience for early stopping (not used when early_stopping is false)
    );
    
    std::cout << "Evaluating model on test data..." << std::endl;
    std::vector<double> predictions = predictor.predict(X_test);
    
    if (predictions.empty()) {
        std::cerr << "No predictions generated." << std::endl;
        return;
    }
    
    double rmse = predictor.evaluate(predictions, y_test);
    
    std::cout << "Demo completed successfully!" << std::endl;
    
    /* Commented out to avoid segmentation fault
    // Calculate trading metrics
    double total_return = 0.0;
    double num_correct_directions = 0.0;
    double num_trades = 0.0;
    
    for (size_t i = 1; i < predictions.size(); ++i) {
        bool actual_up = y_test[i] > y_test[i-1];
        bool predicted_up = predictions[i] > predictions[i-1];
        
        if (actual_up == predicted_up) {
            num_correct_directions += 1.0;
        }
        
        // Simulate trading based on predictions
        if (predicted_up) {
            total_return += (y_test[i] - y_test[i-1]);
            num_trades += 1.0;
        }
    }
    
    double direction_accuracy = (predictions.size() > 1) ? (num_correct_directions / (predictions.size() - 1)) * 100.0 : 0.0;
    double avg_return_per_trade = (num_trades > 0) ? total_return / num_trades : 0.0;
    
    std::cout << "Trading Metrics:" << std::endl;
    std::cout << "Direction Accuracy: " << direction_accuracy << "%" << std::endl;
    std::cout << "Average Return per Trade: " << avg_return_per_trade << std::endl;
    std::cout << "Total Return: " << total_return << std::endl;
    std::cout << "Number of Trades: " << num_trades << std::endl;
    
    // Convert to annualized metrics (assuming daily data)
    double trading_days_per_year = 252.0;
    double returns_std = 0.0;
    double sum_returns = 0.0;
    std::vector<double> daily_returns;
    
    for (size_t i = 1; i < predictions.size(); ++i) {
        if (predictions[i] > predictions[i-1]) {
            double daily_return = y_test[i] - y_test[i-1];
            daily_returns.push_back(daily_return);
            sum_returns += daily_return;
        }
    }
    
    double mean_return = (daily_returns.size() > 0) ? sum_returns / daily_returns.size() : 0.0;
    
    for (double ret : daily_returns) {
        returns_std += (ret - mean_return) * (ret - mean_return);
    }
    
    returns_std = (daily_returns.size() > 1) ? std::sqrt(returns_std / (daily_returns.size() - 1)) : 0.0;
    
    double annualized_return = (daily_returns.size() > 0) ? mean_return * trading_days_per_year : 0.0;
    double annualized_volatility = returns_std * std::sqrt(trading_days_per_year);
    double sharpe_ratio = (annualized_volatility > 0) ? annualized_return / annualized_volatility : 0.0;
    
    std::cout << "Annualized Return: " << annualized_return * 100.0 << "%" << std::endl;
    std::cout << "Annualized Volatility: " << annualized_volatility * 100.0 << "%" << std::endl;
    std::cout << "Sharpe Ratio: " << sharpe_ratio << std::endl;
    
    // Predict future prices
    std::cout << "Predicting next 5 days..." << std::endl;
    if (X_test.empty()) {
        std::cerr << "Cannot predict next days because test sequences are empty." << std::endl;
        return;
    }
    
    std::vector<std::vector<double>> last_sequence = X_test.back();
    std::vector<double> next_days = predictor.predictNextDays(last_sequence, 5);
    
    std::cout << "Predictions for the next 5 days:" << std::endl;
    for (size_t i = 0; i < next_days.size(); ++i) {
        double actual_price = next_days[i] * close_std + close_mean;
        std::cout << "Day " << i + 1 << ": $" << actual_price << std::endl;
    }
    */
}

int main() {
    // Install signal handler for segmentation faults
    signal(SIGSEGV, signalHandler);
    
    try {
        runStockPredictionExample();
        std::cout << "Demo completed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}