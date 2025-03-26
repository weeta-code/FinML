#include "finml/models/lstm_stock_predictor.h"
#include "finml/optimization/loss.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>

namespace finml {
namespace models {

LSTMStockPredictor::LSTMStockPredictor(
    int input_size, 
    int hidden_size, 
    int sequence_length, 
    int output_size,
    int seed
) : input_size_(input_size),
    hidden_size_(hidden_size),
    sequence_length_(sequence_length),
    output_size_(output_size),
    model_(input_size, hidden_size, output_size, 2, 0.5), // 2-layer LSTM with 0.5 dropout rate
    random_engine_(seed) {
    
    // Initialize the LSTM model
    std::cout << "Initialized LSTM Stock Predictor with:"
              << " input_size=" << input_size
              << " hidden_size=" << hidden_size
              << " sequence_length=" << sequence_length
              << " output_size=" << output_size
              << " seed=" << seed << std::endl;
}

void LSTMStockPredictor::train(
    const std::vector<std::vector<double>>& sequences, 
    const std::vector<std::vector<double>>& targets,
    int epochs,
    double learning_rate,
    double dropout_rate,
    int batch_size,
    double validation_split,
    int patience
) {
    if (sequences.empty() || targets.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    if (sequences.size() != targets.size()) {
        throw std::invalid_argument("Number of sequences must match number of targets");
    }
    
    // Split data into training and validation sets
    size_t total_samples = sequences.size();
    size_t validation_samples = static_cast<size_t>(total_samples * validation_split);
    size_t training_samples = total_samples - validation_samples;
    
    std::vector<std::vector<double>> train_sequences(sequences.begin(), sequences.begin() + training_samples);
    std::vector<std::vector<double>> train_targets(targets.begin(), targets.begin() + training_samples);
    
    std::vector<std::vector<double>> val_sequences;
    std::vector<std::vector<double>> val_targets;
    
    if (validation_samples > 0) {
        val_sequences.assign(sequences.begin() + training_samples, sequences.end());
        val_targets.assign(targets.begin() + training_samples, targets.end());
    }
    
    // Early stopping variables
    double best_val_loss = std::numeric_limits<double>::max();
    int no_improvement_count = 0;
    
    std::cout << "Training on " << training_samples << " samples, validating on " 
              << validation_samples << " samples" << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data
        std::vector<size_t> indices(training_samples);
        for (size_t i = 0; i < training_samples; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), random_engine_);
        
        double total_loss = 0.0;
        int batch_count = 0;
        
        // Mini-batch training
        for (size_t batch_start = 0; batch_start < training_samples; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, training_samples);
            size_t batch_size_actual = batch_end - batch_start;
            
            // Process batch
            model_.resetState();
            double batch_loss = 0.0;
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                
                // Convert sequence to matrix format
                auto seq_matrices = sequenceToMatrices(train_sequences[idx]);
                auto target_matrix = valueToMatrix(train_targets[idx]);
                
                // Forward pass
                core::Matrix output = model_.forward(seq_matrices.back());
                
                // Compute loss
                batch_loss += optimization::mse_loss(output, target_matrix);
                
                // Backward pass
                core::Matrix grad = optimization::mse_gradient(output, target_matrix);
                model_.backward(grad);
            }
            
            batch_loss /= batch_size_actual;
            total_loss += batch_loss;
            batch_count++;
            
            // Create a dropout mask function based on dropout_rate
            std::bernoulli_distribution dropout_dist(1.0 - dropout_rate);
            auto dropout_mask = [&]() -> bool { return dropout_dist(random_engine_); };
            
            // Update weights with dropout
            model_.updateWeightsWithDropout(learning_rate, dropout_mask);
        }
        
        total_loss /= batch_count;
        
        // Validation
        double val_loss = 0.0;
        if (validation_samples > 0) {
            model_.resetState();
            
            for (size_t i = 0; i < validation_samples; ++i) {
                auto seq_matrices = sequenceToMatrices(val_sequences[i]);
                auto target_matrix = valueToMatrix(val_targets[i]);
                
                core::Matrix output = model_.forward(seq_matrices.back());
                val_loss += optimization::mse_loss(output, target_matrix);
            }
            
            val_loss /= validation_samples;
            
            // Early stopping check
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
                if (no_improvement_count >= patience) {
                    std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                    break;
                }
            }
        }
        
        // Log progress every 10 epochs
        if ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << ", Loss: " << total_loss;
            
            if (validation_samples > 0) {
                std::cout << ", Val Loss: " << val_loss;
            }
            
            std::cout << std::endl;
        }
    }
    
    std::cout << "Training completed" << std::endl;
}

std::vector<double> LSTMStockPredictor::predictNextDays(
    const std::vector<double>& sequence, 
    int days
) {
    if (static_cast<int>(sequence.size()) < sequence_length_) {
        throw std::invalid_argument("Input sequence must be at least as long as sequence_length");
    }
    
    std::vector<double> result;
    std::vector<double> current_sequence = sequence;
    
    // Reset LSTM state
    model_.resetState();
    
    for (int i = 0; i < days; ++i) {
        // Use the last sequence_length_ values
        std::vector<double> input_sequence(current_sequence.end() - sequence_length_, current_sequence.end());
        
        // Convert to matrix format
        auto input_matrices = sequenceToMatrices(input_sequence);
        
        // Predict
        core::Matrix output = model_.forward(input_matrices.back());
        
        // Extract prediction (take the first output if multiple)
        double prediction = output.at(0, 0)->data;
        
        // Add to result
        result.push_back(prediction);
        
        // Update current sequence for next prediction
        current_sequence.push_back(prediction);
    }
    
    return result;
}

bool LSTMStockPredictor::saveModel(const std::string& filename) {
    // TODO: Implement serialization
    std::cout << "Model saving not implemented yet. Would save to: " << filename << std::endl;
    return false;
}

bool LSTMStockPredictor::loadModel(const std::string& filename) {
    // TODO: Implement deserialization
    std::cout << "Model loading not implemented yet. Would load from: " << filename << std::endl;
    return false;
}

std::vector<core::Matrix> LSTMStockPredictor::sequenceToMatrices(const std::vector<double>& sequence) {
    std::vector<core::Matrix> matrices;
    
    for (size_t i = 0; i < sequence.size(); ++i) {
        core::Matrix m(input_size_, 1);
        for (int j = 0; j < input_size_; ++j) {
            m.at(j, 0) = core::Value::create(sequence[i]);
        }
        matrices.push_back(m);
    }
    
    return matrices;
}

core::Matrix LSTMStockPredictor::valueToMatrix(const std::vector<double>& values) {
    core::Matrix m(output_size_, 1);
    
    for (int i = 0; i < output_size_ && i < static_cast<int>(values.size()); ++i) {
        m.at(i, 0) = core::Value::create(values[i]);
    }
    
    return m;
}

} // namespace models
} // namespace finml 