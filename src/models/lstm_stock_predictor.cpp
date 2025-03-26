#include <finml/models/lstm_stock_predictor.h>
#include <finml/optimization/loss.h>
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
    int num_layers,
    double dropout_rate,
    unsigned int seed
) : input_size_(input_size),
    hidden_size_(hidden_size),
    sequence_length_(sequence_length),
    output_size_(output_size),
    num_layers_(num_layers),
    dropout_rate_(dropout_rate),
    rng_(seed),
    lstm_(input_size, hidden_size, output_size, num_layers, dropout_rate) {
    
    // Initialize random number generator
    if (seed == 0) {
        // Use random seed if 0 is provided
        std::random_device rd;
        rng_.seed(rd());
    }
}

void LSTMStockPredictor::train(
    const std::vector<std::vector<double>>& sequences, 
    const std::vector<std::vector<double>>& targets,
    int epochs,
    double learning_rate,
    double validation_split,
    int patience,
    bool use_dropout
) {
    if (sequences.empty() || targets.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    if (sequences.size() != targets.size()) {
        throw std::invalid_argument("Number of sequences must match number of targets");
    }
    
    if (validation_split < 0.0 || validation_split >= 1.0) {
        throw std::invalid_argument("Validation split must be between 0 and 1");
    }
    
    // Split data into training and validation sets
    size_t train_size = static_cast<size_t>((1.0 - validation_split) * sequences.size());
    
    std::vector<std::vector<double>> train_sequences(sequences.begin(), sequences.begin() + train_size);
    std::vector<std::vector<double>> train_targets(targets.begin(), targets.begin() + train_size);
    
    std::vector<std::vector<double>> val_sequences;
    std::vector<std::vector<double>> val_targets;
    
    if (validation_split > 0.0) {
        val_sequences.assign(sequences.begin() + train_size, sequences.end());
        val_targets.assign(targets.begin() + train_size, targets.end());
    }
    
    // Training variables
    double best_val_loss = std::numeric_limits<double>::max();
    int patience_counter = 0;
    
    // Create dropout mask generator
    std::bernoulli_distribution dropout_dist(1.0 - dropout_rate_);
    auto dropout_mask = [&dropout_dist, this]() -> bool {
        return dropout_dist(rng_);
    };
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data
        std::vector<size_t> indices(train_sequences.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng_);
        
        double total_loss = 0.0;
        int num_batches = 0;
        
        // Training phase
        for (size_t idx : indices) {
            // Reset LSTM state for new sequence
            lstm_.resetState();
            
            // Convert sequence to matrix
            core::Matrix input_matrix = sequenceToMatrix(train_sequences[idx]);
            core::Matrix target_matrix = sequenceToMatrix(train_targets[idx]);
            
            // Forward pass
            core::Matrix output = lstm_.forward(input_matrix);
            
            // Compute loss (MSE)
            double loss = 0.0;
            for (size_t i = 0; i < output.numRows(); ++i) {
                for (size_t j = 0; j < output.numCols(); ++j) {
                    double diff = output.get(i, j) - target_matrix.get(i, j);
                    loss += diff * diff;
                }
            }
            loss /= (output.numRows() * output.numCols());
            
            // Backward pass
            core::Matrix gradient(output.numRows(), output.numCols());
            // Compute gradient of MSE loss
            for (size_t i = 0; i < output.numRows(); ++i) {
                for (size_t j = 0; j < output.numCols(); ++j) {
                    double diff = output.get(i, j) - target_matrix.get(i, j);
                    gradient.set(i, j, 2.0 * diff / (output.numRows() * output.numCols()));
                }
            }
            lstm_.backward(gradient);
            
            // Update weights with or without dropout
            if (use_dropout) {
                lstm_.updateWeightsWithDropout(learning_rate, dropout_mask);
            } else {
                lstm_.updateWeights(learning_rate);
            }
            
            total_loss += loss;
            num_batches++;
        }
        
        double avg_train_loss = total_loss / num_batches;
        
        // Validation phase
        double val_loss = 0.0;
        
        if (!val_sequences.empty()) {
            for (size_t i = 0; i < val_sequences.size(); ++i) {
                // Reset LSTM state for new sequence
                lstm_.resetState();
                
                // Convert sequence to matrix
                core::Matrix input_matrix = sequenceToMatrix(val_sequences[i]);
                core::Matrix target_matrix = sequenceToMatrix(val_targets[i]);
                
                // Forward pass (no backward pass during validation)
                core::Matrix output = lstm_.forward(input_matrix);
                
                // Compute loss (MSE)
                double seq_loss = 0.0;
                for (size_t j = 0; j < output.numRows(); ++j) {
                    for (size_t k = 0; k < output.numCols(); ++k) {
                        double diff = output.get(j, k) - target_matrix.get(j, k);
                        seq_loss += diff * diff;
                    }
                }
                seq_loss /= (output.numRows() * output.numCols());
                
                val_loss += seq_loss;
            }
            
            val_loss /= val_sequences.size();
            
            // Early stopping
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                patience_counter = 0;
            } else {
                patience_counter++;
                if (patience > 0 && patience_counter >= patience) {
                    std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                    break;
                }
            }
        }
        
        // Print progress
        std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                  << " - Loss: " << avg_train_loss;
        
        if (!val_sequences.empty()) {
            std::cout << " - Val Loss: " << val_loss;
        }
        
        std::cout << std::endl;
    }
}

std::vector<double> LSTMStockPredictor::predictNextDays(
    const std::vector<double>& sequence, 
    int num_days
) {
    if (sequence.size() < sequence_length_) {
        throw std::invalid_argument("Input sequence length must be at least sequence_length_");
    }
    
    // Take the last sequence_length_ values from the input sequence
    std::vector<double> current_sequence(
        sequence.end() - sequence_length_,
        sequence.end()
    );
    
    std::vector<double> predictions;
    
    // Reset LSTM state
    lstm_.resetState();
    
    for (int i = 0; i < num_days; ++i) {
        // Convert current sequence to matrix
        core::Matrix input_matrix = valueToMatrix(current_sequence);
        
        // Forward pass
        core::Matrix output = lstm_.forward(input_matrix);
        
        // Get prediction (last value of output)
        double prediction = output.get(output.numRows() - 1, 0);
        predictions.push_back(prediction);
        
        // Update current sequence by removing the first element and adding the prediction
        current_sequence.erase(current_sequence.begin());
        current_sequence.push_back(prediction);
    }
    
    return predictions;
}

bool LSTMStockPredictor::save(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // TODO: Implement serialization
    // For now, just save hyperparameters
    file.write(reinterpret_cast<const char*>(&input_size_), sizeof(input_size_));
    file.write(reinterpret_cast<const char*>(&hidden_size_), sizeof(hidden_size_));
    file.write(reinterpret_cast<const char*>(&sequence_length_), sizeof(sequence_length_));
    file.write(reinterpret_cast<const char*>(&output_size_), sizeof(output_size_));
    file.write(reinterpret_cast<const char*>(&num_layers_), sizeof(num_layers_));
    file.write(reinterpret_cast<const char*>(&dropout_rate_), sizeof(dropout_rate_));
    
    return true;
}

bool LSTMStockPredictor::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return false;
    }
    
    // TODO: Implement deserialization
    // For now, just load hyperparameters
    file.read(reinterpret_cast<char*>(&input_size_), sizeof(input_size_));
    file.read(reinterpret_cast<char*>(&hidden_size_), sizeof(hidden_size_));
    file.read(reinterpret_cast<char*>(&sequence_length_), sizeof(sequence_length_));
    file.read(reinterpret_cast<char*>(&output_size_), sizeof(output_size_));
    file.read(reinterpret_cast<char*>(&num_layers_), sizeof(num_layers_));
    file.read(reinterpret_cast<char*>(&dropout_rate_), sizeof(dropout_rate_));
    
    // Recreate the LSTM model
    lstm_ = LSTM(input_size_, hidden_size_, output_size_, num_layers_, dropout_rate_);
    
    return true;
}

core::Matrix LSTMStockPredictor::sequenceToMatrix(const std::vector<double>& sequence) const {
    core::Matrix matrix(sequence.size(), input_size_);
    
    for (size_t i = 0; i < sequence.size(); ++i) {
        for (size_t j = 0; j < input_size_; ++j) {
            matrix.set(i, j, sequence[i]);
        }
    }
    
    return matrix;
}

core::Matrix LSTMStockPredictor::valueToMatrix(const std::vector<double>& values) const {
    core::Matrix matrix(1, input_size_);
    
    for (size_t j = 0; j < input_size_; ++j) {
        if (j < values.size()) {
            matrix.set(0, j, values[j]);
        } else {
            matrix.set(0, j, 0.0);
        }
    }
    
    return matrix;
}

} // namespace models
} // namespace finml 