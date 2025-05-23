#include "finml/models/lstm_stock_predictor.h"
#include "finml/optim/loss.h"
#include "finml/core/matrix.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace finml {
namespace models {

// Helper function to check matrix dimensions
static bool checkMatrixDimensions(const core::Matrix& matrix, size_t expected_rows, const char* matrix_name) {
    if (matrix.numRows() != expected_rows) {
        std::cerr << "Error: " << matrix_name << " dimensions (" << matrix.numRows() 
                  << ") don't match expected size (" << expected_rows << ")" << std::endl;
        return false;
    }
    return true;
}

// Helper function to create dropout mask
static std::function<bool()> createDropoutMask(double dropout_rate) {
    std::bernoulli_distribution dropout_dist(1.0 - dropout_rate);
    return [&]() -> bool { return dropout_dist(core::getRandomEngine()); };
}

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
    model_(input_size, hidden_size, true, "StockPredictorLSTM") {
    
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
        // Create indices for shuffling
        std::vector<size_t> indices(training_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Use the matrix random number generator for shuffling
        std::shuffle(indices.begin(), indices.end(), core::getRandomEngine());
        
        double total_loss = 0.0;
        int batch_count = 0;
        
        // Mini-batch training
        for (size_t batch_start = 0; batch_start < training_samples; batch_start += batch_size) {
            size_t batch_end = std::min(batch_start + batch_size, training_samples);
            size_t batch_size_actual = batch_end - batch_start;
            
            // Process batch
            model_.reset_state(); // Use reset_state() instead of resetState()
            double batch_loss = 0.0;
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                size_t idx = indices[i];
                
                // Debug output
                std::cout << "Processing sequence " << idx << " with size " << train_sequences[idx].size() << std::endl;
                
                // Convert sequence to matrix format
                auto seq_matrices = sequenceToMatrices(train_sequences[idx]);
                auto target_matrix = valueToMatrix(train_targets[idx]);
                
                // Forward pass through all sequence elements
                core::Matrix output;
                
                // Debug output matrix dimensions
                std::cout << "Processing " << seq_matrices.size() << " matrices in sequence" << std::endl;
                
                for (size_t seq_idx = 0; seq_idx < seq_matrices.size(); ++seq_idx) {
                    const auto& seq_matrix = seq_matrices[seq_idx];
                    std::cout << "  Matrix " << seq_idx << " dimensions: " 
                              << seq_matrix.numRows() << "x" << seq_matrix.numCols() << std::endl;
                    
                    // Ensure LSTM input dimensions match
                    if (seq_matrix.numRows() != input_size_) {
                        std::cerr << "Error: Matrix dimensions (" << seq_matrix.numRows() 
                                  << ") don't match LSTM input size (" << input_size_ << ")" << std::endl;
                        continue;
                    }
                    
                    output = model_.forward(seq_matrix);
                }
                
                // Calculate MSE loss
                double mse = 0.0;
                
                // Debug output target matrix dimensions
                std::cout << "Target matrix dimensions: " 
                          << target_matrix.numRows() << "x" << target_matrix.numCols() << std::endl;
                std::cout << "Output matrix dimensions: " 
                          << output.numRows() << "x" << output.numCols() << std::endl;
                
                // Use the centralized MSE loss function
                mse = optim::mse_loss(output, target_matrix);
                
                batch_loss += mse;
            }
            
            batch_loss /= batch_size_actual;
            total_loss += batch_loss;
            batch_count++;
            
            // Create dropout mask using the same random generator
            auto dropout_mask = createDropoutMask(dropout_rate);
            
            // Print about dropping out 
            std::cout << "Batch " << batch_count << " complete. Applied dropout with rate: " 
                      << dropout_rate << std::endl;
        }
        
        total_loss /= batch_count;
        
        // Validation
        double val_loss = 0.0;
        if (validation_samples > 0) {
            model_.reset_state();
            
            for (size_t i = 0; i < validation_samples; ++i) {
                auto seq_matrices = sequenceToMatrices(val_sequences[i]);
                auto target_matrix = valueToMatrix(val_targets[i]);
                
                // Forward pass through all sequence elements
                core::Matrix output;
                for (const auto& seq_matrix : seq_matrices) {
                    output = model_.forward(seq_matrix);
                }
                
                // Use the centralized MSE loss function
                val_loss += optim::mse_loss(output, target_matrix);
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
    
    // Debug output
    std::cout << "Predicting " << days << " days with sequence of size " << current_sequence.size() << std::endl;
    
    // Reset LSTM state
    model_.reset_state(); // Use reset_state() instead of resetState()
    
    for (int i = 0; i < days; ++i) {
        // Use the last sequence_length_ values
        std::vector<double> input_sequence;
        
        if (current_sequence.size() >= sequence_length_) {
            input_sequence.assign(
                current_sequence.end() - sequence_length_ * input_size_, 
                current_sequence.end()
            );
        } else {
            std::cerr << "Error: Current sequence length " << current_sequence.size() 
                      << " is less than required " << sequence_length_ << std::endl;
            // Pad the sequence if needed
            while (input_sequence.size() < sequence_length_ * input_size_) {
                input_sequence.push_back(0.0);
            }
        }
        
        // Convert to matrix format
        auto input_matrices = sequenceToMatrices(input_sequence);
        
        // Predict
        core::Matrix output;
        
        for (const auto& seq_matrix : input_matrices) {
            if (checkMatrixDimensions(seq_matrix, input_size_, "Input matrix")) {
                output = model_.forward(seq_matrix);
            }
        }
        
        // Extract prediction (take the first output if multiple)
        double prediction = 0.0;
        if (output.numRows() > 0 && output.numCols() > 0 && output.at(0, 0) != nullptr) {
            prediction = output.at(0, 0)->data;
            std::cout << "Predicted value for day " << i + 1 << ": " << prediction << std::endl;
        } else {
            std::cerr << "Error: Invalid output matrix dimensions: " 
                      << output.numRows() << "x" << output.numCols() << std::endl;
        }
        
        // Add to result
        result.push_back(prediction);
        
        // Update current sequence for next prediction
        for (int j = 0; j < input_size_; ++j) {
            current_sequence.push_back(prediction);
        }
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
        if (i % input_size_ == 0 && i / input_size_ < sequence.size() / input_size_) {
            core::Matrix m(input_size_, 1);
            for (int j = 0; j < input_size_ && i + j < sequence.size(); ++j) {
                m.at(j, 0) = core::Value::create(sequence[i + j]);
            }
            matrices.push_back(m);
            i += input_size_ - 1;
        }
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