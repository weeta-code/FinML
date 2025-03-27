#ifndef FINML_MODELS_LSTM_STOCK_PREDICTOR_H
#define FINML_MODELS_LSTM_STOCK_PREDICTOR_H

#include "finml/nn/lstm.h"
#include "finml/core/matrix.h"
#include <vector>
#include <string>
#include <random>

namespace finml {
namespace models {

/**
 * @brief LSTM-based stock price predictor
 * 
 * This class uses an LSTM neural network to predict future stock prices
 * based on historical price sequences.
 */
class LSTMStockPredictor {
public:
    /**
     * @brief Construct a new LSTM Stock Predictor
     * 
     * @param input_size Number of features in input (typically 1 for univariate time series)
     * @param hidden_size Size of hidden state in LSTM cells
     * @param sequence_length Length of input sequences
     * @param output_size Number of outputs (typically 1 for price prediction)
     * @param seed Random seed for weight initialization and dropout
     */
    LSTMStockPredictor(int input_size, 
                      int hidden_size, 
                      int sequence_length, 
                      int output_size,
                      int seed);
    
    /**
     * @brief Train the LSTM model on stock price sequences
     * 
     * @param sequences Input sequences, each of length sequence_length_
     * @param targets Target values corresponding to each sequence
     * @param epochs Number of training epochs
     * @param learning_rate Learning rate for gradient descent
     * @param dropout_rate Dropout rate for regularization
     * @param batch_size Size of mini-batches
     * @param validation_split Fraction of data to use for validation
     * @param patience Number of epochs with no improvement before early stopping
     */
    void train(const std::vector<std::vector<double>>& sequences, 
              const std::vector<std::vector<double>>& targets,
              int epochs = 100, 
              double learning_rate = 0.01,
              double dropout_rate = 0.2,
              int batch_size = 32,
              double validation_split = 0.2,
              int patience = 10);
    
    /**
     * @brief Predict stock prices for future days
     * 
     * @param sequence Input sequence of recent stock prices
     * @param days Number of days to predict into the future
     * @return std::vector<double> Predicted prices for future days
     */
    std::vector<double> predictNextDays(const std::vector<double>& sequence, int days);
    
    /**
     * @brief Save the trained model to a file
     * 
     * @param filename Path to save the model
     * @return bool True if successful, false otherwise
     */
    bool saveModel(const std::string& filename);
    
    /**
     * @brief Load a previously trained model from a file
     * 
     * @param filename Path to the saved model
     * @return bool True if successful, false otherwise
     */
    bool loadModel(const std::string& filename);

private:
    // Convert a sequence of values to a sequence of matrices
    std::vector<core::Matrix> sequenceToMatrices(const std::vector<double>& sequence);
    
    // Convert a vector of values to a matrix
    core::Matrix valueToMatrix(const std::vector<double>& values);
    
    // Model parameters
    int input_size_;
    int hidden_size_;
    int sequence_length_;
    int output_size_;
    
    // The LSTM model
    nn::LSTM model_;
    
    // Random number generator
    std::mt19937 random_engine_;
};

} // namespace models
} // namespace finml

#endif // FINML_MODELS_LSTM_STOCK_PREDICTOR_H 