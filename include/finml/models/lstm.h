#ifndef FINML_MODELS_LSTM_H
#define FINML_MODELS_LSTM_H

#include "finml/nn/lstm.h"
#include "finml/layers/lstm_layer.h"
#include "finml/layers/linear_layer.h"
#include "finml/core/matrix.h"
#include <vector>
#include <functional>

namespace finml {
namespace models {

// LSTM model that can be used for regression/classification tasks
class LSTM {
private:
    int input_size_;
    int hidden_size_;
    int output_size_;
    int num_layers_;
    double dropout_rate_;
    
    std::vector<layers::LSTMLayer> lstm_layers_;
    layers::LinearLayer output_layer_;
    core::Matrix last_hidden_state_;
    
public:
    // Constructor
    LSTM(int input_size, int hidden_size, int output_size, int num_layers = 1, double dropout_rate = 0.0);
    
    // Forward pass
    core::Matrix forward(const core::Matrix& input);
    
    // Backward pass
    void backward(const core::Matrix& gradient);
    
    // Update weights
    void updateWeights(double learning_rate);
    
    // Update weights with dropout
    void updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask);
    
    // Reset internal state
    void resetState();
};

} // namespace models
} // namespace finml

#endif // FINML_MODELS_LSTM_H 