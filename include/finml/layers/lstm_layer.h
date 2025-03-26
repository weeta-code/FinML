#ifndef FINML_LAYERS_LSTM_LAYER_H
#define FINML_LAYERS_LSTM_LAYER_H

#include "finml/core/matrix.h"
#include <functional>

namespace finml {
namespace layers {

class LSTMLayer {
public:
    // Constructor
    LSTMLayer(int input_size, int hidden_size);
    
    // Forward pass
    core::Matrix forward(const core::Matrix& input);
    
    // Backward pass
    core::Matrix backward(const core::Matrix& gradient);
    
    // Update weights
    void updateWeights(double learning_rate);
    
    // Update weights with dropout
    void updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask);
    
    // Reset the hidden state and cell state to zeros
    void resetState();
    
    // Getters for states
    core::Matrix getHiddenState() const;
    core::Matrix getCellState() const;
    core::Matrix getInputGradient() const;

private:
    // Model parameters
    int input_size_;
    int hidden_size_;
    
    // Weights and biases
    core::Matrix W_forget_;  // Forget gate weights
    core::Matrix b_forget_;  // Forget gate bias
    
    core::Matrix W_input_;   // Input gate weights
    core::Matrix b_input_;   // Input gate bias
    
    core::Matrix W_cell_;    // Cell state weights
    core::Matrix b_cell_;    // Cell state bias
    
    core::Matrix W_output_;  // Output gate weights
    core::Matrix b_output_;  // Output gate bias
    
    // Gradients
    core::Matrix dW_forget_;
    core::Matrix db_forget_;
    
    core::Matrix dW_input_;
    core::Matrix db_input_;
    
    core::Matrix dW_cell_;
    core::Matrix db_cell_;
    
    core::Matrix dW_output_;
    core::Matrix db_output_;
    
    // LSTM state
    core::Matrix hidden_state_;
    core::Matrix cell_state_;
    
    // Gradient for input
    core::Matrix d_input_;
    core::Matrix d_hidden_prev_;
    
    // Cache for backward pass
    core::Matrix last_input_;
    core::Matrix last_hidden_state_;
    core::Matrix last_cell_state_;
    
    core::Matrix last_forget_gate_;
    core::Matrix last_input_gate_;
    core::Matrix last_cell_candidate_;
    core::Matrix last_output_gate_;
};

} // namespace layers
} // namespace finml

#endif // FINML_LAYERS_LSTM_LAYER_H 