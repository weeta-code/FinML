#include "finml/layers/lstm_layer.h"
#include <cmath>

using namespace finml;

// Helper function to concatenate two matrices vertically
// This is a helper since verticalConcat is missing from the Matrix class
core::Matrix verticalConcat(const core::Matrix& top, const core::Matrix& bottom) {
    size_t total_rows = top.numRows() + bottom.numRows();
    size_t cols = top.numCols();
    
    if (bottom.numCols() != cols) {
        throw std::invalid_argument("Matrices must have the same number of columns for vertical concatenation");
    }
    
    core::Matrix result(total_rows, cols);
    
    // Copy top matrix
    for (size_t i = 0; i < top.numRows(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = top.at(i, j);
        }
    }
    
    // Copy bottom matrix
    for (size_t i = 0; i < bottom.numRows(); ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i + top.numRows(), j) = bottom.at(i, j);
        }
    }
    
    return result;
}

namespace finml {
namespace layers {

LSTMLayer::LSTMLayer(int input_size, int hidden_size) 
    : input_size_(input_size), hidden_size_(hidden_size),
      W_forget_(core::Matrix::random(hidden_size, input_size + hidden_size, 0.0f, std::sqrt(2.0f / (input_size + hidden_size)))),
      W_input_(core::Matrix::random(hidden_size, input_size + hidden_size, 0.0f, std::sqrt(2.0f / (input_size + hidden_size)))),
      W_cell_(core::Matrix::random(hidden_size, input_size + hidden_size, 0.0f, std::sqrt(2.0f / (input_size + hidden_size)))),
      W_output_(core::Matrix::random(hidden_size, input_size + hidden_size, 0.0f, std::sqrt(2.0f / (input_size + hidden_size)))),
      b_forget_(core::Matrix::ones(hidden_size, 1)),  // Forget gate bias initialized to 1 (common practice)
      b_input_(core::Matrix::zeros(hidden_size, 1)),
      b_cell_(core::Matrix::zeros(hidden_size, 1)),
      b_output_(core::Matrix::zeros(hidden_size, 1)),
      hidden_state_(core::Matrix::zeros(hidden_size, 1)),
      cell_state_(core::Matrix::zeros(hidden_size, 1)) {
    
    // Initialize gradients
    dW_forget_ = core::Matrix::zeros(hidden_size, input_size + hidden_size);
    dW_input_ = core::Matrix::zeros(hidden_size, input_size + hidden_size);
    dW_cell_ = core::Matrix::zeros(hidden_size, input_size + hidden_size);
    dW_output_ = core::Matrix::zeros(hidden_size, input_size + hidden_size);
    db_forget_ = core::Matrix::zeros(hidden_size, 1);
    db_input_ = core::Matrix::zeros(hidden_size, 1);
    db_cell_ = core::Matrix::zeros(hidden_size, 1);
    db_output_ = core::Matrix::zeros(hidden_size, 1);
}

core::Matrix LSTMLayer::forward(const core::Matrix& input) {
    // Cache the input for backward pass
    last_input_ = input;
    last_hidden_state_ = hidden_state_;
    last_cell_state_ = cell_state_;
    
    // Create combined input [x_t, h_{t-1}]
    core::Matrix combined = verticalConcat(input, hidden_state_);
    
    // Forget gate
    core::Matrix forget_gate = core::Matrix::sigmoid(core::Matrix::matmul(W_forget_, combined) + b_forget_);
    
    // Input gate
    core::Matrix input_gate = core::Matrix::sigmoid(core::Matrix::matmul(W_input_, combined) + b_input_);
    
    // Cell candidate
    core::Matrix cell_candidate = core::Matrix::tanh(core::Matrix::matmul(W_cell_, combined) + b_cell_);
    
    // Forget gate * previous cell state
    core::Matrix forget_result = forget_gate % last_cell_state_;
    
    // Input gate * cell candidate
    core::Matrix input_result = input_gate % cell_candidate;
    
    // Update cell state
    cell_state_ = forget_result + input_result;
    
    // Output gate
    core::Matrix output_gate = core::Matrix::sigmoid(core::Matrix::matmul(W_output_, combined) + b_output_);
    
    // Update hidden state
    hidden_state_ = output_gate % core::Matrix::tanh(cell_state_);
    
    // Cache gates for backward pass
    last_forget_gate_ = forget_gate;
    last_input_gate_ = input_gate;
    last_cell_candidate_ = cell_candidate;
    last_output_gate_ = output_gate;
    
    return hidden_state_;
}

core::Matrix LSTMLayer::backward(const core::Matrix& d_hidden) {
    // Gradient from output gate
    core::Matrix d_output_gate = d_hidden % core::Matrix::tanh(last_cell_state_);
    
    // Gradient for tanh(cell_state)
    core::Matrix d_tanh_cell = d_hidden % last_output_gate_;
    
    // Gradient for cell state
    core::Matrix tanh_cell_state = core::Matrix::tanh(last_cell_state_);
    core::Matrix tanh_deriv = core::Matrix::ones(tanh_cell_state.numRows(), tanh_cell_state.numCols());
    tanh_deriv = tanh_deriv - (tanh_cell_state % tanh_cell_state); // 1 - tanh^2(x)
    core::Matrix d_cell = d_tanh_cell % tanh_deriv;
    
    // Gradient for forget gate
    core::Matrix d_forget_gate = d_cell % last_cell_state_;
    
    // Gradient for input gate
    core::Matrix d_input_gate = d_cell % last_cell_candidate_;
    
    // Gradient for cell candidate
    core::Matrix d_cell_candidate = d_cell % last_input_gate_;
    
    // Compute combined input [x_t, h_{t-1}]
    core::Matrix combined = verticalConcat(last_input_, last_hidden_state_);
    
    // Update gradients for weights and biases
    core::Matrix combined_t = core::Matrix::transpose(combined);
    
    dW_forget_ = dW_forget_ + core::Matrix::matmul(d_forget_gate, combined_t);
    db_forget_ = db_forget_ + d_forget_gate;
    
    dW_input_ = dW_input_ + core::Matrix::matmul(d_input_gate, combined_t);
    db_input_ = db_input_ + d_input_gate;
    
    dW_cell_ = dW_cell_ + core::Matrix::matmul(d_cell_candidate, combined_t);
    db_cell_ = db_cell_ + d_cell_candidate;
    
    dW_output_ = dW_output_ + core::Matrix::matmul(d_output_gate, combined_t);
    db_output_ = db_output_ + d_output_gate;
    
    // Compute gradients for input and previous hidden state
    core::Matrix W_forget_t = core::Matrix::transpose(W_forget_);
    core::Matrix W_input_t = core::Matrix::transpose(W_input_);
    core::Matrix W_cell_t = core::Matrix::transpose(W_cell_);
    core::Matrix W_output_t = core::Matrix::transpose(W_output_);
    
    core::Matrix d_combined = 
        core::Matrix::matmul(W_forget_t, d_forget_gate) +
        core::Matrix::matmul(W_input_t, d_input_gate) +
        core::Matrix::matmul(W_cell_t, d_cell_candidate) +
        core::Matrix::matmul(W_output_t, d_output_gate);
    
    // Extract gradients for input and previous hidden state
    d_input_ = core::Matrix(last_input_.numRows(), last_input_.numCols());
    d_hidden_prev_ = core::Matrix(hidden_state_.numRows(), hidden_state_.numCols());
    
    // Copy the gradients from combined gradient
    for (size_t i = 0; i < last_input_.numRows(); ++i) {
        for (size_t j = 0; j < last_input_.numCols(); ++j) {
            d_input_.at(i, j) = d_combined.at(i, j);
        }
    }
    
    for (size_t i = 0; i < hidden_state_.numRows(); ++i) {
        for (size_t j = 0; j < hidden_state_.numCols(); ++j) {
            d_hidden_prev_.at(i, j) = d_combined.at(i + last_input_.numRows(), j);
        }
    }
    
    return d_input_;
}

void LSTMLayer::updateWeights(double learning_rate) {
    // Update weights
    W_forget_ = W_forget_ - core::Matrix::scalarMultiply(dW_forget_, learning_rate);
    W_input_ = W_input_ - core::Matrix::scalarMultiply(dW_input_, learning_rate);
    W_cell_ = W_cell_ - core::Matrix::scalarMultiply(dW_cell_, learning_rate);
    W_output_ = W_output_ - core::Matrix::scalarMultiply(dW_output_, learning_rate);
    
    // Update biases
    b_forget_ = b_forget_ - core::Matrix::scalarMultiply(db_forget_, learning_rate);
    b_input_ = b_input_ - core::Matrix::scalarMultiply(db_input_, learning_rate);
    b_cell_ = b_cell_ - core::Matrix::scalarMultiply(db_cell_, learning_rate);
    b_output_ = b_output_ - core::Matrix::scalarMultiply(db_output_, learning_rate);
    
    // Reset gradients to zero
    dW_forget_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_input_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_cell_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_output_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    db_forget_ = core::Matrix::zeros(hidden_size_, 1);
    db_input_ = core::Matrix::zeros(hidden_size_, 1);
    db_cell_ = core::Matrix::zeros(hidden_size_, 1);
    db_output_ = core::Matrix::zeros(hidden_size_, 1);
}

void LSTMLayer::updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask) {
    // Update weights with dropout
    for (size_t i = 0; i < W_forget_.numRows(); ++i) {
        for (size_t j = 0; j < W_forget_.numCols(); ++j) {
            if (dropout_mask()) {
                double value = W_forget_.at(i, j)->data;
                double gradient = dW_forget_.at(i, j)->data;
                W_forget_.at(i, j) = core::Value::create(value - learning_rate * gradient);
            }
        }
    }
    
    for (size_t i = 0; i < W_input_.numRows(); ++i) {
        for (size_t j = 0; j < W_input_.numCols(); ++j) {
            if (dropout_mask()) {
                double value = W_input_.at(i, j)->data;
                double gradient = dW_input_.at(i, j)->data;
                W_input_.at(i, j) = core::Value::create(value - learning_rate * gradient);
            }
        }
    }
    
    for (size_t i = 0; i < W_cell_.numRows(); ++i) {
        for (size_t j = 0; j < W_cell_.numCols(); ++j) {
            if (dropout_mask()) {
                double value = W_cell_.at(i, j)->data;
                double gradient = dW_cell_.at(i, j)->data;
                W_cell_.at(i, j) = core::Value::create(value - learning_rate * gradient);
            }
        }
    }
    
    for (size_t i = 0; i < W_output_.numRows(); ++i) {
        for (size_t j = 0; j < W_output_.numCols(); ++j) {
            if (dropout_mask()) {
                double value = W_output_.at(i, j)->data;
                double gradient = dW_output_.at(i, j)->data;
                W_output_.at(i, j) = core::Value::create(value - learning_rate * gradient);
            }
        }
    }
    
    // Update biases (biases are not usually dropped out)
    b_forget_ = b_forget_ - core::Matrix::scalarMultiply(db_forget_, learning_rate);
    b_input_ = b_input_ - core::Matrix::scalarMultiply(db_input_, learning_rate);
    b_cell_ = b_cell_ - core::Matrix::scalarMultiply(db_cell_, learning_rate);
    b_output_ = b_output_ - core::Matrix::scalarMultiply(db_output_, learning_rate);
    
    // Reset gradients to zero
    dW_forget_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_input_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_cell_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    dW_output_ = core::Matrix::zeros(hidden_size_, input_size_ + hidden_size_);
    db_forget_ = core::Matrix::zeros(hidden_size_, 1);
    db_input_ = core::Matrix::zeros(hidden_size_, 1);
    db_cell_ = core::Matrix::zeros(hidden_size_, 1);
    db_output_ = core::Matrix::zeros(hidden_size_, 1);
}

void LSTMLayer::resetState() {
    hidden_state_ = core::Matrix::zeros(hidden_size_, 1);
    cell_state_ = core::Matrix::zeros(hidden_size_, 1);
}

core::Matrix LSTMLayer::getHiddenState() const {
    return hidden_state_;
}

core::Matrix LSTMLayer::getCellState() const {
    return cell_state_;
}

core::Matrix LSTMLayer::getInputGradient() const {
    return d_input_;
}

} // namespace layers
} // namespace finml 