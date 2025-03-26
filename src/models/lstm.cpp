#include "finml/models/lstm.h"
#include <iostream>
#include <random>

namespace finml {
namespace models {

LSTM::LSTM(int input_size, int hidden_size, int output_size, int num_layers, double dropout_rate)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      output_size_(output_size),
      num_layers_(num_layers),
      dropout_rate_(dropout_rate),
      output_layer_(hidden_size, output_size) {
    
    // Initialize LSTM layers
    for (int i = 0; i < num_layers; ++i) {
        // First layer takes the input size, subsequent layers take hidden_size as input
        int layer_input_size = (i == 0) ? input_size : hidden_size;
        lstm_layers_.push_back(layers::LSTMLayer(layer_input_size, hidden_size));
    }
}

core::Matrix LSTM::forward(const core::Matrix& input) {
    core::Matrix x = input;
    
    // Pass through LSTM layers
    for (auto& layer : lstm_layers_) {
        x = layer.forward(x);
    }
    
    // Store last hidden state for backpropagation
    last_hidden_state_ = x;
    
    // Pass through output layer (linear transformation)
    core::Matrix output = output_layer_.forward(x);
    return output;
}

void LSTM::backward(const core::Matrix& gradient) {
    // Backward through output layer
    core::Matrix d_hidden = output_layer_.backward(gradient);
    
    // Backward through LSTM layers in reverse order
    for (int i = lstm_layers_.size() - 1; i >= 0; --i) {
        d_hidden = lstm_layers_[i].backward(d_hidden);
    }
}

void LSTM::updateWeights(double learning_rate) {
    // Update output layer weights
    output_layer_.updateWeights(learning_rate);
    
    // Update LSTM layer weights
    for (auto& layer : lstm_layers_) {
        layer.updateWeights(learning_rate);
    }
}

void LSTM::updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask) {
    // Update output layer weights with dropout
    output_layer_.updateWeightsWithDropout(learning_rate, dropout_mask);
    
    // Update LSTM layer weights with dropout
    for (auto& layer : lstm_layers_) {
        layer.updateWeightsWithDropout(learning_rate, dropout_mask);
    }
}

void LSTM::resetState() {
    // Reset state for all LSTM layers
    for (auto& layer : lstm_layers_) {
        layer.resetState();
    }
}

} // namespace models
} // namespace finml 