#include "finml/layers/linear_layer.h"
#include <random>
#include <cmath>

namespace finml {
namespace layers {

// Constructor implementation
LinearLayer::LinearLayer(size_t input_size, size_t output_size) : 
    input_size_(input_size), 
    output_size_(output_size) {
    
    // Xavier/Glorot initialization
    float std_dev = std::sqrt(2.0f / (input_size + output_size));
    
    weights_ = core::Matrix::random(output_size, input_size, 0.0f, std_dev);
    bias_ = core::Matrix::zeros(output_size, 1);
    
    // Initialize gradient matrices
    dW_ = core::Matrix::zeros(output_size, input_size);
    db_ = core::Matrix::zeros(output_size, 1);
}

// Forward pass implementation
core::Matrix LinearLayer::forward(const core::Matrix& input) {
    // Store input for backpropagation
    last_input_ = input;
    
    // Linear transformation: Y = W*X + b
    return core::Matrix::matmul(weights_, input) + bias_;
}

// Backward pass implementation
core::Matrix LinearLayer::backward(const core::Matrix& gradient) {
    // dL/dW = dL/dY * X^T
    core::Matrix input_t = core::Matrix::transpose(last_input_);
    dW_ = core::Matrix::matmul(gradient, input_t);
    
    // dL/db = dL/dY
    db_ = gradient;
    
    // dL/dX = W^T * dL/dY
    core::Matrix weights_t = core::Matrix::transpose(weights_);
    return core::Matrix::matmul(weights_t, gradient);
}

// Update weights implementation
void LinearLayer::updateWeights(double learning_rate) {
    // W = W - learning_rate * dW
    weights_ = weights_ - core::Matrix::scalarMultiply(dW_, learning_rate);
    
    // b = b - learning_rate * db
    bias_ = bias_ - core::Matrix::scalarMultiply(db_, learning_rate);
    
    // Reset gradients
    dW_ = core::Matrix::zeros(output_size_, input_size_);
    db_ = core::Matrix::zeros(output_size_, 1);
}

// Update weights with dropout implementation
void LinearLayer::updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask) {
    // Update weights with dropout
    for (size_t i = 0; i < weights_.numRows(); ++i) {
        for (size_t j = 0; j < weights_.numCols(); ++j) {
            if (dropout_mask()) {
                double weight = weights_.at(i, j)->data;
                double gradient = dW_.at(i, j)->data;
                weights_.at(i, j) = core::Value::create(weight - learning_rate * gradient);
            }
        }
    }
    
    // Update biases (biases are not usually dropped out)
    bias_ = bias_ - core::Matrix::scalarMultiply(db_, learning_rate);
    
    // Reset gradients
    dW_ = core::Matrix::zeros(output_size_, input_size_);
    db_ = core::Matrix::zeros(output_size_, 1);
}

} // namespace layers
} // namespace finml 