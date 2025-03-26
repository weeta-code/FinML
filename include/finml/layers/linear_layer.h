#ifndef FINML_LAYERS_LINEAR_LAYER_H
#define FINML_LAYERS_LINEAR_LAYER_H

#include <finml/core/matrix.h>
#include <functional>

namespace finml {
namespace layers {

class LinearLayer {
public:
    // Constructor
    LinearLayer(size_t input_size, size_t output_size);
    
    // Forward pass
    core::Matrix forward(const core::Matrix& input);
    
    // Backward pass
    core::Matrix backward(const core::Matrix& gradient);
    
    // Update weights
    void updateWeights(double learning_rate);
    
    // Update weights with dropout
    void updateWeightsWithDropout(double learning_rate, std::function<bool()> dropout_mask);
    
    // Getter for weights
    core::Matrix& getWeights() { return weights_; }
    const core::Matrix& getWeights() const { return weights_; }
    
    // Getter for bias
    core::Matrix& getBias() { return bias_; }
    const core::Matrix& getBias() const { return bias_; }
    
    // Getter for weight gradients
    core::Matrix& getWeightGradients() { return dW_; }
    const core::Matrix& getWeightGradients() const { return dW_; }
    
    // Getter for bias gradients
    core::Matrix& getBiasGradients() { return db_; }
    const core::Matrix& getBiasGradients() const { return db_; }

private:
    // Model parameters
    size_t input_size_;
    size_t output_size_;
    
    // Weights and biases
    core::Matrix weights_;
    core::Matrix bias_;
    
    // Gradients
    core::Matrix dW_;
    core::Matrix db_;
    
    // Cache for backward pass
    core::Matrix last_input_;
};

} // namespace layers
} // namespace finml

#endif // FINML_LAYERS_LINEAR_LAYER_H 