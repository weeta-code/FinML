#include "finml/nn/linear.h"
#include <iostream>

namespace finml {
namespace nn {

Linear::Linear(size_t in_features, size_t out_features, bool use_bias, const std::string& name)
    : in_features(in_features), out_features(out_features), use_bias(use_bias), layer_name(name) {
    
    // Initialize weights with Xavier/Glorot initialization
    float stddev = std::sqrt(2.0f / (in_features + out_features));
    weights = core::Matrix::random(out_features, in_features, 0.0f, stddev);
    
    if (use_bias) {
        bias = core::Matrix(out_features, 1);
    }
}

core::Matrix Linear::forward(const core::Matrix& input) {
    if (input.numCols() != 1 && input.numRows() != in_features) {
        throw std::invalid_argument("Input dimensions do not match layer input features");
    }
    
    core::Matrix output = core::Matrix::matmul(weights, input);
    
    if (use_bias) {
        output = core::Matrix::elementWiseAdd(output, bias);
    }
    
    return output;
}

std::vector<core::ValuePtr> Linear::parameters() const {
    std::vector<core::ValuePtr> params = weights.flatten();
    
    if (use_bias) {
        std::vector<core::ValuePtr> bias_params = bias.flatten();
        params.insert(params.end(), bias_params.begin(), bias_params.end());
    }
    
    return params;
}

void Linear::zeroGrad() {
    weights.zeroGrad();
    
    if (use_bias) {
        bias.zeroGrad();
    }
}

std::string Linear::name() const {
    return layer_name;
}

void Linear::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: Linear" << std::endl;
    std::cout << "Input features: " << in_features << std::endl;
    std::cout << "Output features: " << out_features << std::endl;
    std::cout << "Use bias: " << (use_bias ? "true" : "false") << std::endl;
    std::cout << "Parameters: " << parameters().size() << std::endl;
    std::cout << std::endl;
}

const core::Matrix& Linear::getWeights() const {
    return weights;
}

const core::Matrix& Linear::getBias() const {
    if (!use_bias) {
        throw std::runtime_error("Layer does not use bias");
    }
    return bias;
}

} // namespace nn
} // namespace finml 