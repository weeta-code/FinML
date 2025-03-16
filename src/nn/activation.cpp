#include "finml/nn/activation.h"
#include <iostream>

namespace finml {
namespace nn {

// ReLU implementation
ReLU::ReLU(const std::string& name) : layer_name(name) {}

core::Matrix ReLU::forward(const core::Matrix& input) {
    return core::Matrix::relu(input);
}

std::vector<core::ValuePtr> ReLU::parameters() const {
    return {}; // ReLU has no parameters
}

void ReLU::zeroGrad() {
    // No parameters to zero out
}

std::string ReLU::name() const {
    return layer_name;
}

void ReLU::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: ReLU Activation" << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

// LeakyReLU implementation
LeakyReLU::LeakyReLU(float alpha, const std::string& name) : alpha(alpha), layer_name(name) {}

core::Matrix LeakyReLU::forward(const core::Matrix& input) {
    core::Matrix result(input.numRows(), input.numCols());
    
    for (size_t i = 0; i < input.numRows(); ++i) {
        for (size_t j = 0; j < input.numCols(); ++j) {
            const auto& val = input.at(i, j);
            if (val->data > 0) {
                result.at(i, j) = val;
            } else {
                result.at(i, j) = core::Value::multiply(val, core::Value::create(alpha));
            }
        }
    }
    
    return result;
}

std::vector<core::ValuePtr> LeakyReLU::parameters() const {
    return {}; // LeakyReLU has no parameters
}

void LeakyReLU::zeroGrad() {
    // No parameters to zero out
}

std::string LeakyReLU::name() const {
    return layer_name;
}

void LeakyReLU::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: LeakyReLU Activation" << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

// Sigmoid implementation
Sigmoid::Sigmoid(const std::string& name) : layer_name(name) {}

core::Matrix Sigmoid::forward(const core::Matrix& input) {
    return core::Matrix::sigmoid(input);
}

std::vector<core::ValuePtr> Sigmoid::parameters() const {
    return {}; // Sigmoid has no parameters
}

void Sigmoid::zeroGrad() {
    // No parameters to zero out
}

std::string Sigmoid::name() const {
    return layer_name;
}

void Sigmoid::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: Sigmoid Activation" << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

// Tanh implementation
Tanh::Tanh(const std::string& name) : layer_name(name) {}

core::Matrix Tanh::forward(const core::Matrix& input) {
    return core::Matrix::tanh(input);
}

std::vector<core::ValuePtr> Tanh::parameters() const {
    return {}; // Tanh has no parameters
}

void Tanh::zeroGrad() {
    // No parameters to zero out
}

std::string Tanh::name() const {
    return layer_name;
}

void Tanh::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: Tanh Activation" << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

// Softmax implementation
Softmax::Softmax(const std::string& name) : layer_name(name) {}

core::Matrix Softmax::forward(const core::Matrix& input) {
    return core::Matrix::softmax(input);
}

std::vector<core::ValuePtr> Softmax::parameters() const {
    return {}; // Softmax has no parameters
}

void Softmax::zeroGrad() {
    // No parameters to zero out
}

std::string Softmax::name() const {
    return layer_name;
}

void Softmax::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: Softmax Activation" << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

} // namespace nn
} // namespace finml 