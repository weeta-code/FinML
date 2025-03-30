#include "finml/nn/activation.h"
#include "finml/core/matrix.h"
#include <iostream>

namespace finml {
namespace nn {

// Base class for activation functions
class ActivationBase {
protected:
    std::string layer_name;
    
public:
    ActivationBase(const std::string& name) : layer_name(name) {}
    
    virtual void print() const {
        std::cout << "Layer: " << layer_name << std::endl;
        std::cout << "Type: " << getType() << std::endl;
        std::cout << "Parameters: " << getNumParameters() << std::endl;
        std::cout << std::endl;
    }
    
    virtual std::string getType() const = 0;
    virtual size_t getNumParameters() const = 0;
    std::string name() const { return layer_name; }
};

// ReLU implementation
class ReLU : public ActivationBase {
public:
    ReLU(const std::string& name) : ActivationBase(name) {}
    
    core::Matrix forward(const core::Matrix& input) {
        return core::Matrix::relu(input);
    }
    
    std::vector<core::ValuePtr> parameters() const {
        return {}; // ReLU has no parameters
    }
    
    void zeroGrad() {
        // No parameters to zero out
    }
    
    std::string getType() const override { return "ReLU Activation"; }
    size_t getNumParameters() const override { return 0; }
};

// LeakyReLU implementation
class LeakyReLU : public ActivationBase {
private:
    float alpha;
    
public:
    LeakyReLU(float alpha, const std::string& name) : ActivationBase(name), alpha(alpha) {}
    
    core::Matrix forward(const core::Matrix& input) {
        core::Matrix mask = core::Matrix::zeros(input.numRows(), input.numCols());
        
        // Create mask where positive values are 1 and negative values are alpha
        for (size_t i = 0; i < input.numRows(); ++i) {
            for (size_t j = 0; j < input.numCols(); ++j) {
                mask.at(i, j) = core::Value::create(input.at(i, j)->data > 0 ? 1.0f : alpha);
            }
        }
        
        return core::Matrix::elementWiseMultiply(input, mask);
    }
    
    std::vector<core::ValuePtr> parameters() const {
        return {}; // LeakyReLU has no parameters
    }
    
    void zeroGrad() {
        // No parameters to zero out
    }
    
    std::string getType() const override { return "LeakyReLU Activation"; }
    size_t getNumParameters() const override { return 0; }
    
    void print() const override {
        ActivationBase::print();
        std::cout << "Alpha: " << alpha << std::endl;
    }
};

// Sigmoid implementation
class Sigmoid : public ActivationBase {
public:
    Sigmoid(const std::string& name) : ActivationBase(name) {}
    
    core::Matrix forward(const core::Matrix& input) {
        return core::Matrix::sigmoid(input);
    }
    
    std::vector<core::ValuePtr> parameters() const {
        return {}; // Sigmoid has no parameters
    }
    
    void zeroGrad() {
        // No parameters to zero out
    }
    
    std::string getType() const override { return "Sigmoid Activation"; }
    size_t getNumParameters() const override { return 0; }
};

// Tanh implementation
class Tanh : public ActivationBase {
public:
    Tanh(const std::string& name) : ActivationBase(name) {}
    
    core::Matrix forward(const core::Matrix& input) {
        return core::Matrix::tanh(input);
    }
    
    std::vector<core::ValuePtr> parameters() const {
        return {}; // Tanh has no parameters
    }
    
    void zeroGrad() {
        // No parameters to zero out
    }
    
    std::string getType() const override { return "Tanh Activation"; }
    size_t getNumParameters() const override { return 0; }
};

// Softmax implementation
class Softmax : public ActivationBase {
public:
    Softmax(const std::string& name) : ActivationBase(name) {}
    
    core::Matrix forward(const core::Matrix& input) {
        return core::Matrix::softmax(input);
    }
    
    std::vector<core::ValuePtr> parameters() const {
        return {}; // Softmax has no parameters
    }
    
    void zeroGrad() {
        // No parameters to zero out
    }
    
    std::string getType() const override { return "Softmax Activation"; }
    size_t getNumParameters() const override { return 0; }
};

} // namespace nn
} // namespace finml 