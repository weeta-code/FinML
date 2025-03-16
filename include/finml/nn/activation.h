#ifndef FINML_NN_ACTIVATION_H
#define FINML_NN_ACTIVATION_H

#include "finml/nn/layer.h"
#include "finml/core/matrix.h"
#include <string>

namespace finml {
namespace nn {

// ReLU activation layer
class ReLU : public Layer {
private:
    std::string layer_name;

public:
    explicit ReLU(const std::string& name = "ReLU");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
};

// leaky ReLU activation layer
class LeakyReLU : public Layer {
private:
    float alpha;
    std::string layer_name;

public:
    explicit LeakyReLU(float alpha = 0.01f, const std::string& name = "LeakyReLU");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
};

// sigmoid activation layer
class Sigmoid : public Layer {
private:
    std::string layer_name;

public:
    explicit Sigmoid(const std::string& name = "Sigmoid");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
};

// tanh activation layer
class Tanh : public Layer {
private:
    std::string layer_name;

public:
    explicit Tanh(const std::string& name = "Tanh");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
};

// softmax activation layer
class Softmax : public Layer {
private:
    std::string layer_name;

public:
    explicit Softmax(const std::string& name = "Softmax");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
};

} // namespace nn
} // namespace finml

#endif // FINML_NN_ACTIVATION_H 