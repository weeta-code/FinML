#ifndef FINML_NN_CONV_H
#define FINML_NN_CONV_H

#include "finml/nn/layer.h"
#include "finml/core/matrix.h"
#include <string>
#include <vector>
#include <memory>

namespace finml {
namespace nn {

// 1D convolutional layer
class Conv1D : public Layer {
private:
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    bool use_bias;
    std::string layer_name;
    
    // Weights and bias
    std::vector<core::Matrix> kernels; // out_channels x in_channels matrices of size kernel_size x 1
    core::Matrix bias; // out_channels x 1
    
    // Helper function to pad input
    core::Matrix pad(const core::Matrix& input) const;

public:

    Conv1D(
        size_t in_channels, 
        size_t out_channels, 
        size_t kernel_size, 
        size_t stride = 1, 
        size_t padding = 0, 
        bool use_bias = true, 
        const std::string& name = "Conv1D"
    );
    // forward pass
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    
   // Zero out grads
    void zeroGrad() override;
    // get name
    std::string name() const override;
    void print() const override;

    size_t outputSize(size_t input_size) const;
};


class MaxPool1D : public Layer {
private:
    size_t kernel_size;
    size_t stride;
    std::string layer_name;

public:
    
    MaxPool1D(size_t kernel_size, size_t stride = 1, const std::string& name = "MaxPool1D");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;

    size_t outputSize(size_t input_size) const;
};


class AvgPool1D : public Layer {
private:
    size_t kernel_size;
    size_t stride;
    std::string layer_name;

public:
    
    AvgPool1D(size_t kernel_size, size_t stride = 1, const std::string& name = "AvgPool1D");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;

    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
    
    size_t outputSize(size_t input_size) const;
};

} // namespace nn
} // namespace finml

#endif // FINML_NN_CONV_H 