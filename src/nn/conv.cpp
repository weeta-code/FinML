#include "finml/nn/conv.h"
#include "finml/core/matrix.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace finml {
namespace nn {

// Conv1D implementation
Conv1D::Conv1D(
    size_t in_channels, 
    size_t out_channels, 
    size_t kernel_size, 
    size_t stride, 
    size_t padding, 
    bool use_bias, 
    const std::string& name
) : in_channels(in_channels), 
    out_channels(out_channels), 
    kernel_size(kernel_size), 
    stride(stride), 
    padding(padding), 
    use_bias(use_bias), 
    bias(use_bias ? core::Matrix(out_channels, 1) : core::Matrix(0, 0)),
    layer_name(name) {
    
    // Initialize kernels with Xavier/Glorot initialization
    float stddev = std::sqrt(2.0f / (in_channels * kernel_size + out_channels));
    kernels.reserve(out_channels);
    for (size_t i = 0; i < out_channels; ++i) {
        core::Matrix kernel = core::Matrix::random(in_channels, kernel_size, 0.0f, stddev);
        kernels.push_back(kernel);
    }
}

core::Matrix Conv1D::pad(const core::Matrix& input) const {
    if (padding == 0) {
        return input;
    }
    
    size_t padded_cols = input.numCols() + 2 * padding;
    core::Matrix padded(input.numRows(), padded_cols);
    
    // Copy input to padded matrix
    for (size_t i = 0; i < input.numRows(); ++i) {
        for (size_t j = 0; j < input.numCols(); ++j) {
            padded.at(i, j + padding) = input.at(i, j);
        }
        
        // Initialize padding with zeros
        for (size_t j = 0; j < padding; ++j) {
            padded.at(i, j) = core::Value::create(0.0f);
            padded.at(i, j + input.numCols() + padding) = core::Value::create(0.0f);
        }
    }
    
    return padded;
}

core::Matrix Conv1D::forward(const core::Matrix& input) {
    if (input.numRows() != in_channels) {
        throw std::invalid_argument("Input channels do not match layer input channels");
    }
    
    // Pad input
    core::Matrix padded_input = pad(input);
    
    // Calculate output dimensions
    size_t output_length = outputSize(input.numCols());
    core::Matrix output(out_channels, output_length);
    
    // Perform convolution
    for (size_t out_ch = 0; out_ch < out_channels; ++out_ch) {
        const core::Matrix& kernel = kernels[out_ch];
        
        for (size_t i = 0; i < output_length; ++i) {
            size_t start_idx = i * stride;
            
            // Initialize output value
            core::ValuePtr out_val = core::Value::create(0.0f);
            
            // Convolve kernel with input
            for (size_t in_ch = 0; in_ch < in_channels; ++in_ch) {
                for (size_t k = 0; k < kernel_size; ++k) {
                    out_val = core::Value::add(
                        out_val,
                        core::Value::multiply(
                            padded_input.at(in_ch, start_idx + k),
                            kernel.at(in_ch, k)
                        )
                    );
                }
            }
            
            // Add bias if needed
            if (use_bias) {
                out_val = core::Value::add(out_val, bias.at(out_ch, 0));
            }
            
            output.at(out_ch, i) = out_val;
        }
    }
    
    return output;
}

std::vector<core::ValuePtr> Conv1D::parameters() const {
    std::vector<core::ValuePtr> params;
    
    // Kernel parameters
    for (const auto& kernel : kernels) {
        std::vector<core::ValuePtr> kernel_params = kernel.flatten();
        params.insert(params.end(), kernel_params.begin(), kernel_params.end());
    }
    
    // Bias parameters
    if (use_bias) {
        std::vector<core::ValuePtr> bias_params = bias.flatten();
        params.insert(params.end(), bias_params.begin(), bias_params.end());
    }
    
    return params;
}

void Conv1D::zeroGrad() {
    for (auto& kernel : kernels) {
        kernel.zeroGrad();
    }
    
    if (use_bias) {
        bias.zeroGrad();
    }
}

std::string Conv1D::name() const {
    return layer_name;
}

void Conv1D::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: Conv1D" << std::endl;
    std::cout << "Input channels: " << in_channels << std::endl;
    std::cout << "Output channels: " << out_channels << std::endl;
    std::cout << "Kernel size: " << kernel_size << std::endl;
    std::cout << "Stride: " << stride << std::endl;
    std::cout << "Padding: " << padding << std::endl;
    std::cout << "Use bias: " << (use_bias ? "true" : "false") << std::endl;
    std::cout << "Parameters: " << parameters().size() << std::endl;
    std::cout << std::endl;
}

size_t Conv1D::outputSize(size_t input_size) const {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// MaxPool1D implementation
MaxPool1D::MaxPool1D(size_t kernel_size, size_t stride, const std::string& name)
    : kernel_size(kernel_size), stride(stride), layer_name(name) {}

core::Matrix MaxPool1D::forward(const core::Matrix& input) {
    size_t output_length = outputSize(input.numCols());
    core::Matrix output(input.numRows(), output_length);
    
    for (size_t i = 0; i < input.numRows(); ++i) {
        for (size_t j = 0; j < output_length; ++j) {
            size_t start_idx = j * stride;
            
            // Find maximum value in the kernel window
            core::ValuePtr max_val = input.at(i, start_idx);
            for (size_t k = 1; k < kernel_size; ++k) {
                if (start_idx + k < input.numCols()) {
                    if (input.at(i, start_idx + k)->data > max_val->data) {
                        max_val = input.at(i, start_idx + k);
                    }
                }
            }
            
            output.at(i, j) = max_val;
        }
    }
    
    return output;
}

std::vector<core::ValuePtr> MaxPool1D::parameters() const {
    return {}; // MaxPool1D has no parameters
}

void MaxPool1D::zeroGrad() {
    // No parameters to zero out
}

std::string MaxPool1D::name() const {
    return layer_name;
}

void MaxPool1D::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: MaxPool1D" << std::endl;
    std::cout << "Kernel size: " << kernel_size << std::endl;
    std::cout << "Stride: " << stride << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

size_t MaxPool1D::outputSize(size_t input_size) const {
    return (input_size - kernel_size) / stride + 1;
}

// AvgPool1D implementation
AvgPool1D::AvgPool1D(size_t kernel_size, size_t stride, const std::string& name)
    : kernel_size(kernel_size), stride(stride), layer_name(name) {}

core::Matrix AvgPool1D::forward(const core::Matrix& input) {
    size_t output_length = outputSize(input.numCols());
    core::Matrix output(input.numRows(), output_length);
    
    for (size_t i = 0; i < input.numRows(); ++i) {
        for (size_t j = 0; j < output_length; ++j) {
            size_t start_idx = j * stride;
            
            // Calculate average value in the kernel window
            core::ValuePtr sum = core::Value::create(0.0f);
            size_t count = 0;
            
            for (size_t k = 0; k < kernel_size; ++k) {
                if (start_idx + k < input.numCols()) {
                    sum = core::Value::add(sum, input.at(i, start_idx + k));
                    ++count;
                }
            }
            
            // Divide by count to get average
            core::ValuePtr avg = core::Value::divide(sum, core::Value::create(static_cast<float>(count)));
            output.at(i, j) = avg;
        }
    }
    
    return output;
}

std::vector<core::ValuePtr> AvgPool1D::parameters() const {
    return {}; // AvgPool1D has no parameters
}

void AvgPool1D::zeroGrad() {
    // No parameters to zero out
}

std::string AvgPool1D::name() const {
    return layer_name;
}

void AvgPool1D::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: AvgPool1D" << std::endl;
    std::cout << "Kernel size: " << kernel_size << std::endl;
    std::cout << "Stride: " << stride << std::endl;
    std::cout << "Parameters: 0" << std::endl;
    std::cout << std::endl;
}

size_t AvgPool1D::outputSize(size_t input_size) const {
    return (input_size - kernel_size) / stride + 1;
}

} // namespace nn
} // namespace finml 