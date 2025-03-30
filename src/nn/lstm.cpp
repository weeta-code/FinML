#include "finml/nn/lstm.h"
#include <iostream>
#include <cmath>

namespace finml {
namespace nn {

LSTM::LSTM(size_t input_size, size_t hidden_size, bool use_bias, const std::string& name)
    : input_size(input_size), hidden_size(hidden_size), use_bias(use_bias), layer_name(name) {
    
    // Initialize weights with Xavier/Glorot initialization
    float stddev = std::sqrt(2.0f / (input_size + hidden_size));
    
    // Input gate
    W_i = core::Matrix::random(hidden_size, input_size, 0.0f, stddev);
    U_i = core::Matrix::random(hidden_size, hidden_size, 0.0f, stddev);
    if (use_bias) {
        b_i = core::Matrix(hidden_size, 1);
    }
    
    // Forget gate
    W_f = core::Matrix::random(hidden_size, input_size, 0.0f, stddev);
    U_f = core::Matrix::random(hidden_size, hidden_size, 0.0f, stddev);
    if (use_bias) {
        b_f = core::Matrix(hidden_size, 1);
        
        // Initialize forget gate bias to 1.0 (helps with learning long-term dependencies)
        for (size_t i = 0; i < hidden_size; ++i) {
            b_f.at(i, 0) = core::Value::create(1.0f);
        }
    }
    
    // Output gate
    W_o = core::Matrix::random(hidden_size, input_size, 0.0f, stddev);
    U_o = core::Matrix::random(hidden_size, hidden_size, 0.0f, stddev);
    if (use_bias) {
        b_o = core::Matrix(hidden_size, 1);
    }
    
    // Cell gate
    W_g = core::Matrix::random(hidden_size, input_size, 0.0f, stddev);
    U_g = core::Matrix::random(hidden_size, hidden_size, 0.0f, stddev);
    if (use_bias) {
        b_g = core::Matrix(hidden_size, 1);
    }
    
    // Initialize hidden and cell states to zeros
    h = core::Matrix(hidden_size, 1);
    c = core::Matrix(hidden_size, 1);
}

core::Matrix LSTM::forward(const core::Matrix& input) {
    if (input.numCols() != 1) {
        throw std::invalid_argument("LSTM input must be a column vector");
    }
    
    if (input.numRows() != input_size) {
        throw std::invalid_argument("LSTM input size does not match layer input size");
    }
    
    auto [output, new_h, new_c] = forward_with_state(input, h, c);
    
    // Update internal states
    h = new_h;
    c = new_c;
    
    return output;
}

std::tuple<core::Matrix, core::Matrix, core::Matrix> LSTM::forward_with_state(
    const core::Matrix& input, 
    const core::Matrix& h_prev, 
    const core::Matrix& c_prev
) {
    // Input gate
    core::Matrix i_t = core::Matrix::matmul(W_i, input);
    i_t = core::Matrix::elementWiseAdd(i_t, core::Matrix::matmul(U_i, h_prev));
    if (use_bias) {
        i_t = core::Matrix::elementWiseAdd(i_t, b_i);
    }
    i_t = core::Matrix::sigmoid(i_t);
    
    // Forget gate
    core::Matrix f_t = core::Matrix::matmul(W_f, input);
    f_t = core::Matrix::elementWiseAdd(f_t, core::Matrix::matmul(U_f, h_prev));
    if (use_bias) {
        f_t = core::Matrix::elementWiseAdd(f_t, b_f);
    }
    f_t = core::Matrix::sigmoid(f_t);
    
    // Output gate
    core::Matrix o_t = core::Matrix::matmul(W_o, input);
    o_t = core::Matrix::elementWiseAdd(o_t, core::Matrix::matmul(U_o, h_prev));
    if (use_bias) {
        o_t = core::Matrix::elementWiseAdd(o_t, b_o);
    }
    o_t = core::Matrix::sigmoid(o_t);
    
    // Cell gate
    core::Matrix g_t = core::Matrix::matmul(W_g, input);
    g_t = core::Matrix::elementWiseAdd(g_t, core::Matrix::matmul(U_g, h_prev));
    if (use_bias) {
        g_t = core::Matrix::elementWiseAdd(g_t, b_g);
    }
    g_t = core::Matrix::tanh(g_t);
    
    // Cell state update
    core::Matrix c_t = core::Matrix::elementWiseMultiply(f_t, c_prev);
    c_t = core::Matrix::elementWiseAdd(c_t, core::Matrix::elementWiseMultiply(i_t, g_t));
    
    // Hidden state update
    core::Matrix h_t = core::Matrix::elementWiseMultiply(o_t, core::Matrix::tanh(c_t));
    
    return {h_t, h_t, c_t};
}

void LSTM::reset_state() {
    h = core::Matrix(hidden_size, 1);
    c = core::Matrix(hidden_size, 1);
}

std::vector<core::ValuePtr> LSTM::parameters() const {
    std::vector<core::ValuePtr> params;
    
    // Input gate parameters
    std::vector<core::ValuePtr> W_i_params = W_i.flatten();
    std::vector<core::ValuePtr> U_i_params = U_i.flatten();
    params.insert(params.end(), W_i_params.begin(), W_i_params.end());
    params.insert(params.end(), U_i_params.begin(), U_i_params.end());
    
    // Forget gate parameters
    std::vector<core::ValuePtr> W_f_params = W_f.flatten();
    std::vector<core::ValuePtr> U_f_params = U_f.flatten();
    params.insert(params.end(), W_f_params.begin(), W_f_params.end());
    params.insert(params.end(), U_f_params.begin(), U_f_params.end());
    
    // Output gate parameters
    std::vector<core::ValuePtr> W_o_params = W_o.flatten();
    std::vector<core::ValuePtr> U_o_params = U_o.flatten();
    params.insert(params.end(), W_o_params.begin(), W_o_params.end());
    params.insert(params.end(), U_o_params.begin(), U_o_params.end());
    
    // Cell gate parameters
    std::vector<core::ValuePtr> W_g_params = W_g.flatten();
    std::vector<core::ValuePtr> U_g_params = U_g.flatten();
    params.insert(params.end(), W_g_params.begin(), W_g_params.end());
    params.insert(params.end(), U_g_params.begin(), U_g_params.end());
    
    // Bias parameters
    if (use_bias) {
        std::vector<core::ValuePtr> b_i_params = b_i.flatten();
        std::vector<core::ValuePtr> b_f_params = b_f.flatten();
        std::vector<core::ValuePtr> b_o_params = b_o.flatten();
        std::vector<core::ValuePtr> b_g_params = b_g.flatten();
        
        params.insert(params.end(), b_i_params.begin(), b_i_params.end());
        params.insert(params.end(), b_f_params.begin(), b_f_params.end());
        params.insert(params.end(), b_o_params.begin(), b_o_params.end());
        params.insert(params.end(), b_g_params.begin(), b_g_params.end());
    }
    
    return params;
}

void LSTM::zeroGrad() {
    // Zero out gradients for all parameters
    W_i.zeroGrad();
    U_i.zeroGrad();
    W_f.zeroGrad();
    U_f.zeroGrad();
    W_o.zeroGrad();
    U_o.zeroGrad();
    W_g.zeroGrad();
    U_g.zeroGrad();
    
    if (use_bias) {
        b_i.zeroGrad();
        b_f.zeroGrad();
        b_o.zeroGrad();
        b_g.zeroGrad();
    }
}

std::string LSTM::name() const {
    return layer_name;
}

void LSTM::print() const {
    std::cout << "Layer: " << layer_name << std::endl;
    std::cout << "Type: LSTM" << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Hidden size: " << hidden_size << std::endl;
    std::cout << "Use bias: " << (use_bias ? "true" : "false") << std::endl;
    std::cout << "Parameters: " << parameters().size() << std::endl;
    std::cout << std::endl;
}

const core::Matrix& LSTM::getHiddenState() const {
    return h;
}

const core::Matrix& LSTM::getCellState() const {
    return c;
}

void LSTM::setHiddenState(const core::Matrix& hidden_state) {
    if (hidden_state.numRows() != hidden_size || hidden_state.numCols() != 1) {
        throw std::invalid_argument("Hidden state dimensions do not match layer hidden size");
    }
    
    h = hidden_state;
}

void LSTM::setCellState(const core::Matrix& cell_state) {
    if (cell_state.numRows() != hidden_size || cell_state.numCols() != 1) {
        throw std::invalid_argument("Cell state dimensions do not match layer hidden size");
    }
    
    c = cell_state;
}

} // namespace nn
} // namespace finml 