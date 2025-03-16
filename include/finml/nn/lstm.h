#ifndef FINML_NN_LSTM_H
#define FINML_NN_LSTM_H

#include "finml/nn/layer.h"
#include "finml/core/matrix.h"
#include <string>
#include <vector>
#include <memory>
#include <tuple>

namespace finml {
namespace nn {

// Implementing LSTM cells for sequence modeling and layering
class LSTM : public Layer {
private:
    size_t input_size;
    size_t hidden_size;
    bool use_bias;
    std::string layer_name;
    
    // Gates weights and biases
    core::Matrix W_i; // Input gate weights for input
    core::Matrix U_i; // Input gate weights for hidden state
    core::Matrix b_i; // Input gate bias
    
    core::Matrix W_f; // Forget gate weights for input
    core::Matrix U_f; // Forget gate weights for hidden state
    core::Matrix b_f; // Forget gate bias
    
    core::Matrix W_o; // Output gate weights for input
    core::Matrix U_o; // Output gate weights for hidden state
    core::Matrix b_o; // Output gate bias
    
    core::Matrix W_g; // Cell gate weights for input
    core::Matrix U_g; // Cell gate weights for hidden state
    core::Matrix b_g; // Cell gate bias
    
    // Hidden state and cell state
    core::Matrix h;
    core::Matrix c;
    
    // Helper functions
    core::Matrix sigmoid(const core::Matrix& x) const;
    core::Matrix tanh(const core::Matrix& x) const;

public:
    LSTM(size_t input_size, size_t hidden_size, bool use_bias = true, const std::string& name = "LSTM");
    
    core::Matrix forward(const core::Matrix& input) override;
    std::tuple<core::Matrix, core::Matrix, core::Matrix> forward_with_state(
        const core::Matrix& input, 
        const core::Matrix& h_prev, 
        const core::Matrix& c_prev
    );
    
    void reset_state();
    std::vector<core::ValuePtr> parameters() const override;
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;

    const core::Matrix& getHiddenState() const;
    const core::Matrix& getCellState() const;
    void setHiddenState(const core::Matrix& hidden_state);
    void setCellState(const core::Matrix& cell_state);
};

} // namespace nn
} // namespace finml

#endif // FINML_NN_LSTM_H 