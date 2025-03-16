#ifndef FINML_NN_LINEAR_H
#define FINML_NN_LINEAR_H

#include "finml/nn/layer.h"
#include "finml/core/matrix.h"
#include <string>
#include <vector>
#include <memory>

namespace finml {
namespace nn {

// Linear layers for nns fully optimized with weights and bias
class Linear : public Layer {
private:
    size_t in_features;
    size_t out_features;
    core::Matrix weights;
    core::Matrix bias;
    bool use_bias;
    std::string layer_name;

public:
    Linear(size_t in_features, size_t out_features, bool use_bias = true, const std::string& name = "Linear");

    core::Matrix forward(const core::Matrix& input) override;
    std::vector<core::ValuePtr> parameters() const override;
    
    void zeroGrad() override;
    std::string name() const override;
    void print() const override;
    
    const core::Matrix& getWeights() const;
    const core::Matrix& getBias() const;
};

} // namespace nn
} // namespace finml

#endif // FINML_NN_LINEAR_H 