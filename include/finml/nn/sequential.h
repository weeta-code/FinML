#ifndef FINML_NN_SEQUENTIAL_H
#define FINML_NN_SEQUENTIAL_H

#include "finml/nn/layer.h"
#include "finml/core/matrix.h"
#include <vector>
#include <memory>
#include <string>

namespace finml {
namespace nn {

// Sequential container for layers
class Sequential {
private:
    std::vector<LayerPtr> layers;
    std::string model_name;

public:
    explicit Sequential(const std::string& name = "Sequential");
    
    
    Sequential& add(LayerPtr layer);
    
    // Forward pass through all layers
    core::Matrix forward(const core::Matrix& input);
    // Get all trainable parameters of all layers
    std::vector<core::ValuePtr> parameters() const;
    void zeroGrad();
    std::string name() const;
    void print() const;
    size_t size() const;
    
    LayerPtr getLayer(size_t index) const;
};

} // namespace nn
} // namespace finml

#endif // FINML_NN_SEQUENTIAL_H 