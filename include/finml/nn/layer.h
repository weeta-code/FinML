#ifndef FINML_NN_LAYER_H
#define FINML_NN_LAYER_H

#include "finml/core/matrix.h"
#include <vector>
#include <memory>
#include <string>

namespace finml {
namespace nn {


// Layer adt
class Layer {
public:
    virtual ~Layer() = default;
    
    
    virtual core::Matrix forward(const core::Matrix& input) = 0;
    virtual std::vector<core::ValuePtr> parameters() const = 0;

    virtual void zeroGrad() = 0;
    virtual std::string name() const = 0;
    virtual void print() const = 0;
};

using LayerPtr = std::shared_ptr<Layer>;

} // namespace nn
} // namespace finml

#endif // FINML_NN_LAYER_H 