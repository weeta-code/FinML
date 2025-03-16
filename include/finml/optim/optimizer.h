#ifndef FINML_OPTIM_OPTIMIZER_H
#define FINML_OPTIM_OPTIMIZER_H

#include "finml/core/value.h"
#include <vector>
#include <memory>

namespace finml {
namespace optim {

// Optimizer ADT
class Optimizer {
protected:
    std::vector<core::ValuePtr> parameters;
    float learning_rate;

public:

    Optimizer(const std::vector<core::ValuePtr>& parameters, float learning_rate);

    virtual ~Optimizer() = default;
    
    void zero_grad();
    // Update parameters according to gradient descent
    virtual void step() = 0;
    float get_learning_rate() const;
    void set_learning_rate(float lr);
};

} // namespace optim
} // namespace finml

#endif // FINML_OPTIM_OPTIMIZER_H 