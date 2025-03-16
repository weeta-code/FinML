#ifndef FINML_OPTIM_SGD_H
#define FINML_OPTIM_SGD_H

#include "finml/optim/optimizer.h"
#include <vector>

namespace finml {
namespace optim {

// Stochastic Gradient Descent optimizer
class SGD : public Optimizer {
private:
    float momentum;
    float weight_decay;
    std::vector<float> velocity;

public:
    // Constructor
    SGD(
        const std::vector<core::ValuePtr>& parameters, 
        float learning_rate = 0.01f, 
        float momentum = 0.0f, 
        float weight_decay = 0.0f
    );
    
    // Update parameters based on gradients
    void step() override;
};

} // namespace optim
} // namespace finml

#endif // FINML_OPTIM_SGD_H 