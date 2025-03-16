#ifndef FINML_OPTIM_ADAM_H
#define FINML_OPTIM_ADAM_H

#include "finml/optim/optimizer.h"
#include <vector>

namespace finml {
namespace optim {

// Adam optimization algorithm
class Adam : public Optimizer {
private:
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    std::vector<float> m; // First moment estimates
    std::vector<float> v; // Second moment estimates
    size_t t; // Timestep

public:
    // Constructor
    Adam(
        const std::vector<core::ValuePtr>& parameters, 
        float learning_rate = 0.001f, 
        float beta1 = 0.9f, 
        float beta2 = 0.999f, 
        float epsilon = 1e-8f, 
        float weight_decay = 0.0f
    );
    void step() override;
};

} // namespace optim
} // namespace finml

#endif // FINML_OPTIM_ADAM_H 