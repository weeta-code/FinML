#include "finml/optim/sgd.h"

namespace finml {
namespace optim {

SGD::SGD(
    const std::vector<core::ValuePtr>& parameters, 
    float learning_rate, 
    float momentum, 
    float weight_decay
) : Optimizer(parameters, learning_rate), momentum(momentum), weight_decay(weight_decay) {
    velocity.resize(parameters.size(), 0.0f);
}

void SGD::step() {
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        
        // Apply weight decay
        float grad = param->grad;
        if (weight_decay > 0) {
            grad += weight_decay * param->data;
        }
        
        // Apply momentum
        if (momentum > 0) {
            velocity[i] = momentum * velocity[i] + grad;
            grad = velocity[i];
        }
        
        // Update parameter
        param->data -= learning_rate * grad;
    }
}

} // namespace optim
} // namespace finml 