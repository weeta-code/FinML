#include "finml/optim/adam.h"
#include <cmath>

namespace finml {
namespace optim {

Adam::Adam(
    const std::vector<core::ValuePtr>& parameters, 
    float learning_rate, 
    float beta1, 
    float beta2, 
    float epsilon, 
    float weight_decay
) : Optimizer(parameters, learning_rate), 
    beta1(beta1), 
    beta2(beta2), 
    epsilon(epsilon), 
    weight_decay(weight_decay), 
    t(0) {
    
    m.resize(parameters.size(), 0.0f);
    v.resize(parameters.size(), 0.0f);
}

void Adam::step() {
    ++t;
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        auto& param = parameters[i];
        
        // Apply weight decay
        float grad = param->grad;
        if (weight_decay > 0) {
            grad += weight_decay * param->data;
        }
        
        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[i] / (1.0f - std::pow(beta1, t));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[i] / (1.0f - std::pow(beta2, t));
        
        // Update parameters
        param->data -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

} // namespace optim
} // namespace finml 