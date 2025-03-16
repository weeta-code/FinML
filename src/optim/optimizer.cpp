#include "finml/optim/optimizer.h"

namespace finml {
namespace optim {

Optimizer::Optimizer(const std::vector<core::ValuePtr>& parameters, float learning_rate)
    : parameters(parameters), learning_rate(learning_rate) {}

void Optimizer::zero_grad() {
    for (auto& param : parameters) {
        param->grad = 0.0f;
    }
}

float Optimizer::get_learning_rate() const {
    return learning_rate;
}

void Optimizer::set_learning_rate(float lr) {
    learning_rate = lr;
}

} // namespace optim
} // namespace finml 