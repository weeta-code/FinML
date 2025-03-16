#include "finml/core/value.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace finml {
namespace core {

Value::Value(float data, const std::string& op, size_t id)
    : data(data), grad(0.0f), op(op), id(id) {}

Value::~Value() {
    --Value::currentID;
}

ValuePtr Value::create(float data, const std::string& op) {
    return std::make_shared<Value>(data, op, Value::currentID++);
}

ValuePtr Value::add(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = Value::create(lhs->data + rhs->data, "+");
    out->prev = {lhs, rhs};
    out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto lhs = lhs_weak.lock()) {
            if (auto out = out_weak.lock()) {
                lhs->grad += out->grad;
            }
        }
        if (auto rhs = rhs_weak.lock()) {
            if (auto out = out_weak.lock()) {
                rhs->grad += out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::multiply(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = Value::create(lhs->data * rhs->data, "*");
    out->prev = {lhs, rhs};
    out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto lhs = lhs_weak.lock()) {
            if (auto rhs = rhs_weak.lock()) {
                if (auto out = out_weak.lock()) {
                    lhs->grad += rhs->data * out->grad;
                    rhs->grad += lhs->data * out->grad;
                }
            }
        }
    };
    return out;
}

ValuePtr Value::subtract(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto out = Value::create(lhs->data - rhs->data, "-");
    out->prev = {lhs, rhs};
    out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto lhs = lhs_weak.lock()) {
            if (auto out = out_weak.lock()) {
                lhs->grad += out->grad;
            }
        }
        if (auto rhs = rhs_weak.lock()) {
            if (auto out = out_weak.lock()) {
                rhs->grad -= out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::pow(const ValuePtr& base, float exponent) {
    float newValue = std::pow(base->data, exponent);
    auto out = Value::create(newValue, "^");
    out->prev = {base};
    out->backward = [base_weak = std::weak_ptr<Value>(base), out_weak = std::weak_ptr<Value>(out), exponent]() {
        if (auto base = base_weak.lock()) {
            if (auto out = out_weak.lock()) {
                base->grad += exponent * std::pow(base->data, exponent - 1) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::divide(const ValuePtr& lhs, const ValuePtr& rhs) {
    auto reciprocal = pow(rhs, -1.0f);
    return multiply(lhs, reciprocal);
}

ValuePtr Value::neg(const ValuePtr& v) {
    return multiply(v, create(-1.0f));
}

ValuePtr Value::exp(const ValuePtr& v) {
    float newValue = std::exp(v->data);
    auto out = Value::create(newValue, "exp");
    out->prev = {v};
    out->backward = [v_weak = std::weak_ptr<Value>(v), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto v = v_weak.lock()) {
            if (auto out = out_weak.lock()) {
                v->grad += out->data * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::log(const ValuePtr& v) {
    if (v->data <= 0) {
        throw std::domain_error("Cannot take log of non-positive value");
    }
    float newValue = std::log(v->data);
    auto out = Value::create(newValue, "log");
    out->prev = {v};
    out->backward = [v_weak = std::weak_ptr<Value>(v), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto v = v_weak.lock()) {
            if (auto out = out_weak.lock()) {
                v->grad += (1.0f / v->data) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::relu(const ValuePtr& input) {
    float val = std::max(0.0f, input->data);
    auto out = Value::create(val, "ReLU");
    out->prev = {input};
    out->backward = [input_weak = std::weak_ptr<Value>(input), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto input = input_weak.lock()) {
            if (auto out = out_weak.lock()) {
                input->grad += (input->data > 0) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::leakyRelu(const ValuePtr& input, float alpha) {
    float val = input->data > 0 ? input->data : alpha * input->data;
    auto out = Value::create(val, "LeakyReLU");
    out->prev = {input};
    out->backward = [input_weak = std::weak_ptr<Value>(input), out_weak = std::weak_ptr<Value>(out), alpha]() {
        if (auto input = input_weak.lock()) {
            if (auto out = out_weak.lock()) {
                input->grad += (input->data > 0 ? 1.0f : alpha) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::sigmoid(const ValuePtr& input) {
    float x = input->data;
    float t = 1.0f / (1.0f + std::exp(-x));
    auto out = Value::create(t, "Sigmoid");
    out->prev = {input};
    out->backward = [input_weak = std::weak_ptr<Value>(input), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto input = input_weak.lock()) {
            if (auto out = out_weak.lock()) {
                input->grad += out->data * (1.0f - out->data) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::tanh(const ValuePtr& input) {
    float t = std::tanh(input->data);
    auto out = Value::create(t, "tanh");
    out->prev = {input};
    out->backward = [input_weak = std::weak_ptr<Value>(input), out_weak = std::weak_ptr<Value>(out)]() {
        if (auto input = input_weak.lock()) {
            if (auto out = out_weak.lock()) {
                input->grad += (1.0f - out->data * out->data) * out->grad;
            }
        }
    };
    return out;
}

ValuePtr Value::softmax(const std::vector<ValuePtr>& inputs) {
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (const auto& input : inputs) {
        max_val = std::max(max_val, input->data);
    }
    
    // Calculate exp(x_i - max) for each input
    std::vector<float> exps;
    exps.reserve(inputs.size());
    for (const auto& input : inputs) {
        exps.push_back(std::exp(input->data - max_val));
    }
    
    // Calculate sum of exps
    float sum_exp = std::accumulate(exps.begin(), exps.end(), 0.0f);
    
    // Calculate softmax values
    std::vector<ValuePtr> outputs;
    outputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        float softmax_val = exps[i] / sum_exp;
        auto out = Value::create(softmax_val, "softmax");
        out->prev = inputs;
        
        // Store index for backward pass
        size_t idx = i;
        out->backward = [inputs_weak = std::vector<std::weak_ptr<Value>>(inputs.begin(), inputs.end()), 
                         out_weak = std::weak_ptr<Value>(out), idx, exps, sum_exp]() {
            if (auto out = out_weak.lock()) {
                for (size_t j = 0; j < inputs_weak.size(); ++j) {
                    if (auto input = inputs_weak[j].lock()) {
                        float grad = 0.0f;
                        if (j == idx) {
                            grad = out->data * (1.0f - out->data);
                        } else {
                            grad = -out->data * (exps[j] / sum_exp);
                        }
                        input->grad += grad * out->grad;
                    }
                }
            }
        };
        outputs.push_back(out);
    }
    
    return outputs[0]; // Return first output for simplicity
}

void Value::backProp() {
    std::vector<ValuePtr> topo;
    std::unordered_set<ValuePtr, Hash> visited;
    
    auto buildTopo = [&visited, &topo](auto&& self, ValuePtr v) -> void {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->prev) {
                self(self, child);
            }
            topo.push_back(v);
        }
    };
    
    buildTopo(buildTopo, shared_from_this());
    
    grad = 1.0f;
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward) (*it)->backward();
    }
}

void Value::buildTopo(ValuePtr v, std::unordered_set<ValuePtr, Hash>& visited, std::vector<ValuePtr>& topo) {
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        for (const auto& child : v->prev) {
            buildTopo(child, visited, topo);
        }
        topo.push_back(v);
    }
}

void Value::print() const {
    std::cout << "[data=" << data << ", grad=" << grad << ", op=" << op << "]\n";
}

void Value::zeroGrad() {
    grad = 0.0f;
    std::unordered_set<ValuePtr, Hash> visited;
    std::vector<ValuePtr> topo;
    
    buildTopo(shared_from_this(), visited, topo);
    
    for (const auto& v : topo) {
        v->grad = 0.0f;
    }
}

void Value::gradientCheck(const ValuePtr& node, float eps) {
    float original = node->data;
    
    // Compute original loss
    backProp();
    float original_grad = node->grad;
    
    // Perturb node and compute new loss
    node->data = original + eps;
    backProp();
    float perturbed_grad = (node->grad - original_grad) / eps;
    
    // Restore original value
    node->data = original;
    
    std::cout << "Numerical gradient: " << perturbed_grad << "\n";
    std::cout << "Analytical gradient: " << original_grad << "\n";
    std::cout << "Difference: " << std::abs(perturbed_grad - original_grad) << "\n";
}
ValuePtr Value::operator+(const ValuePtr& other) const {
    return add(std::static_pointer_cast<Value>(shared_from_this()), other);
}

ValuePtr Value::operator*(const ValuePtr& other) const {
    return multiply(std::static_pointer_cast<Value>(shared_from_this()), other);
}

ValuePtr Value::operator-(const ValuePtr& other) const {
    return subtract(std::static_pointer_cast<Value>(shared_from_this()), other);
}

ValuePtr Value::operator/(const ValuePtr& other) const {
    return divide(std::static_pointer_cast<Value>(shared_from_this()), other);
}

ValuePtr Value::operator-() const {
    return neg(std::static_pointer_cast<Value>(shared_from_this()));
}

size_t Hash::operator()(const ValuePtr& value) const {
    return std::hash<size_t>()(value->id);
}

} // namespace core
} // namespace finml 