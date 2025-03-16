#include "finml/nn/sequential.h"
#include <iostream>

namespace finml {
namespace nn {

Sequential::Sequential(const std::string& name) : model_name(name) {}

Sequential& Sequential::add(LayerPtr layer) {
    layers.push_back(layer);
    return *this;
}

core::Matrix Sequential::forward(const core::Matrix& input) {
    if (layers.empty()) {
        throw std::runtime_error("Sequential model has no layers");
    }
    
    core::Matrix output = input;
    for (const auto& layer : layers) {
        output = layer->forward(output);
    }
    
    return output;
}

std::vector<core::ValuePtr> Sequential::parameters() const {
    std::vector<core::ValuePtr> params;
    
    for (const auto& layer : layers) {
        std::vector<core::ValuePtr> layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    
    return params;
}

void Sequential::zeroGrad() {
    for (const auto& layer : layers) {
        layer->zeroGrad();
    }
}

std::string Sequential::name() const {
    return model_name;
}

void Sequential::print() const {
    std::cout << "Model: " << model_name << std::endl;
    std::cout << "Layers: " << layers.size() << std::endl;
    std::cout << "Total parameters: " << parameters().size() << std::endl;
    std::cout << std::endl;
    
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i << ": " << std::endl;
        layers[i]->print();
    }
}

size_t Sequential::size() const {
    return layers.size();
}

LayerPtr Sequential::getLayer(size_t index) const {
    if (index >= layers.size()) {
        throw std::out_of_range("Layer index out of bounds");
    }
    
    return layers[index];
}

} // namespace nn
} // namespace finml 