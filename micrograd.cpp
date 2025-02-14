#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <random>

class Value;

using ValuePtr = std::shared_ptr<Value>;

struct Hash {
    size_t operator()(const ValuePtr value) const;
};
static ValuePtr dot(const std::vector<ValuePtr>& a, const std::vector<ValuePtr>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length");
    }
    ValuePtr result = Value::create(0.0);
    for (size_t i = 0; i < a.size(); ++i) {
        result = Value::add(result, Value::multiply(a[i], b[i]));
    }
    return result;
}
// Skeleton Class to be built upon at the moment
class Value : public std::enable_shared_from_this<Value> // Deriving from a shared class because of a shared pointer
{
public:
    inline static size_t currentID = 0; // Increments or Decrements whenever Value gets instantiated or Destroyed; Tracks number of nodes in graph
    float data; // value of class value ex. node a = -1 so data = -1
    float grad; // gradient of data member
    std::string op; // operator used to construct current node
    size_t id; // Unique to the node, so we can distinguish between nodes
    std::vector<ValuePtr> prev; // vector of value, useful for knowing children for back propogation
    std::function<void()> backward;

private:
    Value(float data, const std::string &op, size_t id) : data(data), op(op), id(id) {};

public:
    static ValuePtr create(float data, const std::string &op = "")
    {
        return std::shared_ptr<Value>(new Value(data, op, Value::currentID++)); // Need to expose an API for client construction instead of direct construction, so client doesn't directly call constructur of value
    }

    ~Value(){
        --Value::currentID;
    }

    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data+rhs->data, "+"); // because lhs and rhs are pointers must use ->
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad; // total gradient is equal to output gradient, in this case being out or c
            rhs_weak.lock()->grad += out_weak.lock()->grad; // never want to wipe our old gradient, accumulating our gradients
        };
        return out;
    }

    static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data*rhs->data, "*"); 
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += rhs_weak.lock()->data * out_weak.lock()->grad; 
            rhs_weak.lock()->grad += lhs_weak.lock()->data * out_weak.lock()->grad; 
        };
        return out;
    }

    static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs){
        auto out = Value::create(lhs->data-rhs->data, "-"); 
        out->prev = {lhs, rhs};
        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad; 
            rhs_weak.lock()->grad -= out_weak.lock()->grad; 
        };
        return out;
    }

    // out = base^exponent
    // dL/d(base) = dL/d(out) * d(out)/d(base)
    //            = out->grad * exponent * base^(exponent - 1) basic power rule here

    static ValuePtr pow(const ValuePtr& base, float exponent) {
        float newValue = std::pow(base->data, exponent);
        auto out = Value::create(newValue, "^");
        out->prev = {base};
        out->backward = [base_weak = std::weak_ptr<Value>(base), out_weak = std::weak_ptr<Value>(out), exponent](){
            if (auto base = base_weak.lock()) {
                base->grad += exponent * std::pow(base->data, exponent-1) * out_weak.lock()-> grad;
            }
        };
        return out;
    }

    static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs) {
        auto reciprocal = pow(rhs, -1); // Division is multiplication of the reciprocal, very easy
        return multiply(lhs, reciprocal);
    }


    // Activation Function / Rectified Linear Unit
    // Whenever x > 0 it's value is = x , whenever x < 0 it's value is = 0 grad is 1 when x=x and 0 when x=0
    static ValuePtr relu(const ValuePtr& input) {
        float val = std::max(0.0f, input->data);
        auto out = Value::create(val, "ReLU");
        out->prev = {input};
        out-> backward = [input, out](){
            // 8 *out_grad
            // local_grad * out_grad
            if (input) input-> grad += (out->data > 0) * out->grad; // whenever y<=0 then grad =1 and grad =0 in all toher cases
        };
        return out;
    }

    static ValuePtr sigmoid(const ValuePtr& input) {
        float x = input->data;
        float t = std::exp(x) / (1 + std::exp(x));
        auto out = Value::create(t, "Sigmoid");
        out->prev = {input};
        out->backward = [input, out, t](){
            input->grad += t * (1-t) * out->grad;
        };
        return out;
    }

    static ValuePtr tanh(const ValuePtr& input) {
        float t = std::tanh(input->data);
        auto out = Value::create(t, "tanh");
        out->prev = {input};
        out->backward = [input, out, t]() {
            input->grad += (1 - t * t) * out->grad;
        };
        return out;
    }

    void buildTopo(std::shared_ptr<Value> v, std::unordered_set<std::shared_ptr<Value>, Hash>& visited, std::vector<std::shared_ptr<Value>>& topo){
        if(visited.find(v) == visited.end()){ // checking to see if current value has been visited
            visited.insert(v);
            for(const auto& child : v->prev){ // then looking at it's children
                buildTopo(child, visited, topo); // topologically sorted graph
            }
            topo.push_back(v);
        }
    }

    void backProp(){
        std::vector<std::shared_ptr<Value>> topo;
        std::unordered_set<std::shared_ptr<Value>, Hash> visited; // for no overlap
        
        buildTopo(shared_from_this(), visited, topo); // creating our graph

        this->grad = 1.0f;
        for( auto it = topo.rbegin(); it != topo.rend(); ++it){
            if((*it) -> backward) {
                (*it)-> backward();
            }
        }
    }
    void print(){
        std::cout << "[data=" << this->data << ", grad=" << this-> grad << "]\n";
    }
};

size_t Hash::operator()(const ValuePtr value) const {
    return std::hash<size_t>()(value->id) ^ std::hash<std::string>()(value->op);// Using only the operation string as the hash key
}

enum ActivationType {
    RELU, 
    SIGMOID,
    TANH
};

class Activation {
    static std::shared_ptr<Value> Relu(const std::shared_ptr<Value>& val){
        return Value::relu(val);
    }
    static std::shared_ptr<Value> Sigmoid(const std::shared_ptr<Value>& val){
        return Value::sigmoid(val);
    }
    static std::shared_ptr<Value> Tanh(const std::shared_ptr<Value>& val){
        return Value::tanh(val);
    }

public:
    static inline std::unordered_map<ActivationType, std::shared_ptr<Value>(*)(const std::shared_ptr<Value>&)> mActivationFcn = {
        {ActivationType::RELU, &Relu},
        {ActivationType::SIGMOID, &Sigmoid}
    };
};


float getRandomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(gen);
}

class Neuron {
private:
    std::vector<ValuePtr> weights; // weights per neuron
    ValuePtr bias = Value::create(0.0); // bias 
    const ActivationType activation_t; // activation type (function)

public:
    Neuron(size_t nin, const ActivationType& activation_t) : activation_t(activation_t) {
        for (size_t idx = 0; idx < nin; ++idx) { // takes an input so we know the length of the vector of weights
            weights.emplace_back(Value::create(getRandomFloat())); // need to know the activation type as well
        } // initialize the weights
    }

    // Testing
    Neuron(size_t nin, float val, const ActivationType& activation_t = ActivationType::SIGMOID) : activation_t(activation_t) {
        for (size_t idx = 0; idx < nin; ++idx) {
            weights.emplace_back(Value::create(getRandomFloat()));
        }
    }

    void zeroGrad() {
        for (auto& weight : weights){
            weight->grad = 0;
        }
        bias->grad = 0;
    }


    // Dot Product of a Neuron's weights with the input
    ValuePtr operator()(const std::vector<ValuePtr>& x) { // takes vector as an input
        if (x.size() != weights.size()) {
            throw std::invalid_argument("Vectors must be of the same length"); // checks if the size of the input = size of the weights
        }

        ValuePtr sum = dot(x, weights);

        // Add Bias to our sum bias is always 0 but customizeable
        sum = Value::add(sum, bias);

        // Applying our Activation Function
        const auto& activationFcn = Activation::mActivationFcn.at(activation_t); // look at activation type which is initialized through our constructor
        return activationFcn(sum);
    }

    std::vector<ValuePtr> parameters() const { // returns all parameters, utility function for now
        std::vector<ValuePtr> out;
        out.reserve(weights.size() + 1);

        out.insert(out.end(), weights.begin(), weights.end());
        out.push_back(bias);

        return out;
    }

    void printParameters() const {
        printf("Number of Parameters: %zu\n", weights.size() + 1);
        for (const auto& param : weights) {
            printf("%f, %f\n", param->data, param->grad);
        }
        printf("%f, %f\n", bias->data, bias->grad);
        printf("\n");
    }

    size_t getParametersSize() const {
        return weights.size() + 1;
    }
};

class Layer {
    std::vector<Neuron> neurons;
public:
    Layer(size_t dimOfNeuron, size_t numNeurons, const ActivationType& actType = ActivationType::RELU) {
        for(size_t idx= 0; idx < numNeurons; ++idx){
            this->neurons.emplace_back(dimOfNeuron, actType);
        }
    }

    std::vector<ValuePtr> operator()(const std::vector<ValuePtr>& x) {
        std::vector<ValuePtr> out;
        out.reserve(this->neurons.size()); // reserves the output for each neuron
        std::for_each(this->neurons.begin(), this->neurons.end(), [&out, x = x](auto neuron)mutable {
            out.emplace_back(neuron(x)); // if x is an input basically we use the x value for each neuron and calculate the output
        });
        return out;
    }

    void zeroGrad() {
        for(auto& neuron : this->neurons) {
            neuron.zeroGrad();
        }
    }

    std::vector<Value*> parameters() const{
        std::vector<Value*> params;
        if(params.empty()) {
            for(const auto& neuron : neurons){
                for(const auto& p : neuron.parameters()) {
                    params.push_back(p.get());
                }
            };
        };
        return params;
    }
    
    void print() {
        const auto params = this-> parameters();
        printf("Num parameters: %d\n", (int)params.size());
        for(const auto& p : params) {
            std::cout << &p << " ";
            printf("[data=%f,grad=%lf]\n", p->data, p->grad);
        }
        printf("\n");
    }

};

class LSTMCell {
    // Each parameter being a vector of ValuePtrs
    std::vector<std::vector<ValuePtr>> W_i, U_i; ValuePtr b_i;
    std::vector<std::vector<ValuePtr>> W_f, U_f; ValuePtr b_f;
    std::vector<std::vector<ValuePtr>> W_o, U_o; ValuePtr b_o;
    std::vector<std::vector<ValuePtr>> W_g, U_g; ValuePtr b_g;

    int inputSize;
    int hiddenSize;

    LSTMCell(int inputSize, int hiddenSize) : inputSize(inputSize), hiddenSize(hiddenSize) {
        
    }

};

int main()
{
   Layer l1(100, 20);
   l1.print();
}