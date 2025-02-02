#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <unordered_set>

class Value;

using ValuePtr = std::shared_ptr<Value>;

struct Hash {
    size_t operator()(const ValuePtr value) const;
};
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
            (*it)->print();
        }
    }
    void print(){
        std::cout << "[data=" << this->data << ", grad=" << this-> grad << "]\n";
    }
};

size_t Hash::operator()(const ValuePtr value) const {
    return std::hash<std::string>()(value.get()->op) ^ std::hash<float>()(value.get()->data);  // Using only the operation string as the hash key
}



int main()
{
    auto a = Value::create(1.0, "");
    auto b = Value::create(2.0, "");
    auto c = Value::add(a, b);

    auto d = Value::multiply(c, c);

    assert(c->data == 3.0);
    assert(c->op == "+");

    assert(d->data == 9.0);
    assert(d->op == "*");

    auto loss = Value::add(d, d);

    loss->backProp();
    // auto d = subtract(a-b)

    // auto e = multiply(c*d)

    // a -->
    //        +  --> c --> op --> L dL/da = b * dL/dc
    // b -->
    // auto a = std::shared_ptr<Value>();
}