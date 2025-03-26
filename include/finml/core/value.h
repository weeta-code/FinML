#ifndef FINML_CORE_VALUE_H
#define FINML_CORE_VALUE_H

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>

namespace finml {
namespace core {

class Value;
using ValuePtr = std::shared_ptr<Value>;

struct Hash {
    size_t operator()(const ValuePtr& value) const;
};


class Value : public std::enable_shared_from_this<Value> {
public:
    inline static size_t currentID = 0;
    inline static std::vector<Value*> createdValues;

    float data;
    float grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    std::function<void()> backward;

    ~Value();

    
    static ValuePtr create(float data, const std::string& op = "");

    // Basic arithmetic operations
    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr pow(const ValuePtr& base, float exponent);
    static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr neg(const ValuePtr& v);
    static ValuePtr exp(const ValuePtr& v);
    static ValuePtr log(const ValuePtr& v);

    // Activation functions
    static ValuePtr relu(const ValuePtr& input);
    static ValuePtr sigmoid(const ValuePtr& input);
    static ValuePtr tanh(const ValuePtr& input);
    static ValuePtr leakyRelu(const ValuePtr& input, float alpha = 0.01f);
    static ValuePtr softmax(const std::vector<ValuePtr>& inputs);
    
    // Memory management
    static std::vector<Value*>& getCreatedValues();
    static void clearCreatedValues();

    // Backpropagation
    void backProp();

    // Utility functions
    void print() const;
    void zeroGrad();
    void gradientCheck(const ValuePtr& node, float eps = 1e-4);

    // Operator overloads
    ValuePtr operator+(const ValuePtr& other) const;
    ValuePtr operator*(const ValuePtr& other) const;
    ValuePtr operator-(const ValuePtr& other) const;
    ValuePtr operator/(const ValuePtr& other) const;
    ValuePtr operator-() const;

private:
    Value(float data, const std::string& op, size_t id);
    void buildTopo(ValuePtr v, std::unordered_set<ValuePtr, Hash>& visited, std::vector<ValuePtr>& topo);
};

} // namespace core
} // namespace finml

#endif // FINML_CORE_VALUE_H 