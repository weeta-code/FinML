#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <unordered_set>

class Value;

using ValuePtr = std::shared_ptr<Value>;

class Value : public std::enable_shared_from_this<Value> {
public:
    inline static size_t currentID = 0;
    float data;
    float grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    std::function<void()> backward;

    static std::vector<ValuePtr> create(float data, const std::string& op = "");

    static std::vector<ValuePtr> add(const std::vector<ValuePtr>& lhs, const std::vector<ValuePtr>& rhs);
    static std::vector<ValuePtr> subtract(const std::vector<ValuePtr> lhs, const std::vector<ValuePtr>& rhs);
    static std::vector<ValuePtr> divide(const std::vector<ValuePtr>& lhs, const std::vector<ValuePtr>& rhs);
    static std::vector<ValuePtr> pow(const std::vector<ValuePtr>& base, const std::vector<ValuePtr>& exponent);
    static std::vector<ValuePtr> multiply(const std::vector<ValuePtr>& lhs, const std::vector<ValuePtr>& rhs);
    static std::vector<ValuePtr> relu(const std::vector<ValuePtr>& input);

    void backProp();

private:
    Value(float data, const std::string&op, size_t id);
    void buildTopo(std::vector<ValuePtr> v, std::unordered_set<std::vector<ValuePtr>>& visited, std::vector<std::vector<ValuePtr>>& topo);

};

struct Hash {
    size_t operator()(const std::vector<ValuePtr> value) const;
};