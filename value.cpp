#include <value.h>
#include <unordered_set>
#include <iostream>

Value::Value(float data, const std::string& op, size_t id)
    : data(data), op(op), id(id), grad(0.0) {}

