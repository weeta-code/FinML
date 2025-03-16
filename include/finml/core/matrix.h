#ifndef FINML_CORE_MATRIX_H
#define FINML_CORE_MATRIX_H

#include "finml/core/value.h"
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <random>
#include <functional>
#include <string>

namespace finml {
namespace core {


class Matrix {
private:
    std::vector<std::vector<ValuePtr>> data;
    size_t rows;
    size_t cols;

public:
    
    Matrix(size_t rows, size_t cols);
    Matrix(const std::initializer_list<std::initializer_list<float>>& init_list);
    
    static Matrix random(size_t rows, size_t cols, float mean = 0.0f, float stddev = 1.0f);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);

    // Accessors
    ValuePtr& at(size_t i, size_t j);
    const ValuePtr& at(size_t i, size_t j) const;
    size_t numRows() const { return rows; }
    size_t numCols() const { return cols; }

    // Matrix operations
    static Matrix matmul(const Matrix& a, const Matrix& b);
    static Matrix transpose(const Matrix& m);
    static Matrix elementWiseAdd(const Matrix& a, const Matrix& b);
    static Matrix elementWiseSubtract(const Matrix& a, const Matrix& b);
    static Matrix elementWiseMultiply(const Matrix& a, const Matrix& b);
    static Matrix elementWiseDivide(const Matrix& a, const Matrix& b);
    
    // Scalar operations
    static Matrix scalarAdd(const Matrix& m, float scalar);
    static Matrix scalarMultiply(const Matrix& m, float scalar);
    
    // Element-wise operations
    static Matrix apply(const Matrix& m, std::function<ValuePtr(const ValuePtr&)> func);
    
    // activation functions
    static Matrix relu(const Matrix& m);
    static Matrix sigmoid(const Matrix& m);
    static Matrix tanh(const Matrix& m);
    static Matrix softmax(const Matrix& m);
    
    // Util functions
    void zeroGrad();
    std::vector<ValuePtr> flatten() const;
    void print(const std::string& name = "") const;
    
    // row/col ops
    std::vector<ValuePtr> getRow(size_t i) const;
    std::vector<ValuePtr> getCol(size_t j) const;
    void setRow(size_t i, const std::vector<ValuePtr>& row);
    void setCol(size_t j, const std::vector<ValuePtr>& col);
    
    // Operator overloads
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;  // Matrix multiplication
    Matrix operator%(const Matrix& other) const;  // Element-wise multiplication
    Matrix operator/(const Matrix& other) const;  // Element-wise division
};

ValuePtr dot(const std::vector<ValuePtr>& a, const std::vector<ValuePtr>& b);

} // namespace core
} // namespace finml

#endif // FINML_CORE_MATRIX_H 