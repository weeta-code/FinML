#include "finml/core/matrix.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>

namespace finml {
namespace core {

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows);
    for (auto& row : data) {
        row.resize(cols);
        for (auto& val : row) {
            val = Value::create(0.0f);
        }
    }
}

Matrix::Matrix(const std::initializer_list<std::initializer_list<float>>& init_list) {
    rows = init_list.size();
    if (rows == 0) {
        cols = 0;
        return;
    }
    
    cols = init_list.begin()->size();
    data.resize(rows);
    
    size_t i = 0;
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        
        data[i].resize(cols);
        size_t j = 0;
        for (const auto& val : row) {
            data[i][j] = Value::create(val);
            ++j;
        }
        ++i;
    }
}

Matrix Matrix::random(size_t rows, size_t cols, float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);
    
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = Value::create(dist(gen));
        }
    }
    
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = Value::create(1.0f);
        }
    }
    return result;
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size);
    for (size_t i = 0; i < size; ++i) {
        result.at(i, i) = Value::create(1.0f);
    }
    return result;
}

ValuePtr& Matrix::at(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i][j];
}

const ValuePtr& Matrix::at(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
        throw std::out_of_range("Matrix indices out of bounds");
    }
    return data[i][j];
}

Matrix Matrix::matmul(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    Matrix result(a.rows, b.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < b.cols; ++j) {
            std::vector<ValuePtr> col = b.getCol(j);
            result.at(i, j) = dot(a.getRow(i), col);
        }
    }
    
    return result;
}

Matrix Matrix::transpose(const Matrix& m) {
    Matrix result(m.cols, m.rows);
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result.at(j, i) = m.at(i, j);
        }
    }
    return result;
}

Matrix Matrix::elementWiseAdd(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise addition");
    }
    
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.at(i, j) = Value::add(a.at(i, j), b.at(i, j));
        }
    }
    
    return result;
}

Matrix Matrix::elementWiseSubtract(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise subtraction");
    }
    
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.at(i, j) = Value::subtract(a.at(i, j), b.at(i, j));
        }
    }
    
    return result;
}

Matrix Matrix::elementWiseMultiply(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");
    }
    
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.at(i, j) = Value::multiply(a.at(i, j), b.at(i, j));
        }
    }
    
    return result;
}

Matrix Matrix::elementWiseDivide(const Matrix& a, const Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix dimensions must match for element-wise division");
    }
    
    Matrix result(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            result.at(i, j) = Value::divide(a.at(i, j), b.at(i, j));
        }
    }
    
    return result;
}

Matrix Matrix::scalarAdd(const Matrix& m, float scalar) {
    Matrix result(m.rows, m.cols);
    auto scalarValue = Value::create(scalar);
    
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result.at(i, j) = Value::add(m.at(i, j), scalarValue);
        }
    }
    
    return result;
}

Matrix Matrix::scalarMultiply(const Matrix& m, float scalar) {
    Matrix result(m.rows, m.cols);
    auto scalarValue = Value::create(scalar);
    
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result.at(i, j) = Value::multiply(m.at(i, j), scalarValue);
        }
    }
    
    return result;
}

Matrix Matrix::apply(const Matrix& m, std::function<ValuePtr(const ValuePtr&)> func) {
    Matrix result(m.rows, m.cols);
    
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            result.at(i, j) = func(m.at(i, j));
        }
    }
    
    return result;
}

Matrix Matrix::relu(const Matrix& m) {
    return apply(m, Value::relu);
}

Matrix Matrix::sigmoid(const Matrix& m) {
    return apply(m, Value::sigmoid);
}

Matrix Matrix::tanh(const Matrix& m) {
    return apply(m, Value::tanh);
}

Matrix Matrix::softmax(const Matrix& m) {
    if (m.cols != 1) {
        throw std::invalid_argument("Softmax can only be applied to column vectors");
    }
    
    // Extract values into a vector
    std::vector<ValuePtr> values;
    values.reserve(m.rows);
    for (size_t i = 0; i < m.rows; ++i) {
        values.push_back(m.at(i, 0));
    }
    
    // Find max for numerical stability
    float max_val = -std::numeric_limits<float>::infinity();
    for (const auto& val : values) {
        max_val = std::max(max_val, val->data);
    }
    
    // Calculate exp(x_i - max) for each input
    std::vector<ValuePtr> exps;
    exps.reserve(values.size());
    for (const auto& val : values) {
        exps.push_back(Value::exp(Value::subtract(val, Value::create(max_val))));
    }
    
    // Calculate sum of exps
    ValuePtr sum_exp = Value::create(0.0f);
    for (const auto& exp_val : exps) {
        sum_exp = Value::add(sum_exp, exp_val);
    }
    
    // Calculate softmax values
    Matrix result(m.rows, 1);
    for (size_t i = 0; i < m.rows; ++i) {
        result.at(i, 0) = Value::divide(exps[i], sum_exp);
    }
    
    return result;
}

void Matrix::zeroGrad() {
    for (auto& row : data) {
        for (auto& val : row) {
            val->grad = 0.0f;
        }
    }
}

std::vector<ValuePtr> Matrix::flatten() const {
    std::vector<ValuePtr> result;
    result.reserve(rows * cols);
    
    for (const auto& row : data) {
        result.insert(result.end(), row.begin(), row.end());
    }
    
    return result;
}

void Matrix::print(const std::string& name) const {
    if (!name.empty()) {
        std::cout << name << " = " << std::endl;
    }
    
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[ ";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << data[i][j]->data;
            if (j < cols - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
}

std::vector<ValuePtr> Matrix::getRow(size_t i) const {
    if (i >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    return data[i];
}

std::vector<ValuePtr> Matrix::getCol(size_t j) const {
    if (j >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    std::vector<ValuePtr> result;
    result.reserve(rows);
    
    for (size_t i = 0; i < rows; ++i) {
        result.push_back(data[i][j]);
    }
    
    return result;
}

void Matrix::setRow(size_t i, const std::vector<ValuePtr>& row) {
    if (i >= rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    
    if (row.size() != cols) {
        throw std::invalid_argument("Row size must match matrix columns");
    }
    
    data[i] = row;
}

void Matrix::setCol(size_t j, const std::vector<ValuePtr>& col) {
    if (j >= cols) {
        throw std::out_of_range("Column index out of bounds");
    }
    
    if (col.size() != rows) {
        throw std::invalid_argument("Column size must match matrix rows");
    }
    
    for (size_t i = 0; i < rows; ++i) {
        data[i][j] = col[i];
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    return elementWiseAdd(*this, other);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return elementWiseSubtract(*this, other);
}

Matrix Matrix::operator*(const Matrix& other) const {
    return matmul(*this, other);
}

Matrix Matrix::operator%(const Matrix& other) const {
    return elementWiseMultiply(*this, other);
}

Matrix Matrix::operator/(const Matrix& other) const {
    return elementWiseDivide(*this, other);
}

ValuePtr dot(const std::vector<ValuePtr>& a, const std::vector<ValuePtr>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same length for dot product");
    }
    
    ValuePtr result = Value::create(0.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        result = Value::add(result, Value::multiply(a[i], b[i]));
    }
    
    return result;
}

} // namespace core
} // namespace finml 