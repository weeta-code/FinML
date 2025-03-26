#ifndef FINML_OPTIM_LOSS_H
#define FINML_OPTIM_LOSS_H

#include "finml/core/matrix.h"
#include "finml/core/value.h"
#include <vector>
#include <memory>
#include <cmath>

namespace finml {
namespace optim {

class Loss {
public:
    virtual ~Loss() = default;
    
    /**
     * Calculate the loss between predictions and targets
     * @param predictions The prediction matrix
     * @param targets The target matrix
     * @return A Value representing the loss
     */
    virtual core::ValuePtr forward(const core::Matrix& predictions, const core::Matrix& targets) const = 0;
    
    /**
     * Calculate the gradient of the loss with respect to the predictions
     * @param predictions The prediction matrix
     * @param targets The target matrix
     * @return A Matrix representing the gradient
     */
    virtual core::Matrix backward(const core::Matrix& predictions, const core::Matrix& targets) const = 0;
};

class MSELoss : public Loss {
public:
    MSELoss() = default;
    
    core::ValuePtr forward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        if (predictions.numRows() != targets.numRows() || predictions.numCols() != targets.numCols()) {
            throw std::invalid_argument("Predictions and targets dimensions do not match");
        }
        
        core::ValuePtr sum = core::Value::create(0.0f, "sum");
        size_t n = predictions.numRows() * predictions.numCols();
        
        for (size_t i = 0; i < predictions.numRows(); ++i) {
            for (size_t j = 0; j < predictions.numCols(); ++j) {
                core::ValuePtr diff = core::Value::subtract(predictions.at(i, j), targets.at(i, j));
                core::ValuePtr squared = core::Value::multiply(diff, diff);
                sum = core::Value::add(sum, squared);
            }
        }
        
        return core::Value::divide(sum, core::Value::create(static_cast<float>(n), "n"));
    }
    
    core::Matrix backward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        if (predictions.numRows() != targets.numRows() || predictions.numCols() != targets.numCols()) {
            throw std::invalid_argument("Predictions and targets dimensions do not match");
        }
        
        core::Matrix grad(predictions.numRows(), predictions.numCols());
        float n = static_cast<float>(predictions.numRows() * predictions.numCols());
        
        for (size_t i = 0; i < predictions.numRows(); ++i) {
            for (size_t j = 0; j < predictions.numCols(); ++j) {
                core::ValuePtr diff = core::Value::subtract(predictions.at(i, j), targets.at(i, j));
                grad.at(i, j) = core::Value::multiply(diff, core::Value::create(2.0f / n, "scale"));
            }
        }
        
        return grad;
    }
};

class RMSELoss : public Loss {
public:
    RMSELoss() = default;
    
    core::ValuePtr forward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        core::ValuePtr mse = MSELoss().forward(predictions, targets);
        return core::Value::pow(mse, 0.5f);
    }
    
    core::Matrix backward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        if (predictions.numRows() != targets.numRows() || predictions.numCols() != targets.numCols()) {
            throw std::invalid_argument("Predictions and targets dimensions do not match");
        }
        
        core::Matrix grad = MSELoss().backward(predictions, targets);
        core::ValuePtr mse = MSELoss().forward(predictions, targets);
        core::ValuePtr rmse = core::Value::pow(mse, 0.5f);
        
        float scale = 0.5f / rmse->data;
        
        for (size_t i = 0; i < grad.numRows(); ++i) {
            for (size_t j = 0; j < grad.numCols(); ++j) {
                grad.at(i, j) = core::Value::multiply(grad.at(i, j), core::Value::create(scale, "scale"));
            }
        }
        
        return grad;
    }
};

class MAELoss : public Loss {
public:
    MAELoss() = default;
    
    core::ValuePtr forward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        if (predictions.numRows() != targets.numRows() || predictions.numCols() != targets.numCols()) {
            throw std::invalid_argument("Predictions and targets dimensions do not match");
        }
        
        core::ValuePtr sum = core::Value::create(0.0f, "sum");
        size_t n = predictions.numRows() * predictions.numCols();
        
        for (size_t i = 0; i < predictions.numRows(); ++i) {
            for (size_t j = 0; j < predictions.numCols(); ++j) {
                core::ValuePtr diff = core::Value::subtract(predictions.at(i, j), targets.at(i, j));
                core::ValuePtr abs_diff = core::Value::pow(
                    core::Value::add(
                        core::Value::multiply(diff, diff),
                        core::Value::create(1e-10f, "epsilon")
                    ),
                    0.5f
                );
                sum = core::Value::add(sum, abs_diff);
            }
        }
        
        return core::Value::divide(sum, core::Value::create(static_cast<float>(n), "n"));
    }
    
    core::Matrix backward(const core::Matrix& predictions, const core::Matrix& targets) const override {
        if (predictions.numRows() != targets.numRows() || predictions.numCols() != targets.numCols()) {
            throw std::invalid_argument("Predictions and targets dimensions do not match");
        }
        
        core::Matrix grad(predictions.numRows(), predictions.numCols());
        float n = static_cast<float>(predictions.numRows() * predictions.numCols());
        
        for (size_t i = 0; i < predictions.numRows(); ++i) {
            for (size_t j = 0; j < predictions.numCols(); ++j) {
                core::ValuePtr diff = core::Value::subtract(predictions.at(i, j), targets.at(i, j));
                float abs_val = std::sqrt(diff->data * diff->data + 1e-10f);
                float sign = diff->data >= 0 ? 1.0f : -1.0f;
                grad.at(i, j) = core::Value::create(sign / n, "grad");
            }
        }
        
        return grad;
    }
};

// Utility functions to compute loss values directly without creating Loss objects
inline core::ValuePtr mse_loss(const core::Matrix& predictions, const core::Matrix& targets) {
    return MSELoss().forward(predictions, targets);
}

inline core::ValuePtr rmse_loss(const core::Matrix& predictions, const core::Matrix& targets) {
    return RMSELoss().forward(predictions, targets);
}

inline core::ValuePtr mae_loss(const core::Matrix& predictions, const core::Matrix& targets) {
    return MAELoss().forward(predictions, targets);
}

// Helper function to compute metrics from raw values
inline double computeRMSE(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) {
        throw std::invalid_argument("Invalid input vectors");
    }
    
    double sum_squared_error = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (std::isnan(predictions[i]) || std::isnan(targets[i])) {
            continue;
        }
        double error = predictions[i] - targets[i];
        sum_squared_error += error * error;
        count++;
    }
    
    if (count == 0) return 0.0;
    return std::sqrt(sum_squared_error / count);
}

inline double computeMAE(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) {
        throw std::invalid_argument("Invalid input vectors");
    }
    
    double sum_abs_error = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < predictions.size(); ++i) {
        if (std::isnan(predictions[i]) || std::isnan(targets[i])) {
            continue;
        }
        sum_abs_error += std::fabs(predictions[i] - targets[i]);
        count++;
    }
    
    if (count == 0) return 0.0;
    return sum_abs_error / count;
}

inline double computePercentageError(const std::vector<double>& predictions, const std::vector<double>& targets) {
    if (predictions.size() != targets.size() || predictions.empty()) {
        throw std::invalid_argument("Invalid input vectors");
    }
    
    double rmse = computeRMSE(predictions, targets);
    
    double sum_targets = 0.0;
    size_t count = 0;
    
    for (size_t i = 0; i < targets.size(); ++i) {
        if (!std::isnan(targets[i])) {
            sum_targets += std::fabs(targets[i]);
            count++;
        }
    }
    
    if (count == 0) return 0.0;
    double mean_target = sum_targets / count;
    
    return (mean_target != 0.0) ? (rmse / mean_target) * 100.0 : 0.0;
}

} // namespace optim
} // namespace finml

#endif // FINML_OPTIM_LOSS_H 