#include "finml/data/timeseries.h"
#include "finml/core/matrix.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <numeric>
#include <random>

namespace finml {
namespace data {

// Helper function to parse date string
std::chrono::system_clock::time_point parseDate(const std::string& date_str, const std::string& date_format) {
    std::tm tm = {};
    
    // Default format is yyyy-MM-dd
    if (date_format.empty() || date_format == "%Y-%m-%d") {
        // Parse YYYY-MM-DD format manually
        int year = 0, month = 0, day = 0;
        
        // Try to parse the date with more detailed error messages
        std::string::size_type firstDash = date_str.find('-');
        std::string::size_type secondDash = date_str.find('-', firstDash + 1);
        
        if (firstDash == std::string::npos || secondDash == std::string::npos) {
            throw std::invalid_argument("Failed to parse date '" + date_str + "', expected format 'YYYY-MM-DD'");
        }
        
        year = std::stoi(date_str.substr(0, firstDash));
        month = std::stoi(date_str.substr(firstDash + 1, secondDash - firstDash - 1));
        day = std::stoi(date_str.substr(secondDash + 1));
        
        // Validate the date components
        if (year < 1900 || year > 2100 || month < 1 || month > 12 || day < 1 || day > 31) {
            throw std::invalid_argument("Invalid date components in '" + date_str + "'");
        }
        
        tm.tm_year = year - 1900; // Adjust year (tm_year is years since 1900)
        tm.tm_mon = month - 1;    // Adjust month (tm_mon is 0-11)
        tm.tm_mday = day;         // Day of month (1-31)
    } else {
        // Use the specified format
        std::istringstream date_ss(date_str);
        date_ss >> std::get_time(&tm, date_format.c_str());
        if (date_ss.fail()) {
            throw std::invalid_argument("Failed to parse date: '" + date_str + "' with format '" + date_format + "'");
        }
    }
    
    // Default to noon to avoid timezone issues
    tm.tm_hour = 12;
    tm.tm_min = 0;
    tm.tm_sec = 0;
    
    // Validate the parsed time
    std::time_t time = std::mktime(&tm);
    if (time == -1) {
        throw std::invalid_argument("Invalid date/time from '" + date_str + "'");
    }
    
    return std::chrono::system_clock::from_time_t(time);
}

TimeSeries::TimeSeries(const std::string& symbol) : symbol(symbol) {}

bool TimeSeries::loadFromCSV(const std::string& filename, bool has_header, const std::string& date_format) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    std::string line;
    
    // Skip header if needed
    if (has_header) {
        std::getline(file, line);
    }
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        
        // Parse date
        std::getline(ss, token, ',');
        
        try {
            auto time_point = parseDate(token, date_format);
            
            // Parse OHLCV
            TimeSeriesPoint point;
            point.timestamp = time_point;
            
            std::getline(ss, token, ',');
            point.open = std::stod(token);
            
            std::getline(ss, token, ',');
            point.high = std::stod(token);
            
            std::getline(ss, token, ',');
            point.low = std::stod(token);
            
            std::getline(ss, token, ',');
            point.close = std::stod(token);
            
            std::getline(ss, token, ',');
            point.volume = std::stod(token);
            
            // Add data point
            data.push_back(point);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing data: " << e.what() << " on line: " << line << std::endl;
            continue;
        }
    }
    
    if (data.empty()) {
        std::cerr << "Warning: No data points were loaded from " << filename << std::endl;
    } else {
        std::cout << "Successfully loaded " << data.size() << " data points from " << filename << std::endl;
    }
    
    return !data.empty();
}

bool TimeSeries::saveToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    // Write header
    file << "Date,Open,High,Low,Close,Volume";
    for (const auto& feature : feature_names) {
        file << "," << feature;
    }
    file << std::endl;
    
    // Write data
    for (const auto& point : data) {
        auto time_t = std::chrono::system_clock::to_time_t(point.timestamp);
        std::tm tm = *std::localtime(&time_t);
        
        file << std::put_time(&tm, "%Y-%m-%d") << ",";
        file << point.open << ",";
        file << point.high << ",";
        file << point.low << ",";
        file << point.close << ",";
        file << point.volume;
        
        for (const auto& feature : feature_names) {
            auto it = point.features.find(feature);
            if (it != point.features.end()) {
                file << "," << it->second;
            } else {
                file << ",";
            }
        }
        
        file << std::endl;
    }
    
    return true;
}

void TimeSeries::addDataPoint(const TimeSeriesPoint& point) {
    data.push_back(point);
    
    // Update feature names
    for (const auto& feature : point.features) {
        if (std::find(feature_names.begin(), feature_names.end(), feature.first) == feature_names.end()) {
            feature_names.push_back(feature.first);
        }
    }
}

const TimeSeriesPoint& TimeSeries::getDataPoint(size_t index) const {
    if (index >= data.size()) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

size_t TimeSeries::size() const {
    return data.size();
}

const std::string& TimeSeries::getSymbol() const {
    return symbol;
}

bool TimeSeries::calculateIndicator(const std::string& indicator_name, const std::unordered_map<std::string, double>& params) {
    if (data.empty()) {
        return false;
    }
    
    // Add the indicator to feature names if not already present
    if (std::find(feature_names.begin(), feature_names.end(), indicator_name) == feature_names.end()) {
        feature_names.push_back(indicator_name);
    }
    
    if (indicator_name == "SMA") {
        // Simple Moving Average
        size_t period = static_cast<size_t>(params.at("period"));
        if (period >= data.size()) {
            return false;
        }
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < period - 1) {
                data[i].features[indicator_name] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double sum = 0.0;
                for (size_t j = 0; j < period; ++j) {
                    sum += data[i - j].close;
                }
                data[i].features[indicator_name] = sum / period;
            }
        }
        
        return true;
    } else if (indicator_name == "EMA") {
        // Exponential Moving Average
        size_t period = static_cast<size_t>(params.at("period"));
        if (period >= data.size()) {
            return false;
        }
        
        double alpha = 2.0 / (period + 1.0);
        double ema = data[0].close;
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < period - 1) {
                data[i].features[indicator_name] = std::numeric_limits<double>::quiet_NaN();
            } else if (i == period - 1) {
                double sum = 0.0;
                for (size_t j = 0; j < period; ++j) {
                    sum += data[j].close;
                }
                ema = sum / period;
                data[i].features[indicator_name] = ema;
            } else {
                ema = alpha * data[i].close + (1.0 - alpha) * ema;
                data[i].features[indicator_name] = ema;
            }
        }
        
        return true;
    } else if (indicator_name == "RSI") {
        // Relative Strength Index
        size_t period = static_cast<size_t>(params.at("period"));
        if (period >= data.size()) {
            return false;
        }
        
        std::vector<double> gains;
        std::vector<double> losses;
        
        for (size_t i = 1; i < data.size(); ++i) {
            double change = data[i].close - data[i - 1].close;
            if (change > 0) {
                gains.push_back(change);
                losses.push_back(0.0);
            } else {
                gains.push_back(0.0);
                losses.push_back(-change);
            }
        }
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < period) {
                data[i].features[indicator_name] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double avg_gain = 0.0;
                double avg_loss = 0.0;
                
                for (size_t j = 0; j < period; ++j) {
                    avg_gain += gains[i - period + j];
                    avg_loss += losses[i - period + j];
                }
                
                avg_gain /= period;
                avg_loss /= period;
                
                if (avg_loss == 0.0) {
                    data[i].features[indicator_name] = 100.0;
                } else {
                    double rs = avg_gain / avg_loss;
                    data[i].features[indicator_name] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }
        
        return true;
    } else if (indicator_name == "MACD") {
        // Moving Average Convergence Divergence
        size_t fast_period = static_cast<size_t>(params.at("fast_period"));
        size_t slow_period = static_cast<size_t>(params.at("slow_period"));
        size_t signal_period = static_cast<size_t>(params.at("signal_period"));
        
        if (slow_period >= data.size()) {
            return false;
        }
        
        // Calculate fast EMA
        std::unordered_map<std::string, double> fast_params = {{"period", static_cast<double>(fast_period)}};
        calculateIndicator("EMA_fast", fast_params);
        
        // Calculate slow EMA
        std::unordered_map<std::string, double> slow_params = {{"period", static_cast<double>(slow_period)}};
        calculateIndicator("EMA_slow", slow_params);
        
        // Calculate MACD line
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < slow_period - 1) {
                data[i].features[indicator_name] = std::numeric_limits<double>::quiet_NaN();
            } else {
                data[i].features[indicator_name] = data[i].features["EMA_fast"] - data[i].features["EMA_slow"];
            }
        }
        
        // Calculate signal line
        std::vector<double> macd_line;
        for (size_t i = slow_period - 1; i < data.size(); ++i) {
            macd_line.push_back(data[i].features[indicator_name]);
        }
        
        double alpha = 2.0 / (signal_period + 1.0);
        double signal = std::accumulate(macd_line.begin(), macd_line.begin() + signal_period, 0.0) / signal_period;
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < slow_period + signal_period - 2) {
                data[i].features[indicator_name + "_signal"] = std::numeric_limits<double>::quiet_NaN();
            } else if (i == slow_period + signal_period - 2) {
                data[i].features[indicator_name + "_signal"] = signal;
            } else {
                signal = alpha * data[i].features[indicator_name] + (1.0 - alpha) * signal;
                data[i].features[indicator_name + "_signal"] = signal;
            }
        }
        
        // Calculate histogram
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < slow_period + signal_period - 2) {
                data[i].features[indicator_name + "_hist"] = std::numeric_limits<double>::quiet_NaN();
            } else {
                data[i].features[indicator_name + "_hist"] = data[i].features[indicator_name] - data[i].features[indicator_name + "_signal"];
            }
        }
        
        // Add signal and histogram to feature names
        if (std::find(feature_names.begin(), feature_names.end(), indicator_name + "_signal") == feature_names.end()) {
            feature_names.push_back(indicator_name + "_signal");
        }
        if (std::find(feature_names.begin(), feature_names.end(), indicator_name + "_hist") == feature_names.end()) {
            feature_names.push_back(indicator_name + "_hist");
        }
        
        return true;
    } else if (indicator_name == "Bollinger") {
        // Bollinger Bands
        size_t period = static_cast<size_t>(params.at("period"));
        double num_std_dev = params.at("num_std_dev");
        
        if (period >= data.size()) {
            return false;
        }
        
        // Calculate SMA
        std::unordered_map<std::string, double> sma_params = {{"period", static_cast<double>(period)}};
        calculateIndicator("SMA", sma_params);
        
        // Calculate standard deviation
        for (size_t i = 0; i < data.size(); ++i) {
            if (i < period - 1) {
                data[i].features[indicator_name + "_upper"] = std::numeric_limits<double>::quiet_NaN();
                data[i].features[indicator_name + "_lower"] = std::numeric_limits<double>::quiet_NaN();
            } else {
                double sum_sq_diff = 0.0;
                for (size_t j = 0; j < period; ++j) {
                    double diff = data[i - j].close - data[i].features["SMA"];
                    sum_sq_diff += diff * diff;
                }
                double std_dev = std::sqrt(sum_sq_diff / period);
                
                data[i].features[indicator_name + "_upper"] = data[i].features["SMA"] + num_std_dev * std_dev;
                data[i].features[indicator_name + "_lower"] = data[i].features["SMA"] - num_std_dev * std_dev;
            }
        }
        
        // Add upper and lower bands to feature names
        if (std::find(feature_names.begin(), feature_names.end(), indicator_name + "_upper") == feature_names.end()) {
            feature_names.push_back(indicator_name + "_upper");
        }
        if (std::find(feature_names.begin(), feature_names.end(), indicator_name + "_lower") == feature_names.end()) {
            feature_names.push_back(indicator_name + "_lower");
        }
        
        return true;
    }
    
    return false;
}

std::pair<double, double> TimeSeries::normalizeZScore(const std::string& feature_name) {
    if (data.empty()) {
        return {0.0, 1.0};
    }
    
    // Calculate mean
    double sum = 0.0;
    size_t count = 0;
    
    for (const auto& point : data) {
        auto it = point.features.find(feature_name);
        if (it != point.features.end() && !std::isnan(it->second)) {
            sum += it->second;
            ++count;
        }
    }
    
    if (count == 0) {
        return {0.0, 1.0};
    }
    
    double mean = sum / count;
    
    // Calculate standard deviation
    double sum_sq_diff = 0.0;
    
    for (const auto& point : data) {
        auto it = point.features.find(feature_name);
        if (it != point.features.end() && !std::isnan(it->second)) {
            double diff = it->second - mean;
            sum_sq_diff += diff * diff;
        }
    }
    
    double std_dev = std::sqrt(sum_sq_diff / count);
    
    if (std_dev == 0.0) {
        return {mean, 1.0};
    }
    
    // Normalize data
    for (auto& point : data) {
        auto it = point.features.find(feature_name);
        if (it != point.features.end() && !std::isnan(it->second)) {
            it->second = (it->second - mean) / std_dev;
        }
    }
    
    return {mean, std_dev};
}

std::pair<double, double> TimeSeries::normalizeMinMax(const std::string& feature_name) {
    if (data.empty()) {
        return {0.0, 1.0};
    }
    
    // Find min and max
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::lowest();
    
    for (const auto& point : data) {
        auto it = point.features.find(feature_name);
        if (it != point.features.end() && !std::isnan(it->second)) {
            min_val = std::min(min_val, it->second);
            max_val = std::max(max_val, it->second);
        }
    }
    
    if (min_val == max_val) {
        return {min_val, max_val};
    }
    
    // Normalize data
    for (auto& point : data) {
        auto it = point.features.find(feature_name);
        if (it != point.features.end() && !std::isnan(it->second)) {
            it->second = (it->second - min_val) / (max_val - min_val);
        }
    }
    
    return {min_val, max_val};
}

std::pair<TimeSeries, TimeSeries> TimeSeries::trainTestSplit(double train_ratio) const {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("Train ratio must be between 0 and 1");
    }
    
    TimeSeries train_set(symbol);
    TimeSeries test_set(symbol);
    
    // Create indices and shuffle using matrix random generator
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), core::getRandomEngine());
    
    size_t train_size = static_cast<size_t>(data.size() * train_ratio);
    
    // Split data into train and test sets
    for (size_t i = 0; i < train_size; ++i) {
        train_set.data.push_back(data[indices[i]]);
    }
    
    for (size_t i = train_size; i < data.size(); ++i) {
        test_set.data.push_back(data[indices[i]]);
    }
    
    return {train_set, test_set};
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<double>> TimeSeries::createSequences(
    size_t sequence_length,
    const std::string& target_feature,
    const std::vector<std::string>& feature_names
) const {
    std::vector<std::vector<std::vector<double>>> sequences;
    std::vector<double> targets;
    
    if (data.size() <= sequence_length) {
        return {sequences, targets};
    }
    
    std::vector<std::string> features_to_use;
    if (feature_names.empty()) {
        // Use all features
        features_to_use = this->feature_names;
    } else {
        features_to_use = feature_names;
    }
    
    for (size_t i = 0; i <= data.size() - sequence_length - 1; ++i) {
        std::vector<std::vector<double>> sequence;
        
        for (size_t j = 0; j < sequence_length; ++j) {
            std::vector<double> features;
            
            // Add OHLCV data
            features.push_back(data[i + j].open);
            features.push_back(data[i + j].high);
            features.push_back(data[i + j].low);
            features.push_back(data[i + j].close);
            features.push_back(data[i + j].volume);
            
            // Add custom features
            for (const auto& feature : features_to_use) {
                auto it = data[i + j].features.find(feature);
                if (it != data[i + j].features.end()) {
                    features.push_back(it->second);
                } else {
                    features.push_back(0.0);
                }
            }
            
            sequence.push_back(features);
        }
        
        sequences.push_back(sequence);
        
        // Add target
        auto it = data[i + sequence_length].features.find(target_feature);
        if (it != data[i + sequence_length].features.end()) {
            targets.push_back(it->second);
        } else {
            targets.push_back(data[i + sequence_length].close);
        }
    }
    
    return {sequences, targets};
}

std::vector<size_t> TimeSeries::detectPattern(const std::string& pattern_name, const std::unordered_map<std::string, double>& params) const {
    std::vector<size_t> indices;
    
    if (data.size() < 5) {
        return indices;
    }
    
    if (pattern_name == "HeadAndShoulders") {
        // Head and Shoulders pattern detection
        double threshold = params.at("threshold");
        
        for (size_t i = 4; i < data.size(); ++i) {
            // Check for left shoulder, head, and right shoulder
            if (data[i - 4].close < data[i - 3].close &&
                data[i - 3].close > data[i - 2].close &&
                data[i - 2].close < data[i - 1].close &&
                data[i - 1].close > data[i].close &&
                data[i - 3].close < data[i - 1].close &&
                std::abs(data[i - 4].close - data[i].close) < threshold) {
                indices.push_back(i);
            }
        }
    } else if (pattern_name == "DoubleTop") {
        // Double Top pattern detection
        double threshold = params.at("threshold");
        
        for (size_t i = 4; i < data.size(); ++i) {
            // Check for two peaks with similar heights
            if (data[i - 4].close < data[i - 3].close &&
                data[i - 3].close > data[i - 2].close &&
                data[i - 2].close < data[i - 1].close &&
                data[i - 1].close > data[i].close &&
                std::abs(data[i - 3].close - data[i - 1].close) < threshold) {
                indices.push_back(i);
            }
        }
    } else if (pattern_name == "DoubleBottom") {
        // Double Bottom pattern detection
        double threshold = params.at("threshold");
        
        for (size_t i = 4; i < data.size(); ++i) {
            // Check for two troughs with similar heights
            if (data[i - 4].close > data[i - 3].close &&
                data[i - 3].close < data[i - 2].close &&
                data[i - 2].close > data[i - 1].close &&
                data[i - 1].close < data[i].close &&
                std::abs(data[i - 3].close - data[i - 1].close) < threshold) {
                indices.push_back(i);
            }
        }
    } else if (pattern_name == "BullishEngulfing") {
        // Bullish Engulfing pattern detection
        for (size_t i = 1; i < data.size(); ++i) {
            // Check for bearish candle followed by bullish candle that engulfs it
            if (data[i - 1].close < data[i - 1].open &&
                data[i].close > data[i].open &&
                data[i].open < data[i - 1].close &&
                data[i].close > data[i - 1].open) {
                indices.push_back(i);
            }
        }
    } else if (pattern_name == "BearishEngulfing") {
        // Bearish Engulfing pattern detection
        for (size_t i = 1; i < data.size(); ++i) {
            // Check for bullish candle followed by bearish candle that engulfs it
            if (data[i - 1].close > data[i - 1].open &&
                data[i].close < data[i].open &&
                data[i].open > data[i - 1].close &&
                data[i].close < data[i - 1].open) {
                indices.push_back(i);
            }
        }
    }
    
    return indices;
}

} // namespace data
} // namespace finml 