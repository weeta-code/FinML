#ifndef FINML_DATA_TIMESERIES_H
#define FINML_DATA_TIMESERIES_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace finml {
namespace data {

/**
 * @brief Represents a single data point in a time series
 */
struct TimeSeriesPoint {
    std::chrono::system_clock::time_point timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::unordered_map<std::string, double> features;
};

/**
 * @brief Class for handling financial time series data
 */
class TimeSeries {
private:
    std::string symbol;
    std::vector<TimeSeriesPoint> data;
    std::vector<std::string> feature_names;

public:
    /**
     * @brief Construct a new TimeSeries object
     * 
     * @param symbol Symbol of the financial instrument
     */
    explicit TimeSeries(const std::string& symbol);
    
    /**
     * @brief Load data from a CSV file
     * 
     * @param filename Path to the CSV file
     * @param has_header Whether the CSV file has a header row
     * @param date_format Format of the date column
     * @return bool Whether the data was loaded successfully
     */
    bool loadFromCSV(const std::string& filename, bool has_header = true, const std::string& date_format = "%Y-%m-%d");
    
    /**
     * @brief Save data to a CSV file
     * 
     * @param filename Path to the CSV file
     * @return bool Whether the data was saved successfully
     */
    bool saveToCSV(const std::string& filename) const;
    
    /**
     * @brief Add a data point to the time series
     * 
     * @param point Data point to add
     */
    void addDataPoint(const TimeSeriesPoint& point);
    
    /**
     * @brief Get the data point at the given index
     * 
     * @param index Index of the data point
     * @return const TimeSeriesPoint& Data point at the given index
     */
    const TimeSeriesPoint& getDataPoint(size_t index) const;
    
    /**
     * @brief Get the number of data points in the time series
     * 
     * @return size_t Number of data points
     */
    size_t size() const;
    
    /**
     * @brief Get the symbol of the financial instrument
     * 
     * @return const std::string& Symbol
     */
    const std::string& getSymbol() const;
    
    /**
     * @brief Calculate technical indicators and add them as features
     * 
     * @param indicator_name Name of the indicator
     * @param params Parameters for the indicator calculation
     * @return bool Whether the indicator was calculated successfully
     */
    bool calculateIndicator(const std::string& indicator_name, const std::unordered_map<std::string, double>& params = {});
    
    /**
     * @brief Normalize the data using z-score normalization
     * 
     * @param feature_name Name of the feature to normalize
     * @return std::pair<double, double> Mean and standard deviation of the feature
     */
    std::pair<double, double> normalizeZScore(const std::string& feature_name);
    
    /**
     * @brief Normalize the data using min-max normalization
     * 
     * @param feature_name Name of the feature to normalize
     * @return std::pair<double, double> Min and max of the feature
     */
    std::pair<double, double> normalizeMinMax(const std::string& feature_name);
    
    /**
     * @brief Split the data into training and testing sets
     * 
     * @param train_ratio Ratio of data to use for training
     * @return std::pair<TimeSeries, TimeSeries> Training and testing sets
     */
    std::pair<TimeSeries, TimeSeries> trainTestSplit(double train_ratio = 0.8) const;
    
    /**
     * @brief Create sequences of data for sequence modeling
     * 
     * @param sequence_length Length of each sequence
     * @param target_feature Name of the target feature
     * @param feature_names Names of the features to include
     * @return std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<double>> Sequences and targets
     */
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<double>> createSequences(
        size_t sequence_length,
        const std::string& target_feature,
        const std::vector<std::string>& feature_names = {}
    ) const;
    
    /**
     * @brief Detect patterns in the time series
     * 
     * @param pattern_name Name of the pattern to detect
     * @param params Parameters for the pattern detection
     * @return std::vector<size_t> Indices where the pattern was detected
     */
    std::vector<size_t> detectPattern(const std::string& pattern_name, const std::unordered_map<std::string, double>& params = {}) const;
};

} // namespace data
} // namespace finml

#endif // FINML_DATA_TIMESERIES_H 