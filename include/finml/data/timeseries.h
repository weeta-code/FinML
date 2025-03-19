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


struct TimeSeriesPoint {
    std::chrono::system_clock::time_point timestamp;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::unordered_map<std::string, double> features;
};


class TimeSeries {
private:
    std::string symbol;
    std::vector<TimeSeriesPoint> data;
    std::vector<std::string> feature_names;

public:
    explicit TimeSeries(const std::string& symbol);

    bool loadFromCSV(const std::string& filename, bool has_header = true, const std::string& date_format = "%Y-%m-%d");
    bool saveToCSV(const std::string& filename) const;
    
    void addDataPoint(const TimeSeriesPoint& point);
    const TimeSeriesPoint& getDataPoint(size_t index) const;
    
    size_t size() const;
    const std::string& getSymbol() const;

    bool calculateIndicator(const std::string& indicator_name, const std::unordered_map<std::string, double>& params = {});
    std::pair<double, double> normalizeZScore(const std::string& feature_name);
    std::pair<double, double> normalizeMinMax(const std::string& feature_name);
    std::pair<TimeSeries, TimeSeries> trainTestSplit(double train_ratio = 0.8) const;
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<double>> createSequences(
        size_t sequence_length,
        const std::string& target_feature,
        const std::vector<std::string>& feature_names = {}
    ) const;
    std::vector<size_t> detectPattern(const std::string& pattern_name, const std::unordered_map<std::string, double>& params = {}) const;
};

} // namespace data
} // namespace finml

#endif // FINML_DATA_TIMESERIES_H 