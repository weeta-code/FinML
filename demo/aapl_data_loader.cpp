#include "finml/data/timeseries.h"
#include <iostream>
#include <fstream>
#include <string>
#include <curl/curl.h>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <filesystem>

// Callback function for curl to write data
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to download Yahoo Finance data
bool downloadYahooFinanceData(const std::string& symbol, 
                             const std::string& start_date, 
                             const std::string& end_date, 
                             const std::string& output_file) {
    // Initialize curl
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize curl" << std::endl;
        return false;
    }
    
    // Convert dates to Unix timestamp
    std::tm start_tm = {}, end_tm = {};
    std::istringstream start_ss(start_date), end_ss(end_date);
    start_ss >> std::get_time(&start_tm, "%Y-%m-%d");
    end_ss >> std::get_time(&end_tm, "%Y-%m-%d");
    
    std::time_t start_time = std::mktime(&start_tm);
    std::time_t end_time = std::mktime(&end_tm);
    
    std::string url = "https://query1.finance.yahoo.com/v7/finance/download/" + 
                     symbol + "?period1=" + std::to_string(start_time) + 
                     "&period2=" + std::to_string(end_time) + 
                     "&interval=1d&events=history";
    
    // Setup curl request
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    // Check for errors
    if (res != CURLE_OK) {
        std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    // Write response to file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        curl_easy_cleanup(curl);
        return false;
    }
    
    out_file << response_data;
    out_file.close();
    
    // Cleanup
    curl_easy_cleanup(curl);
    
    std::cout << "Data downloaded successfully to " << output_file << std::endl;
    return true;
}

// Function to load AAPL data into a TimeSeries object
finml::data::TimeSeries loadAppleStockData(const std::string& csv_file) {
    // Create TimeSeries object
    finml::data::TimeSeries apple_data("AAPL");
    
    // Load data from CSV file
    if (!apple_data.loadFromCSV(csv_file)) {
        throw std::runtime_error("Failed to load data from " + csv_file);
    }
    
    // Calculate technical indicators
    apple_data.calculateIndicator("SMA", {{"period", 20}});   // 20-day Simple Moving Average
    apple_data.calculateIndicator("RSI", {{"period", 14}});   // 14-day Relative Strength Index
    apple_data.calculateIndicator("MACD", {
        {"fast_period", 12}, 
        {"slow_period", 26}, 
        {"signal_period", 9}
    });  // Moving Average Convergence Divergence
    
    // Normalize features
    apple_data.normalizeZScore("close");
    apple_data.normalizeZScore("volume");
    
    return apple_data;
}

// Main function
int main() {
    // Initialize curl globally
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    // Create data directory if it doesn't exist
    std::filesystem::create_directories("data");
    
    // Download AAPL data from 2023-01-01 to 2024-05-01
    const std::string output_file = "data/aapl_2023_2024.csv";
    if (!std::filesystem::exists(output_file)) {
        downloadYahooFinanceData("AAPL", "2023-01-01", "2024-05-01", output_file);
    } else {
        std::cout << "Data file already exists, skipping download" << std::endl;
    }
    
    // Load data into TimeSeries object
    finml::data::TimeSeries apple_data = loadAppleStockData(output_file);
    
    // Print some stats
    std::cout << "Loaded " << apple_data.size() << " data points for " << apple_data.getSymbol() << std::endl;
    
    // Split into training and test sets
    auto [train_data, test_data] = apple_data.trainTestSplit(0.8);
    std::cout << "Training set size: " << train_data.size() << std::endl;
    std::cout << "Test set size: " << test_data.size() << std::endl;
    
    // Create sequences for LSTM training (sequence_length=30, predict 'close' price)
    // Use the following features: open, high, low, close, volume, SMA, RSI, MACD
    auto [X_train, y_train] = train_data.createSequences(
        30, "close", {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14", "MACD"}
    );
    
    auto [X_test, y_test] = test_data.createSequences(
        30, "close", {"open", "high", "low", "close", "volume", "SMA_20", "RSI_14", "MACD"}
    );
    
    // Print sequence dimensions
    std::cout << "X_train size: " << X_train.size() << " sequences, each with " 
              << X_train[0].size() << " time steps and " 
              << X_train[0][0].size() << " features" << std::endl;
    
    std::cout << "y_train size: " << y_train.size() << std::endl;
    
    // Detect patterns in the data
    std::vector<size_t> double_top_patterns = apple_data.detectPattern("double_top", {{"threshold", 0.02}});
    std::vector<size_t> double_bottom_patterns = apple_data.detectPattern("double_bottom", {{"threshold", 0.02}});
    
    std::cout << "Detected " << double_top_patterns.size() << " double top patterns" << std::endl;
    std::cout << "Detected " << double_bottom_patterns.size() << " double bottom patterns" << std::endl;
    
    // Cleanup
    curl_global_cleanup();
    
    return 0;
} 