#include "finml/nn/conv.h"
#include "finml/nn/linear.h"
#include "finml/nn/activation.h"
#include "finml/nn/sequential.h"
#include "finml/core/matrix.h"
#include "finml/optim/adam.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <filesystem>
#include <random>
#include <algorithm>
#include <cmath>

// Simple structure to hold a labeled image
struct LabeledPattern {
    finml::core::Matrix image;
    bool is_bullish;  // true for bullish/up pattern, false for bearish/down pattern
    std::string pattern_name;
};

class StockPatternCNN {
private:
    size_t input_channels;   // Usually 1 for grayscale, 3 for RGB
    size_t input_height;     // Height of input image
    size_t input_width;      // Width of input image
    finml::nn::Sequential model;
    
public:
    StockPatternCNN(
        size_t input_channels = 1,
        size_t input_height = 60,
        size_t input_width = 200
    ) : input_channels(input_channels),
        input_height(input_height),
        input_width(input_width),
        model("StockPatternCNN") {
        
        // Build the CNN model
        
        // First convolutional layer
        auto conv1 = std::make_shared<finml::nn::Conv1D>(
            input_channels, 16, 3, 1, 1, true, "Conv1"
        );
        model.add(conv1);
        
        // ReLU activation
        auto relu1 = std::make_shared<finml::nn::ReLU>("ReLU1");
        model.add(relu1);
        
        // Max pooling
        auto pool1 = std::make_shared<finml::nn::MaxPool1D>(2, 2, "Pool1");
        model.add(pool1);
        
        // Second convolutional layer
        auto conv2 = std::make_shared<finml::nn::Conv1D>(
            16, 32, 3, 1, 1, true, "Conv2"
        );
        model.add(conv2);
        
        // ReLU activation
        auto relu2 = std::make_shared<finml::nn::ReLU>("ReLU2");
        model.add(relu2);
        
        // Max pooling
        auto pool2 = std::make_shared<finml::nn::MaxPool1D>(2, 2, "Pool2");
        model.add(pool2);
        
        // Third convolutional layer
        auto conv3 = std::make_shared<finml::nn::Conv1D>(
            32, 64, 3, 1, 1, true, "Conv3"
        );
        model.add(conv3);
        
        // ReLU activation
        auto relu3 = std::make_shared<finml::nn::ReLU>("ReLU3");
        model.add(relu3);
        
        // Max pooling
        auto pool3 = std::make_shared<finml::nn::MaxPool1D>(2, 2, "Pool3");
        model.add(pool3);
        
        // Calculate the size of the flattened output from the last pooling layer
        size_t flattened_size = 64 * (input_width / 8);
        
        // Fully connected layer
        auto fc1 = std::make_shared<finml::nn::Linear>(flattened_size, 128, true, "FC1");
        model.add(fc1);
        
        // ReLU activation
        auto relu4 = std::make_shared<finml::nn::ReLU>("ReLU4");
        model.add(relu4);
        
        // Output layer
        auto fc2 = std::make_shared<finml::nn::Linear>(128, 1, true, "Output");
        model.add(fc2);
        
        // Sigmoid activation for binary classification
        auto sigmoid = std::make_shared<finml::nn::Sigmoid>("Sigmoid");
        model.add(sigmoid);
        
        // Print model architecture
        model.print();
    }
    
    void train(
        const std::vector<LabeledPattern>& training_data,
        size_t epochs = 50,
        double learning_rate = 0.001,
        size_t batch_size = 16
    ) {
        if (training_data.empty()) {
            std::cerr << "No training data provided" << std::endl;
            return;
        }
        
        // Create optimizer
        finml::optim::Adam optimizer(model.parameters(), learning_rate);
        
        // Training loop
        size_t num_samples = training_data.size();
        size_t batches = (num_samples + batch_size - 1) / batch_size;
        
        std::cout << "Starting training with " << num_samples << " samples, " << batches << " batches" << std::endl;
        
        // Track loss and accuracy
        std::vector<double> epoch_losses;
        std::vector<double> epoch_accuracies;
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Shuffle data
            std::vector<size_t> indices(num_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
            
            double epoch_loss = 0.0;
            size_t correct_predictions = 0;
            
            // Process batches
            for (size_t batch = 0; batch < batches; ++batch) {
                // Reset gradients
                model.zeroGrad();
                
                double batch_loss = 0.0;
                size_t batch_start = batch * batch_size;
                size_t batch_end = std::min((batch + 1) * batch_size, num_samples);
                size_t batch_actual_size = batch_end - batch_start;
                
                // Process each sample in the batch
                for (size_t i = batch_start; i < batch_end; ++i) {
                    size_t idx = indices[i];
                    const LabeledPattern& sample = training_data[idx];
                    
                    // Forward pass
                    finml::core::Matrix output = model.forward(sample.image);
                    
                    // Calculate loss (binary cross entropy)
                    finml::core::Matrix target(1, 1);
                    target.at(0, 0) = finml::core::Value::create(sample.is_bullish ? 1.0f : 0.0f);
                    
                    // Implement binary cross entropy loss directly
                    float y = sample.is_bullish ? 1.0f : 0.0f;
                    float y_pred = output.at(0, 0)->data;
                    // Clip prediction to avoid log(0)
                    y_pred = std::max(std::min(y_pred, 1.0f - 1e-7f), 1e-7f);
                    float loss = -(y * std::log(y_pred) + (1 - y) * std::log(1 - y_pred));
                    
                    // Create loss value
                    finml::core::Matrix loss_val(1, 1);
                    loss_val.at(0, 0) = finml::core::Value::create(loss);
                    double sample_loss = loss_val.at(0, 0)->data;
                    batch_loss += sample_loss;
                    
                    // Track accuracy
                    bool predicted_bullish = output.at(0, 0)->data > 0.5f;
                    if (predicted_bullish == sample.is_bullish) {
                        correct_predictions++;
                    }
                    
                    // Backward pass
                    loss_val.at(0, 0)->backward();
                }
                
                // Update weights
                optimizer.step();
                
                // Average batch loss
                batch_loss /= batch_actual_size;
                epoch_loss += batch_loss;
                
                // Print progress for every 5 batches
                if ((batch + 1) % 5 == 0 || batch == batches - 1) {
                    std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                              << ", Batch " << batch + 1 << "/" << batches 
                              << ", Loss: " << batch_loss << std::endl;
                }
            }
            
            // Average epoch loss
            epoch_loss /= batches;
            epoch_losses.push_back(epoch_loss);
            
            // Calculate accuracy
            double accuracy = static_cast<double>(correct_predictions) / num_samples;
            epoch_accuracies.push_back(accuracy);
            
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " completed, Loss: " << epoch_loss 
                      << ", Accuracy: " << (accuracy * 100.0) << "%" << std::endl;
        }
        
        // Save the training history to a file
        std::ofstream history_file("cnn_training_history.csv");
        if (history_file.is_open()) {
            history_file << "Epoch,Loss,Accuracy" << std::endl;
            for (size_t i = 0; i < epoch_losses.size(); ++i) {
                history_file << i + 1 << "," << epoch_losses[i] << "," << epoch_accuracies[i] << std::endl;
            }
            history_file.close();
            std::cout << "Training history saved to cnn_training_history.csv" << std::endl;
        }
    }
    
    std::pair<bool, float> predict(const finml::core::Matrix& image) {
        // Forward pass
        finml::core::Matrix output = model.forward(image);
        
        // Get prediction probability
        float probability = output.at(0, 0)->data;
        
        // Classify as bullish if probability > 0.5
        bool is_bullish = probability > 0.5f;
        
        return {is_bullish, probability};
    }
    
    void evaluate(const std::vector<LabeledPattern>& test_data) {
        if (test_data.empty()) {
            std::cerr << "No test data provided" << std::endl;
            return;
        }
        
        size_t num_correct = 0;
        size_t total = test_data.size();
        
        // Confusion matrix: [TN, FP, FN, TP]
        std::vector<size_t> confusion_matrix(4, 0);
        
        for (const auto& sample : test_data) {
            // Get prediction
            auto [predicted_bullish, probability] = predict(sample.image);
            
            // Update confusion matrix
            // True Negative: Predicted bearish (0) and is bearish (0)
            // False Positive: Predicted bullish (1) but is bearish (0)
            // False Negative: Predicted bearish (0) but is bullish (1)
            // True Positive: Predicted bullish (1) and is bullish (1)
            if (!predicted_bullish && !sample.is_bullish) {
                confusion_matrix[0]++; // TN
            } else if (predicted_bullish && !sample.is_bullish) {
                confusion_matrix[1]++; // FP
            } else if (!predicted_bullish && sample.is_bullish) {
                confusion_matrix[2]++; // FN
            } else { // predicted_bullish && sample.is_bullish
                confusion_matrix[3]++; // TP
            }
            
            // Count correct predictions
            if (predicted_bullish == sample.is_bullish) {
                num_correct++;
            }
        }
        
        // Calculate metrics
        double accuracy = static_cast<double>(num_correct) / total;
        double precision = static_cast<double>(confusion_matrix[3]) / 
                          (confusion_matrix[3] + confusion_matrix[1] + 1e-10);
        double recall = static_cast<double>(confusion_matrix[3]) / 
                       (confusion_matrix[3] + confusion_matrix[2] + 1e-10);
        double f1_score = 2.0 * precision * recall / (precision + recall + 1e-10);
        
        // Print evaluation results
        std::cout << "Evaluation Results:" << std::endl;
        std::cout << "Total samples: " << total << std::endl;
        std::cout << "Correct predictions: " << num_correct << std::endl;
        std::cout << "Accuracy: " << (accuracy * 100.0) << "%" << std::endl;
        std::cout << "Precision: " << (precision * 100.0) << "%" << std::endl;
        std::cout << "Recall: " << (recall * 100.0) << "%" << std::endl;
        std::cout << "F1 Score: " << (f1_score * 100.0) << "%" << std::endl;
        
        // Print confusion matrix
        std::cout << "\nConfusion Matrix:" << std::endl;
        std::cout << "TN: " << confusion_matrix[0] << " | FP: " << confusion_matrix[1] << std::endl;
        std::cout << "FN: " << confusion_matrix[2] << " | TP: " << confusion_matrix[3] << std::endl;
    }
};

// Helper function to load stock chart pattern data from images
std::vector<LabeledPattern> loadPatternData(const std::string& data_dir) {
    std::vector<LabeledPattern> patterns;
    
    // Define pattern types to look for
    std::vector<std::string> pattern_types = {
        "double_top", "double_bottom", "head_shoulders", "reverse_head_shoulders",
        "bullish_flag", "bearish_flag", "triangle"
    };
    
    // Mapping of pattern types to bullish/bearish
    std::unordered_map<std::string, bool> is_bullish_map = {
        {"double_top", false},
        {"double_bottom", true},
        {"head_shoulders", false},
        {"reverse_head_shoulders", true},
        {"bullish_flag", true},
        {"bearish_flag", false},
        {"triangle_ascending", true},
        {"triangle_descending", false},
        {"triangle_symmetric", false}  // Consider symmetric triangles as bearish for this example
    };
    
    // Check if directory exists
    if (!std::filesystem::exists(data_dir)) {
        std::cerr << "Data directory not found: " << data_dir << std::endl;
        return patterns;
    }
    
    // For demo purposes, create synthetic patterns
    std::cout << "Creating synthetic pattern data..." << std::endl;
    
    // Create dummy data
    for (size_t i = 0; i < 100; ++i) {
        LabeledPattern pattern;
        
        // Create a simple image (1 channel, 60x200)
        pattern.image = finml::core::Matrix(1, 200);
        
        // Fill with random data to simulate a chart
        for (size_t j = 0; j < 200; ++j) {
            // Sine wave with noise for a simple pattern
            float value = std::sin(j * 0.1f) + ((rand() % 100) / 500.0f - 0.1f);
            pattern.image.at(0, j) = finml::core::Value::create(value);
        }
        
        // Randomly assign a pattern type
        size_t pattern_idx = rand() % pattern_types.size();
        pattern.pattern_name = pattern_types[pattern_idx];
        
        // Set whether it's bullish or bearish
        pattern.is_bullish = is_bullish_map[pattern.pattern_name];
        
        patterns.push_back(pattern);
    }
    
    std::cout << "Created " << patterns.size() << " synthetic patterns" << std::endl;
    
    return patterns;
}

// Example of using the StockPatternCNN
void runPatternRecognitionExample() {
    // Load pattern data
    std::vector<LabeledPattern> pattern_data = loadPatternData("data/patterns");
    
    // Split into training and test sets (80% train, 20% test)
    size_t train_size = pattern_data.size() * 0.8;
    std::vector<LabeledPattern> train_data(pattern_data.begin(), pattern_data.begin() + train_size);
    std::vector<LabeledPattern> test_data(pattern_data.begin() + train_size, pattern_data.end());
    
    std::cout << "Training set size: " << train_data.size() << std::endl;
    std::cout << "Test set size: " << test_data.size() << std::endl;
    
    // Create and train model
    StockPatternCNN cnn_model(1, 60, 200);
    
    std::cout << "Training CNN model..." << std::endl;
    cnn_model.train(train_data, 20, 0.001, 8);
    
    // Evaluate model
    std::cout << "Evaluating model..." << std::endl;
    cnn_model.evaluate(test_data);
    
    // Sample prediction
    if (!test_data.empty()) {
        const auto& sample = test_data[0];
        auto [predicted_bullish, probability] = cnn_model.predict(sample.image);
        
        std::cout << "\nSample Prediction:" << std::endl;
        std::cout << "Pattern: " << sample.pattern_name << std::endl;
        std::cout << "Actual: " << (sample.is_bullish ? "Bullish" : "Bearish") << std::endl;
        std::cout << "Predicted: " << (predicted_bullish ? "Bullish" : "Bearish") << std::endl;
        std::cout << "Confidence: " << (probability * 100.0) << "%" << std::endl;
    }
}

int main() {
    runPatternRecognitionExample();
    return 0;
} 