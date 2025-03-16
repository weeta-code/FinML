# FinML: Financial Machine Learning Library

FinML is a C++ library for financial time series analysis, neural networks, and options trading. It provides tools for building and training neural networks (including LSTM and CNN), analyzing financial time series data, and pricing options.

## Features

- **Neural Networks**
  - Dense (fully connected) layers
  - Convolutional layers (1D and 2D)
  - LSTM layers for sequence modeling
  - Activation functions (ReLU, Sigmoid, Tanh, Softmax)
  - Dropout for regularization
  - Sequential container for building networks

- **Optimizers**
  - Stochastic Gradient Descent (SGD) with momentum and weight decay
  - Adam optimizer

- **Financial Time Series Analysis**
  - Loading and saving data from/to CSV files
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - Data normalization
  - Pattern detection
  - Train/test splitting
  - Sequence creation for time series forecasting

- **Options Pricing and Analysis**
  - Black-Scholes model
  - Binomial Tree model
  - Monte Carlo simulation
  - Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
  - Implied volatility calculation
  - Options strategies (spreads, straddles, strangles, etc.)

## Building the Library

### Prerequisites

- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.10 or higher

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/finml.git
   cd finml
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure and build:
   ```bash
   cmake ..
   make
   ```

4. (Optional) Install:
   ```bash
   make install
   ```

## Usage Examples

### Basic Neural Network

```cpp
#include "finml/nn/sequential.h"
#include "finml/nn/layers/dense.h"
#include "finml/nn/layers/activation.h"
#include "finml/optim/sgd.h"

using namespace finml;

// Create a sequential model
nn::Sequential model;

// Add layers
model.add(new nn::Dense(input_size, hidden_size));
model.add(new nn::ReLU());
model.add(new nn::Dense(hidden_size, output_size));
model.add(new nn::Sigmoid());

// Create optimizer
optim::SGD optimizer(learning_rate, momentum, weight_decay);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    for (const auto& [x, y] : training_data) {
        // Forward pass
        std::vector<double> prediction = model.forward(x);
        
        // Calculate loss gradient
        std::vector<double> grad_output = calculate_loss_gradient(prediction, y);
        
        // Backward pass
        model.backward(grad_output);
        
        // Update parameters
        optimizer.step(model.parameters());
    }
}
```

### LSTM for Stock Price Prediction

```cpp
#include "finml/nn/sequential.h"
#include "finml/nn/layers/lstm.h"
#include "finml/nn/layers/dense.h"
#include "finml/optim/adam.h"
#include "finml/data/timeseries.h"

using namespace finml;

// Load financial time series data
data::TimeSeries timeseries;
timeseries.loadFromCSV("stock_data.csv");

// Calculate technical indicators
timeseries.calculateIndicator("SMA", 20);
timeseries.calculateIndicator("RSI", 14);

// Normalize data
timeseries.normalizeZScore();

// Create sequences for LSTM
int sequence_length = 30;
int forecast_horizon = 1;
auto [X, y] = timeseries.createSequences(sequence_length, forecast_horizon, 
                                         {"Close", "Volume", "SMA_20", "RSI_14"});

// Create LSTM model
nn::Sequential model;
model.add(new nn::LSTM(input_features, hidden_size, true));
model.add(new nn::LSTM(hidden_size, hidden_size, false));
model.add(new nn::Dense(hidden_size, 1));

// Create optimizer
optim::Adam optimizer(0.001);

// Train the model
// ... (similar to the basic neural network example)
```

### CNN for Pattern Recognition

```cpp
#include "finml/nn/sequential.h"
#include "finml/nn/layers/conv.h"
#include "finml/nn/layers/pooling.h"
#include "finml/nn/layers/dense.h"
#include "finml/nn/layers/flatten.h"
#include "finml/data/timeseries.h"

using namespace finml;

// Create CNN model
nn::Sequential model;

// Add convolutional layers
model.add(new nn::Conv2D(input_channels, 32, 3, 1, 1));
model.add(new nn::ReLU());
model.add(new nn::MaxPool2D(2, 2));

model.add(new nn::Conv2D(32, 64, 3, 1, 1));
model.add(new nn::ReLU());
model.add(new nn::MaxPool2D(2, 2));

// Add fully connected layers
model.add(new nn::Flatten());
model.add(new nn::Dense(flattened_size, 128));
model.add(new nn::ReLU());
model.add(new nn::Dense(128, num_classes));
model.add(new nn::Softmax());

// Train the model
// ... (similar to previous examples)
```

### Options Pricing

```cpp
#include "finml/options/pricing.h"

using namespace finml;

// Set parameters
double S = 100.0;  // Current stock price
double K = 100.0;  // Strike price
double r = 0.05;   // Risk-free interest rate
double sigma = 0.2; // Volatility
double T = 1.0;    // Time to maturity (in years)

// Create pricing model
options::BlackScholes model;

// Calculate option price
double call_price = model.price(S, K, r, sigma, T, options::OptionType::CALL);
double put_price = model.price(S, K, r, sigma, T, options::OptionType::PUT);

// Calculate Greeks
double delta = model.delta(S, K, r, sigma, T, options::OptionType::CALL);
double gamma = model.gamma(S, K, r, sigma, T, options::OptionType::CALL);
double theta = model.theta(S, K, r, sigma, T, options::OptionType::CALL);
double vega = model.vega(S, K, r, sigma, T, options::OptionType::CALL);
double rho = model.rho(S, K, r, sigma, T, options::OptionType::CALL);
```

### Options Strategies

```cpp
#include "finml/options/strategies.h"

using namespace finml;

// Set parameters
double S = 100.0;  // Current stock price
double K = 100.0;  // Strike price
double r = 0.05;   // Risk-free interest rate
double sigma = 0.2; // Volatility
double T = 1.0;    // Time to maturity (in years)

// Create a bull call spread strategy
options::OptionsStrategy strategy = options::createBullCallSpread(S, K * 0.95, K * 1.05, T, r, sigma);

// Analyze the strategy
double value = strategy.value(S, r, sigma);
double delta = strategy.delta(S, r, sigma);
std::vector<double> break_even_points = strategy.breakEvenPoints();
double max_profit = strategy.maxProfit();
double max_loss = strategy.maxLoss();
```

## Examples

The library comes with several examples that demonstrate its functionality:

- `nn_example.cpp`: Basic neural network example (XOR problem)
- `lstm_stock_prediction.cpp`: Stock price prediction using LSTM
- `cnn_pattern_recognition.cpp`: Chart pattern recognition using CNN
- `options_example.cpp`: Options pricing and strategies

To build and run the examples:

```bash
cd build
make
./examples/nn_example
./examples/lstm_stock_prediction stock_data.csv
./examples/cnn_pattern_recognition stock_data.csv
./examples/options_example
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 