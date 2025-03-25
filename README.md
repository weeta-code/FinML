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

# Stock Trading Model with LSTM and CNN

The included demo demonstrates a hybrid stock trading model that combines LSTM (Long Short-Term Memory) networks for time series price prediction and CNN (Convolutional Neural Network) for chart pattern recognition. The demo uses Apple (AAPL) stock data from 2023-2024.

## Components

### 1. Time Series Data Handling

The `TimeSeries` class in `finml::data` provides functionality for:
- Loading stock price data from CSV files
- Calculating technical indicators (SMA, RSI, MACD)
- Normalizing data
- Creating sequences for LSTM training
- Detecting patterns in the data

### 2. LSTM Model for Price Prediction

The LSTM model:
- Takes sequences of price data as input (Open, High, Low, Close, Volume, technical indicators)
- Predicts future price movements
- Uses multiple stacked LSTM layers for deep pattern recognition
- Outputs a predicted price for the next N days

### 3. CNN Model for Pattern Recognition

The CNN model:
- Takes price chart images as input
- Identifies common chart patterns (double tops/bottoms, head and shoulders, flags, etc.)
- Classifies patterns as bullish or bearish
- Provides a confidence score for the identified pattern

### 4. Hybrid Trading Model

The hybrid model:
- Combines signals from both the LSTM and CNN models
- Weighs signals based on configurable parameters
- Adjusts trading decisions based on market volatility and risk tolerance
- Generates BUY/SELL/HOLD signals with confidence levels
- Tracks performance metrics (returns, win rate, Sharpe ratio, etc.)

## Training Data Format

### LSTM Training Data
The LSTM model requires sequences of price and indicator data:

```
Input:
  Sequence of N days, each with M features:
  [
    [open_1, high_1, low_1, close_1, volume_1, sma_1, rsi_1, ...],
    [open_2, high_2, low_2, close_2, volume_2, sma_2, rsi_2, ...],
    ...
    [open_N, high_N, low_N, close_N, volume_N, sma_N, rsi_N, ...]
  ]

Output:
  Target price (typically the close price of the next day)
```

### CNN Training Data
The CNN model requires labeled chart images:

```
Input:
  Chart image showing price action over a specific period
  (typically represented as a Matrix of pixel values)

Output:
  Binary label: bullish (1) or bearish (0)
  Pattern type (optional): double top, head and shoulders, etc.
```

## Performance Metrics

The demo calculates and reports:
- Total return
- Win rate (percentage of profitable trades)
- Maximum drawdown
- Sharpe ratio
- Alpha
- Beta

## Training the Models

To train the LSTM model:
- Historical price data is split into training and test sets
- Data is normalized and sequences are created
- The model is trained to minimize mean squared error between predicted and actual prices

To train the CNN model:
- Chart images are labeled as bullish or bearish based on the resulting price action
- Images are split into training and test sets
- The model is trained to minimize binary cross-entropy loss

## How to Run the Demo

1. Compile the project (make sure you have the necessary dependencies installed)
2. Run the data loader to download AAPL stock data:
   ```
   ./bin/aapl_data_loader
   ```
3. Run the hybrid trading model demo:
   ```
   ./bin/hybrid_trading_model
   ```

## Notes on Real-World Application

In a real trading environment:
- More extensive data preprocessing would be required
- Model hyperparameters would need careful tuning
- Additional risk management techniques should be implemented
- Model performance should be regularly evaluated against new data
- Transaction costs, slippage, and market impact would need to be considered

## References and Further Reading

1. Long Short-Term Memory Networks: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
2. CNN for Pattern Recognition: [Applying Deep Learning to Enhance Momentum Trading Strategies](https://arxiv.org/abs/1607.04318)
3. Technical Analysis: [Encyclopedia of Chart Patterns](https://www.wiley.com/en-us/Encyclopedia+of+Chart+Patterns%2C+2nd+Edition-p-9780471668268)
4. Performance Metrics: [Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp) 