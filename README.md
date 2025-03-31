# FinML: Financial Machine Learning Library

A comprehensive library for applying machine learning techniques to financial markets, specializing in options pricing, volatility modeling, and arbitrage detection.

## Overview

FinML implements state-of-the-art machine learning models tailored for financial applications, with a focus on:

- Volatility surface modeling and forecasting
- Arbitrage detection in options markets
- Neural network-based trading strategies
- Time series prediction for financial data

## Key Features

### Volatility Surface Arbitrage Detection

The library provides tools to:

- Build and visualize complete implied volatility surfaces
- Detect calendar spread and butterfly arbitrage opportunities
- Predict future volatility surfaces using CNN-LSTM hybrid models
- Calculate no-arbitrage bounds for implied volatility

### Neural Network Architectures

- **LSTM**: For time series prediction of financial data
- **CNN**: For pattern recognition 
- **Hybrid Models**: Combining convolutional layers for spatial features with LSTM layers for temporal dynamics

### Options Pricing

- Black-Scholes model implementation
- Implied volatility calculation
- Greeks estimation
- Monte Carlo simulation for pricing complex derivatives

## Getting Started

### Prerequisites

- C++17 compatible compiler
- CMake 3.10 or higher
- Python 3.8+ (for visualization scripts)
- Required Python packages: pandas, numpy, matplotlib, seaborn

### Building the Library

```bash
mkdir build && cd build
cmake ..
make
```

### Running the Demos

Several demo applications are provided to showcase the library's capabilities:

1. **Volatility Surface Arbitrage Detection**:
```bash
./bin/volatility_surface_arbitrage
```

2. **LSTM Options Arbitrage**:
```bash
./bin/lstm_options_arbitrage
```

3. **Hybrid Trading Model**:
```bash
./bin/run_hybrid_trading_model
```

## Visualizing Results

The library outputs results to CSV files that can be visualized using the provided Python scripts:

```bash
python ../python/visualize_vol_surface.py
python ../python/analyze_arbitrage.py
```

These scripts generate comprehensive visualizations including:

- 3D volatility surfaces
- Arbitrage opportunity heatmaps
- Volatility smile plots
- Prediction error analysis
- No-arbitrage bound comparisons

## Volatility Surface Arbitrage Detection

The volatility surface arbitrage detection module implements two primary types of arbitrage detection:

### Calendar Spread Arbitrage

Calendar spread arbitrage occurs when the implied volatility for a longer-term option is lower than that of a shorter-term option with the same strike price. This violates the principle that implied volatility should generally increase with time to maturity.

The detector checks for:
- Decreasing volatility across increasing maturities
- Quantifies the magnitude of the arbitrage opportunity
- Provides detailed descriptions of each identified opportunity

### Butterfly Arbitrage

Butterfly arbitrage occurs when the volatility smile exhibits a non-convex shape across strike prices. This violates the principle that the implied volatility curve should be convex.

The detector:
- Checks for violations of convexity in the volatility smile
- Quantifies the arbitrage magnitude using weighted averages
- Identifies the specific strikes and maturities where arbitrage exists

## CNN-LSTM Hybrid Architecture

The volatility surface prediction model uses a hybrid architecture that combines:

1. **Convolutional layers** to capture the spatial structure of the volatility surface across strikes and maturities
2. **LSTM layers** to model the temporal evolution of these surfaces
3. **Self-attention mechanisms** to weight the importance of different regions of the surface

This approach enables the model to:
- Learn the typical shapes of volatility smiles and term structures
- Understand how these structures evolve over time
- Identify potential arbitrage opportunities before they become apparent in the market

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The SSVI (Surface Stochastic Volatility Inspired) model by Gatheral and Jacquier
- Research on no-arbitrage conditions for volatility surfaces
- The deep learning community for advances in time series forecasting techniques 