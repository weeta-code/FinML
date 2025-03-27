#ifndef FINML_MODELS_VOLATILITY_SURFACE_PREDICTOR_H
#define FINML_MODELS_VOLATILITY_SURFACE_PREDICTOR_H

#include "finml/nn/conv.h"
#include "finml/nn/lstm.h"
#include "finml/nn/linear.h"
#include "finml/nn/sequential.h"
#include "finml/core/matrix.h"
#include "finml/options/pricing.h"
#include "finml/options/implied_volatility.h"
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <functional>
#include <unordered_map>

namespace finml {
namespace models {

// Structure to represent a point on the volatility surface
struct VolatilitySurfacePoint {
    double strike;      // Option strike price
    double maturity;    // Time to maturity in years
    double volatility;  // Implied volatility
    options::OptionType type; // Option type (CALL or PUT)
};

// Structure to represent a volatility surface
struct VolatilitySurface {
    std::vector<double> strikes;  // Unique strike prices
    std::vector<double> maturities; // Unique maturities
    std::vector<std::vector<double>> call_volatilities; // 2D matrix [maturity_idx][strike_idx]
    std::vector<std::vector<double>> put_volatilities;  // 2D matrix [maturity_idx][strike_idx]
    double spot_price; // Underlying spot price
    double timestamp;  // Time when the surface was created/observed
};

// Arbitrage opportunity types
enum class ArbitrageType {
    CALENDAR_SPREAD,  // Arbitrage across maturities
    BUTTERFLY,        // Arbitrage across strikes (convexity)
    VERTICAL_SPREAD,  // Arbitrage in vertical spreads
    CALL_PUT_PARITY   // Call-put parity violation
};

// Structure to represent an arbitrage opportunity in the volatility surface
struct VolatilityArbitrageOpportunity {
    ArbitrageType type;
    double magnitude;  // Arbitrage magnitude (e.g., profit per contract)
    std::vector<VolatilitySurfacePoint> involved_points;
    std::string description;
    double timestamp;
};

// SSVI parameterization for arbitrage-free volatility surfaces
struct SSVIParameters {
    double theta;     // Overall level parameter
    double rho;       // Correlation parameter (-1 <= rho <= 1)
    double eta;       // ATM volatility of vol
    double lambda;    // Density of the wings
    double a;         // Left wing parameter 
    double b;         // Right wing parameter
};

// Class for volatility surface prediction using CNN-LSTM hybrid architecture
class VolatilitySurfacePredictor {
private:
    // Model architecture components
    std::unique_ptr<nn::Conv1D> spatial_features_extractor;  // CNN for spatial features
    std::unique_ptr<nn::LSTM> temporal_predictor;           // LSTM for time series forecasting
    std::unique_ptr<nn::Linear> output_layer;               // Final projection layer
    std::unique_ptr<nn::Sequential> decoder;                // Decoder network
    
    // Model parameters
    int strike_dims;              // Number of strike prices in the grid
    int maturity_dims;            // Number of maturities in the grid
    int time_window;              // Number of historical surfaces to use
    int forecast_horizon;         // Number of steps to forecast ahead
    double learning_rate;         // Learning rate for training
    double dropout_rate;          // Dropout rate for regularization
    
    // SSVI model for arbitrage-free parameterization
    std::function<double(double, double, const SSVIParameters&)> ssvi_model;
    SSVIParameters current_ssvi_params;
    
    // Random number generator for initialization/dropout
    std::mt19937 random_engine;
    
    // Methods
    core::Matrix surfaceToMatrix(const VolatilitySurface& surface);
    VolatilitySurface matrixToSurface(const core::Matrix& matrix, 
                                      const std::vector<double>& strikes,
                                      const std::vector<double>& maturities,
                                      double spot_price);
    
    // Fit SSVI parameters to a volatility surface
    SSVIParameters fitSSVIParameters(const VolatilitySurface& surface);
    
    // Generate an arbitrage-free surface using SSVI parameters
    VolatilitySurface generateArbitrageFreeSSVISurface(const SSVIParameters& params,
                                                      const std::vector<double>& strikes,
                                                      const std::vector<double>& maturities,
                                                      double spot_price);
                                                      
    // Check for different types of arbitrage                      
    std::vector<VolatilityArbitrageOpportunity> detectCalendarArbitrage(const VolatilitySurface& surface);
    std::vector<VolatilityArbitrageOpportunity> detectButterflyArbitrage(const VolatilitySurface& surface);
    std::vector<VolatilityArbitrageOpportunity> detectVerticalArbitrage(const VolatilitySurface& surface);
    std::vector<VolatilityArbitrageOpportunity> detectCallPutParityArbitrage(const VolatilitySurface& surface);
    
public:
    // Constructor
    VolatilitySurfacePredictor(
        int strike_dims,
        int maturity_dims,
        int time_window,
        int forecast_horizon = 1,
        int hidden_size = 128,
        int cnn_filters = 32,
        int cnn_kernel_size = 3,
        double learning_rate = 0.001,
        double dropout_rate = 0.2,
        int seed = 42
    );
    
    // Train the model on historical volatility surfaces
    void train(
        const std::vector<VolatilitySurface>& historical_surfaces,
        int epochs,
        int batch_size = 16,
        double validation_split = 0.2
    );
    
    // Predict future volatility surfaces
    std::vector<VolatilitySurface> predictSurfaces(
        const std::vector<VolatilitySurface>& recent_surfaces,
        int num_steps = 1
    );
    
    // Analyze a volatility surface for arbitrage opportunities
    std::vector<VolatilityArbitrageOpportunity> detectArbitrageOpportunities(
        const VolatilitySurface& surface,
        double threshold = 0.001
    );
    
    // Convert raw market option prices to implied volatility surface
    VolatilitySurface createVolatilitySurfaceFromMarketPrices(
        const std::vector<std::vector<double>>& call_prices,
        const std::vector<std::vector<double>>& put_prices,
        const std::vector<double>& strikes,
        const std::vector<double>& maturities,
        double spot_price,
        double risk_free_rate
    );
    
    // Save and load model weights
    bool saveModel(const std::string& filename);
    bool loadModel(const std::string& filename);
    
    // Calculate no-arbitrage bounds for volatility surfaces
    std::pair<VolatilitySurface, VolatilitySurface> calculateNoArbitrageBounds(
        const VolatilitySurface& surface
    );
    
    // Generate synthetic volatility surfaces with arbitrage opportunities for testing
    VolatilitySurface generateSyntheticSurface(
        const std::vector<double>& strikes,
        const std::vector<double>& maturities,
        double spot_price,
        bool include_arbitrage_opportunities = false,
        double arbitrage_magnitude = 0.05
    );
};

} // namespace models
} // namespace finml

#endif // FINML_MODELS_VOLATILITY_SURFACE_PREDICTOR_H 