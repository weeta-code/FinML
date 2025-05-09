cmake_minimum_required(VERSION 3.10)
project(FinML)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files for the library
file(GLOB_RECURSE LIB_SOURCES 
    "src/core/*.cpp"
    "src/models/*.cpp"
    "src/layers/*.cpp"
    "src/nn/*.cpp"
    "src/optim/*.cpp"
    "src/data/*.cpp"
    "src/options/*.cpp"
)

# Create the library
add_library(finml ${LIB_SOURCES})

# Create the demo executables
add_executable(run_hybrid_trading_model demo/run_hybrid_trading_model.cpp)
target_link_libraries(run_hybrid_trading_model finml)

add_executable(lstm_options_arbitrage demo/lstm_options_arbitrage.cpp)
target_link_libraries(lstm_options_arbitrage finml)

# Add the volatility surface arbitrage demo
add_executable(volatility_surface_arbitrage demo/volatility_surface_arbitrage.cpp)
target_link_libraries(volatility_surface_arbitrage finml)

# Add any dependencies (e.g., threading)
find_package(Threads REQUIRED)
target_link_libraries(finml Threads::Threads)
target_link_libraries(run_hybrid_trading_model Threads::Threads)
target_link_libraries(lstm_options_arbitrage Threads::Threads)
target_link_libraries(volatility_surface_arbitrage Threads::Threads)

# Copy data files to build directory
file(COPY data/ DESTINATION ${CMAKE_BINARY_DIR}/data)

# Output binary to build/bin
set_target_properties(run_hybrid_trading_model PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(lstm_options_arbitrage PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_target_properties(volatility_surface_arbitrage PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add compile options
target_compile_options(finml PRIVATE -Wall -Wextra)
target_compile_options(run_hybrid_trading_model PRIVATE -Wall -Wextra)
target_compile_options(lstm_options_arbitrage PRIVATE -Wall -Wextra)
target_compile_options(volatility_surface_arbitrage PRIVATE -Wall -Wextra)

# Print status
message(STATUS "FinML project configured with C++${CMAKE_CXX_STANDARD}")

# Add examples directory
# add_subdirectory(examples)

# Add tests directory
# add_subdirectory(tests)

# Installation
install(TARGETS finml
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include) 