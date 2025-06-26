#include "../include/FeedForward.h"
#include "../include/Utils.h"
#include <cmath>

FeedForward::FeedForward(const std::string& intermediate_weight_path,
                        const std::string& intermediate_bias_path,
                        const std::string& output_weight_path,
                        const std::string& output_bias_path) {
    loadWeights(intermediate_weight_path, intermediate_bias_path,
                output_weight_path, output_bias_path);
}

void FeedForward::loadWeights(const std::string& intermediate_weight_path,
                             const std::string& intermediate_bias_path,
                             const std::string& output_weight_path,
                             const std::string& output_bias_path) {
    intermediate_weight = loadCSVtoMatrix(intermediate_weight_path);
    intermediate_bias = loadCSVtoVector(intermediate_bias_path);
    output_weight = loadCSVtoMatrix(output_weight_path);
    output_bias = loadCSVtoVector(output_bias_path);
}

Eigen::MatrixXf FeedForward::gelu(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf output = input;
    int rows = input.rows();
    int cols = input.cols();
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float x = input(i, j);
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            float inner = sqrt(2.0f / M_PI) * (x + 0.044715f * x * x * x);
            output(i, j) = 0.5f * x * (1.0f + std::tanh(inner));
        }
    }
    return output;
}

Eigen::MatrixXf FeedForward::forward(const Eigen::MatrixXf& input) {
    // First linear layer
    Eigen::MatrixXf intermediate = input * intermediate_weight.transpose();
    intermediate.rowwise() += intermediate_bias.transpose();
    
    // GELU activation
    intermediate = gelu(intermediate);
    
    // Second linear layer
    Eigen::MatrixXf output = intermediate * output_weight.transpose();
    output.rowwise() += output_bias.transpose();
    
    return output;
}
