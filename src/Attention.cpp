#include "../include/Attention.h"
#include "../include/Utils.h"
#include <cmath>
#include <algorithm>

MultiHeadAttention::MultiHeadAttention(const std::string& query_weight_path,
                                      const std::string& query_bias_path,
                                      const std::string& key_weight_path,
                                      const std::string& key_bias_path,
                                      const std::string& value_weight_path,
                                      const std::string& value_bias_path,
                                      const std::string& output_weight_path,
                                      const std::string& output_bias_path,
                                      int hidden_size, int num_heads)
    : hidden_size(hidden_size), num_heads(num_heads) {
    head_dim = hidden_size / num_heads;
    loadWeights(query_weight_path, query_bias_path, key_weight_path, key_bias_path,
                value_weight_path, value_bias_path, output_weight_path, output_bias_path);
}

void MultiHeadAttention::loadWeights(const std::string& query_weight_path,
                                    const std::string& query_bias_path,
                                    const std::string& key_weight_path,
                                    const std::string& key_bias_path,
                                    const std::string& value_weight_path,
                                    const std::string& value_bias_path,
                                    const std::string& output_weight_path,
                                    const std::string& output_bias_path) {
    query_weight = loadCSVtoMatrix(query_weight_path);
    query_bias = loadCSVtoVector(query_bias_path);
    key_weight = loadCSVtoMatrix(key_weight_path);
    key_bias = loadCSVtoVector(key_bias_path);
    value_weight = loadCSVtoMatrix(value_weight_path);
    value_bias = loadCSVtoVector(value_bias_path);
    output_weight = loadCSVtoMatrix(output_weight_path);
    output_bias = loadCSVtoVector(output_bias_path);
}

Eigen::MatrixXf MultiHeadAttention::softmax(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf output = input;
    int rows = input.rows();
    int cols = input.cols();
    
    for (int i = 0; i < rows; ++i) {
        Eigen::RowVectorXf row = input.row(i);
        float max_val = row.maxCoeff();
        
        for (int j = 0; j < cols; ++j) {
            output(i, j) = std::exp(row(j) - max_val);
        }
        
        float sum = output.row(i).sum();
        output.row(i) /= sum;
    }
    return output;
}

Eigen::MatrixXf MultiHeadAttention::forward(const Eigen::MatrixXf& input) {
    int seq_len = input.rows();
    int hidden_size = input.cols();
    
    // Linear transformations for Q, K, V
    Eigen::MatrixXf Q = input * query_weight.transpose();
    Q.rowwise() += query_bias.transpose();
    
    Eigen::MatrixXf K = input * key_weight.transpose();
    K.rowwise() += key_bias.transpose();
    
    Eigen::MatrixXf V = input * value_weight.transpose();
    V.rowwise() += value_bias.transpose();
    
    // Reshape for multi-head attention
    // For simplicity, we'll implement single-head attention first
    // In full implementation, you'd split into heads here
    
    // Attention scores: Q * K^T / sqrt(head_dim)
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Eigen::MatrixXf attention_scores = Q * K.transpose() * scale;
    
    // Apply softmax
    Eigen::MatrixXf attention_weights = softmax(attention_scores);
    
    // Apply attention to values
    Eigen::MatrixXf context = attention_weights * V;
    
    // Output projection
    Eigen::MatrixXf output = context * output_weight.transpose();
    output.rowwise() += output_bias.transpose();
    
    return output;
}
