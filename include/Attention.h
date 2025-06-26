#ifndef ATTENTION_H
#define ATTENTION_H

#pragma once
#include <Eigen/Dense>
#include <string>

class MultiHeadAttention {
public:
    MultiHeadAttention(const std::string& query_weight_path,
                      const std::string& query_bias_path,
                      const std::string& key_weight_path,
                      const std::string& key_bias_path,
                      const std::string& value_weight_path,
                      const std::string& value_bias_path,
                      const std::string& output_weight_path,
                      const std::string& output_bias_path,
                      int hidden_size = 384,
                      int num_heads = 12);
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
private:
    Eigen::MatrixXf query_weight, key_weight, value_weight, output_weight;
    Eigen::VectorXf query_bias, key_bias, value_bias, output_bias;
    int hidden_size;
    int num_heads;
    int head_dim;
    
    void loadWeights(const std::string& query_weight_path,
                    const std::string& query_bias_path,
                    const std::string& key_weight_path,
                    const std::string& key_bias_path,
                    const std::string& value_weight_path,
                    const std::string& value_bias_path,
                    const std::string& output_weight_path,
                    const std::string& output_bias_path);
    
    Eigen::MatrixXf softmax(const Eigen::MatrixXf& input);
    Eigen::MatrixXf reshape_for_attention(const Eigen::MatrixXf& input);
    Eigen::MatrixXf reshape_from_attention(const Eigen::MatrixXf& input);
};

#endif //ATTENTION_H
