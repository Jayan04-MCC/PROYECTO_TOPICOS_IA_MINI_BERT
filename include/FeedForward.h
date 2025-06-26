#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#pragma once
#include <Eigen/Dense>
#include <string>

class FeedForward {
public:
    FeedForward(const std::string& intermediate_weight_path,
                const std::string& intermediate_bias_path,
                const std::string& output_weight_path,
                const std::string& output_bias_path);
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
private:
    Eigen::MatrixXf intermediate_weight, output_weight;
    Eigen::VectorXf intermediate_bias, output_bias;
    
    void loadWeights(const std::string& intermediate_weight_path,
                    const std::string& intermediate_bias_path,
                    const std::string& output_weight_path,
                    const std::string& output_bias_path);
    
    Eigen::MatrixXf gelu(const Eigen::MatrixXf& input);
};

#endif //FEEDFORWARD_H
