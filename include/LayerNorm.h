#ifndef LAYERNORM_H
#define LAYERNORM_H

#pragma once
#include <Eigen/Dense>
#include <string>

class LayerNorm {
public:
    LayerNorm(const std::string& weight_path, const std::string& bias_path);
    LayerNorm(int dim);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
private:
    Eigen::VectorXf weight;
    Eigen::VectorXf bias;
    float eps = 1e-5;
    void loadWeights(const std::string& weight_path, const std::string& bias_path);
};

#endif //LAYERNORM_H
