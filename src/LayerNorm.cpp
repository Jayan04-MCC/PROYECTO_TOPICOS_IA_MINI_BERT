#include "../include/LayerNorm.h"
#include "../include/Utils.h"
#include <cmath>

LayerNorm::LayerNorm(const std::string& weight_path, const std::string& bias_path) {
    loadWeights(weight_path, bias_path);
}

LayerNorm::LayerNorm(int dim) {
    weight = Eigen::VectorXf::Ones(dim);
    bias = Eigen::VectorXf::Zero(dim);
}

void LayerNorm::loadWeights(const std::string& weight_path, const std::string& bias_path) {
    weight = loadCSVtoVector(weight_path);
    bias = loadCSVtoVector(bias_path);
}

Eigen::MatrixXf LayerNorm::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf output = input;
    int seq_len = input.rows();
    int dim = input.cols();
    
    for (int i = 0; i < seq_len; ++i) {
        Eigen::RowVectorXf row = input.row(i);
        float mean = row.mean();
        float var = (row.array() - mean).square().mean();
        
        for (int j = 0; j < dim; ++j) {
            output(i, j) = ((row(j) - mean) / std::sqrt(var + eps)) * weight(j) + bias(j);
        }
    }
    return output;
}
