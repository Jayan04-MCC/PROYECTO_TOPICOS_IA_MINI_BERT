//
// Created by JAYAN on 23/06/2025.
//

#include "../include/TransformerEncoder.h"
#include <Eigen/Dense>
TransformerEncoder::TransformerEncoder(const std::string &base_path) {
    for (int i = 0; i < TransformerEncoder::num_layer; ++i) {
        std::string prefix = base_path + "/encoder_layer_" + std::to_string(i);
        layers.emplace_back(layers.emplace_back(TransformerBlock(prefix)));
    }
}

Eigen::MatrixXf TransformerEncoder::forward(const Eigen::MatrixXf &x_in) {
    Eigen::MatrixXf x = x_in;
    for (int i = 0; i < layers.size(); ++i) {
        x = layers[i].forward(x);
    }
    return x;
}

