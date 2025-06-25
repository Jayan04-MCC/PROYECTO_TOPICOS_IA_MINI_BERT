//
// Created by JAYAN on 23/06/2025.
//

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#pragma once
#include <Eigen/Dense>

class TransformerBlock {
public:
    TransformerBlock(const std::string& prefix_path);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x);

private:
    // Atención
    Eigen::MatrixXf W_q, W_k, W_v, W_o;
    Eigen::VectorXf b_q, b_k, b_v, b_o;
    Eigen::VectorXf ln1_weight, ln1_bias;

    // Feedforward
    Eigen::MatrixXf W1, W2;
    Eigen::VectorXf b1, b2;
    Eigen::VectorXf ln2_weight, ln2_bias;

    Eigen::MatrixXf selfAttention(const Eigen::MatrixXf& x);
    Eigen::MatrixXf feedForward(const Eigen::MatrixXf& x);
    Eigen::MatrixXf applyLayerNorm(const Eigen::MatrixXf& x, const Eigen::VectorXf& weight, const Eigen::VectorXf& bias);
    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x);
};





#endif //TRANSFORMER_H
