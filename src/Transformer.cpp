//
// Created by JAYAN on 23/06/2025.
//
#include <cmath>
#include "../include/Transformer.h"
#include "../include/Utils.h"
#include <iostream>
TransformerBlock::TransformerBlock(const std::string& prefix)
{
    // Cargar pesos de atención
    W_q = loadCSVtoMatrix(prefix + "_attention_self_query_weight.csv");
    b_q = loadCSVtoVector(prefix + "_attention_self_query_bias.csv");

    W_k = loadCSVtoMatrix(prefix + "_attention_self_key_weight.csv");
    b_k = loadCSVtoVector(prefix + "_attention_self_key_bias.csv");

    W_v = loadCSVtoMatrix(prefix + "_attention_self_value_weight.csv");
    b_v = loadCSVtoVector(prefix + "_attention_self_value_bias.csv");

    W_o = loadCSVtoMatrix(prefix + "_attention_output_dense_weight.csv");
    b_o = loadCSVtoVector(prefix + "_attention_output_dense_bias.csv");

    ln1_weight = loadCSVtoVector(prefix + "_attention_output_LayerNorm_weight.csv");
    ln1_bias   = loadCSVtoVector(prefix + "_attention_output_LayerNorm_bias.csv");

    // Feedforward
    W1 = loadCSVtoMatrix(prefix + "_intermediate_dense_weight.csv").transpose();
    b1 = loadCSVtoVector(prefix + "_intermediate_dense_bias.csv");

    W2 = loadCSVtoMatrix(prefix + "_output_dense_weight.csv").transpose();
    b2 = loadCSVtoVector(prefix + "_output_dense_bias.csv");

    ln2_weight = loadCSVtoVector(prefix + "_output_LayerNorm_weight.csv");
    ln2_bias   = loadCSVtoVector(prefix + "_output_LayerNorm_bias.csv");
}

Eigen::MatrixXf TransformerBlock::forward(const Eigen::MatrixXf& x)
{
    Eigen::MatrixXf attn_out = x;
    Eigen::MatrixXf out = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    Eigen::MatrixXf attn = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    Eigen::MatrixXf ff = Eigen::MatrixXf::Zero(x.rows(), x.cols());
    // Attention + residual + norm
        attn = selfAttention(x);
    // normalizacion
        attn_out = applyLayerNorm(x + attn, ln1_weight, ln1_bias);
    // Feedforward + residual + norm
        ff = feedForward(attn_out);
    // normalizacion
        out = applyLayerNorm(attn_out + ff, ln2_weight, ln2_bias);
    return out;
}


Eigen::MatrixXf TransformerBlock::selfAttention(const Eigen::MatrixXf& x)
{
    std::cout << "x: " << x.rows() << "x" << x.cols() << std::endl;
    std::cout << "W_q: " << W_q.rows() << "x" << W_q.cols() << std::endl;

    Eigen::MatrixXf Q = (x * W_q).rowwise() + b_q.transpose();
    Eigen::MatrixXf K = (x * W_k).rowwise() + b_k.transpose();
    Eigen::MatrixXf V = (x * W_v).rowwise() + b_v.transpose();

    int d_k = Q.cols();
    Eigen::MatrixXf scores = (Q * K.transpose()) / std::sqrt(float(d_k));
    Eigen::MatrixXf weights = softmax(scores);
    Eigen::MatrixXf context = weights * V;

    Eigen::MatrixXf output = (context * W_o).rowwise() + b_o.transpose();
    return output;

}

Eigen::MatrixXf TransformerBlock::feedForward(const Eigen::MatrixXf& x)
{
    std::cout << "x: " << x.rows() << "x" << x.cols() << std::endl;
    std::cout << "W1: " << W1.rows() << "x" << W1.cols() << std::endl;
    std::cout << "W2: " << W2.rows() << "x" << W2.cols() << std::endl;
    Eigen::MatrixXf h = (x * W1).rowwise() + b1.transpose();
    std::cout << "h: " << h.rows() << "x" << h.cols() << std::endl;
    h = h.array().max(0); // ReLU
    Eigen::MatrixXf out = (h * W2).rowwise() + b2.transpose();
    return out;
}

Eigen::MatrixXf TransformerBlock::applyLayerNorm(const Eigen::MatrixXf& x, const Eigen::VectorXf& weight, const Eigen::VectorXf& bias)
{
    Eigen::MatrixXf out = x;
    for (int i = 0; i < x.rows(); ++i) {
        Eigen::RowVectorXf row = x.row(i);
        float mean = row.mean();
        float var = (row.array() - mean).square().mean();
        float eps = 1e-5;

            for (int j = 0; j < x.cols(); ++j) {
                out(i, j) = ((row(j) - mean) / std::sqrt(var + eps)) * weight(j) + bias(j);
            }
    }
    return out;
}

Eigen::MatrixXf TransformerBlock::softmax(const Eigen::MatrixXf& x)
{
    Eigen::MatrixXf result = x;
    for (int i = 0; i < x.rows(); ++i) {
        float max_val = x.row(i).maxCoeff();
        Eigen::VectorXf exps = (x.row(i).array() - max_val).exp();
        result.row(i) = (exps / exps.sum()).transpose();
    }
    return result;
}