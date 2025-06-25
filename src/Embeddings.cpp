//
// Created by JAYAN on 23/06/2025.
//
#include "../include/Embeddings.h"
#include "../include/Utils.h"
#include "cmath"
EmbeddingLayer::EmbeddingLayer( const std::string &word_path,
                                const std::string &position_path,
                                const std::string &ln_weight_path,
                                const std::string &ln_bias_path)
{
    word_embeddings = loadCSVtoMatrix(word_path);
    position_embeddings = loadCSVtoMatrix(position_path);
    ln_weight = loadCSVtoVector(ln_weight_path);
    ln_bias = loadCSVtoVector(ln_bias_path);
}
Eigen::MatrixXf EmbeddingLayer::applyLayerNorm(const Eigen::MatrixXf &input) {// input: (seq_len, dim)
Eigen::MatrixXf output = input;
int seq_len = input.rows();
int dim = input.cols();

for (int i = 0; i < seq_len; ++i) {
    Eigen::RowVectorXf row = input.row(i);
    float mean = row.mean();
    float var  = (row.array() - mean).square().mean();
    float eps  = 1e-5;

    for (int j = 0; j < dim; ++j) {
        output(i, j) = ((row(j) - mean) / std::sqrt(var + eps)) * ln_weight(j) + ln_bias(j);
    }
}
return output;
}

Eigen::MatrixXf EmbeddingLayer::forward(const std::vector<int> &token_ids) {

    int seq_len = token_ids.size();
    int dim = word_embeddings.cols();
    Eigen::MatrixXf output(seq_len, dim);

    for (int i = 0; i < seq_len; ++i) {
        // Word embedding
        Eigen::RowVectorXf token_vec = word_embeddings.row(token_ids[i]);
        // Position embedding
        Eigen::RowVectorXf pos_vec   = position_embeddings.row(i);
        // Combine
        output.row(i) = token_vec + pos_vec; //sumatoria del embedding de los vectores con su positional encoding
    }

    return applyLayerNorm(output);
}

