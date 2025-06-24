//
// Created by JAYAN on 23/06/2025.
//

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>

class EmbeddingLayer {
public:
    EmbeddingLayer(const std::string& word_path,
    const std::string& position_path,
    const std::string& ln_weight_path,
    const std::string& ln_bias_path);
    Eigen::MatrixXf forward(const std::vector<int>& token_ids);
private:
    Eigen::MatrixXf word_embeddings; // (30522, 384)
    Eigen::MatrixXf position_embeddings; // (512, 384)
    Eigen::VectorXf ln_weight; // (384,)
    Eigen::VectorXf ln_bias; // (384,)
    Eigen::MatrixXf applyLayerNorm(const Eigen::MatrixXf& input);
};


#endif //EMBEDDINGS_H
