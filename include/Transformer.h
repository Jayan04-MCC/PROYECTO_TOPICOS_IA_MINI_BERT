#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#pragma once
#include <Eigen/Dense>
#include <string>
#include <memory>
#include "Attention.h"
#include "FeedForward.h"
#include "LayerNorm.h"

class TransformerBlock {
public:
    TransformerBlock(int layer_idx, const std::string& weights_base_path);
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
private:
    std::unique_ptr<MultiHeadAttention> attention;
    std::unique_ptr<FeedForward> feedforward;
    std::unique_ptr<LayerNorm> attention_layernorm;
    std::unique_ptr<LayerNorm> output_layernorm;
    
    std::string buildPath(const std::string& base_path, int layer_idx, const std::string& component);
};

class BERTEncoder {
public:
    BERTEncoder(const std::string& weights_base_path, int num_layers = 6);
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input);
    
private:
    std::vector<std::unique_ptr<TransformerBlock>> layers;
    int num_layers;
};

#endif //TRANSFORMER_H
