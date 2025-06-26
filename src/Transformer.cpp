#include "../include/Transformer.h"
#include <iostream>

std::string TransformerBlock::buildPath(const std::string& base_path, int layer_idx, const std::string& component) {
    return base_path + "encoder_layer_" + std::to_string(layer_idx) + "_" + component + ".csv";
}

TransformerBlock::TransformerBlock(int layer_idx, const std::string& weights_base_path) {
    // Build paths for attention weights
    std::string query_weight = buildPath(weights_base_path, layer_idx, "attention_self_query_weight");
    std::string query_bias = buildPath(weights_base_path, layer_idx, "attention_self_query_bias");
    std::string key_weight = buildPath(weights_base_path, layer_idx, "attention_self_key_weight");
    std::string key_bias = buildPath(weights_base_path, layer_idx, "attention_self_key_bias");
    std::string value_weight = buildPath(weights_base_path, layer_idx, "attention_self_value_weight");
    std::string value_bias = buildPath(weights_base_path, layer_idx, "attention_self_value_bias");
    std::string attn_output_weight = buildPath(weights_base_path, layer_idx, "attention_output_dense_weight");
    std::string attn_output_bias = buildPath(weights_base_path, layer_idx, "attention_output_dense_bias");
    
    // Build paths for feedforward weights
    std::string intermediate_weight = buildPath(weights_base_path, layer_idx, "intermediate_dense_weight");
    std::string intermediate_bias = buildPath(weights_base_path, layer_idx, "intermediate_dense_bias");
    std::string ff_output_weight = buildPath(weights_base_path, layer_idx, "output_dense_weight");
    std::string ff_output_bias = buildPath(weights_base_path, layer_idx, "output_dense_bias");
    
    // Build paths for layer norms
    std::string attn_ln_weight = buildPath(weights_base_path, layer_idx, "attention_output_LayerNorm_weight");
    std::string attn_ln_bias = buildPath(weights_base_path, layer_idx, "attention_output_LayerNorm_bias");
    std::string output_ln_weight = buildPath(weights_base_path, layer_idx, "output_LayerNorm_weight");
    std::string output_ln_bias = buildPath(weights_base_path, layer_idx, "output_LayerNorm_bias");
    
    // Initialize components
    attention = std::make_unique<MultiHeadAttention>(
        query_weight, query_bias, key_weight, key_bias,
        value_weight, value_bias, attn_output_weight, attn_output_bias
    );
    
    feedforward = std::make_unique<FeedForward>(
        intermediate_weight, intermediate_bias, ff_output_weight, ff_output_bias
    );
    
    attention_layernorm = std::make_unique<LayerNorm>(attn_ln_weight, attn_ln_bias);
    output_layernorm = std::make_unique<LayerNorm>(output_ln_weight, output_ln_bias);
}

Eigen::MatrixXf TransformerBlock::forward(const Eigen::MatrixXf& input) {
    // Self-attention + residual connection + layer norm
    Eigen::MatrixXf attn_output = attention->forward(input);
    Eigen::MatrixXf attn_residual = input + attn_output;
    Eigen::MatrixXf attn_normed = attention_layernorm->forward(attn_residual);
    
    // Feed-forward + residual connection + layer norm
    Eigen::MatrixXf ff_output = feedforward->forward(attn_normed);
    Eigen::MatrixXf ff_residual = attn_normed + ff_output;
    Eigen::MatrixXf final_output = output_layernorm->forward(ff_residual);
    
    return final_output;
}

BERTEncoder::BERTEncoder(const std::string& weights_base_path, int num_layers) 
    : num_layers(num_layers) {
    layers.reserve(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(std::make_unique<TransformerBlock>(i, weights_base_path));
    }
}

Eigen::MatrixXf BERTEncoder::forward(const Eigen::MatrixXf& input) {
    Eigen::MatrixXf output = input;
    
    for (int i = 0; i < num_layers; ++i) {
        output = layers[i]->forward(output);
        std::cout << "Layer " << i << " output shape: " << output.rows() << "x" << output.cols() << std::endl;
    }
    
    return output;
}
