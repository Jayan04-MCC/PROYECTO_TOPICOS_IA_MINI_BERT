//
// Created by JAYAN on 23/06/2025.
//

#ifndef ATTENTION_H
#define ATTENTION_H

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "../../include/Transformer.h"
class TransformerEncoder {
public:
    int num_layer = 6;
    std::vector<TransformerBlock> layers;
    TransformerEncoder(const std::string& base_path);
    Eigen::MatrixXf forward(const Eigen::MatrixXf& x_in);
};


#endif //ATTENTION_H
