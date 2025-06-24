//
// Created by JAYAN on 23/06/2025.
//

#ifndef UTILS_H
#define UTILS_H
#pragma once
#include <Eigen/Dense>
#include <string>

Eigen::MatrixXf loadCSVtoMatrix(const std::string& filename);
Eigen::VectorXf loadCSVtoVector(const std::string& filename);


#endif //UTILS_H
