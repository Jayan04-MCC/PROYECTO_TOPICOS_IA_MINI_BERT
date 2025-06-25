//
// Created by JAYAN on 23/06/2025.
//
#include "../include/Utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>

Eigen::MatrixXf loadCSVtoMatrix(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<float>> values;

    if (!file.is_open())
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> row;

        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stof(cell));
        }

        if (!row.empty())
            values.push_back(row);
    }

    int rows = values.size();
    int cols = values[0].size();
    Eigen::MatrixXf mat(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = values[i][j];

    return mat;
}

Eigen::VectorXf loadCSVtoVector(const std::string& filename) {
    Eigen::MatrixXf mat = loadCSVtoMatrix(filename);
    if (mat.rows() == 1)
        return mat.row(0).transpose();  // fila → vector columna
    else if (mat.cols() == 1)
        return mat.col(0);
    else
        throw std::runtime_error("El CSV no es un vector 1D válido: " + filename);
}
