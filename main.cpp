#include <iostream>
#include "D:\PROYECTO-TOPICOS-IA\include\Utils.h"

int main() {
    try {
        Eigen::MatrixXf W = loadCSVtoMatrix(
            "D:/PROYECTO-TOPICOS-IA/Weights/pesos_comprimidos/content/pesos_csv/embeddings_word_embeddings_weight.csv");
        std::cout << "Matriz cargada: " << W.rows() << "x" << W.cols() << std::endl;
        std::cout << "Primera fila:\n" << W.row(0) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
