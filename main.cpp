#include <iostream>
#include "include/Utils.h"
#include "include/Tokenizer.h"
#include "include/Embeddings.h"
#include "include/Config.h"
int main() {

    Config& config = Config::getInstance();
    EmbeddingLayer emb(config.getWordEmbeddingsPath(), 
                       config.getPositionEmbeddingsPath(),
                       config.getLayerNormWeightPath(), 
                       config.getLayerNormBiasPath());
    tokenizerWord();
    //capturamos los ids
    Eigen::VectorXf ids = loadCSVtoVector(config.getTokensPath());
    std::vector<int> tokens(ids.data(),ids.data()+ ids.size());  // Token IDs de entrada
    for (int token : tokens) {
        std::cout << token <<" ,";
    }
    std::vector tokens2 = {101, 6207, 2003, 2919, 102};
    Eigen::MatrixXf out = emb.forward(tokens);

    std::cout << "palabras vectorizadas " << out.rows() << "x" << out.cols() << std::endl;
    std::cout << out.row(0).head(5) << std::endl;

}
