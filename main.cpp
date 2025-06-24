#include <iostream>
#include "D:\PROYECTO-TOPICOS-IA\include\Utils.h"
#include "D:\PROYECTO-TOPICOS-IA\include\Tokenizer.h"
#include "D:\PROYECTO-TOPICOS-IA\include\Embeddings.h"
std::string PATH = "D:/PROYECTO-TOPICOS-IA/Weights/pesos_comprimidos/content/pesos_csv/";
std::string WORD_PATH =PATH + "embeddings_word_embeddings_weight.csv";
std::string POSITION_PATH =PATH +"embeddings_position_embeddings_weight.csv";
std::string IN_WEIGHT_PATH =PATH +"embeddings_LayerNorm_weight.csv";
std::string IN_BIAS_PATH =PATH +"embeddings_LayerNorm_bias.csv";
int main() {

    EmbeddingLayer emb(WORD_PATH, POSITION_PATH, IN_WEIGHT_PATH, IN_BIAS_PATH);
    tokenizerWord();
    //capturamos los ids
    Eigen::VectorXf ids = loadCSVtoVector("D:/PROYECTO-TOPICOS-IA/tokens.csv");
    std::vector<int> tokens(ids.data(),ids.data()+ ids.size());  // Token IDs de entrada
    for (int token : tokens) {
        std::cout << token <<" ,";
    }
    std::vector tokens2 = {101, 6207, 2003, 2919, 102};
    Eigen::MatrixXf out = emb.forward(tokens);

    std::cout << "palabras vectorizadas " << out.rows() << "x" << out.cols() << std::endl;
    std::cout << out.row(0).head(5) << std::endl;

}
