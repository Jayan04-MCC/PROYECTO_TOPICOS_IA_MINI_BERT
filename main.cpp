 #include <iostream>
#include "../../include/Utils.h"
#include "../../include/Tokenizer.h"
#include "../../include/Embeddings.h"
#include "../../include/TransformerEncoder.h"
std::string PATH = "D:/PROYECTO-TOPICOS-IA/Weights/pesos_comprimidos/content/pesos_csv/";
std::string WORD_PATH =PATH + "embeddings_word_embeddings_weight.csv";
std::string POSITION_PATH =PATH +"embeddings_position_embeddings_weight.csv";
std::string IN_WEIGHT_PATH =PATH +"embeddings_LayerNorm_weight.csv";
std::string IN_BIAS_PATH =PATH +"embeddings_LayerNorm_bias.csv";

Eigen::VectorXf mean_pooling(const Eigen::MatrixXf& token_embeddings) {
    return token_embeddings.colwise().mean();
}

float cosine_similarity(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    float dot = a.dot(b);
    float norm_a = a.norm();
    float norm_b = b.norm();
    return dot / (norm_a * norm_b + 1e-8);
}
int main() {

    EmbeddingLayer emb(WORD_PATH, POSITION_PATH, IN_WEIGHT_PATH, IN_BIAS_PATH);
    tokenizerWord();
    // ------------------ Frase 1 -------------------
    Eigen::VectorXf ids1 = loadCSVtoVector("D:/PROYECTO-TOPICOS-IA/cmake-build-debug/tokens1.csv");
    std::vector<int> tokens1(ids1.data(),ids1.data()+ ids1.size());  // Token IDs de entrada
    Eigen::MatrixXf emb_out1 = emb.forward(tokens1);
    TransformerEncoder encoder1(PATH);
    Eigen::MatrixXf out1 = encoder1.forward(emb_out1);
    Eigen::VectorXf vec1 = mean_pooling(out1);

    // ------------------ Frase 2 -------------------
    Eigen::VectorXf ids2 = loadCSVtoVector("D:/PROYECTO-TOPICOS-IA/cmake-build-debug/tokens2.csv");
    std::vector<int> tokens2(ids2.data(),ids2.data()+ ids2.size());  // Token IDs de entrada
    Eigen::MatrixXf emb_out2 = emb.forward(tokens2);
    TransformerEncoder encoder2(PATH);
    Eigen::MatrixXf out2 = encoder2.forward(emb_out2);
    Eigen::VectorXf vec2 = mean_pooling(out2);

    // ------------------ Similitud -------------------
    float sim = cosine_similarity(vec1, vec2);
    std::cout << "Similitud semántica: " << sim * 100.0f << "%" << std::endl;

    return 0;
}