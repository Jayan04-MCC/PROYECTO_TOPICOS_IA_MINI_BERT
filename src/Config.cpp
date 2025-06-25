#include "../include/Config.h"

Config& Config::getInstance() {
    static Config instance;
    if (instance.project_path.empty()) {
        instance.initializePaths();
    }
    return instance;
}

void Config::initializePaths() {
    project_path = std::filesystem::current_path().string() + "/";
}

std::string Config::getProjectPath() const {
    return project_path;
}

std::string Config::getWeightsPath() const {
    return project_path + weights_dir;
}

std::string Config::getTokensPath() const {
    return project_path + "tokens.csv";
}

std::string Config::getVocabPath() const {
    return project_path + "words/vocab.txt";
}

std::string Config::getWordEmbeddingsPath() const {
    return getWeightsPath() + "embeddings_word_embeddings_weight.csv";
}

std::string Config::getPositionEmbeddingsPath() const {
    return getWeightsPath() + "embeddings_position_embeddings_weight.csv";
}

std::string Config::getLayerNormWeightPath() const {
    return getWeightsPath() + "embeddings_LayerNorm_weight.csv";
}

std::string Config::getLayerNormBiasPath() const {
    return getWeightsPath() + "embeddings_LayerNorm_bias.csv";
}

void Config::setWeightsDirectory(const std::string& path) {
    weights_dir = path;
    if (!weights_dir.empty() && weights_dir.back() != '/') {
        weights_dir += "/";
    }
}

void Config::setProjectDirectory(const std::string& path) {
    project_path = path;
    if (!project_path.empty() && project_path.back() != '/') {
        project_path += "/";
    }
}