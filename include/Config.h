#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <filesystem>
#include <map>

class Config {
public:
    static Config& getInstance();
    
    // Path getters
    std::string getProjectPath() const;
    std::string getWeightsPath() const;
    std::string getTokensPath() const;
    std::string getVocabPath() const;
    
    // Weight file paths
    std::string getWordEmbeddingsPath() const;
    std::string getPositionEmbeddingsPath() const;
    std::string getLayerNormWeightPath() const;
    std::string getLayerNormBiasPath() const;
    
    // Configuration setters
    void setWeightsDirectory(const std::string& path);
    void setProjectDirectory(const std::string& path);

private:
    Config() = default;
    std::string project_path;
    std::string weights_dir = "weights/pesos_comprimidos/content/pesos_csv/";
    
    void initializePaths();
};

#endif // CONFIG_H