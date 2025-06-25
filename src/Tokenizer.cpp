//
// Created by JAYAN on 23/06/2025.
//
#include "../include/Tokenizer.h"
#include "../include/Config.h"
#include <filesystem>

void tokenizerWord() {
    Config& config = Config::getInstance();
    std::string pythonScript = config.getProjectPath() + "tokenizer.py";
    
    // Verificar si el script existe
    if (!std::filesystem::exists(pythonScript)) {
        std::cerr << "Error: No se encontró tokenizer.py en " << pythonScript << std::endl;
        return;
    }
    
    std::string command = "python -u \"" + pythonScript + "\"";
    system(command.c_str());
    
    std::cout << "Función tokenizerWord llamada." << std::endl;
}