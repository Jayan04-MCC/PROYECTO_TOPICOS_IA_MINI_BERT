cmake_minimum_required(VERSION 3.16)
project(PROYECTO_TOPICOS_IA)

set(CMAKE_CXX_STANDARD 20)

add_executable(PROYECTO_TOPICOS_IA main.cpp
        include/Tokenizer.h
        include/Embeddings.h
        include/Transformer.h
        include/LayerNorm.h
        include/Attention.h
        include/FeedForward.h
        include/Utils.h
        include/Config.h
        src/Tokenizer.cpp
        src/Embeddings.cpp
        src/Transformer.cpp
        src/LayerNorm.cpp
        src/Attention.cpp
        src/FeedForward.cpp
        src/Utils.cpp
        src/Config.cpp)
include_directories(eigen-3.3.9)
