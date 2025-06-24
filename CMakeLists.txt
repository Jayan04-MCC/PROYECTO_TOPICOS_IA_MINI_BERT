cmake_minimum_required(VERSION 3.30)
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
        src/Tokenizer.cpp
        src/Embeddings.cpp
        src/Transformer.cpp
        src/LayerNorm.cpp
        src/Attention.cpp
        src/FeedForward.cpp
        src/Utils.cpp)
include_directories(third_party/eigen-3.3.9)
