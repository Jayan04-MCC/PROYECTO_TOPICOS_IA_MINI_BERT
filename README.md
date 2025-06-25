# Mini-BERT C++ Implementation

Implementación de un encoder BERT mini en C++ usando Eigen, basado en el modelo `sentence-transformers/all-MiniLM-L6-v2`.

## 🚀 Setup Rápido

### Opción 1: Script automático
```bash
# Crear entorno virtual
./setup_simple.sh

# Luego ejecutar (todo en una línea):
source venv/bin/activate && pip install -r requirements.txt && python3 download_weigths.py
```

### Opción 2: Manual paso a paso
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Descargar pesos del modelo
```bash
# Asegúrate de tener el entorno virtual activado
python3 download_weigths.py
```

### 3. Instalar Eigen (C++)
```bash
# Ubuntu/Debian
sudo apt install libeigen3-dev

# macOS
brew install eigen

# O descargar manualmente desde https://eigen.tuxfamily.org/
```

### 4. Compilar proyecto
```bash
# Primera vez - configurar y compilar
mkdir build && cd build
cmake ..
make

# Recompilaciones posteriores (solo si cambias código C++)
cd build && make
```

### 5. Ejecutar
```bash
# Tokenizar una frase (con entorno virtual activado)
python3 tokenizer.py

# Ejecutar el modelo C++ (desde directorio build)
./PROYECTO_TOPICOS_IA

# O desde directorio raíz
./build/PROYECTO_TOPICOS_IA
```

## 📁 Estructura del Proyecto

```
├── include/           # Headers C++
│   ├── Config.h      # Sistema de configuración
│   ├── Embeddings.h  # Capa de embeddings
│   └── ...
├── src/              # Implementaciones C++
├── weights/          # Pesos del modelo (generado)
├── words/            # Vocabulario
│   └── vocab.txt
├── download_weigths.py # Script de descarga automática
├── tokenizer.py      # Tokenizador Python
└── requirements.txt  # Dependencias Python
```

## 🧠 Arquitectura

- **Embeddings**: Word + Position + LayerNorm
- **Transformer**: 6 capas con atención multi-head
- **Dimensiones**: 384-dimensional embeddings
- **Vocabulario**: 30,522 tokens

## 📝 Uso

```cpp
// Cargar configuración
Config& config = Config::getInstance();

// Crear capa de embeddings
EmbeddingLayer emb(
    config.getWordEmbeddingsPath(),
    config.getPositionEmbeddingsPath(),
    config.getLayerNormWeightPath(),
    config.getLayerNormBiasPath()
);

// Tokenizar y procesar
std::vector<int> tokens = {101, 6207, 2003, 2919, 102}; // [CLS] hello world [SEP]
Eigen::MatrixXf embeddings = emb.forward(tokens);
```

## 🔧 Configuración

El proyecto usa un sistema de configuración flexible que detecta automáticamente los paths:

- **Pesos**: `weights/pesos_comprimidos/content/pesos_csv/`
- **Tokens**: `tokens.csv` 
- **Vocabulario**: `words/vocab.txt`

## ⚡ Comandos de Desarrollo

```bash
# Solo recompilar después de cambios en C++
cd build && make

# Solo ejecutar (si ya está compilado)
./build/PROYECTO_TOPICOS_IA

# Regenerar tokens y ejecutar
source venv/bin/activate && python3 tokenizer.py && ./build/PROYECTO_TOPICOS_IA
```

## 📊 Estado del Desarrollo

- ✅ Sistema de configuración
- ✅ Carga de pesos CSV
- ✅ Embeddings (Word + Position + LayerNorm)
- ⏳ Capas Transformer
- ⏳ Atención Multi-Head
- ⏳ Feed-Forward
- ⏳ Pipeline completo

## 🎯 Objetivo

Crear un encoder BERT funcional capaz de:
- Procesar frases en lenguaje natural
- Generar embeddings semánticos
- Calcular similitud entre frases

## 📋 Requisitos

- **C++20** o superior
- **Eigen 3.3+** para operaciones matriciales
- **Python 3.7+** para descarga de pesos y tokenización
- **CMake 3.16+** para compilación