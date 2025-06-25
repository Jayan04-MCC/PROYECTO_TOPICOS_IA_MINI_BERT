#!/usr/bin/env python3
"""
Script automatizado para descargar y convertir los pesos del modelo all-MiniLM-L6-v2
de HuggingFace a formato CSV para usar en C++.

Uso: python download_weights.py
"""

import os
import sys
import numpy as np
from pathlib import Path

def check_dependencies():
    """Verifica que las dependencias estén instaladas"""
    try:
        from sentence_transformers import SentenceTransformer
        from safetensors.torch import load_file
        import torch
        print("✅ Todas las dependencias están instaladas")
        return True
    except ImportError as e:
        print(f"❌ Falta dependencia: {e}")
        print("Instala las dependencias con: pip install -r requirements.txt")
        return False

def download_model(model_name='all-MiniLM-L6-v2'):
    """Descarga el modelo de SentenceTransformers"""
    print(f"📥 Descargando modelo {model_name}...")
    from sentence_transformers import SentenceTransformer
    
    model_dir = Path(model_name)
    if model_dir.exists():
        print(f"✅ Modelo ya existe en {model_dir}")
    else:
        model = SentenceTransformer(model_name)
        model.save(str(model_dir))
        print(f"✅ Modelo descargado en {model_dir}")
    
    return model_dir

def convert_to_csv(model_dir, output_dir="weights/pesos_comprimidos/content/pesos_csv"):
    """Convierte los pesos safetensors a CSV"""
    from safetensors.torch import load_file
    
    # Buscar archivo safetensors
    safetensors_path = model_dir / "model.safetensors"
    if not safetensors_path.exists():
        # Buscar en subdirectorios
        for path in model_dir.rglob("*.safetensors"):
            safetensors_path = path
            break
        else:
            print(f"❌ No se encontró archivo safetensors en {model_dir}")
            return False
    
    print(f"📄 Cargando pesos desde {safetensors_path}")
    state_dict = load_file(str(safetensors_path))
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio de salida: {output_path}")
    
    # Convertir cada tensor a CSV
    converted_count = 0
    for name, tensor in state_dict.items():
        array = tensor.cpu().numpy()
        clean_name = name.replace('.', '_').replace('/', '_')
        filename = output_path / f"{clean_name}.csv"
        
        # Guardar como CSV
        np.savetxt(filename, array, delimiter=",", fmt='%.8f')
        print(f"💾 {filename.name} - Shape: {array.shape}")
        converted_count += 1
    
    print(f"✅ Convertidos {converted_count} archivos a CSV")
    return True

def main():
    """Función principal"""
    print("🚀 Iniciando descarga y conversión de pesos del modelo all-MiniLM-L6-v2")
    print("="*70)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    try:
        # Descargar modelo
        model_dir = download_model()
        
        # Convertir a CSV
        if convert_to_csv(model_dir):
            print("="*70)
            print("🎉 ¡Proceso completado exitosamente!")
            print("Los pesos están listos para usar en el proyecto C++")
        else:
            print("❌ Error en la conversión")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()