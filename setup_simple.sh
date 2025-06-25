#!/bin/bash
# Script de setup simple y robusto

echo "🚀 Configurando proyecto Mini-BERT..."
echo "======================================"

# Verificar que python3 esté disponible
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 no está instalado"
    exit 1
fi

echo "✅ Usando $(python3 --version)"

# Limpiar entorno virtual anterior si existe
if [ -d "venv" ]; then
    echo "🧹 Limpiando entorno virtual anterior..."
    rm -rf venv
fi

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv venv

# Verificar que se creó correctamente
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Error: No se pudo crear el entorno virtual"
    exit 1
fi

echo "✅ Entorno virtual creado"

# Mensaje para el usuario
echo ""
echo "🎯 Ahora ejecuta estos comandos:"
echo "======================================"
echo "# Activar entorno virtual:"
echo "source venv/bin/activate"
echo ""
echo "# Instalar dependencias:"
echo "pip install -r requirements.txt"
echo ""
echo "# Descargar pesos:"
echo "python3 download_weigths.py"
echo ""
echo "🔄 O ejecuta todo de una vez:"
echo "source venv/bin/activate && pip install -r requirements.txt && python3 download_weigths.py"