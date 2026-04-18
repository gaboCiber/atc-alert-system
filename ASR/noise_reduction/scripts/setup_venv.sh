#!/bin/bash
# Script para configurar el entorno virtual de Python 3.9 para DeepFilterNet
# 
# Uso:
#   chmod +x setup_venv.sh
#   ./setup_venv.sh [ruta_al_venv]
#
# Por defecto crea el venv en ./.venv-deepfilter

set -e  # Exit on error

# Configuración
VENV_PATH="${1:-.venv-deepfilter}"
VENV_NAME=$(basename "$VENV_PATH")

echo "=============================================="
echo "Setup de Entorno Virtual para DeepFilterNet"
echo "=============================================="
echo ""
echo "Ruta del venv: $VENV_PATH"
echo ""

# Verificar que uv está instalado
if ! command -v uv &> /dev/null; then
    echo "ERROR: 'uv' no está instalado."
    echo "Por favor instala uv primero: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✓ uv encontrado"
echo ""

# Verificar que Python 3.9 está disponible
echo "Verificando Python 3.9..."
if uv python find 3.9 &> /dev/null; then
    echo "✓ Python 3.9 disponible"
else
    echo "Instalando Python 3.9..."
    uv python install 3.9
fi
echo ""

# Crear entorno virtual
echo "Creando entorno virtual en $VENV_PATH..."
uv venv "$VENV_PATH" --python 3.9
echo "✓ Entorno virtual creado"
echo ""

# Función para ejecutar comandos en el venv
run_in_venv() {
    uv pip install --python "$VENV_PATH/bin/python" "$@"
}

# Instalar PyTorch
echo "Instalando PyTorch..."
run_in_venv torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
echo "✓ PyTorch instalado"
echo ""

# Instalar maturin (necesario para compilar deepfilterlib)
echo "Instalando maturin..."
run_in_venv maturin
echo "✓ maturin instalado"
echo ""

# Instalar deepfilterlib
echo "Instalando deepfilterlib..."
run_in_venv deepfilterlib==0.5.6 --no-build-isolation
echo "✓ deepfilterlib instalado"
echo ""

# Instalar deepfilternet
echo "Instalando deepfilternet..."
run_in_venv deepfilternet --no-build-isolation
echo "✓ deepfilternet instalado"
echo ""

# Verificación
echo "Verificando instalación..."
if "$VENV_PATH/bin/python" -c "from df.enhance import enhance, init_df" 2>/dev/null; then
    echo "✓ DeepFilterNet importa correctamente"
else
    echo "⚠️  Advertencia: No se pudo importar DeepFilterNet"
    echo "   Intenta activar el venv manualmente y verificar:"
    echo "   source $VENV_PATH/bin/activate"
    echo "   python -c 'from df.enhance import enhance, init_df'"
fi
echo ""

# Resumen
echo "=============================================="
echo "Setup completado!"
echo "=============================================="
echo ""
echo "Para usar DeepFilterNet en el pipeline ASR:"
echo ""
echo "  python -m ASR.transcription.cli \\"
echo "      --model whisperatc \\"
echo "      --input ./audio \\"
echo "      --output results.csv \\"
echo "      --noise-reduction \\"
echo "      --deepfilter-venv ./$VENV_NAME"
echo ""
echo "O en Python:"
echo ""
echo "  from ASR.noise_reduction import DeepFilterNetWrapper"
echo "  wrapper = DeepFilterNetWrapper(venv_path='./$VENV_NAME')"
echo "  cleaned = wrapper.clean_audio('input.wav')"
echo ""
