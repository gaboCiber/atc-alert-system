@echo off
REM Script para configurar el entorno virtual de Python 3.9 para DeepFilterNet (Windows)
REM 
REM Uso:
REM   setup_venv.bat [ruta_al_venv]
REM
REM Por defecto crea el venv en .venv-deepfilter

setlocal enabledelayedexpansion

REM Configuración
if "%~1"=="" (
    set "VENV_PATH=.venv-deepfilter"
) else (
    set "VENV_PATH=%~1"
)

for %%I in ("%VENV_PATH%") do set "VENV_NAME=%%~nxI"

echo ==============================================
echo Setup de Entorno Virtual para DeepFilterNet
echo ==============================================
echo.
echo Ruta del venv: %VENV_PATH%
echo.

REM Verificar que uv está instalado
uv --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'uv' no esta instalado.
    echo Por favor instala uv primero: https://github.com/astral-sh/uv
    exit /b 1
)

echo [ok] uv encontrado
echo.

REM Verificar que Python 3.9 está disponible
uv python find 3.9 >nul 2>&1
if errorlevel 1 (
    echo Instalando Python 3.9...
    uv python install 3.9
) else (
    echo [ok] Python 3.9 disponible
)
echo.

REM Crear entorno virtual
echo Creando entorno virtual en %VENV_PATH%...
uv venv "%VENV_PATH%" --python 3.9
echo [ok] Entorno virtual creado
echo.

REM Instalar PyTorch
echo Instalando PyTorch...
uv pip install --python "%VENV_PATH%\Scripts\python.exe" torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html
echo [ok] PyTorch instalado
echo.

REM Instalar maturin
echo Instalando maturin...
uv pip install --python "%VENV_PATH%\Scripts\python.exe" maturin
echo [ok] maturin instalado
echo.

REM Instalar deepfilterlib
echo Instalando deepfilterlib...
uv pip install --python "%VENV_PATH%\Scripts\python.exe" deepfilterlib==0.5.6 --no-build-isolation
echo [ok] deepfilterlib instalado
echo.

REM Instalar deepfilternet
echo Instalando deepfilternet...
uv pip install --python "%VENV_PATH%\Scripts\python.exe" deepfilternet --no-build-isolation
echo [ok] deepfilternet instalado
echo.

REM Verificación
echo Verificando instalacion...
"%VENV_PATH%\Scripts\python.exe" -c "from df.enhance import enhance, init_df" >nul 2>&1
if errorlevel 1 (
    echo [ADVERTENCIA] No se pudo importar DeepFilterNet
    echo Intenta activar el venv manualmente y verificar.
) else (
    echo [ok] DeepFilterNet importa correctamente
)
echo.

REM Resumen
echo ==============================================
echo Setup completado!
echo ==============================================
echo.
echo Para usar DeepFilterNet en el pipeline ASR:
echo.
echo   python -m ASR.transcription.cli ^
echo       --model whisperatc ^
echo       --input ./audio ^
echo       --output results.csv ^
echo       --noise-reduction ^
echo       --deepfilter-venv ./%VENV_NAME%
echo.
echo O en Python:
echo.
echo   from ASR.noise_reduction import DeepFilterNetWrapper
"  echo   wrapper = DeepFilterNetWrapper(venv_path='./%VENV_NAME%')"
echo   cleaned = wrapper.clean_audio('input.wav')
echo.

endlocal
