@echo off
REM ============================================================
REM  Waste VLM Dataset – Windows Environment Setup
REM ============================================================
echo.
echo ============================================================
echo   Waste VLM Dataset - Environment Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
if not exist "venv" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment already exists.
)

REM Activate
echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo [4/5] Installing dependencies...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [INFO] NVIDIA GPU detected. Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] No NVIDIA GPU detected. Installing standard PyTorch...
    pip install torch torchvision
)
pip install -r requirements.txt

REM Create data directories
echo [5/5] Creating directory structure...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\raw\TACO" mkdir "data\raw\TACO"
if not exist "data\raw\zerowaste" mkdir "data\raw\zerowaste"
if not exist "data\processed\train" mkdir "data\processed\train"
if not exist "data\processed\val" mkdir "data\processed\val"
if not exist "data\processed\test" mkdir "data\processed\test"
if not exist "data\annotations" mkdir "data\annotations"

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo   Next steps:
echo   1. Activate the environment:  venv\Scripts\activate
echo   2. Download datasets:
echo      - TACO: cd data\raw ^& git clone https://github.com/pedropro/TACO.git
echo      - ZeroWaste: download from https://github.com/dbash/zerowaste
echo   3. Run data preparation:  python scripts\01_prepare_data.py
echo   4. Launch Streamlit app:  streamlit run app\streamlit_app.py
echo.
echo   OR use the Colab notebook for GPU caption generation:
echo      notebooks\waste_vlm_colab_setup.ipynb
echo.
pause
