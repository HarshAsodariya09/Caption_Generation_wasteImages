#!/bin/bash
# ============================================================
#  Waste VLM Dataset – Linux/Mac Environment Setup
# ============================================================
set -e

echo ""
echo "============================================================"
echo "  Waste VLM Dataset - Environment Setup"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/5] Virtual environment already exists."
fi

# Activate
echo "[2/5] Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "[3/5] Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "[4/5] Installing dependencies..."
pip install -r requirements.txt

# Create data directories
echo "[5/5] Creating directory structure..."
mkdir -p data/raw/TACO
mkdir -p data/raw/zerowaste
mkdir -p data/processed/{train,val,test}
mkdir -p data/annotations

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "  1. Activate:  source venv/bin/activate"
echo "  2. Download datasets (see README.md)"
echo "  3. Run:  python scripts/01_prepare_data.py"
echo "  4. Launch:  streamlit run app/streamlit_app.py"
echo ""
