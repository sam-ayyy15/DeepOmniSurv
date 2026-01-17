#!/bin/bash

# DeepOmicsSurv Setup and Run Script
# This script sets up the environment and runs the experiment

set -e  # Exit on error

echo "=========================================="
echo "DeepOmicsSurv Setup and Run Script"
echo "=========================================="
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 is not installed!"
    exit 1
fi
echo ""

# Step 2: Create virtual environment (if it doesn't exist)
echo "Step 2: Setting up virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
else
    echo "Virtual environment already exists!"
fi
echo ""

# Step 3: Activate virtual environment
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo ""

# Step 4: Upgrade pip
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "pip upgraded!"
echo ""

# Step 5: Install dependencies
echo "Step 5: Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt --quiet
echo "Dependencies installed!"
echo ""

# Step 6: Verify installation
echo "Step 6: Verifying installation..."
python3 -c "import numpy, pandas, tensorflow, lifelines, shap; print('✓ All dependencies installed successfully!')"
echo ""

# Step 7: Check data files
echo "Step 7: Checking data files..."
if [ -d "tcga_data" ] && [ -f "tcga_data/clinical_data.csv" ]; then
    echo "✓ Data files found!"
    echo "  - clinical_data.csv"
    echo "  - mrna_expression.csv"
    echo "  - dna_methylation.csv"
    echo "  - cna_data.csv"
else
    echo "⚠ Warning: Data files not found. The script will attempt to generate them."
fi
echo ""

# Step 8: Run the experiment
echo "=========================================="
echo "Starting DeepOmicsSurv Experiment"
echo "=========================================="
echo ""
echo "Note: This may take 30-90 minutes depending on your hardware."
echo ""

python3 deepomicsurv_implementation.py

echo ""
echo "=========================================="
echo "Experiment completed!"
echo "=========================================="
echo ""
echo "Check the following files for results:"
echo "  - shap_importance.png (if SHAP analysis succeeded)"
echo "  - survival_curves.png (if survival analysis succeeded)"
echo ""

