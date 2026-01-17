# DeepOmniSurv
Deep Learning-Based Model for Survival Prediction of Oral Cancer.

Developers:
- Samay Shetty
- Swanjith AK

## Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup and execution script
./setup_and_run.sh
```

This script will:
1. Create a virtual environment
2. Install all dependencies
3. Verify the installation
4. Run the experiment

### Option 2: Manual Setup

#### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 3: Verify Data Files

The script requires data files in the `tcga_data/` directory:
- `clinical_data.csv`
- `mrna_expression.csv`
- `dna_methylation.csv`
- `cna_data.csv`

If these files don't exist, the script will automatically generate them.

#### Step 4: Run the Experiment

```bash
python3 deepomicsurv_implementation.py
```

## Requirements

- Python 3.8 or higher (3.9-3.11 recommended)
- 8GB+ RAM recommended
- 30-90 minutes runtime (depending on hardware)

## Dependencies

All dependencies are listed in `requirements.txt`:
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- tensorflow >= 2.8.0
- lifelines >= 0.26.0
- shap >= 0.40.0

## Output Files

After running the experiment, you'll get:
- `shap_importance.png` - SHAP feature importance visualization
- `survival_curves.png` - Kaplan-Meier survival curves
- Console output with performance metrics (C-index, Brier scores, etc.)

## What the Script Does

1. **Data Loading**: Loads TCGA-HNSC data from CSV files
2. **Preprocessing**: Preprocesses clinical and omics data
3. **Dimensionality Reduction**: Applies autoencoders to reduce omics data dimensionality
4. **Model Training**: Trains DeepOmicsSurv on multiple data combinations:
   - Clinical only
   - mRNA + Clinical
   - DNA Methylation + Clinical
   - CNA + Clinical
   - Multi-omics (all omics + clinical)
5. **Evaluation**: Calculates performance metrics (C-index, Brier scores)
6. **SHAP Analysis**: Generates feature importance plots
7. **Survival Analysis**: Creates Kaplan-Meier survival curves
8. **Baseline Comparison**: Compares with DeepSurv, CNN, and RNN baselines

## Troubleshooting

### Memory Errors
If you encounter memory errors, you can reduce:
- Batch size (line 659: change `batch_size=32` to `batch_size=16`)
- Autoencoder epochs (line 558: change `epochs=300` to `epochs=100`)

### TensorFlow GPU Issues
If you have GPU issues, install CPU-only TensorFlow:
```bash
pip uninstall tensorflow
pip install tensorflow-cpu
```

### SHAP Errors
SHAP analysis is optional. If it fails, the script will continue without it.

## Code Fixes Applied

The following fixes have been applied to ensure compatibility:
- Fixed pandas `fillna(method=...)` deprecation
- Removed invalid `restore_best_weights` parameter from EarlyStopping
- Fixed SHAP initialization for non-Jupyter environments
- Fixed import for data generator module
- Fixed matplotlib `plt.show()` for non-interactive environments

## Citation

Based on: "DeepOmicsSurv: a deep learning-based model for survival prediction of oral cancer"
