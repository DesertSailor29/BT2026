#!/bin/bash
set -e  # Exit on any error

echo "🎯 Setting up nnUNet v2 environment (VENV - isolated)..."

# Use current working directory
BASE_DIR=$(pwd)
echo "📁 Working in: $BASE_DIR"

# Create nnUNet standard dirs
mkdir -p {nnUNet_raw,nnUNet_preprocessed,nnUNet_results}

# Export nnUNet paths
export nnUNet_raw="$BASE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnUNet_preprocessed" 
export nnUNet_results="$BASE_DIR/nnUNet_results"

# CREATE & ACTIVATE VENV
python3 -m venv venv
source venv/bin/activate
echo "✅ venv: $(which python)"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Check CUDA
echo "🔍 GPU/CUDA..."
nvidia-smi

# Install PyTorch FIRST (exact versions, NO +cu121 suffix)
pip install \
  "torch==2.3.0" "torchvision==0.18.0" \
  --index-url https://download.pytorch.org/whl/cu121

# Install nnU-Net + your script deps
pip install \
  "nnunetv2==2.6.4" \
  "SimpleITK==2.5.3" "nibabel==5.4.2" "pandas==3.0.1" \
  "scikit-image==0.26.0" "scipy==1.17.1" "seaborn==0.13.2" \
  "matplotlib==3.10.8" "plotly==6.6.0" "kaleido==1.2.0"

# Verify installation
nnUNetv2_print_all_installed_packages || echo "✓ nnU-Net ready"

# Test imports
python -c "
import torch, nnunetv2, nibabel, SimpleITK, pandas, plotly
print('✅ All packages OK!')
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
"

echo "✅ Setup complete!"
echo "🚀 Run: source venv/bin/activate && python your_train_script.py"
echo "📁 Results: $BASE_DIR/nnUNet_results/"