#!/bin/bash
set -e  # Exit on any error

echo "🎯 Setting up nnUNet v2 environment (system-wide install)..."

# Use current working directory
BASE_DIR=$(pwd)
echo "📁 Working in: $BASE_DIR"

# Create nnUNet standard dirs
mkdir -p {nnUNet_raw,nnUNet_preprocessed,nnUNet_results}

# Export nnUNet paths (nnUNet v2 expects these)
export nnUNet_raw="$BASE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnUNet_preprocessed" 
export nnUNet_results="$BASE_DIR/nnUNet_results"

# Upgrade system pip
python3 -m pip install --user --upgrade pip setuptools wheel

# Check CUDA version first
echo "🔍 Checking GPU/CUDA..."
nvidia-smi

# Verify PyTorch + CUDA before install
python3 -c "import torch; print(f'PyTorch ready: {torch.__version__} CUDA: {torch.version.cuda} Available: {torch.cuda.is_available()}')"

# Install EXACT matching packages from your original venv
pip3 install --user \
  "torch==2.3.0+cu121" "torchvision==0.18.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121

pip3 install --user \
  "nnunetv2==2.6.4" \
  "SimpleITK==2.5.3" "nibabel==5.4.2" "pandas==3.0.1" \
  "scikit-image==0.26.0" "scipy==1.17.1" "seaborn==0.13.2" \
  "matplotlib==3.10.8" "plotly==6.6.0" "kaleido==1.2.0"

# Verify installation  
nnUNetv2_print_all_installed_packages || echo "Note: print_all command may vary by version"

# Test imports
python3 -c "
import torch, nnunetv2, nibabel, SimpleITK, pandas, plotly
print('✅ All core packages imported successfully!')
print(f'GPU ready: {torch.cuda.is_available()}')
"

echo "✅ Setup complete!"
echo ""
echo "📁 Verify your LiTS/ and Maisi/ folders exist here:"
ls -la LiTS/ Maisi/ 2>/dev/null || echo "📁 Copy LiTS/ Maisi/ folders here first"
echo ""
echo "🚀 Run: nohup python3 your_train_script.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "📊 Monitor: tail -f training_*.log"
echo "🖥️  Results: $BASE_DIR/nnUNet_results/"
echo "💡 Tip: Add 'export nnUNet_*=...' to ~/.bashrc for persistence"