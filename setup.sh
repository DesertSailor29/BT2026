#!/bin/bash
set -e  # Exit on any error

echo "🎯 Setting up nnUNet v2 environment (local dir)..."

# Use current working directory
BASE_DIR=$(pwd)
echo "📁 Working in: $BASE_DIR"

# Create nnUNet standard dirs
mkdir -p {nnUNet_raw,nnUNet_preprocessed,nnUNet_results}

# Export nnUNet paths (nnUNet v2 expects these)
export nnUNet_raw="$BASE_DIR/nnUNet_raw"
export nnUNet_preprocessed="$BASE_DIR/nnUNet_preprocessed" 
export nnUNet_results="$BASE_DIR/nnUNet_results"

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Check CUDA version first
echo "🔍 Checking GPU/CUDA..."
nvidia-smi

# Install PyTorch (cu118 safe for most UCloud V100s)
python -m pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118

# Install nnUNet v2
python -m pip install "nnunetv2>=2.5.0"

# Verify installation  
nnUNetv2_print_installed_packages

echo "✅ Setup complete!"
echo ""
echo "📁 Verify your LiTS/ and Maisi/ folders exist here:"
ls -la LiTS/ Maisi/
echo ""
echo "🚀 Run: nohup python train_nnunet.py > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "📊 Monitor: tail -f training_*.log"
echo "🖥️  Results: $BASE_DIR/nnUNet_results/"
