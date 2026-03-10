#!/bin/bash
set -e  # Exit on any error

echo "🎯 Setting up nnUNet v2 environment for UCloud V100..."

# Set home and nnUNet dirs (adjust if needed)
export HOME=/workflows/default-storage/$USER
export BASE_DIR=$HOME/nnunet
mkdir -p $BASE_DIR/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}

# Create Python virtual environment
python3 -m venv $BASE_DIR/venv
source $BASE_DIR/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install nnUNet v2 + dependencies (V100 CUDA 11.8 compatible)
pip install torch>=2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nnunetv2>=2.5.0
pip install nibabel SimpleITK batchgenerators scikit-image tqdm

# Make script executable
chmod +x $BASE_DIR/train_nnunet.py

# Add to PATH for convenience
export PATH="$BASE_DIR/venv/bin:$PATH"

# Verify installation
nnUNetv2_print_installed_packages || echo "nnUNetv2 commands ready!"

echo "✅ Setup complete!"
echo ""
echo "📁 Copy your LiTS/ and Maisi/ folders to $BASE_DIR/"
echo "🚀 Run: cd $BASE_DIR && nohup python train_nnunet.py > nnunet_training_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo "📊 Monitor: tail -f nnunet_training_*.log"
