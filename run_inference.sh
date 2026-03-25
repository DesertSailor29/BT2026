#!/bin/bash


# === ENVIRONMENT SETUP ===
export nnUNet_raw="Predictions/nnUNet_raw"
export nnUNet_preprocessed="Predictions/nnUNet_preprocessed" 
export nnUNet_results="./nnUNet_results"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed"

# === CONFIG ===
INPUT_DIR="./LiTS/imagesTs"
OUTPUT_ROOT="./Predictions"
NNUNET_RESULTS_PATH="./nnUNet_results"  # CHANGE THIS to your full path if needed

# List ALL your dataset IDs here (add more as needed):
DATASETS=(
    #"Dataset001_LiTS"
    #"Dataset002_LiTSMaisiCombined"
    #"Dataset003_LiTS_Full"
    #"Dataset004_LiTSMaisiFullMixed"
    "Dataset005_LiTSMaisi100_50"
    "Dataset006_LiTSMaisi100_80"
)

echo "Running inference for ${#DATASETS[@]} datasets on $INPUT_DIR"
echo "Output will be in $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

# === LOOP OVER ALL MODELS ===
for DATASET in "${DATASETS[@]}"; do
    MODEL_DIR="$NNUNET_RESULTS_PATH/$DATASET/nnUNetTrainer__nnUNetPlans__3d_fullres"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "⚠️  Skipping $DATASET (folder not found: $MODEL_DIR)"
        continue
    fi
    
    OUTPUT_DIR="$OUTPUT_ROOT/${DATASET}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "🔄 Running $DATASET"
    echo "  Input: $INPUT_DIR"
    echo "  Model: $MODEL_DIR"
    echo " Output: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    # nnU-Net v2 inference command
    nnUNetv2_predict \
        -i "$INPUT_DIR" \
        -o "$OUTPUT_DIR" \
        -d "$DATASET" \
        -c 3d_fullres \
        -f 0 \
        -chk checkpoint_best.pth \
        -npp 1 -nps 1
        
    echo "✅ Done: $DATASET -> $OUTPUT_DIR"
    echo ""
done

echo "🎉 All inference complete! Check $OUTPUT_ROOT/"
ls -la "$OUTPUT_ROOT/"
