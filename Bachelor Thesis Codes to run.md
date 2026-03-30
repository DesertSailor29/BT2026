---
Parent: "[[Bachelor Thesis]]"
---

# Training
```
pkill -f train_nnunet.py
rm -f training_*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_debug_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_debug_*.log
```

```
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_FULL_LITS$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_FULL_LITS*.log
```

```
rm -f training_FULL_DATASET*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_FULL_DATASET$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_FULL_DATASET*.log
```

```
rm -f training_MIXED_100_50*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_MIXED_100_50$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_MIXED_100_50*.log
```

```
rm -f training_MIXED_100_80*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_MIXED_100_80$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_MIXED_100_80*.log
```

```
rm -f training_MIXED_100_30*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python train_nnunet.py" > training_MIXED_80_AND_30$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f training_MIXED_80_AND_30*.log
```
# Inference
```
chmod +x run_inference.sh
source venv/bin/activate
nohup bash -c 'source venv/bin/activate && ./run_inference.sh' > inference_MIXED_DATASETS_7_100_30$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f inference_*.log
```

# Comparison

```
rm -f compare*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python compare.py" > compare$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f compare*.log
```

```
rm -f compare*.log*
source venv/bin/activate
nohup bash -c "source venv/bin/activate && python compare_figures.py" > compare3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
tail -f compare3_*.log
```


nohup bash -c '
source venv/bin/activate &&
export nnUNet_raw="/work/Bachelor Thesis/nnUNet_raw" &&
export nnUNet_preprocessed="/work/Bachelor Thesis/nnUNet_preprocessed" &&
export nnUNet_results="/work/Bachelor Thesis/nnUNet_results" &&
nnUNetv2_train 6 3d_fullres 0 --c
' > training_MIXED_100_80_$(date +%Y%m%d%H%M%S).log 2>&1 &
tail -f training_MIXED_100_80*.log
