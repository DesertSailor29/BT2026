# Training curves per pancreas arm (table view of training_curves.png)

Smoothed = centred 51-epoch rolling mean. Validation = 43 real MSD07 cases. Written by `nnunet/training_curves.py`.

| run | epochs | lowest smoothed val loss (epoch) | val loss at last smoothed epoch | gap at last smoothed epoch | best smoothed tumour pseudo Dice (epoch) | best smoothed pancreas pseudo Dice |
|---|---|---|---|---|---|---|
| D102 f0 all | 1000 | -0.077 (449) | -0.030 | 0.619 | 0.312 (858) | 0.254 |
| D102 f1 all_rep2 | 1000 | -0.087 (454) | 0.003 | 0.655 | 0.314 (736) | 0.279 |
