# Training curves per liver arm (table view of training_curves.png)

Smoothed = centred 51-epoch rolling mean. Validation = 20 real LiTS cases. Written by `nnunet/training_curves.py`.

| run | epochs | lowest smoothed val loss (epoch) | val loss at last smoothed epoch | gap at last smoothed epoch | best smoothed tumour pseudo Dice (epoch) | best smoothed liver pseudo Dice |
|---|---|---|---|---|---|---|
| D101 f0 all | 1000 | -0.434 (911) | -0.407 | 0.159 | 0.561 (991) | 0.904 |
| D101 f1 all_rep2 | 1000 | -0.476 (999) | -0.476 | 0.096 | 0.614 (999) | 0.912 |
| D101 f2 ccReject | 1000 | -0.390 (756) | -0.327 | 0.350 | 0.501 (943) | 0.901 |
| D103 f0 ccRepair | 602 | -0.407 (555) | -0.389 | 0.178 | 0.535 (596) | 0.898 |
