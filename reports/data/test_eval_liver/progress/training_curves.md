# Training curves per liver arm (table view of training_curves.png)

Smoothed = centred 51-epoch rolling mean. Validation = 20 real LiTS cases. Written by `nnunet/training_curves.py`.

| run | epochs | lowest smoothed val loss (epoch) | val loss at last smoothed epoch | gap at last smoothed epoch | best smoothed tumour pseudo Dice (epoch) | best smoothed liver pseudo Dice |
|---|---|---|---|---|---|---|
| D101 f0 all | 1000 | -0.434 (911) | -0.407 | 0.159 | 0.561 (991) | 0.904 |
| D101 f1 all_rep2 | 1000 | -0.476 (999) | -0.476 | 0.096 | 0.614 (999) | 0.912 |
| D101 f2 ccReject | 1000 | -0.390 (756) | -0.327 | 0.350 | 0.501 (943) | 0.901 |
| D101 f3 all_rep3 | 1000 | -0.452 (995) | -0.450 | 0.126 | 0.588 (999) | 0.913 |
| D103 f0 ccRepair | 1000 | -0.424 (854) | -0.386 | 0.224 | 0.578 (922) | 0.903 |
| D103 f1 ccRepairDel | 1000 | -0.443 (623) | -0.383 | 0.258 | 0.559 (929) | 0.910 |
| D104 f0 ccAucRepair | 1000 | -0.444 (966) | -0.429 | 0.183 | 0.582 (948) | 0.908 |
| D104 f1 ccAucRepair_rep2 | 1000 | -0.461 (992) | -0.450 | 0.160 | 0.594 (997) | 0.908 |
| D104 f2 ccAucRepair_sizeLarge | 1000 | -0.448 (429) | -0.419 | 0.256 | 0.556 (991) | 0.906 |
| D104 f3 ccAucRepair_sizeRand | 1000 | -0.414 (808) | -0.359 | 0.228 | 0.566 (951) | 0.905 |
| D104 f4 ccAucRepair_sizeSmall | 1000 | -0.233 (549) | -0.188 | 0.334 | 0.066 (843) | 0.903 |
| D104 f5 ccAucRepair_sizeLarge_rep2 | 1000 | -0.467 (891) | -0.426 | 0.246 | 0.577 (890) | 0.914 |
| D113 f0 posRepair | 1000 | -0.443 (944) | -0.401 | 0.169 | 0.583 (990) | 0.907 |
| D113 f1 posRepair_rep2 | 1000 | -0.460 (998) | -0.460 | 0.112 | 0.603 (999) | 0.908 |
