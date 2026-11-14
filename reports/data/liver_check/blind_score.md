# Blind liver review: marks vs liver flags

Marks: `reports/figures/liver_review/blind/marks.csv`. Key: 60 codes.

| | ok | bad | unsure | missing |
|---|---|---|---|---|
| flagged | 28 | 0 | 2 | 0 |
| not flagged | 29 | 1 | 0 | 0 |

Of the livers marked bad, flagged: 0/1. 
Of the flagged livers marked ok/bad, bad: 0/28.

| code | volume | flagged | verdict | measures outside the real val range |
|---|---|---|---|---|
| B01 | liver_0281 | no | ok |  |
| B02 | liver_0297 | yes | ok | border_step_hu low (56.5) |
| B03 | liver_0953 | no | ok |  |
| B04 | liver_0675 | yes | ok | border_step_hu low (37.2) |
| B05 | liver_0997 | yes | ok | frac_fat high (0.0019) |
| B06 | liver_0603 | yes | ok | frac_fat high (0.0084) |
| B07 | liver_0933 | yes | ok | border_step_hu low (39.8) |
| B08 | liver_0084 | yes | unsure | frac_air high (0.0001) |
| B09 | liver_0136 | no | ok |  |
| B10 | liver_0296 | yes | ok | border_step_hu low (55) |
| B11 | liver_0528 | yes | ok | border_step_hu low (56.5) |
| B12 | liver_0797 | yes | ok | border_step_hu low (51.3) |
| B13 | liver_0726 | no | ok |  |
| B14 | liver_0508 | no | ok |  |
| B15 | liver_0814 | no | ok |  |
| B16 | liver_0471 | no | ok |  |
| B17 | liver_0223 | no | ok |  |
| B18 | liver_0153 | no | ok |  |
| B19 | liver_0244 | no | ok |  |
| B20 | liver_0180 | yes | ok | border_step_hu low (49) |
| B21 | liver_0595 | no | bad |  |
| B22 | liver_0656 | yes | ok | border_step_hu low (53.4) |
| B23 | liver_0823 | no | ok |  |
| B24 | liver_0476 | no | ok |  |
| B25 | liver_0225 | no | ok |  |
| B26 | liver_0626 | no | ok |  |
| B27 | liver_0932 | yes | ok | border_step_hu low (52.3) |
| B28 | liver_0540 | yes | ok | border_step_hu low (55.9) |
| B29 | liver_0580 | yes | ok | border_step_hu low (51.2) |
| B30 | liver_0829 | no | ok |  |
| B31 | liver_0214 | yes | ok | border_step_hu low (41.9) |
| B32 | liver_0533 | yes | ok | frac_fat high (0.0017) |
| B33 | liver_0844 | yes | ok | hu_median low (49), frac_fat high (0.0165) |
| B34 | liver_0380 | no | ok |  |
| B35 | liver_0828 | no | ok |  |
| B36 | liver_0725 | no | ok |  |
| B37 | liver_0280 | yes | ok | border_step_hu low (49.7) |
| B38 | liver_0496 | yes | ok | edge_peak_ratio low (1.594) |
| B39 | liver_0231 | no | ok |  |
| B40 | liver_0269 | no | ok |  |
| B41 | liver_0678 | no | ok |  |
| B42 | liver_0558 | yes | ok | border_step_hu low (45.9) |
| B43 | liver_0832 | no | ok |  |
| B44 | liver_0735 | yes | unsure | frac_fat high (0.006) |
| B45 | liver_0230 | no | ok |  |
| B46 | liver_0914 | yes | ok | border_step_hu low (55.1) |
| B47 | liver_0177 | no | ok |  |
| B48 | liver_0957 | yes | ok | border_step_hu low (43.5) |
| B49 | liver_0542 | no | ok |  |
| B50 | liver_0319 | yes | ok | edge_peak_ratio low (1.688) |
| B51 | liver_0031 | yes | ok | frac_fat high (0.0063) |
| B52 | liver_0949 | yes | ok | border_step_hu low (41.1) |
| B53 | liver_0762 | yes | ok | edge_peak_ratio low (1.689) |
| B54 | liver_0212 | yes | ok | edge_peak_ratio low (1.497) |
| B55 | liver_0789 | no | ok |  |
| B56 | liver_0682 | no | ok |  |
| B57 | liver_0598 | yes | ok | frac_fat high (0.0019) |
| B58 | liver_0092 | no | ok |  |
| B59 | liver_0463 | no | ok |  |
| B60 | liver_0455 | yes | ok | edge_peak_ratio low (1.346) |
