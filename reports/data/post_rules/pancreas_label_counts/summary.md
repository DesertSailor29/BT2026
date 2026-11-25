# B-prep: confirmed pancreas rules applied to the 999 synthetic training labels (D102)

Rule functions of `nnunet/post_rules.py` with the frozen s6 parameters (`nnunet/label_rule_counts.py`). MISSION B: < 50 changed volumes → no pre-training arm. Exploratory.

Spine centre found in 999 / 999 volumes.

| rule | volumes changed | median changed voxels (changed vols) | volumes losing tumour | tumour voxels removed (% of all) | tumour pieces removed | verdict |
|---|---|---|---|---|---|---|
| minLes | 131 | 536 | 131 | 87263 (0.90 %) | 131 | arm possible |
| pos | 5 | 216 | 5 | 1418 (0.01 %) | 5 | generator obeys |
| pos+minLes | 131 | 536 | 131 | 87263 (0.90 %) | 131 | arm possible |
| ccPost | 88 | 1483 | 75 | 331565 (3.42 %) | 75 | reference |
