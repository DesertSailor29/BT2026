# B-prep: confirmed liver rules applied to the 999 synthetic training labels (D101)

Rule functions of `nnunet/post_rules.py` with the frozen s8 parameters (`nnunet/label_rule_counts.py`). MISSION B: < 50 changed volumes → no pre-training arm. Exploratory.

Spine centre found in 999 / 999 volumes.

| rule | volumes changed | median changed voxels (changed vols) | volumes losing tumour | tumour voxels removed (% of all) | tumour pieces removed | verdict |
|---|---|---|---|---|---|---|
| tumTouch | 598 | 335 | 598 | 1659281 (4.64 %) | 1701 | arm possible |
| pos | 84 | 223 | 79 | 124429 (0.35 %) | 123 | arm possible |
| ccPost | 656 | 322 | 598 | 1659548 (4.64 %) | 1704 | reference |
