# A3: frozen post-hoc rules, pancreas test (238 cases)

Rules and controls frozen in `notes/AUTOPILOT.md` (s6) before this run; definitions `nnunet/post_rules.py`, `nnunet/post_rules_test.py`. Each rule vs the saved prediction (`none`), per-case paired difference (bootstrap 95 % CI, Wilcoxon p, Holm over the frozen rules present, per run and for the per-case mean of the two runs). Exploratory.

| rule | kind | run | tumour Dice | vs none [95% CI] | p | Holm p | pancreas Dice | vs none [95% CI] | lesion recall | FP pieces / case |
|---|---|---|---|---|---|---|---|---|---|---|
| none | – | D102 f0 all | 0.183 | – | – | – | 0.209 | – | 0.550 | 16.9 |
| none | – | D102 f1 all_rep2 | 0.172 | – | – | – | 0.173 | – | 0.521 | 5.3 |
| none | – | mean of the 2 runs | 0.177 | – | – | – | 0.191 | – | – | – |
| pos+minLes | frozen | D102 f0 all | 0.195 | +0.012 [+0.005, +0.019] | 2.3e-09 | 2.3e-09 | 0.210 | +0.000 [+0.000, +0.001] | 0.458 | 0.8 |
| pos+minLes | frozen | D102 f1 all_rep2 | 0.174 | +0.002 [-0.005, +0.007] | 3.2e-08 | 3.2e-08 | 0.173 | +0.000 [+0.000, +0.000] | 0.442 | 0.8 |
| pos+minLes | frozen | mean of the 2 runs | 0.184 | +0.007 [+0.002, +0.012] | 1.3e-07 | 1.3e-07 | 0.191 | +0.000 [+0.000, +0.000] | – | – |
| minLes | control | D102 f0 all | 0.194 | +0.012 [+0.005, +0.018] | 2.9e-09 |  | 0.209 | +0.000 [+0.000, +0.000] | 0.458 | 0.9 |
| minLes | control | D102 f1 all_rep2 | 0.173 | +0.001 [-0.006, +0.006] | 3.9e-08 |  | 0.173 | +0.000 [+0.000, +0.000] | 0.442 | 0.8 |
| minLes | control | mean of the 2 runs | 0.183 | +0.006 [+0.001, +0.011] | 1.9e-07 |  | 0.191 | +0.000 [+0.000, +0.000] | – | – |
| pos | control | D102 f0 all | 0.188 | +0.005 [+0.004, +0.007] | 3e-23 |  | 0.210 | +0.000 [+0.000, +0.001] | 0.550 | 11.3 |
| pos | control | D102 f1 all_rep2 | 0.174 | +0.002 [+0.001, +0.003] | 3.7e-15 |  | 0.173 | +0.000 [+0.000, +0.000] | 0.521 | 4.4 |
| pos | control | mean of the 2 runs | 0.181 | +0.004 [+0.003, +0.005] | 6.4e-24 |  | 0.191 | +0.000 [+0.000, +0.000] | – | – |
| tumNear10 | control | D102 f0 all | 0.200 | +0.018 [+0.011, +0.024] | 5.4e-15 |  | 0.209 | +0.000 [+0.000, +0.000] | 0.504 | 1.9 |
| tumNear10 | control | D102 f1 all_rep2 | 0.175 | +0.003 [-0.007, +0.013] | 6.3e-10 |  | 0.173 | +0.000 [+0.000, +0.000] | 0.458 | 1.2 |
| tumNear10 | control | mean of the 2 runs | 0.188 | +0.011 [+0.004, +0.017] | 3.3e-10 |  | 0.191 | +0.000 [+0.000, +0.000] | – | – |
| open1 | control | D102 f0 all | 0.191 | +0.009 [+0.006, +0.011] | 6.4e-13 |  | 0.199 | -0.010 [-0.012, -0.009] | 0.521 | 3.8 |
| open1 | control | D102 f1 all_rep2 | 0.174 | +0.002 [+0.000, +0.003] | 9.4e-05 |  | 0.163 | -0.010 [-0.012, -0.009] | 0.496 | 2.7 |
| open1 | control | mean of the 2 runs | 0.182 | +0.005 [+0.003, +0.007] | 1.8e-10 |  | 0.181 | -0.010 [-0.012, -0.009] | – | – |
| rand_minLes | control | D102 f0 all | 0.025 | -0.158 [-0.190, -0.126] | 2.2e-18 |  | 0.209 | +0.000 [+0.000, +0.000] | 0.062 | 1.3 |
| rand_minLes | control | D102 f1 all_rep2 | 0.054 | -0.118 [-0.148, -0.089] | 6.9e-12 |  | 0.173 | +0.000 [+0.000, +0.000] | 0.142 | 1.1 |
| rand_minLes | control | mean of the 2 runs | 0.039 | -0.138 [-0.166, -0.111] | 5.7e-22 |  | 0.191 | +0.000 [+0.000, +0.000] | – | – |
| rand_pos+minLes | control | D102 f0 all | 0.025 | -0.158 [-0.190, -0.126] | 2.2e-18 |  | 0.209 | +0.000 [+0.000, +0.000] | 0.062 | 1.2 |
| rand_pos+minLes | control | D102 f1 all_rep2 | 0.053 | -0.119 [-0.149, -0.090] | 3.1e-12 |  | 0.173 | +0.000 [+0.000, +0.000] | 0.138 | 1.1 |
| rand_pos+minLes | control | mean of the 2 runs | 0.039 | -0.139 [-0.166, -0.112] | 5.5e-22 |  | 0.191 | +0.000 [+0.000, +0.000] | – | – |
