Run order (Python 3.12, seed 1):
1. analysis.py  - case-study analyses (expects d.pkl = the Kaggle file loaded with pandas)
2. uq.py        - calibration and conformal prediction
3. val.py       - validation experiments (simulated generators, real and synthetic controls)
4. figs_dmkd.py - Figs 2-7;  fig1val.py - Fig 1
Outputs are written to out/ and dm/.
