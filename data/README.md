# Experimental data

Measured polarization curves (columns `I_A,V_V`) copied without modification from:

E.-J. Liu, J.-C. Cheng, *Parameter identification and sensitivity analysis of PEM Fuel Cell Systems via Puma Optimizer*, Energy Reports 13 (2025) 6328–6348, https://doi.org/10.1016/j.egyr.2025.05.071 (open access, CC BY 4.0).

| File | Stack | Source table | Points |
|------|-------|--------------|--------|
| `BCS500W.csv` | BCS 500 W | Table A.2 | 18 |
| `SR12.csv` | Avista SR-12 Modular | Table A.5 | 18 |
| `NedStackPS6.csv` | NedStack PS6 | Table A.1 | 28 |

Stack specifications (`olbpe_pemfc/model.py`) follow Table 1 of the same article. The model implementation reproduces the voltages published in Tables A.1, A.2 and A.5 for the authors' parameter sets to within 1e-4 V (BCS 500 W, NedStack PS6) and 3e-3 V (SR-12; parameters rounded in print); see `tests/test_all.py`.
