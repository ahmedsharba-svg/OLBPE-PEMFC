# OLBPE-PEMFC

Python implementation of **Orthogonal Learning-Based Population Encoding (OLBPE)** for parameter identification of proton exchange membrane fuel cell (PEMFC) stacks with the semi-empirical Amphlett/Mann model (seven parameters ξ1, ξ2, ξ3, ξ4, λ, b, Rc), together with the complete benchmark reported in the paper: four competitor algorithms, 30 independent runs, Wilcoxon rank-sum and Friedman tests, an ablation study and every figure.

> Paper: S. R. Qasim, A. Adlhussain, N. H. Abbas, H. Salam, *Orthogonal Learning-Based Population Encoding (OLBPE) for Accurate and Reproducible PEM Fuel Cell Parameter Estimation* (under review).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahmedsharba-svg/OLBPE-PEMFC/blob/main/reproduce_on_colab.ipynb)

## Run on Google Colab (no installation)
Click the badge above and run the notebook `reproduce_on_colab.ipynb` from top to bottom.

## Installation
```bash
pip install -r requirements.txt      # numpy, scipy, matplotlib
python tests/test_all.py             # self-checks (< 1 min)
```

## Reproducing the paper
```bash
python run_experiments.py            # 5 algorithms x 3 stacks x 30 runs -> results/raw_results.json
python run_ablation.py               # 6 OLBPE variants x 3 stacks x 30 runs -> results/ablation_results.json
python make_tables_figures.py        # Tables A-E (CSV) and all figures (PNG) -> results/
```
Both run scripts accept the number of worker processes as an argument (default: all CPU cores). Run *r* of every algorithm uses the seed `1000 + r`, so the numbers in the paper are reproduced exactly.

## Checking that the results are reproduced
```bash
python verify_results.py --quick                          # 3 runs per algorithm and stack vs. results/ (~3 min)
cp -r results results_paper                               # keep the published results, then re-run everything
python run_experiments.py && python run_ablation.py && python make_tables_figures.py
python verify_results.py --compare results_paper results  # every run and table vs. the published ones
```
Results are identical (relative difference < 1e-8) with NumPy 2.x and SciPy >= 1.11; only run times differ between computers.

## Repository layout
```
olbpe_pemfc/config.py      all experimental settings (runs, budget, bounds, algorithms, ablation variants)
olbpe_pemfc/model.py       Amphlett model, stack specifications, objective and error metrics
olbpe_pemfc/algorithms.py  OLBPE (components can be switched off) and DE, PSO, GWO, WOA
olbpe_pemfc/stats.py       paired Wilcoxon signed-rank (primary) and rank-sum tests with Holm correction, Friedman test
olbpe_pemfc/runner.py      benchmark runner
data/                      measured polarization curves and their source
tests/test_all.py          model validation against published data, statistics, budget, reproducibility
results/                   generated tables and figures
```

## OLBPE in brief
The search runs in the normalised box [0,1]^7, so that parameters of very different magnitude (ξ3 ≈ 1e-5, λ ≈ 20) are treated equally. Each individual is updated by **elite–worst learning (EWL)** or by **elite sensitivity guidance (ESG)**, which samples from the rank-weighted covariance of the elite set and follows the correlated valley of ξ1, ξ2 and ξ3. When the best solution stagnates, **orthogonal learning (OL)** recombines it with a guide vector through the L8(2^7) orthogonal array and factor analysis. Out-of-bound coordinates are handled by **violation-triggered correction (VTC)**, and the last 15 % of the budget is spent on **confidence-weighted refinement (CWR)**.

## Fair-comparison protocol
Identical population size (40), identical budget of 40 000 objective evaluations, identical bounds, objective (RMSE) and seeds for every algorithm; statistics on the final RMSE of 30 runs. Because run r of every algorithm starts from the same initial population, runs are paired: two-sided exact Wilcoxon signed-rank test (α = 0.05, Holm-corrected; rank-sum reported as a robustness check) and Friedman test with runs as blocks. Additional outputs: evaluations-to-target (Table F), residual plots and statistics (Table H) and the identifiability analysis of ξ1–ξ3 (Table G).

## Data
See `data/README.md`. Please cite the original source of the measurements.

## License
MIT — see `LICENSE`.
