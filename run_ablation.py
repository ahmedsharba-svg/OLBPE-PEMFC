"""Ablation study: full OLBPE vs. variants with one component removed.

  python run_ablation.py [workers]
Runs the ablation at the full budget (40 000 evaluations) and at a reduced
budget (5 000 evaluations), where the contribution of each component is
visible before all variants reach the optimum.
-> results/ablation_results.json, results/ablation_results_5k.json
"""
import os, sys
from olbpe_pemfc import config as C
from olbpe_pemfc.runner import run

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    spec = [(n, kw) for n, kw in C.ABLATION.items()]
    run(spec, "results/ablation_results_5k.json", workers, max_fes=C.ABLATION_REDUCED_FES)
    run(spec, "results/ablation_results.json", workers)
