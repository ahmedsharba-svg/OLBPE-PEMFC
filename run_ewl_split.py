"""EWL decomposition: full OLBPE vs. OLBPE without the elite-attraction term and
without the elite-worst term of Eq. (19), at 5 000 and 40 000 evaluations.
Same seeds as the main experiment.
  -> results/ablation_ewl_split_5k.json, results/ablation_ewl_split.json"""
import os, sys
from olbpe_pemfc import config as C
from olbpe_pemfc.runner import run

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    spec = [(n, kw) for n, kw in C.EWL_SPLIT.items()]
    run(spec, "results/ablation_ewl_split_5k.json", workers, max_fes=C.ABLATION_REDUCED_FES)
    run(spec, "results/ablation_ewl_split.json", workers)
