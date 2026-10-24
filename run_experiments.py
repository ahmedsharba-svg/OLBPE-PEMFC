"""Main comparison: all algorithms x all stacks x 30 runs -> results/raw_results.json"""
import os, sys
from olbpe_pemfc import config as C
from olbpe_pemfc.runner import run

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()
    run([(a, None) for a in C.ALGORITHMS], "results/raw_results.json", workers)
