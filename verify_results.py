"""Checks that the results reported in the paper are reproduced on your machine.

  python verify_results.py --quick
      Re-runs runs 1-3 of every algorithm on every stack (about 2-3 minutes)
      and compares each one with the stored results/raw_results.json.

  python verify_results.py --compare results_paper results
      Compares a complete new run (folder `results`) with a copy of the
      published results (folder `results_paper`): every run and every table.
"""
import argparse, csv, json, os, sys
from olbpe_pemfc import config as C
from olbpe_pemfc.runner import _one

TOL = 1e-8   # relative tolerance: only floating-point rounding is allowed


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def quick(runs):
    ref = {(r["stack"], r["algorithm"], r["run"]): r
           for r in json.load(open("results/raw_results.json"))["results"]}
    worst, n = 0.0, 0
    print(f"{'stack':12s} {'alg':6s} run  RMSE now        RMSE paper      rel.diff")
    for s in C.STACKS:
        for a in C.ALGORITHMS:
            for r in runs:
                new, old = _one((s, a, r, None, C.MAX_FES))["RMSE"], ref[(s, a, r)]["RMSE"]
                d = rel(new, old); worst = max(worst, d); n += 1
                print(f"{s:12s} {a:6s} {r:3d}  {new:.10f}  {old:.10f}  {d:.1e}", flush=True)
    ok = worst < TOL
    print(f"\n{n} runs checked, largest relative difference {worst:.1e} -> "
          f"{'REPRODUCED' if ok else 'DIFFERENT'}")
    return ok


def compare(ref_dir, new_dir):
    ok = True
    for fn in ["raw_results.json", "ablation_results_5k.json", "ablation_results.json"]:
        pa, pb = os.path.join(ref_dir, fn), os.path.join(new_dir, fn)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"[skip] {fn} not found in both folders"); continue
        A = {(r["stack"], r["algorithm"], r["run"]): r["RMSE"] for r in json.load(open(pa))["results"]}
        B = {(r["stack"], r["algorithm"], r["run"]): r["RMSE"] for r in json.load(open(pb))["results"]}
        good = set(A) == set(B) and max(rel(B[k], A[k]) for k in A) < TOL
        ok &= good
        print(f"[{'ok' if good else 'DIFF'}] {fn}: {len(A)} runs")
    ta, tb = os.path.join(ref_dir, "tables"), os.path.join(new_dir, "tables")
    for fn in sorted(f for f in os.listdir(ta) if f.endswith(".csv")):
        ra = list(csv.reader(open(os.path.join(ta, fn))))
        rb = list(csv.reader(open(os.path.join(tb, fn))))
        hdr, w, good = ra[0], 0.0, len(ra) == len(rb)
        for x, y in zip(ra[1:], rb[1:]):
            for col, u, v in zip(hdr, x, y):
                if u == v or col == "RT_s":          # run time depends on the computer
                    continue
                try:
                    w = max(w, rel(float(v), float(u)))
                except ValueError:
                    good = False
        good &= w < 1e-6
        ok &= good
        print(f"[{'ok' if good else 'DIFF'}] {fn}  (largest relative difference {w:.1e})")
    print("\nALL RESULTS REPRODUCED" if ok else "\nSOME RESULTS DIFFER - see lines marked DIFF")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--compare", nargs=2, metavar=("REF_DIR", "NEW_DIR"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    sys.exit(0 if quick(range(1, a.runs + 1)) else 1)
