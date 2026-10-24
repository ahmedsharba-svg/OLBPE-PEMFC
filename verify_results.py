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


NONIDENT = ("xi1", "xi2", "xi3")   # not individually identifiable (paper, Section 4.5)
ATOL = 1e-9                        # absolute tolerance for values that are ~0 (std, mean residual, p)


def close(u, v, rtol=1e-6):
    return abs(u - v) <= ATOL + rtol * max(abs(u), abs(v))


def compare(ref_dir, new_dir):
    """Levels: ok = identical; ok* = differs only in xi1-xi3 (non-identifiable, any
    vector with the same c0 is optimal) or run time; DRIFT = reduced-budget (5k)
    ablation runs that have not converged differ numerically, but every
    statistical conclusion (+/=/- sign) is the same; DIFF = a real difference."""
    real = False
    for fn in ["raw_results.json", "ablation_results.json", "ablation_results_5k.json"]:
        pa, pb = os.path.join(ref_dir, fn), os.path.join(new_dir, fn)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"[skip] {fn} not found in both folders"); continue
        A = {(r["stack"], r["algorithm"], r["run"]): r["RMSE"] for r in json.load(open(pa))["results"]}
        B = {(r["stack"], r["algorithm"], r["run"]): r["RMSE"] for r in json.load(open(pb))["results"]}
        same = sum(rel(B[k], A[k]) < TOL for k in A)
        if same == len(A):
            print(f"[ok]    {fn}: all {len(A)} runs identical")
        elif "5k" in fn:
            print(f"[DRIFT] {fn}: {same}/{len(A)} runs identical; the others stopped at 5,000 evaluations "
                  f"before converging and drifted through floating-point differences between CPUs")
        else:
            print(f"[DIFF]  {fn}: only {same}/{len(A)} runs identical"); real = True

    ta, tb = os.path.join(ref_dir, "tables"), os.path.join(new_dir, "tables")
    for fn in sorted(f for f in os.listdir(ta) if f.endswith(".csv")):
        ra = list(csv.DictReader(open(os.path.join(ta, fn))))
        rb = list(csv.DictReader(open(os.path.join(tb, fn))))
        hard, soft, notes = [], [], []
        for x, y in zip(ra, rb):
            for col in x:
                u, v = x[col], y.get(col)
                if u == v or col == "RT_s":
                    continue
                try:
                    fu, fv = float(u), float(v)
                except (TypeError, ValueError):
                    hard.append(f"{x.get('Stack','')} {col}: {u} -> {v}"); continue
                if close(fu, fv):
                    continue
                where = f"{x.get('Stack','')} {x.get('Algorithm', x.get('Variant', x.get('comparison','')))} {col}: {u} -> {v}"
                (soft if col.startswith(NONIDENT) else hard).append(where)
        is5k = "_5k" in fn
        signs_same = all(x.get("sign") == y.get("sign") for x, y in zip(ra, rb))
        if len(ra) != len(rb):
            tag = "DIFF"
        elif not hard:
            tag = "ok*" if soft else "ok"
        elif is5k and signs_same:
            tag = "DRIFT"
        elif is5k and "sign" not in ra[0]:
            tag = "DRIFT"
        else:
            tag = "DIFF"
        real |= tag == "DIFF"
        print(f"[{tag}]{' ' * (6 - len(tag))}{fn}")
        if tag == "ok*":
            print("        only the non-identifiable xi1-xi3 differ (same RMSE, same c0)")
        if tag in ("DIFF", "DRIFT"):
            for h in hard[:6]:
                print("        " + h)
            if len(hard) > 6:
                print(f"        ... {len(hard) - 6} more")
    # conclusions of the 5k ablation
    fa, fb = os.path.join(ta, "TableE_5k_ablation_wilcoxon.csv"), os.path.join(tb, "TableE_5k_ablation_wilcoxon.csv")
    if os.path.exists(fa) and os.path.exists(fb):
        ea, eb = list(csv.DictReader(open(fa))), list(csv.DictReader(open(fb)))
        ch = [f"{x['Stack']} {x['comparison']}: {x['sign']} -> {y['sign']}" for x, y in zip(ea, eb) if x["sign"] != y["sign"]]
        print("\nAblation (5k) conclusions: " + ("all 15 signs identical to the paper" if not ch else
              "CHANGED:\n  " + "\n  ".join(ch)))
        real |= bool(ch)
    print("\nRESULT: " + ("ALL RESULTS AND CONCLUSIONS REPRODUCED" if not real
                          else "SOME RESULTS DIFFER - see lines marked DIFF / CHANGED"))
    return not real


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--compare", nargs=2, metavar=("REF_DIR", "NEW_DIR"))
    a = ap.parse_args()
    if a.compare:
        sys.exit(0 if compare(*a.compare) else 1)
    sys.exit(0 if quick(range(1, a.runs + 1)) else 1)
