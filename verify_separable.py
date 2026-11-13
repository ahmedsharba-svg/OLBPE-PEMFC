"""Independent check of the best-known minima by separable least squares.

For fixed lambda the Amphlett model is linear in c0 = xi1 + xi2*T + xi3*T*ln(CO2),
xi4, Rc and b:
    V/N - E + I*Rm(lambda) = c0 + xi4*T*ln(I) - Rc*I + b*ln(1 - J/Jmax)
so the bound-constrained problem reduces to a one-dimensional scan over lambda,
each point being a convex bounded linear least-squares problem (exact solution).
The bounds of c0 follow exactly from the box of xi1-xi3 (Table 1).

  python verify_separable.py      -> results/tables/TableI_separable.csv
                                     results/tables/lambda_profile.csv
"""
import csv, json, os
import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar
from olbpe_pemfc import config as C
from olbpe_pemfc.model import STACKS, load_data

LB, UB = C.LB, C.UB
GRID = np.linspace(LB[4], UB[4], 14001)       # lambda step 0.001


def profile(name):
    s = STACKS[name]; I, V = load_data(name)
    T, A, N = s["T"], s["A"], s["N"]; J = I / A
    E = 1.229 - 0.85e-3 * (T - 298.15) + 4.3085e-5 * T * (np.log(s["PH2"]) + 0.5 * np.log(s["PO2"]))
    lnC = np.log(s["PO2"] / (5.08e6 * np.exp(-498.0 / T)))
    c0lo = LB[0] + LB[1] * T + UB[2] * T * lnC        # lnC < 0
    c0hi = UB[0] + UB[1] * T + LB[2] * T * lnC
    M = np.column_stack([np.ones_like(I), T * np.log(I), -I, np.log(1 - J / s["Jmax"])])
    lo = [c0lo, LB[3], LB[6], LB[5]]; hi = [c0hi, UB[3], UB[6], UB[5]]

    def solve(lam):
        den = (lam - 0.634 - 3 * J) * np.exp(4.18 * (T - 303.0) / T)
        if np.any(den <= 0):
            return np.inf, None
        Rm = 181.6 * (1 + 0.03 * J + 0.062 * (T / 303.0) ** 2 * J ** 2.5) / den * s["l"] / A
        y = V / N - E + I * Rm
        r = lsq_linear(M, y, bounds=(lo, hi), method="bvls", tol=1e-14)
        return float(np.sqrt(np.mean((N * (M @ r.x - y)) ** 2))), r.x

    prof = np.array([solve(l)[0] for l in GRID])
    # refine every local minimum of the profile with a bounded 1-D search
    mins = []
    for i in range(len(GRID)):
        left = prof[i - 1] if i > 0 else np.inf; right = prof[i + 1] if i < len(GRID) - 1 else np.inf
        if prof[i] <= left and prof[i] <= right and np.isfinite(prof[i]):
            a, b = GRID[max(i - 1, 0)], GRID[min(i + 1, len(GRID) - 1)]
            if b > a:
                o = minimize_scalar(lambda l: solve(l)[0], bounds=(a, b), method="bounded",
                                    options={"xatol": 1e-10})
                lam, f = (o.x, o.fun) if o.fun < prof[i] else (GRID[i], prof[i])
            else:
                lam, f = GRID[i], prof[i]
            if not mins or abs(lam - mins[-1][0]) > 0.05:
                mins.append((lam, f, solve(lam)[1]))
            elif f < mins[-1][1]:
                mins[-1] = (lam, f, solve(lam)[1])
    mins.sort(key=lambda m: m[1])
    return prof, mins, (c0lo, c0hi)


if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)
    raw = json.load(open("results/raw_results.json"))["results"]
    rows, prof_rows = [], {"lambda": GRID}
    for name in C.STACKS:
        prof, mins, _ = profile(name)
        prof_rows[name] = prof
        best_ol = min(r["RMSE"] for r in raw if r["stack"] == name and r["algorithm"] == "OLBPE")
        for k, (lam, f, th) in enumerate(mins[:2]):
            rows.append(dict(Stack=STACKS[name]["label"], basin="global" if k == 0 else "second",
                             lambda_=lam, RMSE=f, c0=th[0], xi4=th[1], Rc=th[2], b=th[3],
                             OLBPE_best_RMSE=best_ol, rel_gap_OLBPE=(best_ol - mins[0][1]) / mins[0][1]))
        print(f"{name:12s} separable LS: RMSE {mins[0][1]:.9f} at lambda={mins[0][0]:.4f} | OLBPE best {best_ol:.9f}"
              + (f" | second basin {mins[1][1]:.6f} at lambda={mins[1][0]:.3f}" if len(mins) > 1 else ""))
    with open("results/tables/TableI_separable.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open("results/tables/lambda_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh); keys = list(prof_rows); w.writerow(keys)
        for i in range(len(GRID)):
            w.writerow([prof_rows[k][i] for k in keys])
