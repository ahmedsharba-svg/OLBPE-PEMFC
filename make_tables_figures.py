"""Builds every table (CSV) and figure (PNG) of the paper from results/*.json.
No number is typed by hand."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from olbpe_pemfc import config as C
from olbpe_pemfc.model import STACKS, load_data, stack_voltage
from olbpe_pemfc.stats import wilcoxon_paired_vs, friedman

TAB, FIG = "results/tables", "results/figures"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.grid": True, "grid.alpha": 0.3})
COLORS = ["#0b4f8a", "#d1495b", "#edae49", "#00798c", "#6a4c93", "#7a7a7a"]


def write(name, rows):
    with open(f"{TAB}/{name}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def matrix(res, stack, names, key="RMSE"):
    return np.column_stack([[r[key] for r in sorted((r for r in res if r["stack"] == stack and r["algorithm"] == n),
                                                    key=lambda r: r["run"])] for n in names])


def best_run(res, stack, name):
    return min((r for r in res if r["stack"] == stack and r["algorithm"] == name), key=lambda r: r["RMSE"])


def convergence(res, stack, names, fname, title, max_fes=C.MAX_FES):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fes = np.linspace(1, max_fes, C.N_CURVE)
    for j, n in enumerate(names):
        cur = np.array([r["curve"] for r in res if r["stack"] == stack and r["algorithm"] == n])
        ax.semilogy(fes / 1000, cur.mean(0), color=COLORS[j % 6], lw=2.2 if j == 0 else 1.3, label=n)
    ax.set_xlabel("Function evaluations (×10³)"); ax.set_ylabel("Mean best RMSE (V)"); ax.set_title(title)
    ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=300); plt.close(fig)


def boxplot(F, names, fname, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.boxplot(F, tick_labels=names, showfliers=True)
    ax.set_ylabel("Final RMSE (V)"); ax.set_title(title)
    if F.max() / F.min() > 20:
        ax.set_yscale("log")
    plt.setp(ax.get_xticklabels(), rotation=20)
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}", dpi=300); plt.close(fig)


summary = {}
# ============================================================ main comparison
res = json.load(open("results/raw_results.json"))["results"]
algs = C.ALGORITHMS
tA, tB, tC = [], [], []
for s in C.STACKS:
    lab = STACKS[s]["label"]
    F = matrix(res, s, algs); T = matrix(res, s, algs, "time")
    pF, chi2, mr = friedman(F, C.TIE_SIG)
    summary[s] = dict(friedman_p=pF, friedman_chi2=chi2)
    for j, n in enumerate(algs):
        b = best_run(res, s, n)
        tA.append(dict(Stack=lab, Algorithm=n, **{k: v for k, v in zip(C.PARAM_NAMES, b["x"])},
                       SSE=b["SSE"], RMSE=b["RMSE"], MAE=b["MAE"], MaxAE=b["MAXAE"]))
        tB.append(dict(Stack=lab, Algorithm=n, Min=F[:, j].min(), Mean=F[:, j].mean(), Median=np.median(F[:, j]),
                       Max=F[:, j].max(), Std=F[:, j].std(ddof=1), RT_s=T[:, j].mean(), FriedmanRank=mr[j]))
    for row in wilcoxon_paired_vs(F, algs, "OLBPE", C.ALPHA, C.TIE_SIG):
        tC.append(dict(Stack=lab, **row))
    # figures
    I, V = load_data(s); b = best_run(res, s, "OLBPE")
    Ig = np.linspace(I.min(), I.max(), 300); Vg = stack_voltage(Ig, np.array(b["x"]), STACKS[s])
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
    ax[0].plot(I, V, "s", ms=5, mfc="#d1495b", mec="#7a1f2b", label="Measured")
    ax[0].plot(Ig, Vg, "-", color="#0b4f8a", lw=1.8, label="OLBPE model")
    ax[0].set_xlabel("Stack current (A)"); ax[0].set_ylabel("Stack voltage (V)"); ax[0].set_title(f"(a) I–V, {lab}")
    ax[1].plot(I, I * V, "s", ms=5, mfc="#d1495b", mec="#7a1f2b", label="Measured")
    ax[1].plot(Ig, Ig * Vg, "-", color="#0b4f8a", lw=1.8, label="OLBPE model")
    ax[1].set_xlabel("Stack current (A)"); ax[1].set_ylabel("Stack power (W)"); ax[1].set_title(f"(b) I–P, {lab}")
    for a in ax: a.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/fit_{s}.png", dpi=300); plt.close(fig)
    convergence(res, s, algs, f"conv_{s}.png", lab)
    boxplot(F, algs, f"box_{s}.png", lab)
write("TableA_best_parameters", tA); write("TableB_statistics", tB); write("TableC_wilcoxon", tC)
wtl = []
for n in algs[1:]:
    sg = [r["sign"] for r in tC if r["comparison"] == f"OLBPE vs {n}"]
    wtl.append(dict(comparison=f"OLBPE vs {n}", wins=sg.count("+"), ties=sg.count("="), losses=sg.count("-")))
write("TableC_wilcoxon_summary", wtl)


# ============================================================ convergence efficiency
tF = []
for s in C.STACKS:
    lab = STACKS[s]["label"]
    fstar = min(r["RMSE"] for r in res if r["stack"] == s)
    fes = np.unique(np.round(np.linspace(1, C.MAX_FES, C.N_CURVE)).astype(int))
    for n in algs:
        Rn = [r for r in res if r["stack"] == s and r["algorithm"] == n]
        hits = []
        for r in Rn:
            c = np.array(r["curve"]); k = np.nonzero(c <= fstar * (1 + 1e-3))[0]
            if len(k): hits.append(fes[k[0]])
        cur = np.array([r["curve"] for r in Rn]).mean(0)
        auc = float(np.trapezoid(np.log10(cur), fes) / C.MAX_FES)
        tF.append(dict(Stack=lab, Algorithm=n, f_star=fstar, success_rate=len(hits) / len(Rn),
                       FE_median=float(np.median(hits)) if hits else np.nan,
                       FE_q25=float(np.percentile(hits, 25)) if hits else np.nan,
                       FE_q75=float(np.percentile(hits, 75)) if hits else np.nan, log_AUC=auc))
write("TableF_FE_to_target", tF)

# ============================================================ residuals
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, s in zip(axs, C.STACKS):
    I, V = load_data(s); b = best_run(res, s, "OLBPE")
    e = V - stack_voltage(I, np.array(b["x"]), STACKS[s])
    ax.axhline(0, color="k", lw=0.8)
    ax.vlines(I, 0, e, color="#0b4f8a", lw=1.2); ax.plot(I, e, "o", color="#d1495b", ms=5)
    ax.set_xlabel("Stack current (A)"); ax.set_ylabel("Residual (V)"); ax.set_title(STACKS[s]["label"])
fig.tight_layout(); fig.savefig(f"{FIG}/residuals.png", dpi=300); plt.close(fig)
tH = []
for s in C.STACKS:
    I, V = load_data(s); b = best_run(res, s, "OLBPE")
    e = V - stack_voltage(I, np.array(b["x"]), STACKS[s])
    k = int(np.argmax(np.abs(e)))
    tH.append(dict(Stack=STACKS[s]["label"], N=len(I), RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                   MaxAE=float(np.abs(e).max()), I_at_MaxAE=float(I[k]), mean_residual=float(e.mean()),
                   std_residual=float(e.std(ddof=1)), MAPE_pct=float(100 * np.mean(np.abs(e / V)))))
write("TableH_residuals", tH)

# ============================================================ identifiability
tG = []
for s in C.STACKS:
    S = STACKS[s]; T = S["T"]; lnC = np.log(S["PO2"] / (5.08e6 * np.exp(-498.0 / T)))
    X = np.array([r["x"] for r in res if r["stack"] == s and r["algorithm"] == "OLBPE"])
    R_ = np.array([r["RMSE"] for r in res if r["stack"] == s and r["algorithm"] == "OLBPE"])
    opt = X[R_ <= R_.min() * (1 + 1e-6)]
    c0 = opt[:, 0] + opt[:, 1] * T + opt[:, 2] * T * lnC
    row = dict(Stack=S["label"], n_runs_at_optimum=len(opt), c0_mean=c0.mean(), c0_range=np.ptp(c0))
    for j, nm in enumerate(C.PARAM_NAMES):
        row[f"{nm}_min"] = opt[:, j].min(); row[f"{nm}_max"] = opt[:, j].max()
    tG.append(row)
write("TableG_identifiability", tG)

# ============================================================ ablation
def ablation(path, tag, budget_label):
    J = json.load(open(path)); ra = J["results"]; mf = J["config"]["max_fes"]
    vars_ = list(C.ABLATION.keys())
    tD, tE = [], []
    for s in C.STACKS:
        lab = STACKS[s]["label"]
        F = matrix(ra, s, vars_)
        pF, chi2, mr = friedman(F, C.TIE_SIG)
        summary[s].update({f"abl{tag}_friedman_p": pF, f"abl{tag}_friedman_chi2": chi2})
        for j, n in enumerate(vars_):
            tD.append(dict(Stack=lab, Variant=n, Min=F[:, j].min(), Mean=F[:, j].mean(), Std=F[:, j].std(ddof=1),
                           Max=F[:, j].max(), DeltaMean_pct=100 * (F[:, j].mean() - F[:, 0].mean()) / F[:, 0].mean(),
                           FriedmanRank=mr[j]))
        for row in wilcoxon_paired_vs(F, vars_, vars_[0], C.ALPHA, C.TIE_SIG):
            tE.append(dict(Stack=lab, **row))
        convergence(ra, s, vars_, f"ablation{tag}_conv_{s}.png", f"Ablation ({budget_label}) - {lab}", mf)
        boxplot(F, vars_, f"ablation{tag}_box_{s}.png", f"Ablation ({budget_label}) - {lab}")
    write(f"TableD{tag}_ablation", tD); write(f"TableE{tag}_ablation_wilcoxon", tE)


if os.path.exists("results/ablation_results.json"):
    ablation("results/ablation_results.json", "", "40 000 FEs")
if os.path.exists("results/ablation_results_5k.json"):
    ablation("results/ablation_results_5k.json", "_5k", "5 000 FEs")

json.dump(summary, open(f"{TAB}/summary.json", "w"), indent=1)
print("Tables in", TAB, "| Figures in", FIG)
