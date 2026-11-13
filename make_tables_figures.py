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
# Journal artwork rules (Springer): sans-serif lettering 8-12 pt at final size, no titles
# inside figures, panels labelled (a), (b), (c); 600 dpi PNG plus vector PDF.
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 8.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.linewidth": 0.7, "lines.linewidth": 1.2})
COLORS = ["#0b4f8a", "#d1495b", "#e09f1f", "#00798c", "#6a4c93", "#7a7a7a", "#3b7d23"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
W2 = 6.7                       # full text width (inches) = 170 mm


def write(name, rows):
    with open(f"{TAB}/{name}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)


def matrix(res, stack, names, key="RMSE"):
    return np.column_stack([[r[key] for r in sorted((r for r in res if r["stack"] == stack and r["algorithm"] == n),
                                                    key=lambda r: r["run"])] for n in names])


def best_run(res, stack, name):
    return min((r for r in res if r["stack"] == stack and r["algorithm"] == name), key=lambda r: r["RMSE"])


def panel(ax, k):
    ax.text(-0.02, 1.03, f"({'abc'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, fontweight="bold")


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=600); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)


def convergence_fig(res, names, name, max_fes=C.MAX_FES):
    """Three panels (one per stack), mean best-so-far RMSE, logarithmic evaluation axis.
    The curve is drawn from the second recording point (~200 FEs); before that only the initial
    population has been evaluated and a line from FE = 1 would only connect two recording points."""
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55))
    fes = np.linspace(1, max_fes, C.N_CURVE)
    for k, (ax, s) in enumerate(zip(axs, C.STACKS)):
        for j, n in enumerate(names):
            cur = np.array([r["curve"] for r in res if r["stack"] == s and r["algorithm"] == n])
            ax.loglog(fes[1:], cur.mean(0)[1:], color=COLORS[j % 7], ls=STYLES[j % 7], lw=1.7 if j == 0 else 1.1, label=n)
        ax.set_xlabel("Function evaluations")
        if k == 0: ax.set_ylabel("Mean best RMSE (V)")
        ax.set_xlim(fes[1], max_fes); panel(ax, k)   # first recorded point after initialisation (~200 FEs)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(names), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, name)


def box_fig(mats, names, name):
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6))
    for k, (ax, F) in enumerate(zip(axs, mats)):
        ax.boxplot(F, tick_labels=names, showfliers=True, widths=0.6,
                   medianprops=dict(color="#d1495b"), flierprops=dict(markersize=3))
        if F.max() / F.min() > 20: ax.set_yscale("log")
        if k == 0: ax.set_ylabel("Final RMSE (V)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        panel(ax, k)
    fig.tight_layout(); save(fig, name)


summary = {}
# ============================================================ main comparison
res = json.load(open("results/raw_results.json"))["results"]
algs = C.ALGORITHMS
tA, tB, tC = [], [], []
Fall = []
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
    # figure: measured vs modelled I-V and I-P
    I, V = load_data(s); b = best_run(res, s, "OLBPE")
    Ig = np.linspace(I.min(), I.max(), 300); Vg = stack_voltage(Ig, np.array(b["x"]), STACKS[s])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6))
    for k, (yd, ym, yl) in enumerate([(V, Vg, "Stack voltage (V)"), (I * V, Ig * Vg, "Stack power (W)")]):
        ax[k].plot(I, yd, "s", ms=4, mfc="#d1495b", mec="#7a1f2b", mew=0.6, label="Measured")
        ax[k].plot(Ig, ym, "-", color="#0b4f8a", lw=1.5, label="OLBPE model")
        ax[k].set_xlabel("Stack current (A)"); ax[k].set_ylabel(yl); ax[k].legend(); panel(ax[k], k)
    fig.tight_layout(); save(fig, f"fit_{s}")
    Fall.append(F)
convergence_fig(res, algs, "convergence")
box_fig(Fall, algs, "boxplots")
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
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
for k, (ax, s) in enumerate(zip(axs, C.STACKS)):
    I, V = load_data(s); b = best_run(res, s, "OLBPE")
    e = V - stack_voltage(I, np.array(b["x"]), STACKS[s])
    ax.axhline(0, color="k", lw=0.7)
    ax.vlines(I, 0, e, color="#0b4f8a", lw=1.0); ax.plot(I, e, "o", color="#d1495b", ms=3.5)
    ax.set_xlabel("Stack current (A)")
    if k == 0: ax.set_ylabel("Residual (V)")
    panel(ax, k)
fig.tight_layout(); save(fig, "residuals")
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
def ablation(path, tag, variants, fig_names=True):
    J = json.load(open(path)); ra = J["results"]; mf = J["config"]["max_fes"]
    vars_ = list(variants.keys())
    tD, tE, mats = [], [], []
    for s in C.STACKS:
        lab = STACKS[s]["label"]
        F = matrix(ra, s, vars_); mats.append(F)
        pF, chi2, mr = friedman(F, C.TIE_SIG)
        summary[s].update({f"abl{tag}_friedman_p": pF, f"abl{tag}_friedman_chi2": chi2})
        for j, n in enumerate(vars_):
            tD.append(dict(Stack=lab, Variant=n, Min=F[:, j].min(), Mean=F[:, j].mean(), Std=F[:, j].std(ddof=1),
                           Max=F[:, j].max(), DeltaMean_pct=100 * (F[:, j].mean() - F[:, 0].mean()) / F[:, 0].mean(),
                           FriedmanRank=mr[j]))
        for row in wilcoxon_paired_vs(F, vars_, vars_[0], C.ALPHA, C.TIE_SIG):
            tE.append(dict(Stack=lab, **row))
    if fig_names:
        short = ["full"] + [v.replace("w/o ", "w/o ") for v in vars_[1:]]
        convergence_fig(ra, vars_, f"ablation{tag}_convergence", mf)
        box_fig(mats, short, f"ablation{tag}_boxplots")
    return tD, tE


for path, tag, var, out in [("results/ablation_results.json", "", C.ABLATION, "D"),
                            ("results/ablation_results_5k.json", "_5k", C.ABLATION, "D"),
                            ("results/ablation_ewl_split.json", "_ewl", C.EWL_SPLIT, "J"),
                            ("results/ablation_ewl_split_5k.json", "_ewl_5k", C.EWL_SPLIT, "J")]:
    if os.path.exists(path):
        tD, tE = ablation(path, tag, var, fig_names=(out == "D"))
        if out == "D":
            write(f"TableD{tag}_ablation", tD); write(f"TableE{tag}_ablation_wilcoxon", tE)
        else:
            write(f"TableJ{tag.replace('_ewl', '')}_ewl_split", tD)
            write(f"TableK{tag.replace('_ewl', '')}_ewl_split_wilcoxon", tE)

# ============================================================ lambda profile (separable LS check)
if os.path.exists(f"{TAB}/lambda_profile.csv"):
    P = np.genfromtxt(f"{TAB}/lambda_profile.csv", delimiter=",", names=True)
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
    for k, (ax, s) in enumerate(zip(axs, C.STACKS)):
        ax.plot(P["lambda"], P[s], color="#0b4f8a")
        ax.set_xlabel(r"$\lambda$")
        if k == 0: ax.set_ylabel("Min. RMSE over other parameters (V)")
        panel(ax, k)
    fig.tight_layout(); save(fig, "lambda_profile")

json.dump(summary, open(f"{TAB}/summary.json", "w"), indent=1)
print("Tables in", TAB, "| Figures in", FIG)
