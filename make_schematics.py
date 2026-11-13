"""Schematic figures of the paper (Figs. 1-3), drawn for this work.
  python make_schematics.py -> results/figures/fig1_structure, fig2_polarization, fig3_procedure (.png 600 dpi, .pdf)"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch
from olbpe_pemfc.model import STACKS

FIG = "results/figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "font.size": 8.5})
ink = "#1f2933"; BLUE = "#2b6cb0"; ORANGE = "#c05621"; GREEN = "#2f855a"; GREY = "#4a5568"


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=600); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig)


def arrow(ax, x1, y1, x2, y2, c=ink, lw=1.1, ms=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw,
                                 shrinkA=0, shrinkB=0))


def line(ax, xs, ys, lw=1.0, c=ink):
    ax.plot(xs, ys, color=c, lw=lw, solid_capstyle="butt")


def resistor(ax, xa, xb, y, label, dy=0.27):
    n = 6; xs = [xa + (xb - xa) * i / (2 * n) for i in range(2 * n + 1)]
    ys = [y] + [y + (0.09 if i % 2 else -0.09) for i in range(1, 2 * n)] + [y]
    ax.plot(xs, ys, color=ink, lw=1.0); ax.text((xa + xb) / 2, y + dy, label, ha="center", fontsize=9)


# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig = plt.figure(figsize=(6.7, 3.0))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10.9); ax.set_ylim(0, 4.75); ax.axis("off")
    # (a) cell structure
    y0, y1 = 0.95, 3.6
    layers = [(0.85, 0.55, "#cfd8e3", "Gas\nchannel"), (1.40, 0.62, "#7ea6d6", "Anode\n(GDL+CL)"),
              (2.02, 0.56, "#f2c14e", "Mem-\nbrane"), (2.58, 0.62, "#e58f84", "Cathode\n(CL+GDL)"),
              (3.20, 0.55, "#cfd8e3", "Gas\nchannel")]
    for k, (x, w, c, lab) in enumerate(layers):
        ax.add_patch(Rectangle((x, y0), w, y1 - y0, fc=c, ec=ink, lw=0.7))
        yl = y0 - 0.07 if k % 2 == 0 else y0 - 0.07
        if k % 2:
            line(ax, [x + w / 2, x + w / 2], [y0, y0 - 0.42], 0.6, GREY)
            ax.text(x + w / 2, y0 - 0.45, lab.split("\n")[0], ha="center", va="top", fontsize=8)
        else:
            ax.text(x + w / 2, y0 - 0.07, lab, ha="center", va="top", fontsize=8)
    arrow(ax, 1.125, 0.95, 1.125, 3.35, BLUE); ax.text(0.80, 1.0, "H$_2$ in", fontsize=8, color=BLUE, ha="right")
    ax.text(0.80, 3.2, "H$_2$ out", fontsize=8, color=BLUE, ha="right")
    arrow(ax, 3.475, 0.95, 3.475, 3.35, ORANGE); ax.text(3.82, 1.0, "O$_2$/air in", fontsize=8, color=ORANGE)
    ax.text(3.82, 3.05, "H$_2$O +\nheat out", fontsize=8, color=ORANGE)
    for yy in (2.7, 2.95, 3.2):
        arrow(ax, 1.9, yy, 2.68, yy, GREEN, 1.1, 7)
    ax.text(2.30, 3.3, "H$^+$", ha="center", fontsize=8.5, color=GREEN)
    ax.text(1.71, 1.05, "H$_2$ → 2H$^+$ + 2e$^-$", fontsize=8, ha="center", va="bottom", rotation=90)
    ax.text(2.89, 1.05, "½O$_2$ + 2H$^+$ + 2e$^-$ → H$_2$O", fontsize=8, ha="center", va="bottom", rotation=90)
    line(ax, [1.71, 1.71, 2.0], [y1, 4.25, 4.25]); line(ax, [2.6, 2.89, 2.89], [4.25, 4.25, y1])
    ax.add_patch(Rectangle((2.0, 4.05), 0.6, 0.4, fc="white", ec=ink, lw=1.0))
    ax.text(2.3, 4.25, "Load", ha="center", va="center", fontsize=8)
    arrow(ax, 1.71, 3.62, 1.71, 4.12, BLUE, 1.0, 7); ax.text(1.60, 3.82, "e$^-$", fontsize=8.5, color=BLUE, ha="right")
    arrow(ax, 2.89, 4.12, 2.89, 3.62, BLUE, 1.0, 7); ax.text(3.0, 3.82, "e$^-$", fontsize=8.5, color=BLUE)
    ax.text(0.05, 4.5, "(a)", fontsize=9.5, weight="bold")
    # (b) equivalent circuit
    ax.text(4.95, 4.5, "(b)", fontsize=9.5, weight="bold")
    yt, yb, xs_ = 3.35, 0.95, 5.75
    line(ax, [xs_, xs_], [yb, 1.85]); line(ax, [xs_, xs_], [2.45, yt])
    for yy, w, lw in [(2.45, 0.22, 1.4), (2.29, 0.12, 2.6), (2.13, 0.22, 1.4), (1.97, 0.12, 2.6)]:
        line(ax, [xs_ - w, xs_ + w], [yy, yy], lw)
    line(ax, [xs_, xs_], [1.85, 1.97])
    ax.text(xs_ - 0.3, 2.2, "$E_{Nernst}$", ha="right", va="center", fontsize=9)
    ax.text(xs_ + 0.27, 2.43, "+", fontsize=8.5); ax.text(xs_ + 0.27, 1.88, "−", fontsize=8.5)
    xa, xb = 6.2, 8.2; ytop = 3.95; yc = 2.75
    line(ax, [xs_, xa], [yt, yt]); line(ax, [xa, xa], [yc, ytop]); line(ax, [xb, xb], [yc, ytop])
    line(ax, [xa, 6.35], [ytop, ytop]); resistor(ax, 6.35, 7.05, ytop, "$R_{act}$")
    line(ax, [7.05, 7.35], [ytop, ytop]); resistor(ax, 7.35, 8.05, ytop, "$R_{con}$"); line(ax, [8.05, xb], [ytop, ytop])
    line(ax, [xa, 7.13], [yc, yc]); line(ax, [7.27, xb], [yc, yc])
    line(ax, [7.13, 7.13], [yc - 0.18, yc + 0.18], 1.5); line(ax, [7.27, 7.27], [yc - 0.18, yc + 0.18], 1.5)
    ax.text(7.2, yc - 0.45, "$C$", ha="center", fontsize=9)
    line(ax, [xb, 8.45], [yt, yt]); resistor(ax, 8.45, 9.1, yt, "$R_{ohm}$")
    xl = 10.05
    line(ax, [9.1, xl], [yt, yt])
    arrow(ax, 9.55, yt + 0.22, 9.9, yt + 0.22, ink, 1.0, 7); ax.text(9.72, yt + 0.32, "$I$", ha="center", fontsize=9)
    line(ax, [xl, xl], [yt, 2.65]); line(ax, [xl, xl], [1.65, yb]); line(ax, [xs_, xl], [yb, yb])
    ax.add_patch(Rectangle((xl - 0.12, 1.65), 0.24, 1.0, fc="white", ec=ink, lw=1.0))
    arrow(ax, xl - 0.3, 1.55, xl + 0.3, 2.75, ink, 0.9, 6)
    ax.text(xl + 0.38, 2.15, "Load", fontsize=8, ha="left", va="center")
    xt = 9.45
    for y in (yt, yb):
        ax.add_patch(Circle((xt, y), 0.05, fc="white", ec=ink, lw=0.9, zorder=3))
    arrow(ax, xt, yb + 0.1, xt, yt - 0.1, GREY, 0.9, 7)
    ax.text(xt - 0.12, 2.15, "$V_{stack}$", ha="right", va="center", fontsize=9, color=GREY)
    save(fig, "fig1_structure")


# ------------------------------------------------------------------ Fig. 2
def fig2():
    """Polarization curve of one BCS 500 W cell and the decomposition of the voltage losses,
    computed with the model of Section 2 and the best OLBPE parameters."""
    s = STACKS["BCS500W"]
    raw = json.load(open("results/raw_results.json"))["results"]
    x = min((r for r in raw if r["stack"] == "BCS500W" and r["algorithm"] == "OLBPE"), key=lambda r: r["RMSE"])["x"]
    xi1, xi2, xi3, xi4, lam, b, Rc = x
    T, A = s["T"], s["A"]
    J = np.linspace(0.002, 0.995 * s["Jmax"], 600); I = J * A
    E = 1.229 - 0.85e-3 * (T - 298.15) + 4.3085e-5 * T * (np.log(s["PH2"]) + 0.5 * np.log(s["PO2"]))
    CO2 = s["PO2"] / (5.08e6 * np.exp(-498.0 / T))
    Va = -(xi1 + xi2 * T + xi3 * T * np.log(CO2) + xi4 * T * np.log(I))
    rho = 181.6 * (1 + 0.03 * J + 0.062 * (T / 303) ** 2 * J ** 2.5) / ((lam - 0.634 - 3 * J) * np.exp(4.18 * (T - 303) / T))
    Vo = I * (rho * s["l"] / A + Rc); Vc = -b * np.log(1 - J / s["Jmax"])
    V = E - Va - Vo - Vc
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.axhline(E, color=ink, ls="--", lw=1.0)
    ax.text(0.012, E + 0.015, "$E_{Nernst}$", fontsize=8, va="bottom")
    ax.fill_between(J, E, E - Va, color="#f6c4bd", lw=0, label="Activation loss $V_{act}$")
    ax.fill_between(J, E - Va, E - Va - Vo, color="#f2d58a", lw=0, label="Ohmic loss $V_{ohm}$")
    ax.fill_between(J, E - Va - Vo, V, color="#9cc3e6", lw=0, label="Concentration loss $V_{con}$")
    ax.plot(J, V, color="#b3261e", lw=1.8, label="Cell voltage $V_{cell}$")
    ax.set_xlabel("Current density $J$ (A cm$^{-2}$)"); ax.set_ylabel("Voltage (V)")
    ax.set_xlim(0, s["Jmax"]); ax.set_ylim(0, E + 0.12); ax.grid(alpha=0.3)
    for xv in (0.04, 0.38):
        ax.axvline(xv, color=GREY, ls=":", lw=0.9)
    for xm, txt in [(0.02, "Activ.\nregion"), (0.21, "Ohmic region"), (0.425, "Mass-\ntransport\nregion")]:
        ax.text(xm, E - 0.03, txt, ha="center", va="top", fontsize=7.5, color=GREY)
    ax2 = ax.twinx(); ax2.plot(J, J * V, color=BLUE, lw=1.4, ls="-.", label="Power density")
    ax2.set_ylabel("Power density (W cm$^{-2}$)"); ax2.set_ylim(0, (J * V).max() * 1.6)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 1)); save(fig, "fig2_polarization")


# ------------------------------------------------------------------ Fig. 3
def fig3():
    fig = plt.figure(figsize=(6.7, 2.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4.0); ax.axis("off")

    def box(x, y, w, h, txt, fc="#eef3f8", ec="#0b4f8a", round_=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12" if round_ else "square,pad=0.02",
                                    fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.3, linespacing=1.3)
    box(0.15, 2.75, 2.3, 1.0, "Measured data\n$(I_i, V_{exp,i})$, $i=1..N_s$", fc="#f7f7f7", ec=GREY)
    box(0.15, 0.25, 2.3, 1.15, "Operating conditions\n$T$, $P_{H2}$, $P_{O2}$, $N$, $A$, $l$,\n$J_{max}$ (Table 3)", fc="#f7f7f7", ec=GREY)
    box(3.05, 1.35, 1.75, 1.15, "PEMFC model\n(Eqs. 4–11)\n$V_{sim}(I; x)$")
    ax.add_patch(Circle((5.75, 1.925), 0.55, fc="#fdf2e0", ec=ORANGE, lw=1.1))
    ax.text(5.75, 1.925, "RMSE\n(Eq. 15)", ha="center", va="center", fontsize=8.3)
    box(6.75, 1.35, 1.45, 1.15, "OLBPE\n(Algorithm 1)", fc="#e8f3ec", ec=GREEN)
    box(8.55, 1.2, 1.35, 1.45, "Verification:\nI–V, I–P,\nresiduals,\nstatistics", fc="#f7f7f7", ec=GREY)
    arrow(ax, 1.3, 1.4, 1.3, 1.925); line(ax, [1.3, 3.05], [1.925, 1.925]); arrow(ax, 2.9, 1.925, 3.05, 1.925)
    line(ax, [2.45, 2.75], [3.25, 3.25]); line(ax, [2.75, 2.75], [3.25, 3.25])
    arrow(ax, 4.8, 1.925, 5.2, 1.925); ax.text(5.0, 2.05, "$V_{sim}$", ha="center", fontsize=8)
    line(ax, [2.45, 5.75], [3.25, 3.25]); arrow(ax, 5.75, 3.25, 5.75, 2.475); ax.text(5.85, 2.85, "$V_{exp}$", fontsize=8)
    arrow(ax, 6.3, 1.925, 6.75, 1.925)
    arrow(ax, 8.2, 1.925, 8.55, 1.925); ax.text(8.375, 2.05, "$x^*$", ha="center", fontsize=8.5)
    line(ax, [7.475, 7.475], [1.35, 0.55]); line(ax, [7.475, 3.925], [0.55, 0.55]); arrow(ax, 3.925, 0.55, 3.925, 1.35)
    ax.text(5.7, 0.45, "candidate parameters $x = [\\xi_1, \\xi_2, \\xi_3, \\xi_4, \\lambda, b, R_c]$", ha="center",
            va="top", fontsize=8)
    save(fig, "fig3_procedure")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); print("schematics written")
