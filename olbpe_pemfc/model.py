"""Semi-empirical Amphlett/Mann PEMFC stack model and stack specifications.

Specifications and polarization data: Liu & Cheng, Energy Reports 13 (2025)
6328-6348, Table 1 and Tables A.1, A.2, A.5 (doi:10.1016/j.egyr.2025.05.071).
"""
from pathlib import Path
import numpy as np

STACKS = {
    #               cells   area cm2  thickness cm  PH2 atm   PO2 atm  Jmax A/cm2  T K
    "BCS500W":     dict(N=32, A=64.0,  l=178e-4, PH2=1.0,     PO2=0.2095, Jmax=0.469, T=333.0,
                        label="BCS 500 W"),
    "SR12":        dict(N=48, A=62.5,  l=25e-4,  PH2=1.47628, PO2=0.2095, Jmax=0.672, T=323.15,
                        label="SR-12 Modular"),
    "NedStackPS6": dict(N=65, A=240.0, l=178e-4, PH2=1.0,     PO2=1.0,    Jmax=1.4,   T=343.0,
                        label="NedStack PS6"),
}
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def stack_voltage(I, x, s):
    """Stack voltage (V). x = [xi1, xi2, xi3, xi4, lambda, b, Rc]; I in A.

    Invalid points (J >= Jmax or lambda - 0.634 - 3J <= 0) return NaN.
    """
    I = np.asarray(I, dtype=float)
    xi1, xi2, xi3, xi4, lam, b, Rc = x
    T, A = s["T"], s["A"]
    J = I / A
    E = 1.229 - 0.85e-3 * (T - 298.15) + 4.3085e-5 * T * (np.log(s["PH2"]) + 0.5 * np.log(s["PO2"]))
    CO2 = s["PO2"] / (5.08e6 * np.exp(-498.0 / T))
    v_act = -(xi1 + xi2 * T + xi3 * T * np.log(CO2) + xi4 * T * np.log(I))
    den = (lam - 0.634 - 3.0 * J) * np.exp(4.18 * (T - 303.0) / T)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = 181.6 * (1 + 0.03 * J + 0.062 * (T / 303.0) ** 2 * J ** 2.5) / den
        v_ohm = I * (rho * s["l"] / A + Rc)
        arg = 1.0 - J / s["Jmax"]
        v_con = -b * np.log(np.where(arg > 0, arg, np.nan))
        V = s["N"] * (E - v_act - v_ohm - v_con)
    V[(den <= 0) | (arg <= 0)] = np.nan
    return V


def load_data(name):
    d = np.loadtxt(DATA_DIR / f"{name}.csv", delimiter=",", skiprows=1)
    if d.ndim != 2 or len(d) < 5:
        raise ValueError(f"data/{name}.csv is empty or malformed")
    return d[:, 0], d[:, 1]


def metrics(x, I, V, s):
    e = V - stack_voltage(I, x, s)
    return dict(SSE=float(np.sum(e ** 2)), RMSE=float(np.sqrt(np.mean(e ** 2))),
                MAE=float(np.mean(np.abs(e))), MAXAE=float(np.max(np.abs(e))))


def make_objective(name, kind="RMSE"):
    s = STACKS[name]
    I, V = load_data(name)

    def f(x):
        Vs = stack_voltage(I, x, s)
        if not np.all(np.isfinite(Vs)):
            return 1e6
        e = V - Vs
        return float(np.sqrt(np.mean(e * e))) if kind == "RMSE" else float(np.sum(e * e))
    return f
