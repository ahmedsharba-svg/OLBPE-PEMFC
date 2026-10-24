"""OLBPE and the baseline optimisers.

Every optimiser
  * searches the normalised box u in [0, 1]^D (x = LB + u * (UB - LB)),
  * stops after exactly `max_fes` objective evaluations,
  * returns (x_best, f_best, curve) where `curve` holds the best-so-far
    value at `n_curve` equally spaced evaluation counts.
"""
import numpy as np


class Evaluator:
    """Counts evaluations and records the best-so-far convergence curve."""

    def __init__(self, f, lb, ub, max_fes, n_curve=200):
        self.f, self.lb, self.span = f, np.asarray(lb, float), np.asarray(ub, float) - np.asarray(lb, float)
        self.max_fes, self.fe, self.best = int(max_fes), 0, np.inf
        self.cp = np.unique(np.round(np.linspace(1, max_fes, n_curve)).astype(int))
        self.curve = np.full(len(self.cp), np.nan)
        self.ci = 0

    def __call__(self, u):
        v = self.f(self.lb + u * self.span)
        self.fe += 1
        if v < self.best:
            self.best = v
        while self.ci < len(self.cp) and self.fe >= self.cp[self.ci]:
            self.curve[self.ci] = self.best
            self.ci += 1
        return v

    def left(self):
        return self.max_fes - self.fe

    def decode(self, u):
        return self.lb + u * self.span

    def final_curve(self):
        c = self.curve.copy()
        c[np.isnan(c)] = self.best
        return c


# ----------------------------------------------------------------- helpers
def orthogonal_array(d):
    """Two-level orthogonal array L_M(2^Q), Q >= d, entries 1/2, row 0 all ones."""
    u = int(np.ceil(np.log2(d + 1)))
    m = 2 ** u
    return np.array([[1 + bin(r & c).count("1") % 2 for c in range(1, d + 1)] for r in range(m)])


def weighted_cov(X, w):
    mu = w @ X
    Z = X - mu
    C = (Z.T * w) @ Z / max(1.0 - np.sum(w ** 2), 1e-12)
    return (C + C.T) / 2, mu


def safe_chol(C):
    d = C.shape[0]
    try:
        return np.linalg.cholesky(C + 1e-14 * np.eye(d))
    except np.linalg.LinAlgError:
        return np.diag(np.sqrt(np.maximum(np.diag(C), 1e-14)))


# -------------------------------------------------------------------- OLBPE
def olbpe(f, lb, ub, rng, pop_size=40, max_fes=40000, n_curve=200,
          EWL=True, ESG=True, OL=True, VTC=True, CWR=True,
          elite_frac=0.2, cwr_frac=0.15):
    """Orthogonal Learning-Based Population Encoding (see Section 3 of the paper).

    EWL elite-worst learning, ESG elite sensitivity guidance (rank-weighted
    elite covariance), OL stagnation-triggered orthogonal learning with an
    L8(2^7) array, VTC violation-triggered correction, CWR confidence-weighted
    refinement. Each switch can be turned off for the ablation study.
    """
    ev = Evaluator(f, lb, ub, max_fes, n_curve)
    D, N = len(lb), pop_size
    k = max(3, int(np.ceil(elite_frac * N)))
    w = np.log(k + 0.5) - np.log(np.arange(1, k + 1)); w /= w.sum()
    wN = np.log(N + 0.5) - np.log(np.arange(1, N + 1)); wN /= wN.sum()
    OA = orthogonal_array(D); M = len(OA)

    def repair(v, ref):
        lo, hi = v < 0, v > 1
        if VTC:
            v = v.copy()
            v[lo] = ref[lo] * rng.random(lo.sum())
            v[hi] = ref[hi] + (1 - ref[hi]) * rng.random(hi.sum())
            return v
        return np.clip(v, 0, 1)

    U = rng.random((N, D))
    fit = np.array([ev(u) for u in U])

    while ev.left() > 0:
        p = ev.fe / max_fes
        if CWR and p >= 1 - cwr_frac:
            break
        o = np.argsort(fit); U, fit = U[o], fit[o]
        worst, E = U[-1], U[:k]
        a = 2 * (1 - p)
        f_old = fit[0]
        C, _ = weighted_cov(E, w)
        L = safe_chol(C) if ESG else np.eye(D) * np.sqrt(max(np.trace(C) / D, 1e-14))
        for i in range(N):
            if ev.left() <= 0:
                break
            xe = E[rng.integers(k)]
            if rng.random() < 0.5 + 0.3 * (1 - p):           # learning phase
                F = rng.uniform(0.4, 0.9)
                r1, r2 = rng.choice(N, 2, replace=False)
                if EWL:
                    v = U[i] + F * (xe - U[i]) + F * (U[r1] - U[r2]) + 0.25 * a * rng.random() * (xe - worst)
                else:
                    v = U[i] + F * (U[r1] - U[r2])
                mask = rng.random(D) < 0.9
                mask[rng.integers(D)] = True
                v = np.where(mask, v, U[i])
            else:                                            # guided sampling
                v = xe + (0.5 + a / 2) * (L @ rng.standard_normal(D))
            v = repair(v, xe)
            fv = ev(v)
            if fv <= fit[i]:
                U[i], fit[i] = v, fv
        # stagnation-triggered orthogonal learning
        if OL and ev.left() >= M and fit.min() >= f_old:
            ib = int(np.argmin(fit)); b = U[ib].copy()
            g = U[rng.integers(N)]
            if np.any(np.abs(g - b) > 1e-12):
                cand = np.where(OA == 1, b, g)
                fc = np.empty(M); fc[0] = fit[ib]
                for m in range(1, M):
                    fc[m] = ev(cand[m])
                pred = np.array([g[d] if fc[OA[:, d] == 2].mean() < fc[OA[:, d] == 1].mean() else b[d]
                                 for d in range(D)])
                fp = ev(pred)
                allc, allf = np.vstack([cand, pred]), np.append(fc, fp)
                j = int(np.argmin(allf))
                if allf[j] < fit[ib]:
                    U[ib], fit[ib] = allc[j], allf[j]

    o = np.argsort(fit); U, fit = U[o], fit[o]
    b, fb = U[0].copy(), fit[0]
    if CWR and ev.left() > 0:                                # confidence-weighted refinement
        Ln = safe_chol(weighted_cov(U, wN)[0])
        sig, succ, tries = 1.0, 0, 0
        while ev.left() > 0:
            v = np.clip(b + sig * (Ln @ rng.standard_normal(D)), 0, 1)
            fv = ev(v); tries += 1
            if fv < fb:
                b, fb = v, fv; succ += 1
            if tries == 10:
                sig = min(max(sig * (1.5 if succ / tries > 0.2 else 0.7), 1e-6), 2.0)
                succ = tries = 0
    return ev.decode(b), fb, ev.final_curve()


# ---------------------------------------------------------------- baselines
def de(f, lb, ub, rng, pop_size=40, max_fes=40000, n_curve=200, F=0.5, CR=0.9):
    """DE/rand/1/bin (Storn & Price, 1997)."""
    ev = Evaluator(f, lb, ub, max_fes, n_curve); D, N = len(lb), pop_size
    X = rng.random((N, D)); fit = np.array([ev(x) for x in X])
    while ev.left() > 0:
        for i in range(N):
            if ev.left() <= 0:
                break
            a, b, c = rng.choice([j for j in range(N) if j != i], 3, replace=False)
            v = X[a] + F * (X[b] - X[c])
            mask = rng.random(D) < CR; mask[rng.integers(D)] = True
            u = np.clip(np.where(mask, v, X[i]), 0, 1)
            fu = ev(u)
            if fu <= fit[i]:
                X[i], fit[i] = u, fu
    i = int(np.argmin(fit))
    return ev.decode(X[i]), fit[i], ev.final_curve()


def pso(f, lb, ub, rng, pop_size=40, max_fes=40000, n_curve=200):
    """PSO with linearly decreasing inertia 0.9 -> 0.4, c1 = c2 = 2 (Shi & Eberhart, 1998)."""
    ev = Evaluator(f, lb, ub, max_fes, n_curve); D, N = len(lb), pop_size
    X = rng.random((N, D)); V = np.zeros((N, D)); P = X.copy()
    pf = np.array([ev(x) for x in X]); g = P[np.argmin(pf)].copy(); gf = pf.min()
    while ev.left() > 0:
        w = 0.9 - 0.5 * ev.fe / max_fes
        for i in range(N):
            if ev.left() <= 0:
                break
            V[i] = np.clip(w * V[i] + 2 * rng.random(D) * (P[i] - X[i]) + 2 * rng.random(D) * (g - X[i]), -0.2, 0.2)
            X[i] = np.clip(X[i] + V[i], 0, 1)
            fx = ev(X[i])
            if fx < pf[i]:
                P[i], pf[i] = X[i].copy(), fx
            if fx < gf:
                g, gf = X[i].copy(), fx
    return ev.decode(g), gf, ev.final_curve()


def gwo(f, lb, ub, rng, pop_size=40, max_fes=40000, n_curve=200):
    """Grey wolf optimizer (Mirjalili et al., 2014)."""
    ev = Evaluator(f, lb, ub, max_fes, n_curve); D, N = len(lb), pop_size
    X = rng.random((N, D)); fit = np.array([ev(x) for x in X])
    bi = int(np.argmin(fit)); best, fbest = X[bi].copy(), fit[bi]
    while ev.left() > 0:
        Lw = X[np.argsort(fit)[:3]].copy()
        a = 2 * (1 - ev.fe / max_fes)
        for i in range(N):
            if ev.left() <= 0:
                break
            nx = np.zeros(D)
            for Lj in Lw:
                A = 2 * a * rng.random(D) - a; Cc = 2 * rng.random(D)
                nx += Lj - A * np.abs(Cc * Lj - X[i])
            X[i] = np.clip(nx / 3, 0, 1); fit[i] = ev(X[i])
            if fit[i] < fbest:
                best, fbest = X[i].copy(), fit[i]
    return ev.decode(best), fbest, ev.final_curve()


def woa(f, lb, ub, rng, pop_size=40, max_fes=40000, n_curve=200):
    """Whale optimization algorithm (Mirjalili & Lewis, 2016), spiral constant b = 1."""
    ev = Evaluator(f, lb, ub, max_fes, n_curve); D, N = len(lb), pop_size
    X = rng.random((N, D)); fit = np.array([ev(x) for x in X])
    bi = int(np.argmin(fit)); g, gf = X[bi].copy(), fit[bi]
    while ev.left() > 0:
        a = 2 * (1 - ev.fe / max_fes)
        for i in range(N):
            if ev.left() <= 0:
                break
            A = 2 * a * rng.random() - a; Cc = 2 * rng.random()
            if rng.random() < 0.5:
                ref = g if abs(A) < 1 else X[rng.integers(N)]
                X[i] = ref - A * np.abs(Cc * ref - X[i])
            else:
                l = rng.uniform(-1, 1)
                X[i] = np.abs(g - X[i]) * np.exp(l) * np.cos(2 * np.pi * l) + g
            X[i] = np.clip(X[i], 0, 1); fit[i] = ev(X[i])
            if fit[i] < gf:
                g, gf = X[i].copy(), fit[i]
    return ev.decode(g), gf, ev.final_curve()


REGISTRY = {"OLBPE": olbpe, "DE": de, "PSO": pso, "GWO": gwo, "WOA": woa}
