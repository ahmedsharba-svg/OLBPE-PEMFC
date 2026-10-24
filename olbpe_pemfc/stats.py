"""Non-parametric tests used in the paper."""
import numpy as np
from scipy import stats


def sigfig(x, n=10):
    x = np.asarray(x, float)
    out = x.copy()
    nz = (x != 0) & np.isfinite(x)
    d = np.floor(np.log10(np.abs(x[nz])))
    s = 10.0 ** (n - 1 - d)
    out[nz] = np.round(x[nz] * s) / s
    return out


def holm(p):
    p = np.asarray(p, float); m = len(p)
    o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[o]))
    out = np.empty(m); out[o] = adj
    return out


def ranksum(x, y):
    """Two-sided Wilcoxon rank-sum (Mann-Whitney U), normal approximation with
    tie and continuity correction. Returns p, z (z < 0: x has smaller ranks)."""
    x, y = np.asarray(x), np.asarray(y)
    if np.all(np.concatenate([x, y]) == x[0]):
        return 1.0, 0.0
    res = stats.mannwhitneyu(x, y, alternative="two-sided", use_continuity=True, method="asymptotic")
    n1, n2 = len(x), len(y)
    r = stats.rankdata(np.concatenate([x, y]))
    W = r[:n1].sum(); mu = n1 * (n1 + n2 + 1) / 2
    _, t = np.unique(r, return_counts=True)
    n = n1 + n2
    s2 = n1 * n2 / 12 * ((n + 1) - np.sum(t ** 3 - t) / (n * (n - 1)))
    z = 0.0 if s2 <= 0 else (W - mu - 0.5 * np.sign(W - mu)) / np.sqrt(s2)
    return float(res.pvalue), float(z)


def wilcoxon_vs(F, names, ref, alpha=0.05, tie_sig=10):
    """F: runs x k matrix. Compare column `ref` with every other column."""
    F = sigfig(F, tie_sig)
    ir = names.index(ref)
    others = [j for j in range(len(names)) if j != ir]
    p, z = zip(*[ranksum(F[:, ir], F[:, j]) for j in others])
    ph = holm(p)
    rows = []
    for j, pj, pjh, zj in zip(others, p, ph, z):
        sign = "=" if pjh >= alpha else ("+" if zj < 0 else "-")
        rows.append(dict(comparison=f"{ref} vs {names[j]}", p=pj, p_holm=pjh, z=zj,
                         median_ref=float(np.median(F[:, ir])), median_other=float(np.median(F[:, j])),
                         sign=sign))
    return rows


def friedman(F, tie_sig=10):
    """Friedman test over runs (blocks) and algorithms (columns): p, chi2, mean ranks."""
    F = sigfig(F, tie_sig)
    R = np.vstack([stats.rankdata(row) for row in F])
    if np.all(np.ptp(F, axis=1) == 0):
        return 1.0, 0.0, R.mean(0)
    chi2, p = stats.friedmanchisquare(*F.T)
    return float(p), float(chi2), R.mean(0)


def _exact_signrank_p(r, rp):
    """Exact two-sided p of the signed-rank statistic R+ = rp for (possibly tied)
    ranks r, from the permutation distribution over the 2^n sign assignments
    (dynamic programming on doubled ranks; exact also with tied ranks)."""
    w = np.round(2 * np.asarray(r)).astype(int)
    tot = int(w.sum())
    dist = np.zeros(tot + 1); dist[0] = 1.0
    for wi in w:
        new = dist.copy(); new[wi:] += dist[:tot + 1 - wi]; dist = new
    dist /= dist.sum()
    k = int(round(2 * rp))
    p = 2 * min(dist[:k + 1].sum(), dist[k:].sum())
    return float(min(1.0, p))


def signrank(x, y, exact_max_n=50):
    """Two-sided Wilcoxon signed-rank test for paired runs (same seed -> same
    initial population). Zero differences are discarded (Wilcoxon's method).
    Exact permutation p-value when the number of non-zero differences is
    <= exact_max_n (always the case for 30 runs), normal approximation otherwise.
    Returns p, R+ (sum of ranks where x > y), R- (x < y), n_nonzero."""
    d = np.asarray(x, float) - np.asarray(y, float)
    nz = d[d != 0]
    n = len(nz)
    if n == 0:
        return 1.0, 0.0, 0.0, 0
    r = stats.rankdata(np.abs(nz))
    rp, rm = float(r[nz > 0].sum()), float(r[nz < 0].sum())
    if n <= exact_max_n:
        p = _exact_signrank_p(r, rp)
    else:
        p = float(stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="approx").pvalue)
    return p, rp, rm, n


def wilcoxon_paired_vs(F, names, ref, alpha=0.05, tie_sig=10):
    """Paired comparison of column `ref` with every other column (Holm-corrected).
    Also reports the rank-sum p-value as a robustness check."""
    F = sigfig(F, tie_sig)
    ir = names.index(ref)
    others = [j for j in range(len(names)) if j != ir]
    out = [signrank(F[:, ir], F[:, j]) for j in others]
    ph = holm([o[0] for o in out])
    prs = holm([ranksum(F[:, ir], F[:, j])[0] for j in others])
    rows = []
    for j, (p, rp, rm, n), pj, pr in zip(others, out, ph, prs):
        # R- large  -> ref smaller (better) in most pairs
        sign = "=" if pj >= alpha else ("+" if rm > rp else "-")
        rows.append(dict(comparison=f"{ref} vs {names[j]}", p=p, p_holm=pj, R_plus=rp, R_minus=rm,
                         n_nonzero=n, wins=int(np.sum(F[:, ir] < F[:, j])), ties=int(np.sum(F[:, ir] == F[:, j])),
                         losses=int(np.sum(F[:, ir] > F[:, j])), p_holm_ranksum=pr, sign=sign))
    return rows
