"""Self-checks: python -m pytest tests  (or: python tests/test_all.py)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from scipy import stats
from olbpe_pemfc import config as C
from olbpe_pemfc.model import STACKS, load_data, stack_voltage, metrics, make_objective
from olbpe_pemfc.algorithms import REGISTRY, orthogonal_array
from olbpe_pemfc.stats import ranksum, friedman, holm, signrank


def test_model_reproduces_published_curve():
    """Puma-optimizer parameters of Liu & Cheng (2025), Table 6, give their SSE 1.1556e-2 on BCS 500 W."""
    x = [-0.85322, 2.17931e-3, 3.6e-5, -1.92893e-4, 20.8145, 0.0161134, 1e-4]
    I, V = load_data("BCS500W")
    assert abs(metrics(x, I, V, STACKS["BCS500W"])["SSE"] - 1.1556e-2) < 2e-5


def test_orthogonal_array():
    OA = orthogonal_array(7)
    assert OA.shape == (8, 7) and (OA[0] == 1).all()
    for a in range(7):
        for b in range(a + 1, 7):
            for u in (1, 2):
                for v in (1, 2):
                    assert ((OA[:, a] == u) & (OA[:, b] == v)).sum() == 2


def test_statistics():
    x, y = np.arange(1, 11), np.arange(6, 16)
    assert abs(ranksum(x, y)[0] - 0.005075392315273923) < 1e-12
    X = np.array([[1, 2, 3], [2, 1, 3], [1, 3, 2], [1, 2, 3], [2, 3, 1]], float)
    p, chi2, _ = friedman(X)
    assert abs(chi2 - 2.8) < 1e-12 and abs(p - stats.friedmanchisquare(*X.T).pvalue) < 1e-12
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    a = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]); b = a + np.array([.5, -.2, .9, .4, .7, .3, .8, -.1, .6, .2])
    assert abs(signrank(a, b)[0] - stats.wilcoxon(a, b, method='exact').pvalue) < 1e-12
    assert abs(signrank([1, 2, 3], [0, 0, 0])[0] - 0.25) < 1e-12   # n = 3, all positive: exact p = 2/8


def test_budget_and_curve():
    f = make_objective("BCS500W")
    for name, fn in REGISTRY.items():
        x, fb, c = fn(f, C.LB, C.UB, np.random.default_rng(0), pop_size=20, max_fes=1500, n_curve=30)
        assert len(c) == 30 and np.all(np.diff(c) <= 1e-15) and abs(c[-1] - fb) < 1e-12, name
        assert np.all(x >= C.LB - 1e-12) and np.all(x <= C.UB + 1e-12), name


def test_reproducible():
    f = make_objective("SR12")
    a = REGISTRY["OLBPE"](f, C.LB, C.UB, np.random.default_rng(5), max_fes=2000)[1]
    b = REGISTRY["OLBPE"](f, C.LB, C.UB, np.random.default_rng(5), max_fes=2000)[1]
    assert a == b


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("[ok]", k)
