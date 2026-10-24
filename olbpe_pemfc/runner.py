"""Runs algorithms x stacks x runs and stores everything needed for the paper.
Results are appended to <out>.partial.jsonl as they finish, so an interrupted
run resumes where it stopped."""
import json, os, time
from multiprocessing import Pool
import numpy as np
from . import config as C
from .model import STACKS, make_objective, load_data, metrics
from .algorithms import REGISTRY, olbpe


def _one(job):
    stack, name, r, kw, max_fes = job
    f = make_objective(stack, C.OBJECTIVE)
    rng = np.random.default_rng(C.SEED0 + r)
    fn = olbpe if kw is not None else REGISTRY[name]
    t0 = time.perf_counter()
    x, fb, curve = fn(f, C.LB, C.UB, rng, pop_size=C.POP_SIZE, max_fes=max_fes,
                      n_curve=C.N_CURVE, **(kw or {}))
    t = time.perf_counter() - t0
    I, V = load_data(stack)
    m = metrics(x, I, V, STACKS[stack])
    return dict(stack=stack, algorithm=name, run=r, seed=C.SEED0 + r, f=fb, time=t,
                x=x.tolist(), curve=curve.tolist(), **m)


def run(jobs_spec, out_file, workers=1, max_fes=None):
    """jobs_spec: list of (name, kwargs or None); kwargs != None -> OLBPE variant."""
    max_fes = max_fes or C.MAX_FES
    part = out_file + ".partial.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            o = json.loads(line); done[(o["stack"], o["algorithm"], o["run"])] = o
    jobs = [(s, n, r, kw, max_fes) for s in C.STACKS for n, kw in jobs_spec
            for r in range(1, C.RUNS + 1) if (s, n, r) not in done]
    with open(part, "a") as ph, Pool(workers) as pool:
        for out in pool.imap(_one, jobs, chunksize=1):
            done[(out["stack"], out["algorithm"], out["run"])] = out
            ph.write(json.dumps(out) + "\n"); ph.flush()
            grp = [o["RMSE"] for k, o in done.items() if k[0] == out["stack"] and k[1] == out["algorithm"]]
            if len(grp) == C.RUNS:
                print(f"{out['stack']:12s} {out['algorithm']:14s} best {min(grp):.6g} "
                      f"mean {np.mean(grp):.6g} std {np.std(grp, ddof=1):.2g}", flush=True)
    res = [done[(s, n, r)] for s in C.STACKS for n, _ in jobs_spec for r in range(1, C.RUNS + 1)]
    with open(out_file, "w") as fh:
        json.dump(dict(config=dict(runs=C.RUNS, pop=C.POP_SIZE, max_fes=max_fes, seed0=C.SEED0,
                                   objective=C.OBJECTIVE, lb=C.LB.tolist(), ub=C.UB.tolist()),
                       results=res), fh)
    os.remove(part)
    return res
