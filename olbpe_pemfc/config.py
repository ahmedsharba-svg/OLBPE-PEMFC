"""Single source of truth for every experimental setting used in the paper."""
import numpy as np

RUNS = 30                 # independent runs per algorithm and stack
POP_SIZE = 40             # population size of every algorithm
MAX_ITER = 1000           # nominal iterations (reported only)
MAX_FES = POP_SIZE * MAX_ITER   # identical evaluation budget for all algorithms
SEED0 = 1000              # run r uses seed SEED0 + r for EVERY algorithm
N_CURVE = 200             # checkpoints of the convergence curve stored per run
OBJECTIVE = "RMSE"        # objective minimised (RMSE, Eq. 15 of the paper)
ALPHA = 0.05              # significance level
TIE_SIG = 10              # values equal to 10 significant digits are ties

# Search space (Table 1 of the paper):  xi1, xi2, xi3, xi4, lambda, b, Rc
PARAM_NAMES = ["xi1", "xi2", "xi3", "xi4", "lambda", "b", "Rc"]
LB = np.array([-1.19969, 1e-3, 3.6e-5, -2.6e-4, 10.0, 0.0136, 1e-4])
UB = np.array([-0.8532,  5e-3, 9.8e-5, -9.54e-5, 24.0, 0.5,    8e-4])

STACKS = ["BCS500W", "SR12", "NedStackPS6"]
ALGORITHMS = ["OLBPE", "DE", "PSO", "GWO", "WOA"]
ABLATION_REDUCED_FES = 5000     # second, reduced budget for the ablation study
ABLATION = {
    "OLBPE (full)": {},
    "w/o EWL": {"EWL": False},
    "w/o ESG": {"ESG": False},
    "w/o OL":  {"OL": False},
    "w/o VTC": {"VTC": False},
    "w/o CWR": {"CWR": False},
}
