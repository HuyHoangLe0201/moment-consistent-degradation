# -*- coding: utf-8 -*-
r"""The published antecedent's scaling, on the grid study of Sec. 5.1.

Fujita and Nagatsuka (2021) give each increment the law
BS(alpha * dLambda_k, beta * dLambda_k).  With Lambda(t) = t, alpha = 0.4,
beta = 1 and the horizon T = 20 split into n = 1, 2, ..., 64 equal steps (the
grid of results/theory/grid_consistency.json), the accumulated moments are in
closed form:

    E[X(T)]   = n * b (1 + a^2/2)              = beta T (1 + alpha^2 s^2 / 2),
    Var[X(T)] = n * b^2 a^2 (1 + 5 a^2 / 4)    = alpha^2 beta^2 T s^3 (1 + 5 alpha^2 s^2 / 4),

with s = T / n, a = alpha s and b = beta s.  Remark 5.1 of the paper quotes the
factor by which each moment varies across the grid and the rate at which the
variance decays in n; this file computes all three.

    python code/study_fujita_scaling.py   ->  results/theory/fujita_scaling.json
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "theory", "fujita_scaling.json")

ALPHA, BETA, T = 0.4, 1.0, 20.0
G = json.load(open(os.path.join(HERE, "..", "results", "theory", "grid_consistency.json")))
n = np.array([r["n"] for r in G["naive"]], float)          # the paper's grid
assert G["alpha"] == ALPHA and G["beta"] == BETA and G["T"] == T

s = T / n
a, b = ALPHA * s, BETA * s
mean = n * b * (1 + a ** 2 / 2)
var = n * b ** 2 * a ** 2 * (1 + 5 * a ** 2 / 4)

slope_ls = float(np.polyfit(np.log(n), np.log(var), 1)[0])       # least squares
slope_end = float(np.log(var[-1] / var[0]) / np.log(n[-1] / n[0]))  # end points

res = {"alpha": ALPHA, "beta": BETA, "T": T, "n": n.tolist(),
       "mean": mean.tolist(), "var": var.tolist(),
       "mean_factor": float(mean.max() / mean.min()),
       "var_factor": float(var.max() / var.min()),
       "var_slope_least_squares": slope_ls,
       "var_slope_end_points": slope_end}
json.dump(res, open(OUT, "w"), indent=1)
print(f"mean varies by {res['mean_factor']:.2f}, variance by {res['var_factor']:.3g}; "
      f"variance slope in n: {slope_ls:.3f} (least squares), {slope_end:.3f} (end points)")
