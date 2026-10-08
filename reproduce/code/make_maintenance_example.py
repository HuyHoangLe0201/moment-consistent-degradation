# -*- coding: utf-8 -*-
r"""What interval scaling does to a replacement decision, from stored results.

The grid study (results/theory/grid_consistency.json) stores the mean and
standard deviation of the simulated first-passage time on five inspection
grids, under interval scaling and under the moment-consistent
parameterisation.  A planner who schedules replacement at the eta-quantile of
the RUL law accepts a risk eta of failing first.  This asks what that risk
really is when the plan comes from the interval-scaled model but the asset
follows the moment-consistent process.

Each RUL law is taken as the inverse Gaussian matched to the simulated mean and
standard deviation, which is the form Theorem (IG RUL approximation) gives it.
No new simulation is run: every number below is a deterministic function of the
stored file, so the table cannot drift from the study it summarises.

    python code/make_maintenance_example.py
"""
from __future__ import annotations

import json
import os

from scipy.stats import invgauss

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ETA = 0.05


def ig(mean, sd):
    lam = mean ** 3 / sd ** 2            # IG(mean, shape) has var mean^3/shape
    return invgauss(mu=mean / lam, scale=lam)


fp = json.load(open(os.path.join(RES, "theory", "grid_consistency.json"),
                    encoding="utf-8"))["first_passage"]
rows = []
for a, b in zip(fp["naive"], fp["consistent"]):
    assert a["steps_per_unit"] == b["steps_per_unit"]
    naive, true = ig(a["mean"], a["sd"]), ig(b["mean"], b["sd"])
    q_naive, q_true = naive.ppf(ETA), true.ppf(ETA)
    rows.append({"steps_per_unit": a["steps_per_unit"],
                 "q_naive": q_naive, "q_consistent": q_true,
                 "true_risk_at_naive_plan": float(true.cdf(q_naive))})

#  Sensitivity.  Inspecting k times more often divides the interval-scaled
#  first-passage variance by k and leaves the mean (Proposition 1), so the
#  carried risk depends only on k and on the shape phi = mean/CV^2 ... of the
#  life law, i.e. phi = lambda/mu of the IG.  In the normal limit it is
#  Phi(z_eta / sqrt(k)), whatever alpha, beta or D.
from scipy.stats import norm                                      # noqa: E402
PHIS = (2.0, 5.0, 10.0, 25.0, 100.0)
KS = (2, 4, 8, 16)
sens = {"phi": list(PHIS), "k": list(KS), "risk": [], "normal_limit": []}
for k in KS:
    sens["normal_limit"].append(float(norm.cdf(norm.ppf(ETA) / k ** 0.5)))
    row = []
    for phi in PHIS:
        true = invgauss(mu=1 / phi, scale=phi)          # IG(mean 1, shape phi)
        naive = invgauss(mu=1 / (phi * k), scale=phi * k)
        row.append(float(true.cdf(naive.ppf(ETA))))
    sens["risk"].append(row)
#  the life law of Table 7, for reference: phi = mean^2/var
b16 = [b for b in fp["consistent"] if b["steps_per_unit"] == 1][0]
sens["phi_table"] = b16["mean"] ** 2 / b16["sd"] ** 2

out = {"eta": ETA, "rows": rows, "sensitivity": sens,
       "note": "IG laws matched to the simulated first-passage mean and sd"}
json.dump(out, open(os.path.join(RES, "theory", "maintenance_example.json"),
                    "w", encoding="utf-8"), indent=2)

lines = [r"\begin{table}[htbp]", r"\centering\small",
         r"\caption{Replacement at the $5\,\%$ quantile on the first-passage study of"
         r" \Cref{tab:grid}, with inverse Gaussian laws matched to the simulated"
         r" moments. Actual risk: probability that a moment-consistent asset fails"
         r" before the interval-scaled plan.}",
         r"\label{tab:maintenance}",
         r"\begin{tabular}{@{}r rr r@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{planned replacement time} & \\",
         r"\cmidrule(lr){2-3}",
         r"steps per unit time & interval scaling & moment-consistent"
         r" & actual risk\\", r"\midrule"]
for r in rows:
    lines.append(f"{r['steps_per_unit']} & {r['q_naive']:.2f} & "
                 f"{r['q_consistent']:.2f} & "
                 f"${100 * r['true_risk_at_naive_plan']:.1f}\\,\\%$\\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
open(os.path.join(RES, "tables", "tab_maintenance.tex"), "w",
     encoding="utf-8", newline="\n").write("\n".join(lines))

for r in rows:
    print(f"  {r['steps_per_unit']:3d} steps/unit: plan {r['q_naive']:.3f} vs "
          f"{r['q_consistent']:.3f}, actual risk "
          f"{100 * r['true_risk_at_naive_plan']:.1f} %")
print("  -> results/tables/tab_maintenance.tex")
