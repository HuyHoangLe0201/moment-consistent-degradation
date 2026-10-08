"""Does a model fitted on one inspection schedule still hold on another?

schedule_change_test() runs the protocol of Section 7.2 of the paper on your own
degradation paths: thin the records to one schedule (h_ref), fit there, predict
on another (h_dep), and compare the stated coverage of the prediction
intervals with what the held-out unit actually did.  Both parameterisations
are run:

    consistent   a step of length h has squared CV  cv2_ref * h_ref / h
    interval     a step of length h has squared CV  cv2_ref     (fixed shape)

Each unit is held out in turn; its rate comes from its own readings up to the
forecast origin.  The uncertainty of that rate (under the model's own variance
claim) and of cv2_ref (scaled inverse chi-square) is carried into every
prediction.  Forecasts of one unit share its path, so the confidence interval
of each coverage resamples units.

The test is for homogeneous paths (no damage coupling, roughly constant rate on
the chosen clock).  For decelerating or accelerating paths pass clock, e.g.
clock=lambda t: t**p, and the test runs on that clock.
"""
from __future__ import annotations

import numpy as np

from .diagnostics import fit_reference, thin
from .families import get_family

__all__ = ["schedule_change_test"]


def _steps(fam, mean, cv2, size, rng):
    a, b = fam.from_moments(mean, cv2 * mean ** 2)
    return fam.sample(a, b, size=size, rng=rng)


def schedule_change_test(paths, h_ref, h_dep, family="BS", horizon=None,
                         origins=None, level=0.9, n_sim=20000, clock=None,
                         n_boot=2000, rng=None):
    """Leave-one-unit-out coverage of level-`level` intervals for the reading at
    `horizon`, issued at each time in `origins` on the h_dep schedule.

    paths    list of (t, x); every path must contain both grids
    horizon  time of the reading to predict (default: last common time)
    origins  forecast origins (default: 1/4, 1/2, 3/4 of the horizon)

    Returns {"consistent": {...}, "interval": {...}} with coverage, its 95 %
    unit-bootstrap interval, mean interval width and the number of forecasts."""
    fam = get_family(family)
    rng = np.random.default_rng(rng)
    tau = (lambda t: np.asarray(t, float)) if clock is None else clock
    paths = [(np.asarray(t, float), np.asarray(x, float)) for t, x in paths]
    if horizon is None:
        horizon = min(t[-1] for t, _ in paths)
    if origins is None:
        origins = [horizon * f for f in (0.25, 0.5, 0.75)]
    a_lo, a_hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {m: [] for m in ("consistent", "interval")}
    for i, (t, x) in enumerate(paths):
        rec = []
        for j, (tj, xj) in enumerate(paths):
            if j != i:
                tt, xx = thin(tj, xj, h_ref)
                rec.append((tau(tt), xx))
        ref = fit_reference(rec, dt_ref=1.0)        # clock units
        tt, xx = thin(t, x, h_dep)
        cl = tau(tt)
        if not np.any(np.isclose(tt, horizon)):
            continue
        obs = xx[np.isclose(tt, horizon)][0]
        for t0 in origins:
            k = np.nonzero(np.isclose(tt, t0))[0]
            if not len(k) or k[0] < 1:
                continue
            k = k[0]
            fut = (tt > t0 + 1e-12) & (tt <= horizon + 1e-12)
            S = np.diff(np.r_[cl[k], cl[fut]])
            S_hist = np.diff(cl[:k + 1])
            span = cl[k] - cl[0]
            rate = (xx[k] - xx[0]) / span
            # ref["cv2"] is the squared CV of a step of one clock unit under the
            # consistent model; interval scaling keeps the squared CV of the
            # record step, cv2 / S_ref, whatever the step length
            S_ref = np.mean([np.mean(np.diff(c)) for c, _ in rec])
            for m in out:
                cv2 = ref["cv2"] * ref["df"] / rng.chisquare(ref["df"], (n_sim, 1))
                if m == "consistent":
                    c_step, c_hist = cv2 / S, cv2 / S_hist
                else:
                    c_step = cv2 / S_ref * np.ones_like(S)
                    c_hist = cv2 / S_ref * np.ones_like(S_hist)
                rel2 = np.sum(S_hist ** 2 * c_hist, axis=1, keepdims=True) / span ** 2
                r = rate * np.exp(np.sqrt(rel2) * rng.standard_normal((n_sim, 1)) - rel2 / 2)
                steps = _steps(fam, np.broadcast_to(S, c_step.shape) * 1.0, c_step,
                               c_step.shape, rng)
                end = xx[k] + (steps * r).sum(axis=1)
                lo, hi = np.quantile(end, [a_lo, a_hi])
                out[m].append((i, bool(lo <= obs <= hi), float(hi - lo)))
    res = {}
    for m, rows in out.items():
        if not rows:
            raise ValueError("no forecasts: check that horizon and origins lie on the h_dep grid")
        units = sorted({u for u, _, _ in rows})
        hits = {u: [h for v, h, _ in rows if v == u] for u in units}
        boots = [np.mean([h for u in rng.choice(units, len(units)) for h in hits[u]])
                 for _ in range(n_boot)]
        res[m] = {"coverage": float(np.mean([h for _, h, _ in rows])),
                  "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                  "width": float(np.mean([w for _, _, w in rows])),
                  "n_forecasts": len(rows), "level": level}
    return res
