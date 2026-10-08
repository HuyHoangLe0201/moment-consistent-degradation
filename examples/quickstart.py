"""Quick start: use the method on your own degradation data.

    pip install -e .
    python examples/quickstart.py

Steps
  1. why interval scaling fails, in three lines
  2. one step of any length, in moment-consistent form, for five families
  3. the RUL on any inspection schedule (Algorithm 1), with damage coupling
  4. your data: fit the reference law, check the requirement, test a schedule change
"""
import numpy as np

import mcdeg as M

rng = np.random.default_rng(0)

# ── 1. the problem ───────────────────────────────────────────────────────────
# Reference law: one step of dt_ref = 1 h has mean 0.05 and CV 0.3.
mu, var = 0.05, (0.3 * 0.05) ** 2
print("1. accumulated variance over 8 h, by inspection interval")
for dt in (4.0, 1.0, 0.25):
    n = int(8 / dt)
    for name, f in (("consistent", M.consistent_params), ("interval  ", M.interval_params)):
        a, b = f("BS", mu, var, dt)
        X = M.BS.sample(a, b, size=(50000, n), rng=rng).sum(axis=1)
        print(f"   dt = {dt:4.2f} h  {name}  Var = {X.var():.2e}")
print("   consistent: the same on every grid; interval scaling: shrinks with dt\n")

# ── 2. one step, any family ─────────────────────────────────────────────────
print("2. parameters of a 15-minute step (s = 0.25)")
for fam in (M.BS, M.WEIBULL, M.GAMMA, M.IG, M.LOGNORMAL):
    shape, scale = M.consistent_params(fam, mu, var, 0.25)
    m, v = fam.moments(shape, scale)
    print(f"   {fam.name:9s} shape {float(shape):8.4f}  scale {float(scale):.5f}"
          f"  mean {float(m):.4f}  var {float(v):.2e}")
print("   BS has a finest grid: squared CV must stay below 5, i.e."
      f" s > {var / mu**2 / 5:.4f} here\n")

# ── 3. RUL on any schedule (Algorithm 1) ─────────────────────────────────────
# Paris-coupled damage: the step scale grows as q(x)^gamma, q = 1 + x / x_ref.
print("3. RUL to D = 2.0, gamma = 1.0, x_ref = 1, inspected every 0.5 h")
for x_now in (0.0, 1.6):
    p = M.predict_rul("BS", x_now, 2.0, mu, var, t_k=100.0, eta=0.05,
                      dt=0.5, gamma=1.0, x_ref=1.0, rng=1)
    print(f"   from x = {x_now}: mean {p.mean:6.2f} h, sd {p.sd:5.2f} h, "
          f"replace at t = {p.replace_at:6.2f} h  [{p.method}, n_d = {p.n_d:.1f}]")
print()

# ── 4. your data ─────────────────────────────────────────────────────────────
# A list of (t, x) pairs, one per unit.  Here: 20 simulated units with their own
# rates, read every hour for 32 hours.  Replace this with your records, e.g.
#   import pandas as pd
#   df = pd.read_csv("my_data.csv")          # columns: unit, time, value
#   paths = [(g.time.to_numpy(), g.value.to_numpy())
#            for _, g in df.sort_values("time").groupby("unit")]
paths = []
for _ in range(20):
    rate = rng.uniform(0.6, 1.4)
    a, b = M.consistent_params("BS", rate * mu, rate ** 2 * var, 1.0)
    x = np.r_[0.0, np.cumsum(M.BS.sample(a, b, size=32, rng=rng))]
    paths.append((np.arange(33.0), x))

ref = M.fit_reference(paths, dt_ref=1.0)
print(f"4a. reference law: mean step {ref['mu']:.4f}, squared CV {ref['cv2']:.4f}"
      f" (true {var / mu**2:.4f})")

b = M.variance_growth_exponent(paths, n_boot=500, rng=2)
print(f"4b. variance growth exponent b = {b['b']:.2f}"
      f"  [{b['b_lo']:.2f}, {b['b_hi']:.2f}]   (consistent: 1, interval scaling: 2)")
# For decelerating paths use a clock, e.g. clock=lambda t: t**0.8;
# for readings rounded to a resolution r, pass resolution=r.

res = M.schedule_change_test(paths, h_ref=4.0, h_dep=1.0, horizon=32.0,
                             origins=[8.0, 16.0, 24.0], n_sim=5000, n_boot=500, rng=3)
print("4c. records every 4 h, monitoring every 1 h: coverage of 90 % intervals")
for m, r in res.items():
    print(f"    {m:10s} {r['coverage']:.0%}  [{r['ci'][0]:.0%}, {r['ci'][1]:.0%}]"
          f"  mean width {r['width']:.3f}  ({r['n_forecasts']} forecasts)")
