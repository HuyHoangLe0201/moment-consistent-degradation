# mcdeg — degradation models that do not depend on the inspection schedule

Companion code of

> H. H. Le and K.-A. Nguyen, *Inspection-Schedule Consistency for Degradation
> Processes with Non-Additive Increments: Birnbaum–Saunders and Weibull Laws*
> (submitted).

It contains

* **`mcdeg`** — a small Python package for using the method on your own data:
  moment-consistent increment laws for five families, the remaining-useful-life
  (RUL) law under damage coupling, Algorithm 1 of the paper, and two
  diagnostics that tell you whether your data need it;
* **`reproduce/`** — the scripts that regenerate every number, table and figure
  of the paper.

## The problem in one paragraph

A degradation model is fitted on records taken on one inspection schedule and
then used on assets inspected on others. If the increment law is not closed
under convolution — Birnbaum–Saunders (BS), Weibull, lognormal — the obvious way
to give it a time argument is to let its scale absorb the interval. The
accumulated mean is then right on every schedule, but the accumulated variance
is proportional to the interval: an asset inspected ten times more often is
credited with a tenth of its degradation variance, and its RUL intervals become
far too narrow. Point accuracy cannot detect this.

The fix is to let both parameters depend on the interval so that the mean
**and the variance** add up over any partition (*moment consistency*). For a
two-parameter scale family this reduces to one equation in the shape,

    g(shape_s) = r = v / (m^2 s),        g = squared coefficient of variation,

which has a solution when `r` lies in the range of `g`, and a unique one when
`g` is one-to-one (Proposition 2). For gamma and inverse Gaussian it returns
their Lévy processes; for BS it has a closed form and a finest admissible grid
(`r < 5`); for Weibull it has a root on every grid. Invariance *in law* is
impossible for non-closed families; on coarser grids the consistent law is a
two-moment approximation to the aggregated increments, whose error the paper
measures.

## Install

```bash
pip install -e .            # the package: numpy and scipy only
pip install -e ".[test]"    # plus pytest
pytest                      # 45 tests against the statements of the paper
```

## Use it on your data

```python
import numpy as np
import mcdeg as M

# paths: one (times, readings) pair per unit
paths = [(t1, x1), (t2, x2), ...]

# 1. Does your degradation need moment consistency?  b ≈ 1 says yes,
#    b ≈ 2 would support interval scaling.
M.variance_growth_exponent(paths)            # {'b': ..., 'b_lo': ..., 'b_hi': ...}
#    decelerating paths: clock=lambda t: t**p;  rounded readings: resolution=0.01

# 2. Reference law: mean and variance of one step of length dt_ref
ref = M.fit_reference(paths, dt_ref=1.0)      # mu, var, cv2, per-unit rates

# 3. The law of a step of any length, any family
shape, scale = M.consistent_params("Weibull", ref["mu"], ref["var"], s=0.25)

# 4. RUL from the current state, on any inspection interval (Algorithm 1),
#    with optional Paris-type damage coupling q(x)^gamma
p = M.predict_rul("BS", x_k=0.4, D=1.0, mu=ref["mu"], var=ref["var"],
                  t_k=120.0, eta=0.05, dt=0.5, gamma=1.5, x_ref=1.0)
p.mean, p.sd, p.replace_at, p.method       # IG closed form or simulation

# 5. Would a model fitted on one schedule hold on another?  Leave-one-unit-out
#    coverage of both parameterisations, with unit-bootstrap intervals.
M.schedule_change_test(paths, h_ref=4.0, h_dep=1.0, family="BS")
```

`examples/quickstart.py` runs all five steps on simulated data. Families:
`"BS"`, `"Weibull"`, `"gamma"`, `"IG"`, `"lognormal"`.

| function | paper |
|---|---|
| `consistent_params`, `interval_params` | Propositions 1–2, Theorem 1, Proposition 6 |
| `psi`, `distances`, `rul_ig_params` | Propositions 7–8 (Lamperti reduction, two distances) |
| `predict_rul` | Algorithm 1 |
| `variance_growth_exponent` | Section 6 |
| `schedule_change_test` | Section 7.2 |

**When to trust the closed form.** The IG approximation of the RUL is used
when about ten or more steps remain and the step scale is small against
`x_ref` (`β/x_ref ≤ 0.1`); `predict_rul` simulates otherwise. Appendix B of the
paper gives explicit error bounds.

**If your increments are correlated** (`b` somewhat above 1), fit on the
coarsest schedule you will predict over: the consistent model is then safe on
every finer one (Remark 6).

## Reproduce the paper

```bash
pip install -e ".[reproduce]"
cd reproduce
python fetch_data.py        # the published data sets (not redistributed here)
python run_all.py           # every result, table and figure (about an hour)
```

`run_all.py` lists which script produces which section, table and figure.
Tables are written to `reproduce/results/tables/` as LaTeX; figures to
`reproduce/figures/out/` as TikZ, with `figures/preview.tex` to view them all.
All simulations are seeded. `results/theory/` ships with the stored simulation
results, so `python run_all.py --quick` rebuilds those tables at once.

### Data

The real-data analyses use five data sets from the R package that accompanies
Meeker & Escobar, *Statistical Methods for Reliability Data* (SMRD,
[github.com/Auburngrads/SMRD](https://github.com/Auburngrads/SMRD)): GaAs lasers,
Device-B, Alloy-A, metal wear and fatigue lives. They are not redistributed;
`fetch_data.py` downloads them from that repository. Results that contain
readings of these data sets are not committed either; they are recomputed.

## Citing

If you use the method or the code, please cite the paper (see
`CITATION.cff`).

## License

MIT, see `LICENSE`.
