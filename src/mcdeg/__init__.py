"""mcdeg -- moment-consistent degradation processes with non-additive increments.

Companion code of
  H. H. Le and K.-A. Nguyen, "Inspection-Schedule Consistency for Degradation
  Processes with Non-Additive Increments: Birnbaum--Saunders and Weibull Laws".

Quick map
  families     BS, WEIBULL, GAMMA, IG, LOGNORMAL; consistent_params(), interval_params()
  coupling     psi(), distances(), rul_ig_params(), simulate_first_passage(), predict_rul()
  diagnostics  fit_reference(), variance_growth_exponent()
  validation   schedule_change_test()
"""
from .families import (BS, WEIBULL, GAMMA, IG, LOGNORMAL, FAMILIES, Family,
                       get_family, consistent_params, interval_params, step_moments)
from .coupling import (q, psi, kappa, distances, rul_ig_params, ig_cdf, ig_ppf,
                       step_law, simulate_first_passage, predict_rul, RULPrediction)
from .diagnostics import fit_reference, variance_growth_exponent, thin
from .validation import schedule_change_test

__version__ = "1.0.0"
