"""Tests of the mcdeg package against the statements of the paper."""
import numpy as np
import pytest
from scipy import integrate

import mcdeg as M

FAMS = [M.BS, M.WEIBULL, M.GAMMA, M.IG, M.LOGNORMAL]


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
@pytest.mark.parametrize("s", [0.25, 1.0, 4.0, 20.0])
def test_consistent_moments_are_additive(fam, s):
    """Proposition 2: a step of size s has mean s*m and variance s*v."""
    m, v = 2.0, 0.6
    a, b = M.consistent_params(fam, m, v, s)
    mean, var = fam.moments(a, b)
    assert np.isclose(mean, s * m, rtol=1e-8)
    assert np.isclose(var, s * v, rtol=1e-6)


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_interval_scaling_shrinks_variance(fam):
    """Proposition 1: under interval scaling a step of size s has variance s^2 v."""
    a, b = M.interval_params(fam, 2.0, 0.6, 0.25)
    assert np.isclose(fam.moments(a, b)[1], 0.25 ** 2 * 0.6, rtol=1e-6)


@pytest.mark.parametrize("fam", FAMS, ids=lambda f: f.name)
def test_shape_inverts_g(fam):
    r = np.array([0.01, 0.1, 0.5, 2.0, 4.5])
    assert np.allclose(fam.g(fam.shape_from_cv2(r)), r, rtol=1e-7)


def test_bs_finest_grid():
    """Theorem 1 / Remark 2: BS has no root at squared CV >= 5."""
    with pytest.raises(ValueError):
        M.BS.shape_from_cv2(5.5, clip=False)
    assert M.BS.cv2_max == 5.0


def test_gamma_recovers_the_gamma_process():
    """Remark 1: consistent gamma steps index the shape by the interval."""
    k, c = M.GAMMA.from_moments(2.0, 0.5)
    ks, cs = M.consistent_params(M.GAMMA, 2.0, 0.5, 3.0)
    assert np.isclose(ks, 3 * k) and np.isclose(cs, c)


@pytest.mark.parametrize("fam", [M.BS, M.WEIBULL], ids=lambda f: f.name)
def test_accumulated_variance_is_grid_free(fam):
    """The sum over a horizon keeps its variance on every grid; interval scaling
    divides it by the number of steps."""
    rng = np.random.default_rng(1)
    m, v, T = 1.0, 0.16, 8.0
    out = {}
    for n in (1, 4, 16):
        s = T / n
        for cons in (True, False):
            f = M.consistent_params if cons else M.interval_params
            a, b = f(fam, m, v, s)
            X = fam.sample(a, b, size=(40000, n), rng=rng).sum(axis=1)
            out[(n, cons)] = X.var()
    for n in (4, 16):
        assert abs(out[(n, True)] / out[(1, True)] - 1) < 0.05
        assert abs(out[(n, False)] / out[(1, False)] - 1 / n) < 0.05


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 1.5])
def test_psi_is_the_lamperti_integral(gamma):
    x_ref, x = 0.8, 2.3
    num = integrate.quad(lambda u: (1 + u / x_ref) ** -gamma, 0, x)[0]
    assert np.isclose(M.psi(x, gamma, x_ref), num, rtol=1e-10)


def test_homogeneous_ig_parameters():
    mean, lam, n_d = M.rul_ig_params(1.0, 11.0, 0.5, 0.04, gamma=0.0, dt_ref=2.0)
    assert np.isclose(mean, 10 / 0.5 * 2.0)
    assert np.isclose(lam, 10 ** 2 * 2.0 / 0.04)
    assert np.isclose(n_d, 20.0)


@pytest.mark.parametrize("gamma", [0.0, 1.0])
def test_ig_approximation_matches_simulation(gamma):
    """Propositions 7-8: IG mean and s.d. against simulated first passages."""
    fam, x_ref = M.BS, 1.0
    a, b = 0.3, 0.02
    mu, var = M.BS.moments(a, b)
    mean, lam, n_d = M.rul_ig_params(0.0, 1.5, mu, var, gamma, x_ref)
    tau = M.simulate_first_passage(fam, 0.0, 1.5, mu, var, gamma=gamma, x_ref=x_ref,
                                   n_paths=20000, rng=3)
    assert n_d > 20
    assert abs(tau.mean() / mean - 1) < 0.01
    assert abs(tau.std() / np.sqrt(mean ** 3 / lam) - 1) < 0.05


def test_predict_rul_switches_to_simulation_near_threshold():
    mu, var = M.BS.moments(0.3, 1.0)
    far = M.predict_rul("BS", 0.0, 50.0, mu, var)
    near = M.predict_rul("BS", 47.0, 50.0, mu, var, rng=1)
    assert far.method == "IG" and near.method == "simulation"
    assert far.replace_at < far.mean


def _simulate_paths(n_units=20, n_steps=32, h=1.0, a=0.3, rng=0):
    """Consistent BS paths with unit-specific rates."""
    rng = np.random.default_rng(rng)
    paths = []
    for _ in range(n_units):
        rate = rng.uniform(0.5, 1.5)
        mu, var = M.BS.moments(a, 1.0)
        aa, bb = M.consistent_params(M.BS, rate * mu, (rate * mu) ** 2 * var / mu ** 2, h)
        x = np.r_[0, np.cumsum(M.BS.sample(aa, bb, size=n_steps, rng=rng))]
        paths.append((np.arange(n_steps + 1) * h, x))
    return paths


def test_fit_reference_recovers_cv2():
    paths = _simulate_paths(n_units=60, rng=4)
    ref = M.fit_reference(paths, dt_ref=1.0)
    assert abs(ref["cv2"] / M.BS.g(0.3) - 1) < 0.15


def test_variance_growth_exponent_is_one_for_consistent_data():
    res = M.variance_growth_exponent(_simulate_paths(n_units=40, rng=5), n_boot=200, rng=0)
    assert res["b_lo"] < 1.0 < res["b_hi"]
    assert res["b_hi"] < 2.0


def test_schedule_change_test_flags_interval_scaling():
    paths = _simulate_paths(n_units=20, n_steps=32, rng=6)
    res = M.schedule_change_test(paths, h_ref=4.0, h_dep=1.0, horizon=32.0,
                                 origins=[8.0, 16.0, 24.0], n_sim=4000, n_boot=200, rng=0)
    assert res["consistent"]["coverage"] > 0.75
    assert res["interval"]["coverage"] < res["consistent"]["coverage"]
    assert res["interval"]["width"] < res["consistent"]["width"]


def test_bs_infeasible_step_raises():
    """No BS step matches both moments when the squared CV reaches 5."""
    with pytest.raises(ValueError):
        M.consistent_params(M.BS, 1.0, 0.5, s=0.05)      # r = 0.5 / 0.05 = 10


def test_predict_rul_simulates_when_steps_are_coarse():
    mu, var = M.BS.moments(0.3, 0.2)                    # mu / x_ref = 0.2 > 0.1
    p = M.predict_rul("BS", 0.0, 20.0, mu, var, gamma=1.0, x_ref=1.0, rng=1)
    assert p.method == "simulation"
