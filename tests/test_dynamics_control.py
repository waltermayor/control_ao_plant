import numpy as np
from conftest import stable_ar2
from scipy.linalg import solve_discrete_are

from aoctrl.control import (
    build_oracle_lqr,
    dare_sda,
    lqr_gain_robust,
    verify_koopman_lqr,
    verify_latent_arx_lqr,
)
from aoctrl.dynamics import fit_ar, identify_ar, predictor_weights, stabilize_ar
from aoctrl.sensors import OpticalGainModel, fit_optical_gain


def simulate_ar(coefs, T, rng):
    p, n = coefs.shape
    X = np.zeros((T, n))
    for t in range(p, T):
        X[t] = sum(coefs[j] * X[t - 1 - j] for j in range(p)) + rng.standard_normal(n)
    return X


def test_fit_ar_recovers_coefficients(rng):
    a = stable_ar2(4, rng)
    X = simulate_ar(a, 20000, rng)
    est = fit_ar(X, 2, len(X))
    assert np.allclose(est, a, atol=0.02)
    m = identify_ar(X, 2)
    assert m.order == 2 and np.allclose(m.innov_var, 1.0, atol=0.1)


def test_stabilize_and_predictor(rng):
    unstable = np.array([[1.2], [-0.1]])
    s = stabilize_ar(unstable, 0.99)
    assert np.max(np.abs(np.roots([1.0, -s[0, 0], -s[1, 0]]))) <= 0.99 + 1e-9
    a = stable_ar2(3, rng)
    assert np.allclose(predictor_weights(a, 1), a)


def test_dare_sda_matches_scipy(rng):
    n, m = 6, 2
    A = rng.standard_normal((n, n)) * 0.3
    B = rng.standard_normal((n, m))
    Q, R = np.eye(n), np.eye(m)
    assert np.allclose(dare_sda(A, B, Q, R)[0], solve_discrete_are(A, B, Q, R), atol=1e-8)
    K, info = lqr_gain_robust(A, B, Q, R)
    assert info["rho_cl"] < 1


def test_oracle_dare_equals_closed_form(rng):
    n = 5
    a = stable_ar2(n, rng)
    W = np.eye(n)
    for D in (1, 2):
        mo = build_oracle_lqr(a, D, 1e-3, W)
        assert mo.method == "DARE" and mo.dare_vs_closed_form < 1e-3
    mo0 = build_oracle_lqr(a, 0, 1e-3, W)
    # K = G / (1 + rho), G = first block row of F^1 = diag(a1), diag(a2)
    assert np.allclose(np.diag(mo0.K[:, :n]), a[0] / (1 + 1e-3))


def test_latent_lqrs_reproduce_oracle_law(rng):
    """Notebook §8.6 / §9.5 checks: physical dynamics in latent formats -> oracle law."""
    n = 4
    a = stable_ar2(n, rng)
    Wc = np.eye(n)
    for res in (verify_latent_arx_lqr(a, [0, 2], 1e-3, Wc), verify_koopman_lqr(a, [0, 2], 1e-3, Wc)):
        for d, r in res.items():
            assert r["rel_error"] < 1e-6, (d, r)


def test_optical_gain_jacobian_and_fit(rng):
    n = 6
    W = np.eye(n) / n
    og = OpticalGainModel(rng.uniform(0.8, 1.2, n), 1.0, 1e-7, W)
    s = rng.standard_normal(n) * 2e-7
    assert og.jacobian_check(s) < 1e-6
    S = rng.standard_normal((3000, n)) * np.logspace(-8, -6.5, 3000)[:, None]
    R = np.sqrt(np.einsum("ti,ij,tj->t", S, W, S))
    PV = np.array([og.h(x) for x in S])
    fit = fit_optical_gain(PV, S, R, qs=(0.5, 1.0, 2.0), rms_grid=np.logspace(-7.7, -6.3, 29))
    assert fit.q == 1.0 and abs(np.log10(fit.rm) + 7) < 0.06 and fit.loss < fit.loss_linear
