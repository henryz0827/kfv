import numpy as np
from scipy.special import ndtr

from kfv.model import fair_value, fit_ladder, taker_fee, variance_time


def test_variance_time_matches_simulated_average():
    """Simulate one-second diffusion paths and compare the variance of the last-minute mean with the formula."""
    rng = np.random.default_rng(0)
    for tau in (60, 300, 1800):
        steps = rng.standard_normal((40_000, tau))
        path = np.cumsum(steps, axis=1)                    # unit variance per second, prints at 1..tau
        avg = path[:, -60:].mean(axis=1)
        assert abs(avg.var() / variance_time(tau) - 1) < 0.03
    assert abs(variance_time(300) - (300 - 60 + 60 * 61 * 121 / (6 * 3600))) < 1e-12
    assert abs(variance_time(3600) - (3600 - 39.4972)) < 1e-3


def test_variance_time_inside_window():
    rng = np.random.default_rng(1)
    path = np.cumsum(rng.standard_normal((40_000, 20)), axis=1)   # 20 prints left, 40 already known
    assert abs((path.sum(axis=1) / 60).var() / variance_time(20) - 1) < 0.03
    assert variance_time(0) == 0


def test_fair_value_against_simulation():
    rng = np.random.default_rng(2)
    spot, strike, var, tau = 83_000.0, 83_100.0, (0.45 ** 2) / (365 * 86400), 900
    z = rng.standard_normal((200_000, 1)) * np.sqrt(var * (tau - 60))
    inc = rng.standard_normal((200_000, 60)) * np.sqrt(var)
    logs = np.log(spot) - var * tau / 2 + z + np.cumsum(inc, axis=1)
    settle = np.exp(logs).mean(axis=1)
    assert abs((settle > strike).mean() - fair_value(spot, strike, var, tau)) < 0.004
    # ignoring the averaging overstates the remaining variance, which pulls the price toward a half
    naive = ndtr(np.log(spot / strike) / np.sqrt(var * tau))
    assert abs(fair_value(spot, strike, var, tau) - 0.5) > abs(naive - 0.5)


def test_fit_ladder_recovers_level_and_variance():
    level, v = 83_217.0, 0.0031 ** 2
    strikes = np.arange(82_000, 84_500, 100) - 0.01
    mids = ndtr((np.log(level / strikes) - v / 2) / np.sqrt(v))
    f, v_hat, n, rmse = fit_ladder(strikes, mids)
    assert abs(f - level) < 0.01 and abs(v_hat / v - 1) < 1e-6 and n >= 3 and rmse < 1e-6
    assert fit_ladder(strikes[:2], mids[:2]) is None


def test_fee_schedule():
    assert abs(taker_fee(0.5) - 0.0175) < 1e-12                      # 1.75c a contract at 50c, the maximum
    assert taker_fee(0.5, 100, round_up=True) == 1.75
    assert taker_fee(0.05, 100, round_up=True) == 0.34               # 0.3325 rounds up
    assert taker_fee(0.99, 1, round_up=True) == 0.01
