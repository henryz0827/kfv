"""Fair value of a Kalshi "Bitcoin above K" contract, the implied level and volatility of a ladder, and fees."""
import math

import numpy as np
from scipy.special import ndtr, ndtri

WINDOW = 60.0   # seconds of index prints averaged into the settlement value
PRINTS = 60     # one print a second


def variance_time(tau, window=WINDOW, n=PRINTS):
    """Seconds of diffusion variance left in the settlement average, seen `tau` seconds before the close.

    The settlement value is the mean of n prints spaced window/n apart. For a driftless diffusion the mean of
    the first m future prints, each h seconds apart, has variance rate * h * m(m+1)(2m+1) / (6 n^2).
    Before the window opens all n prints are ahead and the path first runs tau - window seconds.
    """
    tau = np.asarray(tau, dtype=float)
    h = window / n
    m = np.clip(np.floor(tau / h + 1e-9), 0, n)                       # prints still to come
    inside = h * m * (m + 1) * (2 * m + 1) / (6 * n * n)
    return np.where(tau >= window, tau - window + inside, inside)


def fair_value(spot, strike, var_per_sec, tau):
    """P(settlement average > strike) for a driftless lognormal path; requires tau >= WINDOW."""
    v = np.asarray(var_per_sec, dtype=float) * variance_time(tau)
    return ndtr((np.log(np.asarray(spot, dtype=float) / strike) - v / 2) / np.sqrt(v))


def fit_ladder(strikes, mids, lo=0.05, hi=0.95, min_strikes=3):
    """Least-squares line through the probit of the quoted mids against log strike.

    Returns (implied settlement level F, implied total variance v, strikes used, rmse of the probit residual),
    or None when fewer than `min_strikes` mids fall inside [lo, hi] or the fitted slope has the wrong sign.
    """
    k, p = np.asarray(strikes, dtype=float), np.asarray(mids, dtype=float)
    keep = (p >= lo) & (p <= hi)
    if keep.sum() < min_strikes:
        return None
    x, y = np.log(k[keep]), ndtri(p[keep])
    slope, intercept = np.polyfit(x, y, 1)
    if slope >= 0:
        return None
    s = -1.0 / slope
    resid = y - (intercept + slope * x)
    return math.exp(intercept * s + s * s / 2), s * s, int(keep.sum()), float(np.sqrt(np.mean(resid ** 2)))


def taker_fee(price, contracts=1.0, rate=0.07, round_up=False):
    """Kalshi taker fee in dollars for one order: rate * contracts * p * (1 - p), rounded up to the next cent."""
    fee = rate * np.asarray(contracts, dtype=float) * np.asarray(price, dtype=float) * (1 - np.asarray(price, dtype=float))
    return np.ceil(fee * 100 - 1e-9) / 100 if round_up else fee
