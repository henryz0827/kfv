"""The four tests of PLAN.md D1.

    python -m kfv.study ledger        T1 ledger of trade prints, full sample
    python -m kfv.study implied       T2 implied against realized, full sample
    python -m kfv.study dev           T3 and T4 on the development window, plus the volatility sensitivity table
    python -m kfv.study final         T3 and T4 on the final window (run once, after the code is frozen)

Each command writes results/<name>.json; kfv.report turns the four files into results/RESULTS.md.
"""
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import ndtri

from . import api
from .model import fair_value, fit_ladder, taker_fee, variance_time

HORIZONS = (45, 30, 15, 5)                       # minutes before the close
SPLIT = api.ts("2026-09-17T00:00:00Z")           # development before, final from here on
VOL_MINUTES = 1440                               # trailing window of the model's volatility
MARGIN = 0.02                                    # edge a taker order must clear on top of its fee
LOT = 100                                        # contracts per backtest order
RESULTS = api.DATA.parent / "results"


# ---------------------------------------------------------------- data
def load_markets():
    m = pd.read_parquet(api.DATA / "markets.parquet")
    m = m[(m.volume > 0) & (m.strike_type == "greater") & m.settle.notna()].copy()
    m["y"] = (m.result == "yes").astype(int)
    return m


def load_spot():
    """Minute grid of spot: value at second t is the last Coinbase trade before t."""
    s = pd.read_parquet(api.DATA / "spot_1m.parquet")
    grid = np.arange(s.ts.min(), s.ts.max() + 60, 60)
    close = pd.Series(s.close.values, index=s.ts.values).reindex(grid).ffill()
    px = pd.Series(close.values, index=grid + 60)                 # known at the end of its minute
    r2 = np.log(px).diff().fillna(0.0) ** 2
    return px, r2.cumsum()


def snapshots(markets, px, cum_r2, vol_minutes=VOL_MINUTES):
    """One row per (strike, horizon) with a two-sided quote and a mid in [2c, 98c]."""
    files = sorted((api.DATA / "candles").glob("*.parquet"))
    c = pd.concat(pd.read_parquet(f) for f in files)
    c = c.merge(markets[["ticker", "event", "strike", "close_ts", "y", "settle"]], on="ticker")
    c["h"] = (c.close_ts - c.ts) // 60
    c = c[c.h.isin(HORIZONS) & (c.bid >= 0.01) & (c.ask <= 0.99) & (c.ask > c.bid)].copy()
    c["mid"] = (c.bid + c.ask) / 2
    c = c[(c.mid >= 0.02) & (c.mid <= 0.98)].copy()
    c["spot"] = px.reindex(c.ts).values
    c["var_sec"] = ((cum_r2.reindex(c.ts).values - cum_r2.reindex(c.ts - 60 * vol_minutes).values)
                    / (60.0 * vol_minutes))
    c["rv_fwd"] = cum_r2.reindex(c.close_ts).values - cum_r2.reindex(c.ts).values     # snapshot to close
    c["fv"] = fair_value(c.spot, c.strike, c.var_sec, c.h * 60.0)
    return c.dropna(subset=["spot", "var_sec", "fv"]).reset_index(drop=True)


def cluster_mean(x, groups, w=None):
    """Mean of x (weighted by w) and its standard error with whole groups as the sampling unit."""
    x, w = np.asarray(x, float), np.ones(len(x)) if w is None else np.asarray(w, float)
    r = (x * w).sum() / w.sum()
    resid = pd.Series((x - r) * w).groupby(np.asarray(groups)).sum()
    g = len(resid)
    se = np.sqrt((resid ** 2).sum() * g / (g - 1)) / w.sum()
    return float(r), float(se), int(g)


# ---------------------------------------------------------------- T1 ledger
def ledger(markets):
    files = sorted((api.DATA / "trades").glob("*.parquet"))
    t = pd.concat(pd.read_parquet(f) for f in files)
    t = t.merge(markets[["ticker", "event", "close_ts", "y"]], on="ticker")
    yes = (t.taker_side == "yes").values
    t["paid"] = np.where(yes, t.yes_price, 1 - t.yes_price)          # price of the side the taker bought
    t["payoff"] = np.where(yes, t.y, 1 - t.y)
    t["gross"] = (t.payoff - t.paid) * t["count"]                    # dollars, before fees
    t["fee"] = taker_fee(t.paid, t["count"])                         # lower bound: no per-order rounding
    t["premium"] = t.paid * t["count"]
    t["mins"] = (t.close_ts - t.time.astype("int64") // 10 ** 9) / 60.0

    def row(d):
        g, se, n_ev = cluster_mean(d.gross / d["count"], d.event, d["count"])
        return dict(trades=int(len(d)), contracts=float(d["count"].sum()), premium=float(d.premium.sum()),
                    gross=float(d.gross.sum()), fees=float(d.fee.sum()),
                    gross_per_contract_c=100 * g, se_c=100 * se, fee_per_contract_c=100 * float(d.fee.sum() / d["count"].sum()),
                    events=n_ev)

    out = dict(all=row(t), sides={k: row(d) for k, d in t.groupby("taker_side")},
               field_combos={"/".join(map(str, k)): int(v) for k, v in
                             t.groupby(["taker_side", "book_side", "outcome_side"]).size().items()})
    price_bins = [0, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 1.0]
    t["price_bin"] = pd.cut(t.paid, price_bins, labels=["1-5c", "6-15c", "16-35c", "36-65c", "66-85c", "86-95c", "96-99c"])
    out["by_price"] = {str(k): row(d) for k, d in t.groupby("price_bin", observed=True)}
    time_bins = [-1, 2, 10, 30, 60, 1e9]
    t["time_bin"] = pd.cut(t.mins, time_bins, labels=["last 2 min", "2-10 min", "10-30 min", "30-60 min", "over 60 min"])
    out["by_time"] = {str(k): row(d) for k, d in t.groupby("time_bin", observed=True)}
    days = (t.close_ts.max() - t.close_ts.min()) / 86400
    out["sample"] = dict(events=int(t.event.nunique()), days=float(days),
                         first=str(pd.Timestamp(t.close_ts.min(), unit="s")), last=str(pd.Timestamp(t.close_ts.max(), unit="s")),
                         series_contracts=float(markets.volume.sum()), series_events=int(markets.event.nunique()))
    return out, t


def taker_side_check(t, markets):
    """Do trades labelled taker 'yes' print at the YES ask and 'no' at the YES bid of the minute before?"""
    files = sorted((api.DATA / "candles").glob("*.parquet"))
    c = pd.concat(pd.read_parquet(f) for f in files)[["ticker", "ts", "bid", "ask"]]
    x = t[["ticker", "time", "yes_price", "taker_side", "count"]].copy()
    x["ts"] = (x.time.astype("int64") // 10 ** 9 // 60) * 60          # end stamp of the previous full minute
    x = x.merge(c, on=["ticker", "ts"]).dropna(subset=["bid", "ask"])
    x = x[(x.ask > x.bid)]
    x["above_mid"] = x.yes_price > (x.bid + x.ask) / 2
    x["below_mid"] = x.yes_price < (x.bid + x.ask) / 2
    yes, no = x[x.taker_side == "yes"], x[x.taker_side == "no"]
    return dict(matched_trades=int(len(x)),
                yes_takers_above_prior_mid=float(yes.above_mid.mean()), yes_takers_below_prior_mid=float(yes.below_mid.mean()),
                no_takers_below_prior_mid=float(no.below_mid.mean()), no_takers_above_prior_mid=float(no.above_mid.mean()))


def markouts(t):
    """Spread paid and what the quote did next, per contract, in the taker's favour when positive.

    For a trade in minute m: `spread` is the trade price against the mid at the start of m (up to a minute old),
    `markout` is the mid at the end of minute m+1 (one to two minutes after the trade) against the trade price.
    Settlement profit = markout + (settlement - later mid), and the last term averages to zero when mids are calibrated,
    so the markout measures the same thing as the settlement ledger with far less noise.
    """
    files = sorted((api.DATA / "candles").glob("*.parquet"))
    c = pd.concat(pd.read_parquet(f) for f in files)[["ticker", "ts", "bid", "ask"]]
    c = c[(c.ask > c.bid) & (c.ask - c.bid <= 0.10)]
    c["mid"] = (c.bid + c.ask) / 2
    x = t[["ticker", "event", "time", "yes_price", "taker_side", "count", "paid", "price_bin", "time_bin"]].copy()
    x["ts"] = (x.time.astype("int64") // 10 ** 9 // 60) * 60
    x = x.merge(c[["ticker", "ts", "mid"]].rename(columns={"mid": "pre"}), on=["ticker", "ts"])
    x = x.merge(c[["ticker", "ts", "mid"]].assign(ts=c.ts - 120).rename(columns={"mid": "post"}), on=["ticker", "ts"])
    sign = np.where(x.taker_side == "yes", 1.0, -1.0)
    x["spread"] = sign * (x.yes_price - x.pre)
    x["markout"] = sign * (x.post - x.yes_price)

    def row(d):
        sp, sp_se, _ = cluster_mean(d.spread, d.event, d["count"])
        mo, mo_se, n_ev = cluster_mean(d.markout, d.event, d["count"])
        return dict(trades=int(len(d)), contracts=float(d["count"].sum()), events=n_ev, spread_paid_c=100 * sp, spread_se_c=100 * sp_se,
                    markout_c=100 * mo, markout_se_c=100 * mo_se, fee_c=float(100 * taker_fee(d.paid, d["count"]).sum() / d["count"].sum()))

    return dict(all=row(x), by_price={str(k): row(d) for k, d in x.groupby("price_bin", observed=True)},
                by_time={str(k): row(d) for k, d in x.groupby("time_bin", observed=True)},
                coverage=float(x["count"].sum() / t["count"].sum()))


# ---------------------------------------------------------------- T2 implied against realized
def implied(snap):
    rows = []
    for (ev, h), d in snap.groupby(["event", "h"]):
        fit = fit_ladder(d.strike.values, d.mid.values)
        if fit is None:
            continue
        level, v, n, rmse = fit
        tau = h * 60.0
        rows.append(dict(event=ev, h=h, close_ts=d.close_ts.iloc[0], level=level, v_imp=v, n=n, rmse=rmse,
                         spot=d.spot.iloc[0], settle=d.settle.iloc[0],
                         v_real=d.rv_fwd.iloc[0] * variance_time(tau) / tau,
                         v_model=d.var_sec.iloc[0] * variance_time(tau)))
    return pd.DataFrame(rows)


def implied_vs_realized(imp):
    out = {}
    day = imp.close_ts // 86400
    for h, d in list(imp.groupby("h")) + [("all", imp)]:
        g = day.loc[d.index]
        d = d.assign(v_move=np.log(d.settle / d.level) ** 2)              # squared move from the implied level to settlement
        sums = d.groupby(g)[["v_imp", "v_real", "v_model", "v_move"]].sum()
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(2000):
            s = sums.iloc[rng.integers(0, len(sums), len(sums))].sum()
            boot.append((s.v_imp / s.v_real, s.v_model / s.v_real, s.v_imp / s.v_move))
        boot = np.array(boot)
        z = np.log(d.settle / d.level) / np.sqrt(d.v_imp)                # the hour's move in implied standard deviations
        ann = np.sqrt(365 * 86400 / variance_time(d.h * 60.0))
        out[str(h)] = dict(
            ladders=int(len(d)), strikes_per_fit=float(d.n.mean()), probit_rmse=float(d.rmse.median()),
            implied_over_realized_var=float(d.v_imp.sum() / d.v_real.sum()),
            ci=[float(np.quantile(boot[:, 0], 0.025)), float(np.quantile(boot[:, 0], 0.975))],
            implied_over_move_var=float(d.v_imp.sum() / d.v_move.sum()),
            move_ci=[float(np.quantile(boot[:, 2], 0.025)), float(np.quantile(boot[:, 2], 0.975))],
            z_var=float((z ** 2).mean()), z_kurtosis=float((z ** 4).mean() / (z ** 2).mean() ** 2),
            z_inside_half=float((z.abs() < 0.5).mean()), z_beyond_two=float((z.abs() > 2).mean()),
            z_beyond_three=float((z.abs() > 3).mean()),
            model_over_realized_var=float(d.v_model.sum() / d.v_real.sum()),
            model_ci=[float(np.quantile(boot[:, 1], 0.025)), float(np.quantile(boot[:, 1], 0.975))],
            implied_vol_median=float((np.sqrt(d.v_imp) * ann).median()),
            realized_vol_rms=float(np.sqrt(d.v_real.sum() / (variance_time(d.h * 60.0)).sum() * 365 * 86400)),
            level_minus_spot_median=float((d.level - d.spot).median()),
            level_minus_spot_mad=float((d.level - d.spot).abs().median()),
            level_abs_err_to_settle=float((d.level - d.settle).abs().median()),
            spot_abs_err_to_settle=float((d.spot - d.settle).abs().median()))
    return out


def calibration(snap):
    """Quoted mid against how often the contract paid, by price band."""
    bins = [0.02, 0.05, 0.15, 0.35, 0.65, 0.85, 0.95, 0.98001]
    labels = ["2-5c", "5-15c", "15-35c", "35-65c", "65-85c", "85-95c", "95-98c"]
    band = pd.cut(snap.mid, bins, labels=labels, right=False)
    out = {}
    for k, d in snap.groupby(band, observed=True):
        diff, se, _ = cluster_mean(d.y - d.mid, d.event)
        out[str(k)] = dict(quotes=int(len(d)), mid_c=float(100 * d.mid.mean()), paid_pct=float(100 * d.y.mean()),
                           diff_c=100 * diff, se_c=100 * se)
    return out


# ---------------------------------------------------------------- T3 information, T4 taker rule
def information(snap):
    z_mkt = ndtri(snap.mid.values)
    z_mod = ndtri(np.clip(snap.fv.values, 1e-4, 1 - 1e-4))
    X = sm.add_constant(np.column_stack([z_mkt, z_mod - z_mkt]))
    groups = pd.factorize(snap.event)[0]
    res = sm.Probit(snap.y.values, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    return dict(rows=int(len(snap)), events=int(snap.event.nunique()),
                coef_market=float(res.params[1]), se_market=float(res.bse[1]),
                coef_model_gap=float(res.params[2]), se_model_gap=float(res.bse[2]),
                t_model_gap=float(res.params[2] / res.bse[2]),
                brier_market=float(((snap.mid - snap.y) ** 2).mean()), brier_model=float(((snap.fv - snap.y) ** 2).mean()),
                mean_abs_gap_c=float(100 * (snap.fv - snap.mid).abs().mean()),
                half_spread_c=float(100 * ((snap.ask - snap.bid) / 2).mean()))


def taker_rule(snap):
    no_ask = 1 - snap.bid
    fee_yes = taker_fee(snap.ask, LOT, round_up=True) / LOT
    fee_no = taker_fee(no_ask, LOT, round_up=True) / LOT
    buy_yes = (snap.fv - snap.ask) >= fee_yes + MARGIN
    buy_no = ((1 - snap.fv) - no_ask) >= fee_no + MARGIN
    pnl = np.where(buy_yes, snap.y - snap.ask - fee_yes, np.where(buy_no, (1 - snap.y) - no_ask - fee_no, np.nan))
    gross = np.where(buy_yes, snap.y - snap.ask, np.where(buy_no, (1 - snap.y) - no_ask, np.nan))
    took = buy_yes | buy_no
    if took.sum() < 2:
        return dict(orders=int(took.sum()))
    net, se, n_ev = cluster_mean(pnl[took], snap.event[took])
    g, g_se, _ = cluster_mean(gross[took], snap.event[took])
    return dict(orders=int(took.sum()), of_quotes=int(len(snap)), events=n_ev, buy_yes=int(buy_yes.sum()), buy_no=int(buy_no.sum()),
                net_per_contract_c=100 * net, se_c=100 * se, t=net / se,
                gross_per_contract_c=100 * g, gross_se_c=100 * g_se,
                fee_per_contract_c=100 * float(np.where(buy_yes, fee_yes, fee_no)[took].mean()))


def window_tests(snap):
    out = dict(information=dict(all=information(snap)), taker_rule=dict(all=taker_rule(snap)))
    for h, d in snap.groupby("h"):
        out["information"][str(h)] = information(d)
        out["taker_rule"][str(h)] = taker_rule(d)
    return out


def main(cmd):
    RESULTS.mkdir(exist_ok=True)
    markets = load_markets()
    px, cum_r2 = load_spot()
    if cmd == "ledger":
        out, t = ledger(markets)
        out["taker_side_check"] = taker_side_check(t, markets)
        out["markouts"] = markouts(t)
    elif cmd == "implied":
        snap = snapshots(markets, px, cum_r2)
        out = dict(implied_vs_realized=implied_vs_realized(implied(snap)), calibration=calibration(snap),
                   information=information(snap),
                   snapshots=dict(rows=int(len(snap)), events=int(snap.event.nunique()),
                                  half_spread_c=float(100 * ((snap.ask - snap.bid) / 2).mean())))
    else:
        snap = snapshots(markets, px, cum_r2)
        part = snap[snap.close_ts < SPLIT] if cmd == "dev" else snap[snap.close_ts >= SPLIT]
        out = window_tests(part)
        out["window"] = dict(first=str(pd.Timestamp(part.close_ts.min(), unit="s")), last=str(pd.Timestamp(part.close_ts.max(), unit="s")))
        if cmd == "dev":
            out["vol_sensitivity"] = {}
            for minutes in (60, 360, 1440):
                s = snapshots(markets, px, cum_r2, minutes)
                s = s[s.close_ts < SPLIT]
                out["vol_sensitivity"][str(minutes)] = dict(information=information(s), taker_rule=taker_rule(s))
    (RESULTS / f"{cmd}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
