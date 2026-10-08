"""Kalshi against Polymarket on the same question: "Bitcoin above K at noon Eastern".

Polymarket lists a daily ladder (strikes $2,000 apart) that resolves on the close of the Binance BTC/USDT one-minute
candle labelled 12:00 ET. Kalshi's hourly ladder that closes at 12:00 ET resolves on the average of the sixty
one-second prints of the CF Benchmarks index before 12:00. Same strike, same hour, two different fixings.

    python -m kfv.cross_venue download      Polymarket events + minute prices, Binance noon candles -> data/
    python -m kfv.cross_venue               the comparison -> results/cross_venue.json
"""
import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from . import api
from .download import END, START
from scipy.special import ndtr

from .model import fit_ladder, taker_fee, variance_time
from .study import HORIZONS, RESULTS, cluster_mean, load_markets

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
BINANCE = "https://data-api.binance.vision"


def noon_utc(day):
    """Unix time of 12:00 US Eastern on `day`; the whole study window is in daylight time (UTC-4)."""
    return int(dt.datetime(day.year, day.month, day.day, 16, tzinfo=dt.timezone.utc).timestamp())


def download():
    rows, hist, fix = [], [], []
    day = dt.datetime.utcfromtimestamp(START).date()
    while noon_utc(day) <= END:
        noon = noon_utc(day)
        slug = f"bitcoin-above-on-{day.strftime('%B').lower()}-{day.day}-{day.year}"
        ev = api.get(f"{GAMMA}/events", {"slug": slug}, pause=0.15)
        k = api.get(f"{BINANCE}/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1m", "startTime": (noon - 3600) * 1000,
                                               "limit": 62}, pause=0.15)
        fix.append(dict(noon=noon, **{f"close_{int(c[0]) // 1000 - noon}": float(c[4]) for c in k}))
        if ev and ev[0]["endDate"].startswith(day.isoformat()):
            for m in ev[0]["markets"]:
                strike = float(m["groupItemTitle"].replace(",", ""))
                token = json.loads(m["clobTokenIds"])[0]
                prices = json.loads(m["outcomePrices"]) if m.get("outcomePrices") else [None]
                rows.append(dict(noon=noon, strike=strike, token=token, volume=float(m.get("volume") or 0),
                                 pm_yes=None if prices[0] is None else float(prices[0]) > 0.5))
        day += dt.timedelta(days=1)
    pm = pd.DataFrame(rows)
    spot = pd.read_parquet(api.DATA / "spot_1m.parquet").set_index("ts").close
    pm["spot_1h"] = spot.reindex(pm.noon - 3600 - 60).values
    near = pm[(pm.strike - pm.spot_1h).abs() <= 1500]                 # the strikes Kalshi's hourly ladder also quotes
    for r in near.itertuples():
        h = api.get(f"{CLOB}/prices-history", {"market": r.token, "startTs": r.noon - 3600, "endTs": r.noon, "fidelity": 1},
                    pause=0.15).get("history", [])
        hist += [(r.noon, r.strike, x["t"], x["p"]) for x in h]
    pm.to_parquet(api.DATA / "polymarket_markets.parquet")
    pd.DataFrame(hist, columns=["noon", "strike", "t", "p"]).to_parquet(api.DATA / "polymarket_prices.parquet")
    pd.DataFrame(fix).to_parquet(api.DATA / "binance_noon.parquet")
    print(len(pm), "Polymarket strikes,", len(near), "near the money,", len(hist), "price points")


def main():
    pm = pd.read_parquet(api.DATA / "polymarket_markets.parquet")
    px = pd.read_parquet(api.DATA / "polymarket_prices.parquet")
    fix = pd.read_parquet(api.DATA / "binance_noon.parquet")
    mk = load_markets()
    mk = mk[mk.close_ts.isin(pm.noon.unique()) & (mk.close_ts - mk.open_ts == 3600)]      # Kalshi's 12:00 ET hourly events
    c = pd.concat(pd.read_parquet(f) for f in sorted((api.DATA / "candles").glob("*.parquet")))
    c = c.merge(mk[["ticker", "strike", "close_ts", "y", "settle"]], on="ticker")
    c["k"] = (c.strike + 0.01).round(2)                                                    # 83,999.99 on Kalshi is 84,000

    # -- same strike, same minute: Polymarket price against the Kalshi two-sided mid
    pairs = []
    binance = fix.set_index("noon")
    for h in HORIZONS:
        t = c[(c.close_ts - c.ts == 60 * h) & (c.bid >= 0.01) & (c.ask <= 0.99) & (c.ask > c.bid)]
        for noon, ladder in t.groupby("close_ts"):
            fit = fit_ladder(ladder.strike.values, ((ladder.bid + ladder.ask) / 2).values)
            for r in ladder.itertuples():
                p = px[(px.noon == noon) & (px.strike == r.k) & (px.t <= r.ts) & (px.t > r.ts - 120)]
                if not len(p):
                    continue
                row = dict(noon=noon, strike=r.k, h=h, kalshi_bid=r.bid, kalshi_ask=r.ask, kalshi=(r.bid + r.ask) / 2,
                           poly=float(p.sort_values("t").p.iloc[-1]), y=r.y)
                if fit is not None:
                    # the same contract on Polymarket's fixing: Binance spot instead of the implied index level, and the
                    # variance of a single print at 12:01:00 instead of the average of the minute before 12:00:00
                    level, v, _, _ = fit
                    tau = 60.0 * h
                    v_poly = v * (tau + 60) / variance_time(tau)
                    b = binance.loc[noon, f"close_{-60 * h - 60}"]             # last Binance trade before the snapshot
                    row.update(level=level, binance=b, repriced=float(ndtr((np.log(b / r.k) - v_poly / 2) / np.sqrt(v_poly))),
                               kalshi_line=float(ndtr((np.log(level / r.k) - v / 2) / np.sqrt(v))))
                pairs.append(row)
    pairs = pd.DataFrame(pairs)
    pairs["gap"] = pairs.poly - pairs.kalshi
    pairs["resid"] = pairs.poly - (pairs.kalshi + pairs.repriced - pairs.kalshi_line)   # gap left after the fixing adjustment
    # cost for a taker to cross to the Polymarket price on Kalshi alone: half-spread plus the fee
    pairs["kalshi_cost"] = (pairs.kalshi_ask - pairs.kalshi_bid) / 2 + taker_fee(pairs.kalshi)
    live = pairs[(pairs.kalshi >= 0.05) & (pairs.kalshi <= 0.95)]
    mean_gap, se_gap, days = cluster_mean(live.gap, live.noon)
    out = dict(pairs=int(len(pairs)), live_pairs=int(len(live)), days=int(live.noon.nunique()),
               mean_gap_c=100 * mean_gap, se_gap_c=100 * se_gap,
               mean_abs_gap_c=float(100 * live.gap.abs().mean()), median_abs_gap_c=float(100 * live.gap.abs().median()),
               p90_abs_gap_c=float(100 * live.gap.abs().quantile(0.9)),
               mean_kalshi_cost_c=float(100 * live.kalshi_cost.mean()),
               share_gap_over_kalshi_cost=float((live.gap.abs() > live.kalshi_cost).mean()),
               by_horizon={str(h): dict(pairs=int(len(d)), mean_abs_gap_c=float(100 * d.gap.abs().mean()))
                           for h, d in live.groupby("h")})

    adj = live.dropna(subset=["resid"])
    m_res, se_res, _ = cluster_mean(adj.resid, adj.noon)
    m_raw, se_raw, _ = cluster_mean(adj.gap, adj.noon)
    out["repriced"] = dict(pairs=int(len(adj)), days=int(adj.noon.nunique()),
                           binance_minus_implied_level_median=float((adj.binance - adj.level).median()),
                           mean_gap_c=100 * m_raw, se_gap_c=100 * se_raw, mean_abs_gap_c=float(100 * adj.gap.abs().mean()),
                           mean_resid_c=100 * m_res, se_resid_c=100 * se_res, mean_abs_resid_c=float(100 * adj.resid.abs().mean()),
                           median_abs_resid_c=float(100 * adj.resid.abs().median()),
                           share_resid_over_kalshi_cost=float((adj.resid.abs() > adj.kalshi_cost).mean()))
    pairs.to_csv(RESULTS / "cross_venue_pairs.csv", index=False)

    # -- the two fixings: Kalshi's index average before 12:00:00 against Binance's close of the 12:00 candle
    settle = mk.drop_duplicates("close_ts").set_index("close_ts").settle
    f = fix.assign(kalshi=fix.noon.map(settle)).dropna(subset=["kalshi", "close_0"])
    d = f.close_0 - f.kalshi                           # close_0 = last Binance trade before 12:01:00
    d_same = f["close_-60"] - f.kalshi                 # last Binance trade before 12:00:00, the end of Kalshi's window
    out["fixings"] = dict(days=int(len(f)), median_diff=float(d.median()), median_abs_diff=float(d.abs().median()),
                          p90_abs_diff=float(d.abs().quantile(0.9)), max_abs_diff=float(d.abs().max()),
                          median_abs_diff_same_minute=float(d_same.abs().median()),
                          median_diff_same_minute=float(d_same.median()))
    # how often would one strike pay on one venue and not the other, for every $100 Kalshi strike that day
    k_all = mk.merge(f[["noon", "close_0"]], left_on="close_ts", right_on="noon")
    split = (k_all.settle > k_all.strike) != (k_all.close_0 > k_all.strike + 0.01)
    by_day = split.groupby(k_all.close_ts).any()
    out["fixings"].update(days_with_a_split_100_strike=int(by_day.sum()), strikes_split=int(split.sum()))
    # the listed Polymarket strikes that had a matching Kalshi contract
    both = pm.merge(mk.assign(k=(mk.strike + 0.01).round(2))[["close_ts", "k", "y"]], left_on=["noon", "strike"],
                    right_on=["close_ts", "k"]).dropna(subset=["pm_yes"])
    diff = both[both.pm_yes.astype(int) != both.y].merge(f[["noon", "close_0", "kalshi"]], on="noon")
    out["listed"] = dict(matched_strikes=int(len(both)), resolved_differently=int(len(diff)),
                         cases=[dict(date=str(pd.Timestamp(r.noon, unit="s").date()), strike=r.strike, polymarket_yes=bool(r.pm_yes),
                                     kalshi_yes=bool(r.y), binance_close=r.close_0, kalshi_settlement=r.kalshi)
                                for r in diff.itertuples()])
    (RESULTS / "cross_venue.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    download() if sys.argv[1:] == ["download"] else main()
