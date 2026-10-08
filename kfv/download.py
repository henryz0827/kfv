"""Download the study sample into data/. Re-running skips what is already on disk.

    python -m kfv.download markets | spot | candles | trades

Sample (fixed in PLAN.md before any outcome was looked at):
  series  KXBTCD, the hourly "Bitcoin above K" ladders
  window  events closing 2026-08-08 00:00 UTC .. 2026-10-07 23:00 UTC (the range the live API still serves)
  candles every event, every strike that traded, the last 60 minutes before the close
  trades  every fifth hourly event (hour index mod 5 == 0, so all 24 hours of the day are covered), all strikes
"""
import random
import sys

import pandas as pd

from . import api

SERIES = "KXBTCD"
START = api.ts("2026-08-08T00:00:00Z")
END = api.ts("2026-10-07T23:00:00Z")
D = api.DATA


def do_markets():
    out = D / "markets.parquet"
    if out.exists():
        return
    parts, day = [], 86400
    for s in range(START, END + 1, day):
        e = min(s + day - 1, END)
        df = api.markets(SERIES, s, e)
        parts.append(df)
        print(pd.Timestamp(s, unit="s"), len(df), flush=True)
    pd.concat(parts).drop_duplicates("ticker").to_parquet(out)


def do_spot():
    out = D / "spot_1m.parquet"
    if out.exists():
        return
    api.spot_1m(START - 2 * 86400, END + 600).to_parquet(out)


def traded_by_event():
    m = pd.read_parquet(D / "markets.parquet")
    m = m[(m.volume > 0) & (m.strike_type == "greater")]
    return {ev: g for ev, g in m.groupby("event")}


def do_candles():
    (D / "candles").mkdir(exist_ok=True)
    ev = traded_by_event()
    days = {}
    for e, g in ev.items():
        days.setdefault(pd.Timestamp(g.close_ts.iloc[0], unit="s").strftime("%Y%m%d"), []).append(g)
    for day, groups in sorted(days.items()):
        out = D / "candles" / f"{day}.parquet"
        if out.exists():
            continue
        parts = [api.candles(list(g.ticker), int(g.close_ts.iloc[0]) - 3600, int(g.close_ts.iloc[0])) for g in groups]
        pd.concat(parts).to_parquet(out)
        print("candles", day, sum(len(p) for p in parts), flush=True)


def trade_sample_events():
    ev = traded_by_event()
    return sorted(e for e, g in ev.items() if ((int(g.close_ts.iloc[0]) - START) // 3600) % 5 == 0)


def do_trades():
    (D / "trades").mkdir(exist_ok=True)
    ev = traded_by_event()
    todo = trade_sample_events()
    random.Random(0).shuffle(todo)          # fixed shuffle, so an interrupted download is still spread over the window
    for e in todo:
        out = D / "trades" / f"{e}.parquet"
        if out.exists():
            continue
        parts = [api.trades(t) for t in ev[e].ticker]
        df = pd.concat(parts)
        df.to_parquet(out)
        print("trades", e, len(df), flush=True)


if __name__ == "__main__":
    {"markets": do_markets, "spot": do_spot, "candles": do_candles, "trades": do_trades}[sys.argv[1]]()
