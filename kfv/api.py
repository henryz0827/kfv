"""Public market-data clients: Kalshi (markets, 1-minute bid/ask candles, trade prints) and Coinbase (1-minute spot).

No account or key is needed for any call here. Everything is cached under data/ so the study is re-runnable offline.
"""
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
COINBASE = "https://api.exchange.coinbase.com"
DATA = Path(__file__).resolve().parent.parent / "data"

_session = requests.Session()
_session.headers.update({"User-Agent": "kfv-research/0.1", "Accept": "application/json"})


def get(url, params=None, pause=0.12, tries=10):
    """GET with a polite pause and backoff on 429 / 5xx."""
    for k in range(tries):
        try:
            r = _session.get(url, params=params, timeout=40)
        except requests.RequestException:
            time.sleep(2.0 * (k + 1))
            continue
        if r.status_code == 200:
            time.sleep(pause)
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(0.6 * (k + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {url} {params}")


def ts(iso):
    return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


# ---------------------------------------------------------------- Kalshi
def markets(series, start_ts, end_ts):
    """Every settled market of `series` closing in [start_ts, end_ts]."""
    rows, cursor = [], None
    while True:
        p = {"series_ticker": series, "status": "settled", "min_close_ts": start_ts, "max_close_ts": end_ts,
             "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = get(f"{KALSHI}/markets", p)
        for m in d["markets"]:
            rows.append(dict(ticker=m["ticker"], event=m["event_ticker"], strike=m.get("floor_strike"),
                             strike_type=m.get("strike_type"), result=m["result"],
                             settle=float(m["expiration_value"]) if m.get("expiration_value") else None,
                             open_ts=ts(m["open_time"]), close_ts=ts(m["close_time"]),
                             volume=float(m.get("volume_fp") or 0)))
        cursor = d.get("cursor")
        if not cursor or not d["markets"]:
            return pd.DataFrame(rows)


def candles(tickers, start_ts, end_ts):
    """1-minute candles for up to 100 markets: closing YES bid / ask and traded volume per minute."""
    rows = []
    for i in range(0, len(tickers), 100):
        d = get(f"{KALSHI}/markets/candlesticks",
                {"market_tickers": ",".join(tickers[i:i + 100]), "start_ts": start_ts, "end_ts": end_ts,
                 "period_interval": 1})
        for m in d.get("markets", []):
            for c in m.get("candlesticks", []):
                bid, ask = c.get("yes_bid", {}).get("close_dollars"), c.get("yes_ask", {}).get("close_dollars")
                rows.append((m["market_ticker"], c["end_period_ts"],
                             float(bid) if bid is not None else float("nan"),
                             float(ask) if ask is not None else float("nan"),
                             float(c.get("volume_fp") or 0)))
    return pd.DataFrame(rows, columns=["ticker", "ts", "bid", "ask", "volume"])


def trades(ticker):
    """Every trade print of one market: time, YES price, size, and which side the taker took."""
    rows, cursor = [], None
    while True:
        p = {"ticker": ticker, "limit": 1000}
        if cursor:
            p["cursor"] = cursor
        d = get(f"{KALSHI}/markets/trades", p)
        for t in d["trades"]:
            rows.append((t["ticker"], t["created_time"], float(t["yes_price_dollars"]), float(t["count_fp"]),
                         t["taker_side"], t.get("taker_book_side"), t.get("taker_outcome_side")))
        cursor = d.get("cursor")
        if not cursor or not d["trades"]:
            break
    df = pd.DataFrame(rows, columns=["ticker", "time", "yes_price", "count", "taker_side", "book_side",
                                     "outcome_side"])
    df["time"] = pd.to_datetime(df["time"], format="ISO8601", utc=True)
    return df


# ---------------------------------------------------------------- Coinbase
def spot_1m(start_ts, end_ts, product="BTC-USD"):
    """1-minute candles [start, end]; `ts` is the minute's start, `close` the last trade in that minute."""
    rows, t = [], start_ts
    while t < end_ts:
        e = min(t + 299 * 60, end_ts)
        d = get(f"{COINBASE}/products/{product}/candles",
                {"granularity": 60, "start": dt.datetime.utcfromtimestamp(t).isoformat() + "Z",
                 "end": dt.datetime.utcfromtimestamp(e).isoformat() + "Z"}, pause=0.12)
        rows += [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in d]
        t = e + 60
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
