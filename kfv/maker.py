"""Rest small maker orders on the favourite side of the current hourly Bitcoin ladder.

    python -m kfv.maker quote            show the orders it would place (public data only, no key, nothing sent)
    python -m kfv.maker quote --live     place them
    python -m kfv.maker status           balance, resting orders, positions, fills of the last day
    python -m kfv.maker cancel --live    cancel every resting order this tool placed
    python -m kfv.maker fund --dollars 20 --live     move cash to the exchange shard the Bitcoin markets trade on
    python -m kfv.maker fund --dollars 20 --back --live     and back to the default shard

What it does, and why (results/RESULTS.md): contracts quoted between 85c and 95c paid about 1.7c more than their
mid, takers pay a fee and makers do not, and the fills that hurt a maker come in the last minutes. So the tool
only ever rests a bid, on the side priced 85c to 95c, on at most one strike below spot (buy Yes) and one above
(buy No), and each order cancels itself five minutes before the close.

Limits are constants below; the command line can only make an order smaller.
"""
import argparse
import csv
import json
import math
import time
import uuid

from . import api
from .model import fit_ladder, variance_time

SERIES = "KXBTCD"
MAX_CONTRACTS = 20            # per order
MAX_COST = 25.0               # dollars of premium per run, across its orders
MAX_RESTING = 4               # resting orders of this tool at any time
MINUTES = (12, 50)            # only quote this many minutes before the close
CANCEL_BEFORE_CLOSE = 300     # seconds; orders expire this long before the close
FAVOURITE = (0.85, 0.95)      # price band of the side that is bought
SHARD = 2                     # crypto and commodity markets trade on exchange shard 2; cash sits on shard 0 until moved
MAX_FUND = 50.0               # dollars per `fund` call
TAG = "kfv-"                  # client_order_id prefix, so `cancel` and `status` find this tool's orders
LOG = api.DATA / "live"


def quotes(market):
    """Yes bid and ask in dollars from a market record, whichever pair of fields it carries."""
    def f(key):
        v = market.get(key)
        return float(v) if v not in (None, "") else None
    bid = f("yes_bid_dollars") if f("yes_bid_dollars") is not None else (None if f("no_ask_dollars") is None else 1 - f("no_ask_dollars"))
    ask = f("yes_ask_dollars") if f("yes_ask_dollars") is not None else (None if f("no_bid_dollars") is None else 1 - f("no_bid_dollars"))
    return bid, ask


def current_ladder(now):
    """Markets of the hourly event that closes next, as dicts with strike, yes bid / ask and close time."""
    d = api.get(f"{api.KALSHI}/markets", {"series_ticker": SERIES, "status": "open", "min_close_ts": int(now),
                                          "max_close_ts": int(now) + 2 * 3600, "limit": 1000})
    rows = []
    for m in d["markets"]:
        close, opened = api.ts(m["close_time"]), api.ts(m["open_time"])
        bid, ask = quotes(m)
        if close - opened == 3600 and bid is not None and ask is not None and m.get("strike_type") == "greater":
            rows.append(dict(ticker=m["ticker"], event=m["event_ticker"], strike=float(m["floor_strike"]), bid=bid, ask=ask,
                             close_ts=close))
    if not rows:
        return []
    first = min(r["close_ts"] for r in rows)
    return sorted((r for r in rows if r["close_ts"] == first), key=lambda r: r["strike"])


def plan(ladder, now, contracts=5, budget=10.0, sides=("yes", "no")):
    """Orders to rest, as request payloads plus the facts they were chosen on. Pure: no network, no clock."""
    if not ladder:
        return [], "no hourly ladder is open"
    close = ladder[0]["close_ts"]
    minutes = (close - now) / 60.0
    if not MINUTES[0] <= minutes <= MINUTES[1]:
        return [], f"{minutes:.0f} minutes to the close; this tool quotes between {MINUTES[1]} and {MINUTES[0]} minutes before it"
    live = [r for r in ladder if r["bid"] >= 0.01 and r["ask"] <= 0.99 and r["ask"] > r["bid"]]
    fit = fit_ladder([r["strike"] for r in live], [(r["bid"] + r["ask"]) / 2 for r in live]) if live else None
    note = f"{ladder[0]['event']}, {minutes:.0f} minutes to the close"
    if fit:
        level, v, n, _ = fit
        vol = math.sqrt(v / variance_time(close - now) * 365 * 86400)
        note += f"; ladder implies ${level:,.0f} and {100 * vol:.0f}% annualised volatility from {n} strikes"
    contracts = max(0, min(int(contracts), MAX_CONTRACTS))
    budget = min(float(budget), MAX_COST)
    orders, spent = [], 0.0
    for side in sides:
        # buying Yes rests on the Yes bid; buying No is selling Yes, so it rests on the Yes ask
        priced = [(r, r["bid"] if side == "yes" else 1 - r["ask"]) for r in live]
        band = [(r, p) for r, p in priced if FAVOURITE[0] <= p <= FAVOURITE[1]]
        if not band:
            continue
        r, paying = min(band, key=lambda x: abs(x[1] - 0.90))
        count = min(contracts, int((budget - spent) / paying + 1e-9))
        if count < 1:
            continue
        spent += count * paying
        orders.append(dict(
            payload=dict(ticker=r["ticker"], client_order_id=TAG + str(uuid.uuid4()), side="bid" if side == "yes" else "ask",
                         count=f"{count:.2f}", price=f"{r['bid'] if side == 'yes' else r['ask']:.4f}",
                         expiration_time=int(close - CANCEL_BEFORE_CLOSE), time_in_force="good_till_canceled", post_only=True,
                         self_trade_prevention_type="taker_at_cross", cancel_order_on_pause=True),
            buys=side, strike=r["strike"], paying=paying, cost=count * paying, max_gain=count * (1 - paying),
            mid=(r["bid"] + r["ask"]) / 2, spread=r["ask"] - r["bid"], minutes=minutes))
    return orders, note


def append(name, row):
    LOG.mkdir(parents=True, exist_ok=True)
    path = LOG / name
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_quote(args):
    now = time.time()
    orders, note = plan(current_ladder(now), now, args.contracts, args.budget, args.sides.split(","))
    print(note)
    for o in orders:
        p = o["payload"]
        print(f"  buy {p['count']} {o['buys'].upper()} on ${o['strike']:,.2f} at {100 * o['paying']:.0f}c (rest on the Yes {p['side']} at {p['price']}); "
              f"cost ${o['cost']:.2f}, pays ${o['max_gain']:.2f} more if right; mid {100 * o['mid']:.1f}c, spread {100 * o['spread']:.0f}c")
    if not orders:
        print("  nothing to place")
        return
    if not args.live:
        print("dry run: nothing was sent. Add --live to place these orders.")
        return
    from .account import Account
    acct = Account()
    cash = acct.balance(SHARD)
    resting = [o for o in acct.orders("resting") if str(o.get("client_order_id", "")).startswith(TAG)]
    need = sum(o["cost"] for o in orders)
    if cash < need:
        raise SystemExit(f"${cash:.2f} on exchange shard {SHARD}, where these markets trade, and the orders need ${need:.2f}.\n"
                         f"Move cash there first: python -m kfv.maker fund --dollars 20 --live   (or kalshi.com/account/exchange-indexes)")
    if len(resting) + len(orders) > MAX_RESTING:
        raise SystemExit(f"{len(resting)} orders of this tool are already resting; the limit is {MAX_RESTING}. Run `cancel --live` first.")
    busy = {o["ticker"] for o in resting}
    for o in orders:
        p = o["payload"]
        if p["ticker"] in busy:
            print(f"  skipped {p['ticker']}: an order of this tool is already resting there")
            continue
        try:
            reply = acct.place(p)
        except RuntimeError as err:
            reply = {"error": str(err)}
        print("  sent", p["ticker"], json.dumps(reply))
        append("orders.csv", dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), ticker=p["ticker"], buys=o["buys"],
                                  yes_book_side=p["side"], yes_price=p["price"], paying=round(o["paying"], 4), count=p["count"],
                                  mid=round(o["mid"], 4), spread=round(o["spread"], 4), minutes_to_close=round(o["minutes"], 1),
                                  ladder=note, client_order_id=p["client_order_id"], order_id=reply.get("order_id", ""),
                                  filled=reply.get("fill_count", ""), error=reply.get("error", "")))


def cmd_status(args):
    from .account import Account
    acct = Account()
    print(f"balance ${acct.balance():.2f} in all; ${acct.balance(0):.2f} on the default shard, "
          f"${acct.balance(SHARD):.2f} on shard {SHARD} (crypto), which is what this tool can spend")
    resting = acct.orders("resting")
    print(f"resting orders: {len(resting)}")
    for o in resting:
        mine = "*" if str(o.get("client_order_id", "")).startswith(TAG) else " "
        print(f" {mine} {o.get('ticker')} {o.get('outcome_side')} {o.get('book_side')} yes {o.get('yes_price_dollars')} "
              f"remaining {o.get('remaining_count_fp')} filled {o.get('fill_count_fp')} expires {o.get('expiration_time')}")
    held = acct.positions()
    for key in ("market_positions", "event_positions"):
        open_rows = [p for p in held.get(key, []) if any(str(v) not in ("0", "0.00", "0.0000", "") for k, v in p.items()
                                                          if "position" in k or "exposure" in k)]
        print(f"{key}: {len(open_rows)} open")
        for p in open_rows:
            print("  ", json.dumps(p))
    fills = acct.fills(time.time() - 86400)
    print(f"fills in the last day: {len(fills)}")
    seen = set()
    if (LOG / "fills.csv").exists():
        seen = {r["fill_id"] for r in csv.DictReader(open(LOG / "fills.csv"))}
    for x in fills:
        print(f"   {x.get('created_time')} {x.get('ticker') or x.get('market_ticker')} {x.get('outcome_side')} {x.get('book_side')} "
              f"{x.get('count_fp')} at yes {x.get('yes_price_dollars')} taker={x.get('is_taker')} fee {x.get('fee_cost')}")
        if x.get("fill_id") not in seen:
            append("fills.csv", dict(fill_id=x.get("fill_id"), time=x.get("created_time"), ticker=x.get("ticker") or x.get("market_ticker"),
                                     outcome_side=x.get("outcome_side"), book_side=x.get("book_side"), count=x.get("count_fp"),
                                     yes_price=x.get("yes_price_dollars"), is_taker=x.get("is_taker"), fee=x.get("fee_cost"),
                                     order_id=x.get("order_id")))


def cmd_cancel(args):
    from .account import Account
    acct = Account()
    mine = [o for o in acct.orders("resting") if str(o.get("client_order_id", "")).startswith(TAG)]
    print(f"{len(mine)} resting orders placed by this tool")
    for o in mine:
        if args.live:
            print("  cancelled", o["ticker"], json.dumps(acct.cancel(o["order_id"], o["ticker"])))
        else:
            print("  would cancel", o["ticker"], o["order_id"])
    if mine and not args.live:
        print("dry run: nothing was sent. Add --live to cancel them.")


def cmd_fund(args):
    dollars = min(float(args.dollars), MAX_FUND)
    src, dst = (SHARD, 0) if args.back else (0, SHARD)
    print(f"move ${dollars:.2f} from exchange shard {src} to shard {dst}")
    if dollars <= 0:
        return
    if not args.live:
        print("dry run: nothing was sent. Add --live to move the cash.")
        return
    from .account import Account
    acct = Account()
    have = acct.balance(src)
    if have < dollars:
        raise SystemExit(f"only ${have:.2f} on shard {src}")
    print("  sent", json.dumps(acct.move_cash(dollars, src, dst)))
    print("the transfer is processed in the background; run `status` in a few seconds to see it arrive")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kfv.maker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quote")
    q.add_argument("--live", action="store_true", help="send the orders; without it nothing is sent")
    q.add_argument("--contracts", type=int, default=5, help=f"contracts per order (at most {MAX_CONTRACTS})")
    q.add_argument("--budget", type=float, default=10.0, help=f"dollars of premium for this run (at most {MAX_COST:.0f})")
    q.add_argument("--sides", default="yes,no", help="yes, no or yes,no")
    q.set_defaults(run=cmd_quote)
    sub.add_parser("status").set_defaults(run=cmd_status)
    c = sub.add_parser("cancel")
    c.add_argument("--live", action="store_true")
    c.set_defaults(run=cmd_cancel)
    f = sub.add_parser("fund")
    f.add_argument("--dollars", type=float, required=True, help=f"amount to move (at most {MAX_FUND:.0f} a call)")
    f.add_argument("--back", action="store_true", help="move from the crypto shard back to the default shard")
    f.add_argument("--live", action="store_true")
    f.set_defaults(run=cmd_fund)
    args = ap.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
