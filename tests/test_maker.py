import numpy as np
from scipy.special import ndtr

from kfv import maker
from kfv.account import centicents, message

NOW = 1_800_000_000
CLOSE = NOW + 30 * 60


def ladder(level=83_180.0, sd=0.0025, spread=0.01, close=CLOSE):
    rows = []
    for k in np.arange(82_000, 84_500, 100) - 0.01:
        mid = float(ndtr(np.log(level / k) / sd))
        rows.append(dict(ticker=f"KXBTCD-X-T{k:.2f}", event="KXBTCD-X", strike=float(k), close_ts=close,
                         bid=round(mid - spread / 2, 2), ask=round(mid - spread / 2, 2) + spread))
    return rows


def test_plan_buys_the_favourite_on_each_side_as_a_maker():
    orders, note = maker.plan(ladder(), NOW)
    assert [o["buys"] for o in orders] == ["yes", "no"] and "KXBTCD-X" in note
    yes, no = orders
    assert yes["strike"] < 83_180 < no["strike"]                      # Yes below the implied level, No above it
    for o in orders:
        p = o["payload"]
        assert maker.FAVOURITE[0] <= o["paying"] <= maker.FAVOURITE[1]
        assert p["post_only"] is True and p["time_in_force"] == "good_till_canceled"
        assert p["expiration_time"] == CLOSE - maker.CANCEL_BEFORE_CLOSE
        assert p["client_order_id"].startswith(maker.TAG) and p["count"] == "5.00"
    row = {r["ticker"]: r for r in ladder()}
    # a Yes buy rests on the Yes bid; a No buy is a Yes sell resting on the Yes ask, and pays one minus that
    assert yes["payload"]["side"] == "bid" and abs(float(yes["payload"]["price"]) - row[yes["payload"]["ticker"]]["bid"]) < 1e-9
    assert no["payload"]["side"] == "ask" and abs(float(no["payload"]["price"]) - row[no["payload"]["ticker"]]["ask"]) < 1e-9
    assert abs(no["paying"] - (1 - row[no["payload"]["ticker"]]["ask"])) < 1e-12


def test_plan_respects_the_hard_limits():
    orders, _ = maker.plan(ladder(), NOW, contracts=10_000, budget=10_000)
    assert all(float(o["payload"]["count"]) <= maker.MAX_CONTRACTS for o in orders)
    assert sum(o["cost"] for o in orders) <= maker.MAX_COST + 1e-9
    small, _ = maker.plan(ladder(), NOW, contracts=5, budget=3.0)
    assert sum(o["cost"] for o in small) <= 3.0 and all(float(o["payload"]["count"]) >= 1 for o in small)
    assert maker.plan(ladder(), NOW, budget=0.5)[0] == []


def test_plan_only_quotes_inside_its_time_window():
    assert maker.plan(ladder(close=NOW + 5 * 60), NOW)[0] == []          # too close to the close
    assert maker.plan(ladder(close=NOW + 58 * 60), NOW)[0] == []         # the hour has only just opened
    assert maker.plan([], NOW) == ([], "no hourly ladder is open")


def test_quotes_reads_either_pair_of_fields():
    assert maker.quotes({"yes_bid_dollars": "0.4100", "yes_ask_dollars": "0.4300"}) == (0.41, 0.43)
    bid, ask = maker.quotes({"no_bid_dollars": "0.5700", "no_ask_dollars": "0.5900"})
    assert abs(bid - 0.41) < 1e-12 and abs(ask - 0.43) < 1e-12


def test_signed_message_leaves_out_the_query_string():
    assert message(1234567890000, "get", "/portfolio/orders?status=resting") == "1234567890000GET/trade-api/v2/portfolio/orders"


def test_transfer_amount_is_in_hundredths_of_a_cent():
    assert centicents(10) == 100_000 and centicents(0.01) == 100 and centicents("20") == 200_000
