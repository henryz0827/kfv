import numpy as np
import pandas as pd

from kfv.study import cluster_mean, taker_rule


def test_cluster_mean_treats_an_event_as_one_draw():
    x = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    mean, se_event, n = cluster_mean(x, [0, 0, 0, 0, 1, 1, 1, 1])
    _, se_trade, _ = cluster_mean(x, np.arange(8))
    assert mean == 0 and n == 2 and se_event > 2 * se_trade


def test_taker_rule_buys_the_cheap_side_and_charges_the_fee():
    snap = pd.DataFrame(dict(event=["a", "b", "c"], bid=[0.40, 0.60, 0.49], ask=[0.41, 0.61, 0.51],
                             fv=[0.50, 0.50, 0.50], y=[1, 0, 1]))
    r = taker_rule(snap)
    assert r["orders"] == 2 and r["buy_yes"] == 1 and r["buy_no"] == 1          # the 49/51 quote is inside fee + margin
    # buy YES at 41c and it pays: 59c less the fee; buy NO at 40c and it pays: 60c less the fee
    fee_yes, fee_no = np.ceil(7 * 0.41 * 0.59 * 100 - 1e-9) / 10000, np.ceil(7 * 0.40 * 0.60 * 100 - 1e-9) / 10000
    assert abs(r["net_per_contract_c"] - 100 * ((0.59 - fee_yes) + (0.60 - fee_no)) / 2) < 1e-9
