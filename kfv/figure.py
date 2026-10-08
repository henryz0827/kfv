"""results/summary.png: one hour's ladder, calibration by price band, and the takers' result by price paid.

    python -m kfv.figure
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

from .model import fit_ladder
from .study import RESULTS, load_markets, load_spot, snapshots

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
EXAMPLE, EXAMPLE_H = "KXBTCD-26OCT0714", 30


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    led = json.loads((RESULTS / "ledger.json").read_text())
    imp = json.loads((RESULTS / "implied.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), facecolor=SURFACE)

    # 1. one ladder, one distribution
    ax = axes[0]
    snap = snapshots(load_markets(), *load_spot())
    d = snap[(snap.event == EXAMPLE) & (snap.h == EXAMPLE_H)].sort_values("strike")
    level, v, n, _ = fit_ladder(d.strike.values, d.mid.values)
    grid = np.linspace(d.strike.min() - 60, d.strike.max() + 60, 200)
    style(ax, f"One hour's ladder is one distribution\n{EXAMPLE_H} min before the 2pm EDT close, 7 Oct 2026")
    ax.plot(grid, 100 * ndtr((np.log(level / grid) - v / 2) / np.sqrt(v)), color=MUTED, linewidth=1.4, zorder=2)
    ax.vlines(d.strike, 100 * d.bid, 100 * d.ask, color=BLUE, linewidth=2, zorder=3)
    ax.scatter(d.strike, 100 * d.mid, s=34, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    ax.annotate(f"line fitted to the quotes\nimplied level ${level:,.0f}", xy=(level, 50), xytext=(level + 130, 66),
                fontsize=8.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xlabel("strike ($)", fontsize=8.5, color=MUTED)
    ax.set_ylabel("price of \"Bitcoin above strike\" (cents)", fontsize=8.5, color=MUTED)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")

    # 2. calibration by price band
    ax = axes[1]
    cal = imp["calibration"]
    x = np.arange(len(cal))
    diff = np.array([r["diff_c"] for r in cal.values()])
    se = np.array([r["se_c"] for r in cal.values()])
    style(ax, "Cheap contracts paid less than quoted, dear ones more\npaid minus quoted mid, cents a contract (bar = 1 s.e.)")
    ax.bar(x, diff, width=0.56, color=BLUE, zorder=3)
    ax.errorbar(x, diff, yerr=se, fmt="none", ecolor=INK, elinewidth=1, capsize=0, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)
    for xi, di, si in zip(x, diff, se):
        ax.text(xi, di + np.sign(di) * (si + 0.14), f"{di:+.1f}", ha="center", va="bottom" if di > 0 else "top", fontsize=8.5, color=INK)
    ax.set_xticks(x, list(cal))
    ax.set_xlabel("quoted mid", fontsize=8.5, color=MUTED)
    ax.set_ylim(min(diff - se) - 0.8, max(diff + se) + 0.8)

    # 3. takers by price paid, before and after fees
    ax = axes[2]
    bp = led["by_price"]
    x = np.arange(len(bp))
    gross = np.array([r["gross_per_contract_c"] for r in bp.values()])
    net = gross - np.array([r["fee_per_contract_c"] for r in bp.values()])
    style(ax, "What takers made, by the price they paid\ncents a contract at settlement")
    se3 = np.array([r["se_c"] for r in bp.values()])
    ax.bar(x - 0.2, gross, width=0.36, color=BLUE, zorder=3, label="before fees (bar = 1 s.e.)")
    ax.bar(x + 0.2, net, width=0.36, color=ORANGE, zorder=3, label="after the taker fee")
    ax.errorbar(x - 0.2, gross, yerr=se3, fmt="none", ecolor=INK, elinewidth=1, capsize=0, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.9, zorder=2)
    for xi, nn in zip(x, net):                                  # label the after-fee bars only; the whiskers carry the rest
        ax.text(xi + 0.2, nn + (0.08 if nn > 0 else -0.08), f"{nn:+.1f}", ha="center", va="bottom" if nn > 0 else "top", fontsize=8, color=INK)
    ax.set_xticks(x, list(bp))
    ax.set_xlabel("price the taker paid", fontsize=8.5, color=MUTED)
    lo, hi = min(net.min(), (gross - se3).min()), max(net.max(), (gross + se3).max())
    ax.set_ylim(lo - 0.12 * (hi - lo) - 0.2, hi + 0.3 * (hi - lo) + 0.2)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left", ncols=2, handlelength=1.1, borderaxespad=0.2)

    fig.tight_layout(w_pad=3)
    fig.savefig(RESULTS / "summary.png", dpi=170, facecolor=SURFACE)
    print("wrote", RESULTS / "summary.png")


if __name__ == "__main__":
    main()
