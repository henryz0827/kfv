# Plan and decision log

## D1 (2026-10-08, written before any settlement outcome was analysed)

**Question.** Kalshi lists, every hour, a ladder of binary contracts "Bitcoin above K at the top of the hour"
(series KXBTCD, about 20 traded strikes $100 apart, about a million contracts an hour). What is each contract
worth, who is paid for trading it, and does a fair-value model built only from spot know anything the quoted
market does not?

**Idea.** A ladder of "above K" binaries is a survival function, so the whole ladder is one probability
distribution, and two facts follow with nothing to tune.

1. *The contract is not a digital on spot.* It settles on the average of the 60 one-second prints of the CF
   Benchmarks index in the last minute. For a driftless diffusion with variance rate s2 per second, seen tau
   seconds before the close (tau >= 60), the settlement average has variance
   `s2 * (tau - 60 + 60 * (n+1)(2n+1) / (6 n^2))` with n = 60, which is `s2 * (tau - 39.5 s)`.
   The clock of the contract stops about 40 seconds early. Fair value is
   `Phi((ln(S/K) - v/2) / sqrt(v))` with `v` that variance.
2. *The probit of the ladder is a straight line in log strike.* `Phi^-1(mid_K) = (ln F - v/2 - ln K) / sqrt(v)`,
   so an ordinary least-squares line through the quoted mids gives the market's implied settlement level `F`
   from the intercept and its implied volatility from the slope, using Kalshi quotes alone. Curvature away from
   the line is the market's fat-tail pricing.

**Data.** Kalshi public API (no account): settled markets, 1-minute candles with closing YES bid and ask, trade
prints with the taker's side. Coinbase BTC-USD 1-minute candles for spot. Window: events closing
2026-08-08 00:00 UTC to 2026-10-07 23:00 UTC, which is everything the live endpoints still serve.

**Fixed choices.**
- Snapshots at 45, 30, 15 and 5 minutes before the close. Spot at a snapshot is the last Coinbase trade before it.
- Volatility for the model is the realized variance of the trailing 1,440 one-minute log returns. One estimator,
  chosen here; alternatives are reported only as a sensitivity table on the development window.
- A strike enters a snapshot if it has a two-sided quote and a mid between 2c and 98c. A ladder fit needs at
  least 3 strikes with mid between 5c and 95c.
- Fee: Kalshi's taker schedule for this series, `0.07 * p * (1 - p)` per contract, rounded up to the next cent
  per order. Orders in the backtest are 100 contracts. Makers pay no fee on this series.
- Development window: closes before 2026-09-17 00:00 UTC. Final window: closes from then to 2026-10-07, run once
  after the code is frozen.

**Tests.**
- T1 (description, full sample). Ledger of trade prints on every fifth hourly event: taker profit per contract
  before and after fees, by the price the taker paid and by time to the close. Takers' loss before fees is the
  makers' gain, trade by trade.
- T2 (description, full sample). Implied against realized: implied variance from the ladder fit against the
  realized variance of spot from the snapshot to the close, as a ratio of sums.
- T3 (final window, once). Does the model add information? Logistic regression of the outcome on the market's
  probit and on (model probit - market probit), standard errors clustered by event. The model adds information
  if the second coefficient is positive with t >= 3.
- T4 (final window, once). Taker rule: buy YES at the ask when `FV - ask >= fee + 2c`, buy NO when
  `(1 - FV) - no_ask >= fee + 2c`, at each snapshot, 100 contracts, held to settlement. Passes if net profit per
  contract is positive with t >= 3, clustered by event.

Theory does not predict that T4 passes. A ladder this liquid is quoted by professional market makers who see
the same spot feed; the expected result is that the edge sits with the maker and that the taker fee is larger
than any gap a spot-only model can find. The measurement is the point.

## D2 (2026-10-08, on the development window, before the final window was run)

- The trade download paces at about five requests a second, which is what the public endpoint allows. Sample
  and window are unchanged.
- T3 uses a probit link instead of a logit, so the coefficient on the market's probit reads directly: 1 means
  the quoted prices are calibrated, above 1 means they sit too close to 50c.
- T2 gains a second realized measure, the squared move from the ladder's implied level to the settlement value,
  and the shape of that move in implied standard deviations (variance, kurtosis, share inside half a standard
  deviation, share beyond two and three). Reason: on the development window the implied variance came out below
  the realized variance of one-minute returns while quoted prices were close to calibrated, and those two facts
  are only compatible if the hour's move is more peaked and fatter-tailed than a normal. The new columns test
  that directly.
- T2 also gains a calibration table (quoted mid against how often the contract paid, by price band).
- The model, its volatility window, the snapshots, the fee, the 2c margin and the split date are as in D1.

## D3 (2026-10-08, after the final window was run; descriptive additions only)

- Final window result, run once on the frozen code (hashes and start time in `results/frozen_code_sha256.txt`): T3
  and T4 both fail, as D1 expected. Numbers in `results/RESULTS.md`.
- T1 gains quote-based measures. Settlement profit per contract is exact but noisy, because every taker in an
  hour carries the same coin flip. For each trade the ledger now also records the spread paid against the mid at
  the start of its minute and the markout to the mid one to two minutes later. Settlement profit equals the
  markout plus a term that averages to zero when mids are calibrated, so the two measure the same quantity.
- Cross-venue section (`kfv/cross_venue.py`). Polymarket lists the same question at noon Eastern on a different
  fixing (close of the Binance BTC/USDT one-minute candle labelled 12:00, against Kalshi's average of the sixty
  index prints before 12:00). The section compares the two prices at the same strike and minute, reprices the
  Kalshi quote onto Polymarket's fixing with the ladder's own implied volatility, and counts strikes that
  resolved differently. Nothing here was pre-specified; it is a description of 61 days.
