"""Stop-path + borrow-cost test for the sector-outrunner SHORT (tools/edge_study.py
fade). Close-to-close results ignored the squeeze tail's path; a live short
carries a resting BUY-STOP. Here: short at t+1 OPEN; each later session, if
the HIGH reaches entry x (1 + stop) the cover fills at max(open, stop price)
(a gap through the stop fills at the open, worse); else cover at t+hold CLOSE.
Two legs: "short" = the short alone (-stock return; an overlay on whatever
else the account holds) and "pair" = short stock + long SPY over the SAME
window (SPY - stock; market-neutral, benchmark = cash). Each minus a round-trip cost that stands
in for spread + slippage + borrow. Signals: edge_study.build(126, 60).

RESULT 2026-11-07 (signals 2024-05..2026-10, z>=3 $25M+ ADV, hold 10):
  no stop, pair   : +2.24 / +1.94 / +1.54 / +0.54 % at 0.3/0.6/1.0/2.0% cost,
                    PASS to 1.0% -- but worst trade -314% (unsurvivable).
  buy-stop +15%   : pair +0.94 (0.3%, 2024 +0.04) PASS only at 0.3% cost;
                    +0.64 at 0.6% FAIL (2024 -0.26); +0.24 at 1.0% FAIL.
  stops +10/20/30%: all FAIL at 0.6%. short-alone (no SPY leg): FAIL in every
                    cell (2024 negative) -- part of the "edge" was SPY drift.
  z>=2 and $100M+ variants: FAIL once any stop or >=0.6% cost applies.
  => The fade's mean is many small wins paid for by a squeeze tail; any stop
     a $2.4k account must carry removes it at realistic short costs. NO-SHIP,
     no shadow (a shadow would track a rule that cannot be executed safely).
     Defined-risk variant (put debit spreads) untested; premium-buying lost in
     the options study, and IV is rich exactly after a 3-sigma run.

Usage: py -3 tools/fade_stop_study.py
"""
import os, sys, pickle, datetime as dt
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pit_harness as ph
import edge_study as es

B = es.build(126, 60)
o, c, cols, z, base, adv = (B[k] for k in ("o", "c", "cols", "z", "base", "adv"))
SIG = base & (z >= 2)
syms = sorted(SIG.columns[SIG.any()])
HP = "fade_highs.pkl"
if os.path.exists(HP):
    hi = pickle.load(open(HP, "rb"))
else:
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    sc = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    fr = []
    for i in range(0, len(syms), 400):
        r = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 400], timeframe=TimeFrame.Day,
                                               start=ph.START, end=dt.datetime(2026, 10, 31),
                                               adjustment=Adjustment.SPLIT, feed=DataFeed.SIP))
        fr.append(r.df.reset_index()[["symbol", "timestamp", "high"]])
        print("highs", i, flush=True)
    df = pd.concat(fr)
    df["date"] = df.timestamp.dt.tz_convert("America/New_York").dt.date
    hi = df.pivot_table(index="date", columns="symbol", values="high")
    pickle.dump(hi, open(HP, "wb"))
hi = hi.reindex(index=c.index, columns=syms)
O, C, H = o[syms].values, c[syms].values, hi.values
SO, SC = o["SPY"].values, c["SPY"].values
dates = list(c.index)
zz, aa, sg = z[syms].values, adv[syms].values, SIG[syms].values


def trades(hold, stop, zmin, adv_min):
    out = []
    for i, j in zip(*np.nonzero(sg & (zz >= zmin) & (aa >= adv_min))):
        if i + hold >= len(dates):
            continue
        e = O[i + 1, j]
        if not np.isfinite(e) or e <= 0:
            continue
        px, k_exit = C[i + hold, j], i + hold
        if stop is not None:
            sp = e * (1 + stop)
            for k in range(i + 1, i + hold + 1):
                ok = np.isfinite(H[k, j]) and H[k, j] >= sp
                if ok:
                    px, k_exit = (max(O[k, j], sp) if k > i + 1 else sp), k
                    break
        if not np.isfinite(px):
            continue
        spy = SC[k_exit] / SO[i + 1] - 1
        out.append((dates[i], -(px / e - 1), spy - (px / e - 1)))
    return pd.DataFrame(out, columns=["date", "short", "pair"])


def line(lab, t, cost, leg="pair"):
    r = t[leg] - cost
    lab = f"{lab} {leg}"
    d = r.groupby(t.date).mean()
    yr = r.groupby(t.date.astype(str).str[:4]).mean() * 100
    mid = len(d) // 2
    tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    ok = d.mean() > 0 and tt >= 2 and d.iloc[:mid].mean() > 0 and d.iloc[mid:].mean() > 0 and (yr > 0).all()
    print(f"  {lab:40s} cost {cost*100:.1f}% n {len(r):5d} mean {r.mean()*100:+.2f} med {r.median()*100:+.2f} "
          f"win {(r>0).mean():.0%} worst {r.min()*100:+.0f}% t {tt:+.2f} halves {d.iloc[:mid].mean()*100:+.2f}/"
          f"{d.iloc[mid:].mean()*100:+.2f} yrs { {k: round(v, 2) for k, v in yr.items()} } {'PASS' if ok else 'FAIL'}",
          flush=True)
    return ok


for zmin, amin, alab in ((3, 25e6, "$25M+"), (2, 25e6, "$25M+"), (3, 100e6, "$100M+")):
    print(f"\n=== z >= {zmin}, {alab} ADV, hold 10 ===")
    for stop in (None, 0.30, 0.20, 0.15, 0.10):
        t = trades(10, stop, zmin, amin)
        for cost in ((0.003, 0.006, 0.01, 0.02) if stop in (None, 0.15) else (0.006,)):
            for leg in ("pair", "short"):
                line(f"stop {'none' if stop is None else f'+{stop:.0%}'}", t, cost, leg)
print("\n=== z >= 3, $25M+, stop +15%, hold 5 ===")
line("stop +15% h5", trades(5, 0.15, 3, 25e6), 0.006)
