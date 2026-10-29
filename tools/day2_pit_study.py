"""Point-in-time test of the LIVE Day 2 Continuation rule (_rs_day2_continuation).

RESULT 2026-10-29 (2024-08..2026-10, 0 costs, entry = Day-2 close):
  $5+ / $25M+ ADV, funds excluded: n=147, 32% win, median 0.00%, mean vs SPY
    over the same window +0.80%/trade (t 1.63), halves +0.43/+1.16; same-day
    clustered +0.56% (t 1.09); 0.2% round trip t 1.22; without its 5 best
    trades +0.14%. $100+ (spread lane) n=47 +1.47% (t 1.75), -0.01% sans top 5.
  => The ONLY setup here that does not lose to SPY, but it is a fat-tail bet
  (T2 exits +12%, full stops -4%) and the edge is not distinguishable from
  zero. Keep it live at small size; the kill switch is its verdict.
  Funds (leveraged ETFs) were 39 of 186 liquid trades; with them t reads 2.25.

Usage: py -3 tools/day2_pit_study.py <univ_bars.pkl> <out.json> [fund_syms.json]
  (univ_bars.pkl is what tools/ml_walkforward_study.py caches)
"""
import io, contextlib, json, pickle, sys, datetime as dt
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Users\singh")
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as d
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

close, dvol = pickle.load(open(sys.argv[1], "rb"))
adv = dvol.rolling(20, min_periods=15).mean()
r1 = close.pct_change(fill_method=None)
cand = ((r1 >= 0.04) & (close >= 5) & (adv >= 2e6) & (close <= 5000)).any()
syms = sorted(set(cand.index[cand.values]) | {"SPY"})
print("symbols with a +4% liquid day:", len(syms), flush=True)

dc = d.get_alpaca_data_client()
frames = []
for i in range(0, len(syms), 300):
    r = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 300], timeframe=TimeFrame.Day,
                                           start=dt.datetime(2024, 7, 1), end=dt.datetime(2026, 10, 29),
                                           adjustment=Adjustment.SPLIT, feed=DataFeed.SIP))
    frames.append(r.df)
    print("  fetched", i, flush=True)
bars = pd.concat(frames)
spy = bars.loc["SPY"].copy(); spy.index = [t.date() for t in spy.index.tz_convert("America/New_York")]

def simulate(df, i0, entry, stop, t1, t2):
    trail, be_px, be, t1hit, hold = stop, entry + (entry - stop), False, False, 0
    for j in range(i0 + 1, len(df)):
        H, L, C = df.high.iat[j], df.low.iat[j], df.close.iat[j]
        hold += 1
        if L <= trail:
            return j, trail, "STOP(BE)" if be else "STOP"
        if not be and not t1hit and H >= be_px:
            trail, be = entry, True
        if not t1hit and H >= t1:
            t1hit, trail = True, entry
            continue
        if t1hit and H >= t2:
            return j, t2 * 0.999, "T2"
        if hold >= 3 and not t1hit and abs(C - entry) / entry < 0.005:
            return j, C, "STALL"
        if hold >= d.BACKTEST_MAX_HOLD_BARS:
            return j, C, "TIME"
    return None

trades = []
for s in syms:
    if s == "SPY":
        continue
    try:
        df = bars.loc[s].copy()
    except KeyError:
        continue
    if len(df) < 40:
        continue
    df.index = [t.date() for t in df.index.tz_convert("America/New_York")]
    c, v = df.close, df.volume
    delta = c.diff()
    rsi = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    av20 = v.rolling(20).mean()
    rvol = v / av20.replace(0, np.nan)
    O, H = df.open, df.high
    for i in range(30, len(df) - 1):
        p3c, pO, pH, pC = c.iat[i - 2], O.iat[i - 1], H.iat[i - 1], c.iat[i - 1]
        d1_gap = (pO - p3c) / p3c * 100
        if d1_gap < 4.0:
            continue
        rng = pH - pO
        held = (pC - pO) / rng if rng > 0 else 0
        cc = c.iat[i]
        if not (held >= 0.6 and cc >= pC * 0.98 and rvol.iat[i] >= 1.5 and 45 < rsi.iat[i] < 75
                and (cc - p3c) / p3c * 100 <= d.DAY2_MAX_CUMULATIVE_MOVE_PCT
                and macd.iat[i] > sig.iat[i] and cc * av20.iat[i] >= 500_000):
            continue
        stop = round(pC * 0.97, 2)
        if cc - stop <= 0:
            continue
        risk = cc - stop
        t1 = max(round(cc + 2.0 * risk, 2), round(cc * (1 + d1_gap / 100), 2))
        t2 = max(round(cc + 3.5 * risk, 2), round(cc * (1 + d1_gap / 100 * 1.5), 2))
        res = simulate(df, i, cc, stop, t1, t2)
        if res is None:
            continue
        j, px, why = res
        d0, d1 = df.index[i], df.index[j]
        if d0 not in spy.index or d1 not in spy.index:
            continue
        adv_now = float((c * v).iloc[i - 20:i].mean())
        trades.append(dict(sym=s, date=str(d0), exit=str(d1), why=why, px=float(cc), adv=adv_now,
                           pnl=(px - cc) / cc * 100,
                           spy=(spy.close[d1] - spy.close[d0]) / spy.close[d0] * 100,
                           wl=s in d.WATCHLIST, gap=d1_gap, hold=j - i))
json.dump(trades, open(sys.argv[2], "w"))

T = pd.DataFrame(trades)
if len(sys.argv) > 3:   # a JSON list of fund symbols: leveraged ETFs are not companies
    T = T[~T.sym.isin(set(json.load(open(sys.argv[3]))))]
T["ex"] = T.pnl - T.spy
def rep(name, t):
    if len(t) < 2:
        print(f"  {name:34} n={len(t)}"); return
    t = t.sort_values("date"); h = len(t) // 2
    print(f"  {name:34} n={len(t):4}  win {(t.pnl>0).mean()*100:3.0f}%  mean {t.pnl.mean():+.2f}%  "
          f"vs SPY {t.ex.mean():+.2f}% (t {t.ex.mean()/t.ex.std()*np.sqrt(len(t)):+.2f})  median {t.pnl.median():+.2f}%  "
          f"halves {t.ex[:h].mean():+.2f}/{t.ex[h:].mean():+.2f}  days {t.date.nunique()}")
print(f"\nDay 2 Continuation, {T.date.min()}..{T.date.max()}, 0 costs")
rep("ALL ($500K rule floor)", T)
rep("$5+ & $5M+ ADV", T[(T.px >= 5) & (T.adv >= 5e6)])
rep("$5+ & $25M+ ADV", T[(T.px >= 5) & (T.adv >= 25e6)])
rep("$100+ & $25M+ (spread lane)", T[(T.px >= 100) & (T.adv >= 25e6)])
rep("WATCHLIST (hindsight list)", T[T.wl])
print("exits:", T.why.value_counts().to_dict())
