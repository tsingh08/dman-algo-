"""Point-in-time research harness: the bar every strategy change must clear.

Why this exists: the studies that overturned this repo's claims (WATCHLIST
backtests hindsight-biased; zone rule -0.80%/trade vs SPY; PEAD, earnings gaps
and the ML ranker all ~zero or negative) were written in session scratchpads
and deleted with them. A desk keeps one reproducible harness instead.

Universe: every ACTIVE and INACTIVE (delisted) US equity Alpaca lists, funds
removed, split-adjusted SIP daily bars; liquidity measured from PRIOR bars
only. A rule is a function (open, close, dvol) -> boolean frame of entries
decided at day t's close. Entry is t+1's open, exit t+hold's close, cost per
round trip, and every trade is measured against SPY over the same window.

VERDICT (all required): mean excess > 0, date-clustered t >= 2, BOTH halves
> 0, EVERY calendar year > 0. Anything less is not an edge -- do not ship it.

RESULTS 2026-10-31 (2023-06..2026-10, 0.10% round trip):
  zone  n=3,910 over 566 days: mean +0.04% vs SPY, MEDIAN -2.44%, beats SPY
        45%, clustered t 0.06; halves +0.51/-0.26; 2024 +2.16 2025 +1.89
        2026 -3.68  -> FAIL (no edge; research size is right)
  dip   n=164k: mean -0.15% vs SPY, t -4.43, every year negative -> FAIL

TREND (portfolio bar below; 2017-02..2026-10, total return, 0.05%/turnover):
              CAGR  Sharpe  h1/h2      maxDD   worst month
  SPY         14.9   0.73   0.93/0.52  -33.8   -12.6
  60/40        9.6   0.69   1.09/0.30  -21.0    -7.4   (reference)
  spy_ma10     8.4   0.52   0.94/0.06  -29.2   -10.1   FAIL (monthly signal missed 2020)
  gtaa5        5.8   0.50   0.95/0.07  -11.5    -4.2   FAIL (small hole, no return)
  gem          8.7   0.45   0.56/0.32  -33.8   -12.6   FAIL (whipsawed 2018/2020)
  rotation    12.0   0.72   0.91/0.54  -15.2    -9.3   FAIL, nearest miss: SPY's
              Sharpe with less than half the hole, but -2.9%/yr and h1 under 60/40.
  -> Trend following did NOT earn its keep out of sample. Do not ship it as a
     "safer SPY"; at this account size the cheapest drawdown control is size.

Usage:
  py -3 tools/pit_harness.py zone            # the live breakout-zone rule
  py -3 tools/pit_harness.py dip             # the dip-in-uptrend rule
  py -3 tools/pit_harness.py trend           # 4 trend-following portfolios vs SPY, 60/40
First run downloads ~13k symbols (minutes) -> pit_bars.pkl (gitignored by size).
"""
import math, os, pickle, re, sys, datetime as dt
import numpy as np, pandas as pd

CACHE = "pit_bars.pkl"
START = dt.datetime(2023, 6, 1)
FUND = re.compile(r"\b(ETF|ETN|ETP|Fund|Trust|Leveraged|Inverse|Ultra|ProShares|Direxion|"
                  r"GraniteShares|Tradr|Defiance|T-Rex|YieldMax|Roundhill|iShares|Invesco|SPDR|"
                  r"2X|3X|1\.5X|Daily|Bull|Bear|Notes?)\b", re.I)


def load_bars():
    """(open, close, dollar volume) frames, date x symbol, funds removed (SPY kept)."""
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    K, S = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    tc = TradingClient(K, S, paper=False, raw_data=True)
    syms = {"SPY"}
    for st in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for x in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=st)):
            s = str(x.get("symbol", ""))
            if (x.get("exchange") in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS") and s.isalpha()
                    and not FUND.search(str(x.get("name") or ""))):
                syms.add(s)
    sc = StockHistoricalDataClient(K, S)
    syms, frames = sorted(syms), []
    for i in range(0, len(syms), 400):
        try:
            r = sc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=syms[i:i + 400], timeframe=TimeFrame.Day, start=START,
                end=dt.datetime.now() - dt.timedelta(minutes=20),
                adjustment=Adjustment.SPLIT, feed=DataFeed.SIP))
            frames.append(r.df.reset_index()[["symbol", "timestamp", "open", "close", "volume"]])
        except Exception as e:
            print("chunk", i, type(e).__name__, str(e)[:80], flush=True)
    df = pd.concat(frames)
    df["date"] = df.timestamp.dt.tz_convert("America/New_York").dt.date
    df["dv"] = df.close * df.volume
    out = tuple(df.pivot_table(index="date", columns="symbol", values=v).astype("float32")
                for v in ("open", "close", "dv"))
    pickle.dump(out, open(CACHE, "wb"))
    return out


def evaluate(entries: pd.DataFrame, open_: pd.DataFrame, close: pd.DataFrame,
             hold: int, cost: float = 0.001, spy: str = "SPY") -> dict:
    """Excess-over-SPY stats for entries (bool, date x symbol, decided at t's close)."""
    ent, ext = open_.shift(-1), close.shift(-hold)
    r = ext / ent - 1 - cost
    x = r.sub(ext[spy] / ent[spy] - 1, axis=0)
    trades = x.where(entries.reindex_like(x).fillna(False).astype(bool)).stack().dropna()
    trades = trades[np.isfinite(trades)]
    if trades.empty:
        return {"n": 0}
    daily = trades.groupby(level=0).mean()            # one observation per entry date
    t = (daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
         if len(daily) > 1 and daily.std(ddof=1) > 0 else float("nan"))
    dates = sorted(daily.index)
    mid = dates[len(dates) // 2]
    yrs = trades.groupby(lambda k: str(k[0])[:4]).mean()
    return {"n": int(len(trades)), "days": int(len(daily)),
            "mean": float(trades.mean() * 100), "median": float(trades.median() * 100),
            "beat_spy": float((trades > 0).mean()), "t_clustered": float(t),
            "half1": float(trades[[k[0] < mid for k in trades.index]].mean() * 100),
            "half2": float(trades[[k[0] >= mid for k in trades.index]].mean() * 100),
            "years": {k: round(float(v) * 100, 2) for k, v in yrs.items()}}


def verdict(s: dict) -> tuple[bool, str]:
    """(passes, why). The whole bar, or nothing."""
    if not s.get("n"):
        return False, "no trades"
    fails = []
    if s["mean"] <= 0:
        fails.append(f"mean {s['mean']:+.2f}% <= 0")
    if not s["t_clustered"] >= 2:
        fails.append(f"clustered t {s['t_clustered']:.2f} < 2")
    if s["half1"] <= 0 or s["half2"] <= 0:
        fails.append(f"halves {s['half1']:+.2f}/{s['half2']:+.2f}")
    bad = [y for y, v in s["years"].items() if v <= 0]
    if bad:
        fails.append("losing years " + ",".join(bad))
    return (not fails), ("PASS" if not fails else "FAIL: " + "; ".join(fails))


def _adv(dv):
    return dv.rolling(20, min_periods=15).mean().shift(1)    # prior bars only


def rule_zone(open_, close, dv):
    """Live breakout zone: new 252d closing high, >=100% off the 252d low, >=10%
    over the prior 20-day mean, $5-15, >= $5M/day."""
    hi = close.shift(1).rolling(251, min_periods=240).max()
    lo = close.rolling(252, min_periods=240).min()
    ma20 = close.shift(1).rolling(20).mean()
    return ((close >= hi) & (close / lo - 1 >= 1.0) & (close / ma20 - 1 >= 0.10)
            & (close >= 5) & (close <= 15) & (_adv(dv) >= 5e6))


def rule_dip(open_, close, dv):
    """Dip in uptrend: above MA200 and 2-bar Cutler RSI < 10, $10+, $20M/day."""
    d = close.diff()
    up, dn = d.clip(lower=0).rolling(2).mean(), (-d.clip(upper=0)).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi2 = rsi2.where(dn > 0, 100.0)
    return ((close > close.rolling(200, min_periods=190).mean()) & (rsi2 < 10)
            & (close >= 10) & (_adv(dv) >= 20e6))


RULES = {"zone": (rule_zone, 20), "dip": (rule_dip, 5)}


# ---------------------------------------------------------------------------
# PORTFOLIO RULES: cross-asset trend following, judged as DRAWDOWN CONTROL.
# The per-trade bar above asks "does it beat SPY"; nothing has. These rules
# ask a different question -- "same or better risk-adjusted return with a
# much smaller hole" -- so they get their own bar, fixed BEFORE the first run:
#   max drawdown <= 2/3 of SPY's, AND Sharpe >= SPY's AND >= a static 60/40's
#   (else a dumb bond mix does the job), in BOTH halves of the sample.
# All four rules were published 2007-2014, so 2016+ is out of sample for them.
# Signals at month-end close, traded the NEXT day's close, 0.05% per unit of
# turnover; dividend-adjusted closes; weights held at target between rebalances.
# ---------------------------------------------------------------------------
ETF_CACHE = "pit_etf.pkl"
ETF_START = dt.datetime(2016, 1, 1)
ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "AGG", "SHY", "BIL",
        "GLD", "DBC", "VNQ"]


def load_etf_bars():
    """Total-return closes (dividends reinvested), DatetimeIndex x ETF."""
    if os.path.exists(ETF_CACHE):
        return _unsplit(pickle.load(open(ETF_CACHE, "rb")))
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    sc = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    df = sc.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=ETFS, timeframe=TimeFrame.Day, start=ETF_START,
        end=dt.datetime.now() - dt.timedelta(minutes=20),
        adjustment=Adjustment.ALL, feed=DataFeed.SIP)).df.reset_index()
    df["date"] = pd.to_datetime(df.timestamp.dt.tz_convert("America/New_York").dt.date)
    out = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    pickle.dump(out, open(ETF_CACHE, "wb"))
    return _unsplit(out)


def _unsplit(close: pd.DataFrame) -> pd.DataFrame:
    """Undo split adjustments the feed missed. BIL's 2017-11-30 reverse split
    shows as a +100% day, which made T-bills 'return' 8.9%/yr and wrecked
    every Sharpe and GEM's signal. No diversified ETF moves 40% in a day, so
    such a day is treated as a 0% day and the series is rebuilt from returns."""
    r = close.pct_change()
    r = r.mask(r.abs() > 0.40, 0.0)
    return (1 + r.fillna(0)).cumprod() * close.iloc[0]


def _sma10(m):
    return m.rolling(10).mean()


def port_spy_ma10(m):
    """Faber timing: SPY above its 10-month SMA, else short Treasuries."""
    on = (m["SPY"] > _sma10(m)["SPY"]).astype(float).where(_sma10(m)["SPY"].notna())
    return pd.DataFrame({"SPY": on, "SHY": 1 - on})


def port_gtaa5(m):
    """Faber GTAA-5: 20% each of SPY/EFA/IEF/DBC/VNQ when above its own 10-month SMA."""
    a = ["SPY", "EFA", "IEF", "DBC", "VNQ"]
    on = (m[a] > _sma10(m)[a]).astype(float).where(_sma10(m)[a].notna()) * 0.2
    return on.assign(SHY=0.2 * 5 - on.sum(axis=1).where(on.notna().all(axis=1)))


def port_gem(m):
    """Antonacci dual momentum: 12m SPY > 12m T-bills -> better of SPY/EFA, else AGG."""
    r12 = m / m.shift(12) - 1
    risk_on = r12["SPY"] > r12["BIL"]
    pick = np.where(~risk_on, "AGG", np.where(r12["SPY"] >= r12["EFA"], "SPY", "EFA"))
    w = pd.DataFrame(0.0, index=m.index, columns=["SPY", "EFA", "AGG"])
    for i, p in enumerate(pick):
        w.iloc[i, w.columns.get_loc(p)] = 1.0
    return w.where(r12["SPY"].notna() & r12["BIL"].notna())


def port_rotation(m):
    """Top 3 of 9 asset classes by blended 1/3/6/12m momentum, each 1/3, each
    only when above its own 10-month SMA (else that third sits in SHY)."""
    a = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ"]
    mom = sum(m[a] / m[a].shift(k) - 1 for k in (1, 3, 6, 12)) / 4
    top = mom.rank(axis=1, ascending=False) <= 3
    w = (top & (m[a] > _sma10(m)[a])).astype(float) / 3
    w = w.assign(SHY=1 - w.sum(axis=1))
    return w.where(mom.notna().all(axis=1), axis=0)


def port_6040(m):
    """Reference, not a candidate: static 60% SPY / 40% IEF."""
    return pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=m.index)


def port_spy(m):
    return pd.DataFrame({"SPY": 1.0}, index=m.index)


def run_portfolio(weights_m: pd.DataFrame, close: pd.DataFrame, start=None,
                  cost: float = 0.0005) -> pd.Series:
    """Daily returns of month-end target weights, traded one day late."""
    c = close[weights_m.columns]
    w = weights_m.reindex(c.index).ffill().shift(1)      # decided at t, held from t+1's close
    ret = c.pct_change()
    port = (w.shift(1) * ret).sum(axis=1, min_count=1) - w.fillna(0).diff().abs().sum(axis=1) * cost
    port = port[w.shift(1).notna().all(axis=1)]
    return port[port.index >= start] if start is not None else port


def portfolio_stats(r: pd.Series, rf: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    ex = r - rf.reindex(r.index).fillna(0)
    sh = lambda x: float(x.mean() / x.std() * math.sqrt(252)) if x.std() > 0 else float("nan")
    mid = r.index[len(r) // 2]
    yrs = (1 + r).groupby(r.index.year).prod() - 1
    return {"cagr": float(eq.iloc[-1] ** (252 / len(r)) - 1) * 100,
            "vol": float(r.std() * math.sqrt(252)) * 100,
            "sharpe": sh(ex), "sharpe_h1": sh(ex[ex.index < mid]), "sharpe_h2": sh(ex[ex.index >= mid]),
            "max_dd": float((eq / eq.cummax() - 1).min()) * 100,
            "worst_month": float(((1 + r).groupby(r.index.to_period("M")).prod() - 1).min()) * 100,
            "years": {int(k): round(float(v) * 100, 1) for k, v in yrs.items()}}


def portfolio_verdict(s: dict, spy: dict, ref: dict) -> tuple[bool, str]:
    """The drawdown-control bar, fixed before the first run."""
    fails = []
    if s["max_dd"] < spy["max_dd"] * 2 / 3:
        fails.append(f"max DD {s['max_dd']:.1f}% not <= 2/3 of SPY's {spy['max_dd']:.1f}%")
    for h in ("sharpe_h1", "sharpe_h2"):
        best = max(spy[h], ref[h])
        if not s[h] >= best:
            fails.append(f"{h[-2:]} Sharpe {s[h]:.2f} < {best:.2f}")
    return (not fails), ("PASS" if not fails else "FAIL: " + "; ".join(fails))


PORTFOLIO_RULES = {"spy_ma10": port_spy_ma10, "gtaa5": port_gtaa5,
                   "gem": port_gem, "rotation": port_rotation}


def run_trend():
    close = load_etf_bars()
    me = close.groupby(close.index.to_period("M")).tail(1)
    runs = {k: run_portfolio(f(me), close) for k, f in
            {**PORTFOLIO_RULES, "spy": port_spy, "60/40": port_6040}.items()}
    start = max(r.index[0] for r in runs.values())        # common window for a fair table
    rf = close["BIL"].pct_change()
    st = {k: portfolio_stats(r[r.index >= start], rf) for k, r in runs.items()}
    print(f"window {start.date()} .. {close.index[-1].date()}  (total return, 0.05%/turnover)")
    yrs = sorted(st["spy"]["years"])
    print(f"{'rule':10}{'CAGR':>7}{'vol':>6}{'Sharpe':>7}{'h1':>6}{'h2':>6}{'maxDD':>7}{'wMon':>7}  "
          + " ".join(f"{y:>6}" for y in yrs))
    for k, s in st.items():
        print(f"{k:10}{s['cagr']:7.1f}{s['vol']:6.1f}{s['sharpe']:7.2f}{s['sharpe_h1']:6.2f}"
              f"{s['sharpe_h2']:6.2f}{s['max_dd']:7.1f}{s['worst_month']:7.1f}  "
              + " ".join(f"{s['years'].get(y, float('nan')):6.1f}" for y in yrs))
    for k in PORTFOLIO_RULES:
        print(f"VERDICT {k}:", portfolio_verdict(st[k], st["spy"], st["60/40"])[1])
    return st


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "zone"
    if name == "trend":
        run_trend()
        sys.exit(0)
    fn, hold = RULES[name]
    o, c, v = load_bars()
    s = evaluate(fn(o, c, v), o, c, hold)
    ok, why = verdict(s)
    print(f"{name} (hold {hold}, 0.10% round trip, vs SPY): "
          + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    print("VERDICT:", why)
