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

Usage:
  py -3 tools/pit_harness.py zone            # the live breakout-zone rule
  py -3 tools/pit_harness.py dip             # the dip-in-uptrend rule
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
            & close.between(5, 15) & (_adv(dv) >= 5e6))


def rule_dip(open_, close, dv):
    """Dip in uptrend: above MA200 and 2-bar Cutler RSI < 10, $10+, $20M/day."""
    d = close.diff()
    up, dn = d.clip(lower=0).rolling(2).mean(), (-d.clip(upper=0)).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    rsi2 = rsi2.where(dn > 0, 100.0)
    return ((close > close.rolling(200, min_periods=190).mean()) & (rsi2 < 10)
            & (close >= 10) & (_adv(dv) >= 20e6))


RULES = {"zone": (rule_zone, 20), "dip": (rule_dip, 5)}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "zone"
    fn, hold = RULES[name]
    o, c, v = load_bars()
    s = evaluate(fn(o, c, v), o, c, hold)
    ok, why = verdict(s)
    print(f"{name} (hold {hold}, 0.10% round trip, vs SPY): "
          + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in s.items()))
    print("VERDICT:", why)
