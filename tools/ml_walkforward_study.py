"""Walk-forward ML study: can a model trained ONLY on data available at the time
pick liquid US stocks that beat SPY over the next 5 sessions?

RESULT 2026-10-29 (test 2025-06..2026-10, 70 non-overlapping weeks, monthly
retrain, HistGradientBoosting on 20 trailing price/volume features):
  $5+ / $25M+ ADV universe (this script, default): daily rank-IC +0.003
    (t 0.24) -- no signal. top-5 / 10 / 25 (next close, 0.10%/side):
    -0.49 / +0.22 / +0.26 %/wk vs SPY, every |t| < 0.5; at 0.25%/side top-5
    is -0.79%/wk. Worst week -17..-21% (SPY's worst ~-4%).
  $5+ / $5M+ ADV: IC +0.028 (t 2.2) but the top-N books still ~= SPY (t < 0.2);
    that IC is the model learning to AVOID junk, not to pick winners.
  Dip rule (RSI2<10 & above MA100, 5d hold) on $25M+: -0.91%/wk vs SPY (t -2.5).
  => An ML ranker does not beat SPY here. Do not wire one into live.
  An LLM cannot be backtested honestly on this window (its training data
  contains the outcomes); its forward shadow record is dman_shadow_reviews.json.

DATA CAVEAT: split-adjusted history back-scales serial reverse-splitters into
"$5+" (YYAI shows $173,600), which pollutes any $5 floor and the extreme-move
rules (5d-loser / 20d-winner books read -6..-7%/wk). Use MIN_ADV >= 25e6.

Usage: py -3 tools/ml_walkforward_study.py [min_adv=25e6]   (needs scikit-learn;
first run downloads ~12.7k symbols' daily bars incl. delisted -> univ_bars.pkl)
"""
import json, os, pickle, re, sys, time, datetime as dt
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MIN_ADV = float(sys.argv[1]) if len(sys.argv) > 1 else 25e6
CACHE = "univ_bars.pkl"
# Same fund filter as tools/smallcap_universe_refresh.py.
FUND = re.compile(r"\b(ETF|ETN|ETP|Fund|Trust|Leveraged|Inverse|Ultra|ProShares|Direxion|"
                  r"GraniteShares|Tradr|Defiance|T-Rex|YieldMax|Roundhill|iShares|Invesco|SPDR|"
                  r"2X|3X|1\.5X|Daily|Bull|Bear|Notes?)\b", re.I)


def load_bars():
    """(close, dollar volume, fund symbols) for every active AND inactive US
    equity, split-adjusted, cached to univ_bars.pkl."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    K = os.getenv("APCA_API_KEY_ID"); S = os.getenv("APCA_API_SECRET_KEY")
    tc = TradingClient(K, S, paper=False, raw_data=True)
    syms, funds = set(), set()
    for st in (AssetStatus.ACTIVE, AssetStatus.INACTIVE):
        for x in tc.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=st)):
            s = str(x.get("symbol", ""))
            if x.get("exchange") in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS") and s.isalpha():
                syms.add(s)
                if FUND.search(str(x.get("name") or "")):
                    funds.add(s)
    if os.path.exists(CACHE):
        close, dvol = pickle.load(open(CACHE, "rb"))
        return close, dvol, funds
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    sc = StockHistoricalDataClient(K, S)
    syms, frames = sorted(syms), []
    for i in range(0, len(syms), 400):
        try:
            r = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=syms[i:i + 400], timeframe=TimeFrame.Day,
                                                   start=dt.datetime(2024, 7, 1), end=dt.datetime.now(),
                                                   adjustment=Adjustment.SPLIT, feed=DataFeed.SIP))
            frames.append(r.df.reset_index()[["symbol", "timestamp", "close", "volume"]])
        except Exception as e:
            print("chunk", i, type(e).__name__, str(e)[:80], flush=True)
    df = pd.concat(frames); df["date"] = df.timestamp.dt.tz_convert("America/New_York").dt.date
    close = df.pivot_table(index="date", columns="symbol", values="close")
    dvol = df.assign(dv=df.close * df.volume).pivot_table(index="date", columns="symbol", values="dv")
    pickle.dump((close, dvol), open(CACHE, "wb"))
    return close, dvol, funds


t0 = time.time()
close, dvol, funds = load_bars()
keep = [c for c in close.columns if c not in funds or c == "SPY"]
close, dvol = close[keep].astype("float32"), dvol[keep].astype("float32")
spy = close["SPY"]

ret1 = close.pct_change(fill_method=None)
adv = dvol.rolling(20, min_periods=15).mean()

def rsi_wilder(n):
    up = ret1.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-ret1.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def rsi_cutler2():
    d = close.diff()
    up = d.clip(lower=0).rolling(2).mean()
    dn = (-d.clip(upper=0)).rolling(2).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

F = {
    "r1": ret1,
    "r5": close.pct_change(5, fill_method=None),
    "r20": close.pct_change(20, fill_method=None),
    "r60": close.pct_change(60, fill_method=None),
    "r120": close.pct_change(120, fill_method=None),
    "ma20": close / close.rolling(20).mean() - 1,
    "ma50": close / close.rolling(50).mean() - 1,
    "ma100": close / close.rolling(100).mean() - 1,
    "hi120": close / close.rolling(120, min_periods=60).max() - 1,
    "lo120": close / close.rolling(120, min_periods=60).min() - 1,
    "rsi2": rsi_cutler2(),
    "rsi14": rsi_wilder(14),
    "vol20": ret1.rolling(20).std(),
    "volr": ret1.rolling(5).std() / ret1.rolling(20).std(),
    "rvol": dvol / adv,
    "rvol5": dvol.rolling(5).mean() / adv,
    "ladv": np.log(adv),
    "lpx": np.log(close),
}
# cross-sectional ranks of the momentum/reversal features (regime-free)
for k in ("r5", "r20", "r60"):
    F[k + "_rk"] = F[k].rank(axis=1, pct=True)
FEATS = list(F) + ["spy_r5", "spy_r20"]
spy_r5, spy_r20 = spy.pct_change(5), spy.pct_change(20)

H = 5
fwd = close.shift(-H) / close - 1                    # enter at close t
fwd_next = close.shift(-H - 1) / close.shift(-1) - 1  # enter at close t+1 (conservative)
spy_fwd, spy_fwd_next = spy.shift(-H) / spy - 1, spy.shift(-H - 1) / spy.shift(-1) - 1
universe = (close >= 5) & (adv >= MIN_ADV) & (close <= 5000)
universe["SPY"] = False

dates = list(close.index)
start = 125                                           # features need ~120 bars
print(f"prep {time.time()-t0:.0f}s, dates {dates[start]}..{dates[-1]}", flush=True)

def panel(d):
    m = universe.loc[d]
    syms = m.index[m.values]
    X = pd.DataFrame({k: F[k].loc[d, syms] for k in F})
    X["spy_r5"], X["spy_r20"] = spy_r5.loc[d], spy_r20.loc[d]
    X["y"] = (fwd.loc[d, syms] - spy_fwd.loc[d]).clip(-0.5, 0.5)
    X["fwd"] = fwd.loc[d, syms]
    X["fwd_next"] = fwd_next.loc[d, syms]
    X["spy_fwd"], X["spy_fwd_next"] = spy_fwd.loc[d], spy_fwd_next.loc[d]
    X["date"] = d
    return X

rows = [panel(d) for d in dates[start:]]
P = pd.concat(rows)
P = P.replace([np.inf, -np.inf], np.nan)
print(f"panel {len(P):,} rows, {time.time()-t0:.0f}s", flush=True)

# monthly walk-forward
months = sorted({(d.year, d.month) for d in P.date.unique()})
first_test = 6                                        # >= 6 months of training
preds = []
for mi in range(first_test, len(months)):
    y_, m_ = months[mi]
    test_days = [d for d in dates if (d.year, d.month) == (y_, m_)]
    cutoff = dates[dates.index(test_days[0]) - H - 1]  # last label fully known
    tr = P[(P.date <= cutoff) & P.y.notna()]
    if len(tr) > 400_000:
        tr = tr.sample(400_000, random_state=0)
    mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                        min_samples_leaf=200, l2_regularization=1.0, random_state=0)
    mdl.fit(tr[FEATS], tr.y)
    te = P[P.date.isin(test_days)].copy()
    te["pred"] = mdl.predict(te[FEATS])
    preds.append(te)
    print(f"  {y_}-{m_:02d} train {len(tr):,} test {len(te):,}  {time.time()-t0:.0f}s", flush=True)
T = pd.concat(preds)
T.to_pickle(f"ml_preds_{MIN_ADV:.0e}.pkl")

# ---- evaluation ---------------------------------------------------------
tdates = sorted(T.date.unique())
ic = T.dropna(subset=["fwd"]).groupby("date").apply(lambda g: g.pred.corr(g.fwd, method="spearman"))
print(f"\nTEST {tdates[0]}..{tdates[-1]}  daily rank-IC mean {ic.mean():+.4f}  (t {ic.mean()/ic.std()*np.sqrt(len(ic)/H):.2f}, overlap-adjusted)")

def book(score_col, n, fwd_col, spy_col, cost, asc=False, mask=None):
    """weekly non-overlapping, top-n by score; returns per-week (port, spy, univ)."""
    out = []
    for d in tdates[::H]:
        g = T[(T.date == d)].dropna(subset=[fwd_col])
        if mask is not None:
            g = g[mask(g)]
        if len(g) < n:
            continue
        pick = g.nsmallest(n, score_col) if asc else g.nlargest(n, score_col)
        out.append((d, pick[fwd_col].clip(-0.95, 3).mean() - 2 * cost, g[spy_col].iloc[0], g[fwd_col].clip(-0.95, 3).mean()))
    return pd.DataFrame(out, columns=["date", "port", "spy", "univ"])

def report(name, b):
    ex = b.port - b.spy
    half = len(b) // 2
    cum = (1 + b.port).prod() - 1
    cum_spy = (1 + b.spy).prod() - 1
    print(f"  {name:38} weeks {len(b):3}  mean/wk {b.port.mean()*100:+.2f}%  vs SPY {ex.mean()*100:+.2f}% "
          f"(t {ex.mean()/ex.std()*np.sqrt(len(ex)):+.2f})  beat-SPY wks {(ex>0).mean()*100:.0f}%  "
          f"halves {ex[:half].mean()*100:+.2f}/{ex[half:].mean()*100:+.2f}  "
          f"cum {cum*100:+.1f}% vs SPY {cum_spy*100:+.1f}%  worst wk {b.port.min()*100:+.1f}%")

for cost in (0.001, 0.0025):
    print(f"\n=== cost {cost*100:.2f}%/side ===")
    for n in (5, 10, 25):
        report(f"ML top-{n} (enter same close)", book("pred", n, "fwd", "spy_fwd", cost))
        report(f"ML top-{n} (enter next close)", book("pred", n, "fwd_next", "spy_fwd_next", cost))
    report("Universe equal-weight", book("pred", 1, "fwd", "spy_fwd", 0).assign(port=lambda b: b.univ))
    report("Rule: 5d losers (reversal) top-10", book("r5", 10, "fwd_next", "spy_fwd_next", cost, asc=True))
    report("Rule: 20d winners (momentum) top-10", book("r20", 10, "fwd_next", "spy_fwd_next", cost))
    report("Rule: dip RSI2<10 & >MA100 top-10", book("rsi2", 10, "fwd_next", "spy_fwd_next", cost, asc=True,
                                                     mask=lambda g: g.ma100 > 0))
    report("ML top-10 among $5-$50 only (next close)", book("pred", 10, "fwd_next", "spy_fwd_next", cost,
                                                            mask=lambda g: g.lpx < np.log(50)))
print(f"\ndone {time.time()-t0:.0f}s")
