"""ML v2: an ADAPTIVE walk-forward ranker -- price/volume + market regime (VIX
term structure, breadth, news mood) + per-ticker news sentiment -- judged by
the same bar as tools/pit_harness.py. No look-ahead anywhere:

  * universe: pit_harness bars (delisted included), $10+, prior-20d ADV >= $25M
  * features: known at day t's close (news by the 16:00 ET cutoff, VIX close)
  * label:    entry t+1 OPEN -> exit t+5 CLOSE, minus SPY over the same window
  * retrain:  monthly; train rows end 6 sessions before the test month so
              every training label was fully known on the first test day
  * two schemes: EXPANDING (all history) and ROLLING 12 months -- the Alpaca
    "quant agent" article's adaptivity rule (drop the oldest month, add the
    newest). Rolling is the "adaptive" model; expanding is its control.
  * ablation: price-only vs +regime vs +regime+news, so any lift is attributed

Judged three ways: daily rank-IC; weekly NON-overlapping top-N books vs SPY
(t on weekly excess); and ph.evaluate/ph.verdict on the daily top-10 entries
(mean > 0, clustered t >= 2, both halves > 0, every year > 0). Only a PASS on
the verdict is a ship candidate, and even then: shadow first, never live.

RESULT 2026-11-07 (test 2024-07..2026-10, 1.13M rows, ~1,550 names/day,
117 non-overlapping weeks): 6 of 6 FAIL the bar. NO-SHIP.
  variant                     IC      top10 %/wk vs SPY   harness top10/day
  price / expanding          +0.003   -0.40 (t -0.73)     -0.09  t -0.36
  price / rolling12          +0.009   -0.88 (t -1.61)     -0.60  t -2.45
  +regime / expanding        +0.009   +0.26 (t +0.47)     +0.14  t +0.58  (best; halves -0.02/+0.29)
  +regime / rolling12        +0.006   -0.15 (t -0.25)     -0.35  t -1.36
  +regime+news / expanding   +0.010   +0.07 (t +0.13)     -0.06  t -0.26
  +regime+news / rolling12   +0.005   -0.30 (t -0.49)     -0.36  t -1.37
  * ADAPTIVE (rolling 12m retrain) is WORSE than expanding in all 3 pairs:
    recency chases noise. News adds nothing over regime (matches the 27/27
    sentiment fail). Nothing picks winners.
  * What it CAN do is find losers: bottom-10 -6.6..-7.5%/wk vs SPY, t -6..-7,
    every variant. Mostly a volatility effect (vol20 top decile -0.74%/5d),
    but inside the top-vol decile ML still separates: low half -1.31%/5d
    (t -3.4, every year) vs high half -0.17%. Outside top-vol names: nothing.
  * As a filter on the live zone rule: h5 ML>=median +0.49 (t 1.05) vs all
    -0.12, but h20 (zone's real hold) no better (-0.37 vs +0.09), 2025
    negative either way -> FAIL. Not wired as a zone gate.
  * Regime -> SPY 5d (overlapping days, not a trade test): VIX>=25 +1.74%
    (n 38) vs +0.31% all; VIX/VIX3M>1 +1.19% (n 48); breadth50<0.30 +0.88%;
    news mood z>=2 +0.25% (nothing). Same sign as the VIX>=30 study (t 3.3).

Usage: py -3 tools/ml_v2_study.py            (needs pit_bars.pkl, news_sentiment.pkl)
"""
import os, sys, time, pickle, datetime as dt
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pit_harness as ph
import news_sentiment_study as ns

H, MIN_ADV, MIN_PX, TOPN = 5, 25e6, 10.0, 10
PANEL_START = dt.date(2023, 12, 1)       # ~125 bars after the 2023-06 cache start
FIRST_TEST = dt.date(2024, 7, 1)          # news exists from here; >= 7 train months
t0 = time.time()


def vix_frame(index):
    import yfinance as yf
    p = "pit_vix.pkl"
    if os.path.exists(p):
        v = pd.read_pickle(p)
    else:
        v = yf.download(["^VIX", "^VIX3M"], start="2023-01-01", progress=False)["Close"]
        v.to_pickle(p)
    v.index = [d.date() for d in pd.to_datetime(v.index)]
    v = v.reindex(index).ffill()
    return pd.DataFrame({"vix": v["^VIX"], "vix_ratio": v["^VIX"] / v["^VIX3M"],
                         "vix_chg5": v["^VIX"].pct_change(5)})


o, c, dv = ph.load_bars()
_ever = ((c >= MIN_PX) & (ph._adv(dv) >= MIN_ADV)).any()
_keep = list(_ever.index[_ever.values]) + (["SPY"] if not _ever.get("SPY", False) else [])
o, c, dv = o[_keep], c[_keep], dv[_keep]           # ~12k -> ~2.5k columns: memory
spy = c["SPY"]
adv = ph._adv(dv)
r1 = c.pct_change(fill_method=None)
univ = (c >= MIN_PX) & (adv >= MIN_ADV) & (c <= 5000)
univ["SPY"] = False

# ---- per-ticker news (by the 16:00 ET cutoff) + market mood ---------------
rows = pickle.load(open(ns.CACHE, "rb"))
pos, neg, tot = ns.daily_counts(rows, c.index, c.columns)
news_live = pd.Series(c.index >= ns.START, index=c.index)
mood = neg.sum(axis=1) / tot.sum(axis=1).replace(0, np.nan)
mood_z = (mood - mood.rolling(60, min_periods=30).mean()) / mood.rolling(60, min_periods=30).std()
tot20 = tot.rolling(20, min_periods=10).mean().shift(1)


def rsi_wilder(n):
    up = r1.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-r1.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


d_ = c.diff()
rsi2 = 100 - 100 / (1 + d_.clip(lower=0).rolling(2).mean() / (-d_.clip(upper=0)).rolling(2).mean().replace(0, np.nan))
PRICE = {
    "r1": r1, "r5": c.pct_change(5, fill_method=None), "r20": c.pct_change(20, fill_method=None),
    "r60": c.pct_change(60, fill_method=None), "r120": c.pct_change(120, fill_method=None),
    "gap": o / c.shift(1) - 1, "intraday": c / o - 1,
    "ma20": c / c.rolling(20).mean() - 1, "ma50": c / c.rolling(50).mean() - 1,
    "ma100": c / c.rolling(100).mean() - 1,
    "hi120": c / c.rolling(120, min_periods=60).max() - 1,
    "lo120": c / c.rolling(120, min_periods=60).min() - 1,
    "rsi2": rsi2, "rsi14": rsi_wilder(14),
    "vol20": r1.rolling(20).std(), "volr": r1.rolling(5).std() / r1.rolling(20).std(),
    "rvol": dv / adv, "rvol5": dv.rolling(5).mean() / adv, "ladv": np.log(adv), "lpx": np.log(c),
}
for k in ("r1", "r5", "r20", "r60"):
    PRICE[k + "_rk"] = PRICE[k].where(univ).rank(axis=1, pct=True)
na = lambda f: f.where(news_live, axis=0)       # NaN, not 0, before news exists
NEWS = {
    "n_pos1": na(pos), "n_neg1": na(neg), "n_tot1": na(tot),
    "n_net5": na((pos - neg).rolling(5, min_periods=1).sum()),
    "n_tot5": na(tot.rolling(5, min_periods=1).sum()),
    "n_surge": na(tot / tot20.clip(lower=1)),
}
breadth = ((c > c.rolling(50).mean()) & univ).sum(axis=1).astype(float) / univ.sum(axis=1).astype(float).replace(0, np.nan)
REG = vix_frame(c.index).assign(
    spy_r5=spy.pct_change(5), spy_r20=spy.pct_change(20), spy_ma200=spy / spy.rolling(200).mean() - 1,
    breadth50=breadth, mood_z=mood_z.where(news_live))

ent, ext = o.shift(-1), c.shift(-H)
fwd = ext / ent - 1
spy_fwd = fwd["SPY"]
xs = fwd.sub(spy_fwd, axis=0)

dates = [d for d in c.index if d >= PANEL_START]
parts = []
for d in dates:
    m = univ.loc[d]
    syms = m.index[m.values]
    if len(syms) < 50:
        continue
    X = pd.DataFrame({k: F.loc[d, syms] for k, F in {**PRICE, **NEWS}.items()})
    for k in REG.columns:
        X[k] = REG.at[d, k]
    X["y"] = xs.loc[d, syms].clip(-0.5, 0.5)
    X["ex"] = xs.loc[d, syms]
    X["date"], X["sym"] = d, syms
    parts.append(X.astype({k: "float32" for k in X.columns if k not in ("date", "sym")}))
P = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
print(f"panel {len(P):,} rows x {len(dates)} days, {time.time()-t0:.0f}s", flush=True)

FEATS = {
    "price":            list(PRICE),
    "price+regime":     list(PRICE) + list(REG.columns),
    "price+regime+news": list(PRICE) + list(REG.columns) + list(NEWS),
}
months = sorted({(d.year, d.month) for d in dates if d >= FIRST_TEST})
didx = {d: i for i, d in enumerate(c.index)}


def walk(feats, rolling_months=None):
    out = []
    for y_, m_ in months:
        test = [d for d in dates if (d.year, d.month) == (y_, m_)]
        cutoff = c.index[didx[test[0]] - H - 1]
        lo = (pd.Timestamp(test[0]) - pd.DateOffset(months=rolling_months)).date() if rolling_months else dates[0]
        tr = P[(P.date <= cutoff) & (P.date >= lo) & P.y.notna()]
        if len(tr) > 400_000:
            tr = tr.sample(400_000, random_state=0)
        mdl = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31,
                                            min_samples_leaf=300, l2_regularization=1.0, random_state=0)
        fs = [f for f in feats if tr[f].notna().any()]   # news/mood are all-NaN pre 2024-07
        mdl.fit(tr[fs], tr.y)
        te = P[P.date.isin(test)][["date", "sym", "ex"]].copy()
        te["pred"] = mdl.predict(P.loc[te.index, fs])
        out.append(te)
    return pd.concat(out)


def judge(name, T):
    ic = T.dropna(subset=["ex"]).groupby("date")[["pred", "ex"]].apply(lambda g: g.pred.corr(g.ex, method="spearman"))
    ic_t = ic.mean() / ic.std() * np.sqrt(len(ic) / H)
    td = sorted(T.date.unique())
    wk = []
    for d in td[::H]:
        g = T[T.date == d].dropna(subset=["ex"])
        if len(g) >= 4 * TOPN:
            wk.append((g.nlargest(TOPN, "pred").ex.clip(-0.95, 3).mean() - 0.001,
                       g.nsmallest(TOPN, "pred").ex.clip(-0.95, 3).mean() - 0.001))
    wk = np.array(wk)
    wt = lambda a: a.mean() / a.std(ddof=1) * np.sqrt(len(a))
    # daily top-N entries through the harness bar
    rk = T.groupby("date").pred.rank(ascending=False)
    top = T[rk <= TOPN]
    ent_mask = pd.DataFrame(False, index=c.index, columns=c.columns)
    for d, g in top.groupby("date"):
        ent_mask.loc[d, g.sym.values] = True
    s = ph.evaluate(ent_mask, o, c, H)
    ok, why = ph.verdict(s)
    print(f"{name:34s} IC {ic.mean():+.4f} (t {ic_t:+.2f})  wk top{TOPN} {wk[:,0].mean()*100:+.2f}%/wk vs SPY "
          f"(t {wt(wk[:,0]):+.2f}, n {len(wk)})  bottom{TOPN} {wk[:,1].mean()*100:+.2f} (t {wt(wk[:,1]):+.2f})\n"
          f"{'':34s} harness top{TOPN}/day: n {s.get('n')} mean {s.get('mean',0):+.2f}% med {s.get('median',0):+.2f}% "
          f"t {s.get('t_clustered',0):+.2f} halves {s.get('half1',0):+.2f}/{s.get('half2',0):+.2f} "
          f"yrs {s.get('years')}  -> {why}", flush=True)
    return ok


passes = []
for lab, feats in FEATS.items():
    for scheme, rm in (("expanding", None), ("rolling12", 12)):
        T = walk(feats, rm)
        passes.append((f"{lab} / {scheme}", judge(f"{lab} / {scheme}", T)))
        if lab == "price+regime+news" and scheme == "rolling12":
            T.to_pickle("ml_v2_preds.pkl")

# market-timing view: does the regime predict SPY itself? (the ranker cannot)
sp5 = (spy.shift(-H) / spy - 1) * 100
for lab, m in (("VIX >= 25", REG.vix >= 25), ("VIX/VIX3M > 1.0", REG.vix_ratio > 1.0),
               ("breadth50 < 0.30", REG.breadth50 < 0.30), ("mood_z >= 2", REG.mood_z >= 2)):
    x = sp5[m & (c.index >= FIRST_TEST)].dropna()
    print(f"SPY 5d | {lab:18s} n {len(x):3d} mean {x.mean():+.2f}% vs all {sp5[c.index >= FIRST_TEST].mean():+.2f}%")
print("\nVERDICT:", ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in passes), f"\ndone {time.time()-t0:.0f}s")
