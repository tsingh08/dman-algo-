"""Two edge candidates, judged by the pit_harness bar (mean > 0, t >= 2, both
halves > 0, every year/period > 0). Written 2026-11-07.

  vix   : VIX-spike SPY bounce. Index level, 1993+ (yfinance SPY total return,
          ^VIX, ^VIX3M from 2006). Event = first close at/above a threshold,
          declustered (no new event for `hold` sessions). Enter next OPEN, exit
          close t+hold. Excess = event return minus SPY's unconditional mean
          return over the same hold (the "just own SPY" opportunity).
  pairs : stock-vs-sector-ETF residual mean reversion (the Alpaca article's
          cointegration idea). Each liquid stock is matched to the sector
          ETF it is most correlated with over the trailing 252d (point in
          time, refit monthly), beta from the same window. Signal: 5d
          residual return z <= -2 vs its own 252d history, optionally only
          when the 252d log-price spread passes a Dickey-Fuller test (t <
          -2.86, 5%). Long stock at t+1 open, exit t+hold close; judged LONG-
          ONLY vs SPY (ph.evaluate) and HEDGED (stock minus beta x ETF).

RESULTS 2026-11-07 (0.10% RT long, 0.30% RT short):
  vix   : VIX>=30 NEAR MISS, FAIL. Excess vs SPY +0.73/+1.02/+1.41% at
          5/10/20d (n 155/99/64), every decade and both halves positive, t
          1.89/1.80/1.90 < 2. Every other threshold/filter weaker.
  pairs : LONG the sector laggard (resid5 z <= -2, with/without DF coint,
          z <= -3): all FAIL (~0 vs SPY).
  fade  : SHORT the sector OUTRUNNER (resid5 z >= k), vs SPY, $25M+ ADV,
          fit 126d (2024-05..2026-10, wider sample):
            z>=2 h5  +0.38% t 3.29 | h10 +0.72% t 6.26  (every year > 0)
            z>=3 h5  +1.17% t 5.57 | h10 +2.24% t 8.89  (every year > 0)
          fit 252d (2025-26) agrees. Matches the news study's good-news fade.
          CAVEATS -- why this is SHADOW-ONLY, not live:
            * liquidity-concentrated: $250M+ ADV FAILS (~0); $100M+ marginal
              (h10 z>=2 +0.42 t 2.43, 2nd half +0.01) -> the edge lives where
              borrow is dearest; 0.30% RT may undercount borrow fees.
            * tail: worst single trade -550%, p1 -24..-39% (squeezes). Close
              -to-close only; NO stop-loss path test yet (cache has no highs).
            * executable top-3 $100M+ weekly book: t 1.65 (126d) / 3.03
              (252d), worst week -16..-48%.
            * "hedged" lines are INVALID: the matched ETF can change at a
              month boundary between entry and exit (mixed-ETF prices).
          NEXT: daily-high stop test + borrow sensitivity, then a forward
          shadow (30 trades) before any real short.

Usage: py -3 tools/edge_study.py vix | pairs | fade
"""
import os, sys, pickle, datetime as dt
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pit_harness as ph

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH", "XBI", "KRE", "XHB", "XRT"]


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


# --------------------------------------------------------------------- vix
def run_vix():
    import yfinance as yf
    p = "edge_vix.pkl"
    if os.path.exists(p):
        d = pd.read_pickle(p)
    else:
        d = yf.download(["SPY", "^VIX", "^VIX3M"], start="1993-01-01", auto_adjust=True, progress=False)
        d.to_pickle(p)
    o, c = d["Open"]["SPY"], d["Close"]["SPY"]
    vix, ratio = d["Close"]["^VIX"], d["Close"]["^VIX"] / d["Close"]["^VIX3M"]
    rules = {f"VIX >= {k}": vix >= k for k in (25, 30, 35, 40)}
    rules["VIX/VIX3M > 1.10"] = ratio > 1.10
    rules["VIX 5d chg >= +50%"] = vix.pct_change(5) >= 0.5
    rules["VIX>=30 & SPY>MA200"] = (vix >= 30) & (c > c.rolling(200).mean())
    rules["VIX>=30 & SPY<MA200"] = (vix >= 30) & (c < c.rolling(200).mean())
    for hold in (5, 10, 20):
        fwd = c.shift(-hold) / o.shift(-1) - 1
        base = fwd.mean()
        print(f"\n=== hold {hold}d  (SPY unconditional mean {base*100:+.2f}%, 1993+) ===")
        for lab, m in rules.items():
            idx, last = [], -10**9
            for i in np.flatnonzero(m.fillna(False).values):
                if i - last > hold:
                    idx.append(i); last = i
            ev = fwd.iloc[idx].dropna()
            if len(ev) < 8:
                print(f"  {lab:22s} n {len(ev)} (too few)"); continue
            ex = ev - base
            h = len(ex) // 2
            per = ex.groupby(ex.index.year // 10 * 10).mean() * 100
            ok = ex.mean() > 0 and tstat(ex) >= 2 and ex.iloc[:h].mean() > 0 and ex.iloc[h:].mean() > 0 and (per > 0).all()
            print(f"  {lab:22s} n {len(ev):3d} mean {ev.mean()*100:+.2f}% excess {ex.mean()*100:+.2f} "
                  f"t {tstat(ex):+.2f} up {(ev>0).mean():.0%} worst {ev.min()*100:+.1f}% halves "
                  f"{ex.iloc[:h].mean()*100:+.2f}/{ex.iloc[h:].mean()*100:+.2f} decades "
                  f"{ {int(k): round(v, 2) for k, v in per.items()} } {'PASS' if ok else 'FAIL'}")


# ------------------------------------------------------------------- pairs
def sector_bars():
    p = "edge_sectors.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import Adjustment, DataFeed
    sc = StockHistoricalDataClient(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"))
    df = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=SECTORS, timeframe=TimeFrame.Day,
                                            start=ph.START, end=dt.datetime(2026, 10, 31),
                                            adjustment=Adjustment.SPLIT, feed=DataFeed.SIP)).df.reset_index()
    df["date"] = df.timestamp.dt.tz_convert("America/New_York").dt.date
    out = tuple(df.pivot_table(index="date", columns="symbol", values=v) for v in ("open", "close"))
    pickle.dump(out, open(p, "wb"))
    return out


def df_t(y):
    """Dickey-Fuller t-stat (constant, no lags) of series y."""
    y = y[np.isfinite(y)]
    if len(y) < 120:
        return np.nan
    dy, yl = np.diff(y), y[:-1]
    X = np.column_stack([np.ones_like(yl), yl])
    b, res, *_ = np.linalg.lstsq(X, dy, rcond=None)
    e = dy - X @ b
    s2 = e @ e / (len(dy) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return b[1] / se


def build(W=252, zmin=150):
    """Point-in-time sector match/beta/DF + 5d residual z. Cached per params."""
    cp = f"edge_pairs_W{W}_z{zmin}.pkl"
    if os.path.exists(cp):
        return pickle.load(open(cp, "rb"))
    o, c, dv = ph.load_bars()
    so, sc_ = sector_bars()
    liq = (c >= 10) & (ph._adv(dv) >= 25e6)
    liq["SPY"] = False
    ever = liq.any()
    cols = list(ever.index[ever.values])
    o, c, liq = o[cols + ["SPY"]], c[cols + ["SPY"]], liq[cols + ["SPY"]]
    sc_ = sc_.reindex(c.index).ffill()
    so = so.reindex(c.index)
    r, sr = c[cols].pct_change(fill_method=None), sc_.pct_change(fill_method=None)
    dates = list(c.index)
    # monthly refit: sector match, beta, DF stat (all trailing, point in time)
    month_starts = [i for i in range(W, len(dates)) if dates[i].month != dates[i - 1].month]
    match = pd.DataFrame(index=c.index, columns=cols, dtype=object)
    beta = pd.DataFrame(np.nan, index=c.index, columns=cols)
    dfs = pd.DataFrame(np.nan, index=c.index, columns=cols)
    for k, i in enumerate(month_starts):
        j = month_starts[k + 1] if k + 1 < len(month_starts) else len(dates)
        win_r, win_s = r.iloc[i - W:i], sr.iloc[i - W:i]
        ok = win_r.notna().sum() >= int(0.8 * W)
        cs = [s for s in cols if ok[s]]
        corr = pd.DataFrame({e: win_r[cs].corrwith(win_s[e]) for e in SECTORS if win_s[e].notna().sum() > int(0.8 * W)})
        best = corr.dropna(how="all").idxmax(axis=1)
        for s, e in best.items():
            x, y = win_s[e].values, win_r[s].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < int(0.6 * W):
                continue
            b = np.cov(y[m], x[m])[0, 1] / np.var(x[m], ddof=1)
            spread = np.log(c[s].iloc[i - W:i].values) - b * np.log(sc_[e].iloc[i - W:i].values)
            match.iloc[i:j, match.columns.get_loc(s)] = e
            beta.iloc[i:j, beta.columns.get_loc(s)] = b
            dfs.iloc[i:j, dfs.columns.get_loc(s)] = df_t(spread)
        print(f"  fit {dates[i]} {len(cs)} names", flush=True)
    # residual returns with the (point-in-time) matched ETF
    etf_r = pd.DataFrame(np.nan, index=c.index, columns=cols)
    etf_fwd_close = pd.DataFrame(np.nan, index=c.index, columns=cols)
    etf_open = pd.DataFrame(np.nan, index=c.index, columns=cols)
    for e in SECTORS:
        msk = match == e
        etf_r = etf_r.mask(msk, pd.DataFrame({s: sr[e] for s in cols}))
        etf_open = etf_open.mask(msk, pd.DataFrame({s: so[e] for s in cols}))
        etf_fwd_close = etf_fwd_close.mask(msk, pd.DataFrame({s: sc_[e] for s in cols}))
    resid = r - beta * etf_r
    res5 = resid.rolling(5).sum()
    z = (res5 - res5.rolling(W, min_periods=zmin).mean()) / res5.rolling(W, min_periods=zmin).std()
    base = liq[cols] & beta.notna()
    out = dict(o=o, c=c, cols=cols, z=z, beta=beta, dfs=dfs, base=base, adv=ph._adv(dv)[cols],
               etf_open=etf_open, etf_close=etf_fwd_close)
    pickle.dump(out, open(cp, "wb"))
    return out


def run_pairs():
    B = build()
    o, c, cols, z, beta, dfs, base = (B[k] for k in ("o", "c", "cols", "z", "beta", "dfs", "base"))
    etf_open, etf_fwd_close = B["etf_open"], B["etf_close"]
    cells = {
        "resid5 z <= -2":                base & (z <= -2),
        "resid5 z <= -2 & DF coint":     base & (z <= -2) & (dfs < -2.86),
        "resid5 z <= -3":                base & (z <= -3),
        "resid5 z <= -2 & sector up 5d": base & (z <= -2) & (etf_fwd_close / etf_fwd_close.shift(5) > 1),
        "resid5 z >= +2 (fade check)":   base & (z >= 2),
    }
    for hold in (5, 10):
        ent_s, ext_s = o[cols].shift(-1), c[cols].shift(-hold)
        ent_e, ext_e = etf_open.shift(-1), etf_fwd_close.shift(-hold)
        # hedged: long stock, short beta x ETF (the ETF matched as of day t)
        hedged = (ext_s / ent_s - 1) - beta * (ext_e / ent_e - 1) - 0.002
        print(f"\n=== hold {hold}d ===")
        for lab, m in cells.items():
            full = m.reindex(columns=c.columns, fill_value=False)
            s = ph.evaluate(full, o, c, hold)
            ok, why = ph.verdict(s)
            ht = hedged.where(m).stack().dropna()
            ht = ht[np.isfinite(ht)].clip(-0.9, 2)
            hd = ht.groupby(level=0).mean()
            hy = ht.groupby(lambda k: str(k[0])[:4]).mean() * 100
            mid = len(hd) // 2
            hok = hd.mean() > 0 and tstat(hd) >= 2 and hd.iloc[:mid].mean() > 0 and hd.iloc[mid:].mean() > 0 and (hy > 0).all()
            print(f"  {lab:30s} LONG vs SPY n {s.get('n',0):6d} mean {s.get('mean',0):+.2f} med {s.get('median',0):+.2f} "
                  f"t {s.get('t_clustered',0):+.2f} yrs {s.get('years')} {why[:4]}\n"
                  f"  {'':30s} HEDGED      n {len(ht):6d} mean {ht.mean()*100:+.2f} med {ht.median()*100:+.2f} "
                  f"t {tstat(hd):+.2f} halves {hd.iloc[:mid].mean()*100:+.2f}/{hd.iloc[mid:].mean()*100:+.2f} "
                  f"yrs { {k: round(v, 2) for k, v in hy.items()} } {'PASS' if hok else 'FAIL'}", flush=True)


def run_fade():
    """SHORT the stock that outran its sector (resid5 z >= k). Unclipped, so a
    squeeze counts in full. Short P/L = -(stock ret) [+ beta x ETF if hedged]
    - 0.30% round trip (spread + slippage + a borrow allowance)."""
    COST = 0.003
    for W, zmin in ((252, 150), (126, 60)):
        B = build(W, zmin)
        o, c, cols, z, beta, base, adv = (B[k] for k in ("o", "c", "cols", "z", "beta", "base", "adv"))
        print(f"##### fit window {W}d, z warmup {zmin} #####")
        for hold in (5, 10):
            rs = c[cols].shift(-hold) / o[cols].shift(-1) - 1
            re_ = B["etf_close"].shift(-hold) / B["etf_open"].shift(-1) - 1
            spy = (c["SPY"].shift(-hold) / o["SPY"].shift(-1) - 1)
            legs = {"short vs SPY (short stk, long SPY)": -rs.sub(spy, axis=0) - COST,
                    "short hedged (short stk, long b*ETF)": -(rs - beta.clip(0, 3) * re_) - COST}
            for kz, liqlab, lm in ((2, "$25M+", None), (3, "$25M+", None), (2, "$100M+", 100e6), (2, "$250M+", 250e6)):
                m = base & (z >= kz) & (adv >= lm if lm else True)
                for lab, pl in legs.items():
                    t_ = pl.where(m).stack().dropna()
                    t_ = t_[np.isfinite(t_)]
                    if len(t_) < 50:
                        continue
                    d = t_.groupby(level=0).mean()
                    yr = t_.groupby(lambda k: str(k[0])[:4]).mean() * 100
                    mid = len(d) // 2
                    ok = d.mean() > 0 and tstat(d) >= 2 and d.iloc[:mid].mean() > 0 and d.iloc[mid:].mean() > 0 and (yr > 0).all()
                    print(f"  h{hold:2d} z>={kz} {liqlab:6s} {lab:38s} n {len(t_):6d} days {len(d):3d} mean {t_.mean()*100:+.2f} "
                          f"med {t_.median()*100:+.2f} win {(t_>0).mean():.0%} worst {t_.min()*100:+.0f}% p1 {t_.quantile(.01)*100:+.1f} "
                          f"t {tstat(d):+.2f} halves {d.iloc[:mid].mean()*100:+.2f}/{d.iloc[mid:].mean()*100:+.2f} "
                          f"yrs { {k: round(v, 2) for k, v in yr.items()} } {'PASS' if ok else 'FAIL'}", flush=True)
            # executable book: top-3 by z each WEEK (non-overlapping), hedged with SPY
            if hold == 5:
                pl = legs["short vs SPY (short stk, long SPY)"]
                wk = []
                for d0 in list(z.index)[::5]:
                    zz = z.loc[d0].where(base.loc[d0] & (adv.loc[d0] >= 100e6)).dropna()
                    zz = zz[zz >= 2]
                    if len(zz) >= 3:
                        v = pl.loc[d0, zz.nlargest(3).index].dropna()
                        if len(v):
                            wk.append((d0, v.mean()))
                if wk:
                    w = pd.Series(dict(wk))
                    mid = len(w) // 2
                    print(f"  BOOK top-3 z>=2 $100M+ weekly, short vs SPY: weeks {len(w)} mean {w.mean()*100:+.2f}%/wk "
                          f"t {tstat(w):+.2f} win wks {(w>0).mean():.0%} worst wk {w.min()*100:+.1f}% halves "
                          f"{w.iloc[:mid].mean()*100:+.2f}/{w.iloc[mid:].mean()*100:+.2f} cum {((1+w).prod()-1)*100:+.1f}%")


if __name__ == "__main__":
    {"vix": run_vix, "pairs": run_pairs, "fade": run_fade}[sys.argv[1] if len(sys.argv) > 1 else "vix"]()
