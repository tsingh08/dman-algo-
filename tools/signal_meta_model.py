"""Meta-model on the algo's OWN labelled signals (dman_signal_features.json).

Question: given a signal the scanner already produced, can a model trained
only on EARLIER signals say which ones to take?  This is meta-labelling, not
the cross-sectional stock ranker of ml_walkforward_study.py / ml_v2_study.py
(both failed walk-forward).

Honesty rules, same as the other studies:
  * one row per (ticker, date, setup) -- the log repeats a signal on every
    scan, so the raw 771 rows are ~241 independent signals;
  * expanding walk-forward BY DATE: a day's signals are scored by a model
    that has seen only strictly earlier days whose labels had resolved
    (LABEL_LAG sessions back);
  * the pick rule is fixed before looking: predicted P/L in the top
    TOP_FRAC of the TRAINING predictions' distribution;
  * t-stat on DAILY means (signals on one day are not independent);
  * pass bar: picked mean > 0, t >= 2, positive in both halves, n >= 30.

Usage:  py tools/signal_meta_model.py [--json out.json]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES_FILE = ROOT / "dman_signal_features.json"
MIN_TRAIN_DAYS = 12
LABEL_LAG = 5          # sessions before a label is knowable (5-day horizon)
TOP_FRAC = 0.30
TIER = {"A": 3, "B": 2, "C": 1, "D": 0, "": 0}


def load_rows() -> list[dict]:
    raw = json.load(open(FEATURES_FILE, encoding="utf-8"))
    seen, out = set(), []
    for r in sorted(raw, key=lambda x: x.get("ts", "")):
        if r.get("label_pnl_pct") is None or not r.get("label_outcome"):
            continue
        k = (r["ticker"], r["date"], r["setup"])
        if k in seen:
            continue                       # keep the FIRST sighting only
        seen.add(k)
        out.append(r)
    return out


def featurize(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    bkeys = sorted({k for r in rows for k in (r.get("breakdown") or {})})
    setups = sorted({r["setup"] for r in rows})
    names = (["rr", "rsi", "rvol", "atr_pct", "stop_pct", "beta", "regime_score",
              "mtf_ok", "earnings_ok", "news_boost", "tier", "bull", "hour"]
             + [f"setup={s}" for s in setups] + [f"bd:{k}" for k in bkeys])
    X = []
    for r in rows:
        e = float(r.get("entry") or 0) or 1.0
        bd = r.get("breakdown") or {}
        try:
            hour = int(r["ts"][11:13]) + int(r["ts"][14:16]) / 60.0
        except Exception:
            hour = 10.0
        X.append([
            min(float(r.get("rr") or 0), 10.0), float(r.get("rsi") or 50),
            min(float(r.get("rvol") or 1), 20.0),
            float(r.get("atr") or 0) / e * 100, abs(e - float(r.get("stop") or e)) / e * 100,
            float(r.get("beta") or 1), float(r.get("regime_score") or 0),
            float(bool(r.get("mtf_ok"))), float(bool(r.get("earnings_ok"))),
            float(bool(r.get("news_boost"))), TIER.get(str(r.get("catalyst_tier") or ""), 0),
            float(r.get("regime") == "BULL"), hour,
            *[float(r["setup"] == s) for s in setups],
            *[float(bd.get(k) or 0) for k in bkeys],
        ])
    return np.nan_to_num(np.array(X, dtype=float)), names


def tstat_daily(pnl: np.ndarray, days: list[str]) -> tuple[float, float, int]:
    g = defaultdict(list)
    for p, d in zip(pnl, days):
        g[d].append(p)
    dm = np.array([np.mean(v) for v in g.values()])
    if len(dm) < 3 or dm.std(ddof=1) == 0:
        return float(np.mean(pnl)) if len(pnl) else 0.0, 0.0, len(dm)
    return float(pnl.mean()), float(dm.mean() / (dm.std(ddof=1) / math.sqrt(len(dm)))), len(dm)


def r_multiple(r: dict) -> float:
    """P/L in units of the signal's own stop distance. The raw % label is
    mechanically tied to stop width (a tight stop caps the loss), so a model
    on raw % learns "tight stop = small loss", which is risk, not edge."""
    e = float(r.get("entry") or 0) or 1.0
    sd = abs(e - float(r.get("stop") or e)) / e * 100
    return float(r["label_pnl_pct"]) / max(sd, 0.5)


def walk_forward(rows, X, make_model, target=None) -> dict:
    target = target or (lambda r: float(r["label_pnl_pct"]))
    y = np.clip(np.array([target(r) for r in rows]), -30, 30)
    days = sorted({r["date"] for r in rows})
    d_idx = np.array([days.index(r["date"]) for r in rows])
    pred = np.full(len(rows), np.nan)
    pick = np.zeros(len(rows), dtype=bool)
    for i in range(MIN_TRAIN_DAYS + LABEL_LAG, len(days)):
        tr = d_idx <= i - LABEL_LAG - 1
        te = d_idx == i
        if tr.sum() < 60 or not te.any():
            continue
        m = make_model().fit(X[tr], y[tr])
        cut = np.quantile(m.predict(X[tr]), 1 - TOP_FRAC)
        pred[te] = m.predict(X[te])
        pick[te] = pred[te] >= cut
    scored = ~np.isnan(pred)
    yr = np.array([target(r) for r in rows])
    dd = [r["date"] for r in rows]
    res = {"scored": int(scored.sum()), "picked": int(pick.sum())}
    res["all_mean"], res["all_t"], _ = tstat_daily(yr[scored], [d for d, s in zip(dd, scored) if s])
    if pick.sum():
        res["pick_mean"], res["pick_t"], res["pick_days"] = tstat_daily(
            yr[pick], [d for d, p in zip(dd, pick) if p])
        res["pick_win"] = float((yr[pick] > 0).mean())
        rest = scored & ~pick
        res["rest_mean"] = float(yr[rest].mean()) if rest.any() else 0.0
        pd_sorted = sorted({d for d, p in zip(dd, pick) if p})
        mid = pd_sorted[len(pd_sorted) // 2]
        h1 = pick & np.array([d < mid for d in dd]); h2 = pick & ~np.array([d < mid for d in dd])
        res["half1"] = float(yr[h1].mean()) if h1.any() else 0.0
        res["half2"] = float(yr[h2].mean()) if h2.any() else 0.0
        ok = scored & ~np.isnan(pred)
        res["ic"] = float(np.corrcoef(pred[ok], yr[ok])[0, 1]) if ok.sum() > 5 else 0.0
        res["pass"] = bool(res["pick_mean"] > 0 and res["pick_t"] >= 2 and res["half1"] > 0
                           and res["half2"] > 0 and res["picked"] >= 30)
    else:
        res["pass"] = False
    return res


MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=50.0)),
    "gbm": lambda: GradientBoostingRegressor(n_estimators=80, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, min_samples_leaf=10, random_state=0),
}


def main() -> int:
    rows = load_rows()
    X, names = featurize(rows)
    days = sorted({r["date"] for r in rows})
    print(f"signals {len(rows)} (unique ticker/date/setup) over {len(days)} days "
          f"{days[0]}..{days[-1]}, {X.shape[1]} features")
    y = np.array([float(r["label_pnl_pct"]) for r in rows])
    print(f"baseline: mean {y.mean():+.2f}%  win {np.mean(y > 0):.0%}")
    out = {"n": len(rows), "days": len(days), "baseline_mean": float(y.mean())}
    yR = np.array([r_multiple(r) for r in rows])
    print(f"baseline in R: mean {yR.mean():+.2f}R")
    for name, mk, tgt in [(n, m, t) for t in ("pct", "R") for n, m in MODELS.items()]:
        r = walk_forward(rows, X, mk, r_multiple if tgt == "R" else None)
        name = f"{name}/{tgt}"
        out[name] = r
        if "pick_mean" not in r:
            print(f"{name}: nothing scored"); continue
        print(f"{name}: scored {r['scored']}  picked {r['picked']} on {r['pick_days']} days  "
              f"pick {r['pick_mean']:+.2f} (t {r['pick_t']:+.2f}, win {r['pick_win']:.0%})  "
              f"rest {r['rest_mean']:+.2f}  halves {r['half1']:+.2f}/{r['half2']:+.2f}  "
              f"IC {r['ic']:+.3f}  -> {'PASS' if r['pass'] else 'FAIL'}")
    # In-sample importance, for reading only -- never a reason to ship.
    g = MODELS["gbm"]().fit(X, np.clip(y, -30, 30))
    top = sorted(zip(g.feature_importances_, names), reverse=True)[:8]
    print("gbm in-sample importance:", ", ".join(f"{n} {v:.2f}" for v, n in top))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
