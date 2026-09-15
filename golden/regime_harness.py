"""Characterization harness for get_market_regime().

Its only inputs are fetch_df, _latest_vix3m and _news_sentiment_breadth. They
are frozen once, then replayed truncated to many historical as-of dates so
the BULL / BEAR / CHOP branches all run; the regime dict and printed lines
are compared to a baseline.

    python golden/regime_harness.py capture   # once, network ON
    python golden/regime_harness.py record    # network OFF
    python golden/regime_harness.py check
"""
import sys, io, os, json, pickle, socket, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unittest.mock import patch
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
import pandas as pd

FIX = os.path.join(HERE, "fixtures", "regime_inputs.pkl")
OUT = os.path.join(HERE, "regime.golden.json")
TICKERS = ["SPY", "^VIX", "IWM", "QQQ", "TLT", "UUP"]
N_DATES = 40


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if hasattr(x, "item") and callable(x.item):
        try:
            x = x.item()
        except (ValueError, TypeError):
            return str(x)
    if isinstance(x, float):
        return round(x, 6)
    return x if isinstance(x, (int, str, bool, type(None))) else str(x)


def main(mode):
    if mode == "capture":
        with contextlib.redirect_stdout(io.StringIO()):
            data = {t: a.fetch_df(t, period_days=2200) for t in TICKERS}
            data["_vix3m"] = a._latest_vix3m()
            try:
                data["_news"] = a._news_sentiment_breadth(hours_back=24.0)
            except Exception as e:
                data["_news"] = e
        os.makedirs(os.path.dirname(FIX), exist_ok=True)
        pickle.dump(data, open(FIX, "wb"))
        print("captured", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in data.items() if k[0] != "_"})
        return

    data = pickle.load(open(FIX, "rb"))
    spy = data["SPY"]
    idx = list(spy.index[260::max(1, (len(spy) - 260) // N_DATES)])[:N_DATES] + [spy.index[-1]]
    results = {}

    def _no_net(*_a, **_k):
        raise OSError("network disabled in harness")

    for asof in idx:
        def fdf(ticker, period_days=430, interval="1d"):
            df = data.get(ticker)
            if df is None:
                return None
            cut = df[df.index <= asof]
            return cut.tail(period_days).copy() if len(cut) else None

        def news(hours_back=24.0):
            v = data["_news"]
            if isinstance(v, Exception):
                raise v
            return v

        ps = [patch.object(a, "fetch_df", fdf),
              patch.object(a, "_latest_vix3m", lambda: data["_vix3m"]),
              patch.object(a, "_news_sentiment_breadth", news),
              patch.object(socket.socket, "connect", _no_net)]
        for cache in [k for k, v in vars(a).items() if "REGIME" in k.upper() and isinstance(v, dict)]:
            ps.append(patch.dict(getattr(a, cache), clear=True))
        for p in ps:
            p.start()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                try:
                    r = clean(a.get_market_regime())
                except Exception as e:
                    r = f"{type(e).__name__}: {e}"
        finally:
            for p in reversed(ps):
                p.stop()
        results[str(asof)[:10]] = {"result": r, "stdout": [l.rstrip() for l in buf.getvalue().splitlines() if l.strip()]}

    regimes = [v["result"].get("regime") if isinstance(v["result"], dict) else "ERR" for v in results.values()]
    print("regimes seen:", {g: regimes.count(g) for g in set(map(str, regimes))})
    if mode == "record":
        json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("recorded", OUT)
        return
    gold = json.load(open(OUT, encoding="utf-8"))
    bad = [k for k in set(gold) | set(results) if gold.get(k) != results.get(k)]
    for k in sorted(bad)[:10]:
        print("DIFF", k, "\n   gold:", str(gold.get(k))[:300], "\n   now: ", str(results.get(k))[:300])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")


main(sys.argv[1] if len(sys.argv) > 1 else "check")
