"""Characterization harness for the live-scored backtest pipeline.

Covers _run_pro_backtest_impl, _raw_signals and score_signal together: the
backtest replays every bar of a fixed ticker set through the live scoring
path point-in-time, and every trade record plus printed line is compared to
a baseline.

    python golden/backtest_harness.py capture   # once, network ON: freeze bars
    python golden/backtest_harness.py record    # network OFF, replay frozen bars
    python golden/backtest_harness.py check

The frozen data (golden/fixtures/backtest_bars.pkl) is local only; this
harness is not wired into CI.
"""
import sys, io, os, re, json, pickle, socket, tempfile, contextlib
import datetime as _dt, zoneinfo as _zi
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unittest.mock import patch
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
import yfinance
os.environ.pop("ACCOUNT_SIZE", None)

TICKERS = ["NVDA", "PLTR", "TSLA", "AMD", "SMCI", "COIN", "MSTR", "HOOD", "SOFI", "RIVN",
           "META", "AAPL"]
YEARS = 2
FIX = os.path.join(HERE, "fixtures", "backtest_bars.pkl")
OUT = os.path.join(HERE, "backtest.golden.json")
_ET = _zi.ZoneInfo("America/New_York")
_FIXED = _dt.datetime(2026, 9, 11, 16, 30, tzinfo=_ET)   # a Friday after the close


class _FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED.astimezone(tz) if tz is not None else _FIXED.replace(tzinfo=None)

    @classmethod
    def utcnow(cls):
        return _FIXED.astimezone(_dt.timezone.utc).replace(tzinfo=None)

    @classmethod
    def today(cls):
        return cls.now()


class _FixedDate(_dt.date):
    @classmethod
    def today(cls):
        return _FIXED.date()


def main(mode):
    capture = mode == "capture"
    store = {} if capture else pickle.load(open(FIX, "rb"))
    misses = []

    def frozen(name, fn):
        def _w(*ar, **kw):
            key = (name, repr(ar), repr(sorted(kw.items())))
            if capture:
                if key not in store:
                    try:
                        store[key] = ("ok", fn(*ar, **kw))
                    except Exception as e:
                        store[key] = ("err", e)
            if key not in store:
                misses.append(key)
                raise OSError(f"harness: no frozen data for {key[:2]}")
            kind, val = store[key]
            if kind == "err":
                raise val
            return val.copy() if hasattr(val, "copy") else val
        return _w

    def _no_net(*_a, **_k):
        raise OSError("network disabled in harness")

    tmp = tempfile.mkdtemp(prefix="dman_bt_harness_")
    trk = os.path.join(tmp, "tracker.json")

    class _T(a.WinRateTracker):
        def __init__(self, filepath=None, *x, **k):
            super().__init__(trk, *x, **k)

    seen = []
    _raw, _score = a._raw_signals, a.score_signal

    def _snap(obj):
        d = vars(obj) if hasattr(obj, "__dict__") else {"value": obj}
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in sorted(d.items())
                if isinstance(v, (int, float, str, bool, type(None)))}

    def raw_rec(df, ticker, *x, **k):
        r = _raw(df, ticker, *x, **k)
        if r is not None:
            seen.append(["raw", ticker, str(df.index[-1])[:10], _snap(r)])
        return r

    def score_rec(sig, df, *x, **k):
        r = _score(sig, df, *x, **k)
        seen.append(["score", getattr(sig, "ticker", "?"), str(df.index[-1])[:10], _snap(r)])
        return r

    ps = [patch.object(a, "_raw_signals", raw_rec),
          patch.object(a, "score_signal", score_rec),
          patch.object(a, "datetime", _FixedDatetime),
          patch.object(a, "date", _FixedDate),
          patch.object(a, "_et_today", lambda: _FIXED.date()),
          patch.object(a.time, "sleep", lambda *_: None),
          patch.object(a, "fetch_df", frozen("fetch_df", a.fetch_df)),
          patch.object(a, "_fetch_massive_earnings", frozen("earn", a._fetch_massive_earnings)),
          patch.object(yfinance, "download", frozen("yf.download", yfinance.download)),
          patch.object(a, "WinRateTracker", _T),
          patch.object(a, "send_telegram", lambda *_a, **_k: None)]
    if not capture:
        ps.append(patch.object(socket.socket, "connect", _no_net))
    for p in ps:
        p.start()
    buf = io.StringIO()
    how = "returned"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                a.run_pro_backtest(tickers=TICKERS, years=YEARS, live_scoring=True)
            except Exception as e:
                how = f"{type(e).__name__}: {e}"[:200]
    finally:
        for p in reversed(ps):
            p.stop()
    recs = json.load(open(trk)) if os.path.exists(trk) else []
    if isinstance(recs, dict):
        recs = recs.get("records", recs)
    out = buf.getvalue()
    out = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", out)
    out = re.sub(r"\b\d+(\.\d+)?\s*s(ec(ond)?s?)?\b(?= ?(elapsed|\)|$))", "Ns", out, flags=re.M)
    res = {"how": how, "records": recs, "signals": seen,
           "stdout": [l.rstrip() for l in out.splitlines() if l.strip()]}

    if capture:
        os.makedirs(os.path.dirname(FIX), exist_ok=True)
        pickle.dump(store, open(FIX, "wb"))
        print(f"captured {len(store)} data calls -> {FIX}; how={how}; trades={len(recs)}")
        return
    print(f"frozen-data misses: {len(misses)}  trades={len(recs)}  signals={len(seen)}  how={how[:80]}")
    if mode == "record":
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
        print("recorded", OUT)
        return
    gold = json.load(open(OUT, encoding="utf-8"))
    res = json.loads(json.dumps(res, ensure_ascii=False, default=str))
    bad = [k for k in res if gold.get(k) != res[k]]
    import difflib
    for k in bad:
        g = [json.dumps(x, ensure_ascii=False) for x in gold[k]] if isinstance(gold[k], list) else [str(gold[k])]
        r = [json.dumps(x, ensure_ascii=False) for x in res[k]] if isinstance(res[k], list) else [str(res[k])]
        print("DIFF", k)
        for d in list(difflib.unified_diff(g, r, lineterm="", n=0))[2:10]:
            print("    ", d[:200])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")


main(sys.argv[1] if len(sys.argv) > 1 else "check")
