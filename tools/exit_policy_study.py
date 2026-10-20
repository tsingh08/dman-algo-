"""Replay the labelled signals on real minute paths and price the exit knobs.

    python tools/exit_policy_study.py [cache.pkl]

Run it when the label count has grown, or after a setup is re-enabled, and
read the delta column. It is a MEASUREMENT, not a recommendation: see the
2026-10-20 result in its commit for why the winning row (the tightest trail)
is the one to distrust most.

The recorded blocker was "the sim cannot pick the replacement (biased, never
turns over) -- needs minute paths". Alpaca serves historical minute bars, so
this fetches the actual intraday path of every labelled signal and runs each
candidate exit policy over it, in order, bar by bar. No daily-bar
approximation: a stop and a lock can both be touched inside one daily candle
and only the minute sequence says which came first.
"""
import contextlib, io, json, os, pickle, sys, datetime as dt
from datetime import timezone as tz

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

CACHE = (sys.argv[1] if len(sys.argv) > 1
         else os.path.join(_ROOT, "dman_minute_paths.pkl"))
HORIZON_SESSIONS = 5


def load_signals():
    rows = a._dedupe_feature_rows(json.load(open('dman_signal_features.json')))
    out = []
    for r in rows:
        if r.get('label_pnl_pct') is None:
            continue
        try:
            e, s = float(r.get('entry') or 0), float(r.get('stop') or 0)
            t1 = float(r.get('target1') or 0)
        except (TypeError, ValueError):
            continue
        if e <= 0 or s <= 0 or s >= e:          # LONG only; shorts are not traded
            continue
        out.append({'ticker': r['ticker'], 'date': r['date'], 'entry': e,
                    'stop': s, 'target1': t1,
                    'actual': float(r['label_pnl_pct']),
                    'mfe': float(r.get('label_mfe_pct') or 0),
                    'setup': r.get('setup', '')})
    return out


def fetch_paths(sigs):
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, 'rb'))
    dc = a.get_alpaca_data_client()
    paths = {}
    for i, s in enumerate(sigs):
        key = (s['ticker'], s['date'])
        if key in paths:
            continue
        d0 = dt.date.fromisoformat(s['date'])
        try:
            r = dc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=[s['ticker']], timeframe=TimeFrame.Minute,
                adjustment=Adjustment.SPLIT,
                start=dt.datetime.combine(d0, dt.time(0, 0), tzinfo=tz.utc),
                end=dt.datetime.combine(d0 + dt.timedelta(days=12),
                                        dt.time(23, 59), tzinfo=tz.utc)))
            df = r.df
            if df is None or len(df) == 0:
                paths[key] = None
                continue
            bars = [(str(ix[1])[:10], float(row.high), float(row.low), float(row.close))
                    for ix, row in df.iterrows()]
            paths[key] = bars
        except Exception as exc:
            paths[key] = None
            print('fetch fail', key, type(exc).__name__, str(exc)[:60])
        if (i + 1) % 20 == 0:
            print('fetched %d/%d' % (i + 1, len(sigs)), flush=True)
    pickle.dump(paths, open(CACHE, 'wb'))
    return paths


def session_window(bars, entry_date):
    """Bars from the session AFTER entry_date through HORIZON_SESSIONS of them.

    The signal is known at entry_date's close; the position is live from the
    next session. Same convention as the daily labeller.
    """
    days = sorted({b[0] for b in bars})
    after = [d for d in days if d > entry_date][:HORIZON_SESSIONS]
    if not after:
        return []
    keep = set(after)
    return [b for b in bars if b[0] in keep]


def replay(bars, entry, stop, target1, lock_gain=None, trail_pct=None):
    """Walk the path in order. Returns (pct, reason).

    Within one minute bar the low is assumed hit before the high -- the
    pessimistic order. A bar that spans both a stop and a target is the one
    place a replay can flatter itself, and assuming the good side first is
    how a backtest invents edge.
    """
    cur_stop, peak, locked = stop, entry, False
    for _d, hi, lo, _c in bars:
        if lo <= cur_stop:
            return (cur_stop - entry) / entry * 100.0, 'stop'
        if target1 and hi >= target1:
            return (target1 - entry) / entry * 100.0, 'target'
        peak = max(peak, hi)
        gain = (peak - entry) / entry * 100.0
        if lock_gain is not None and not locked and gain >= lock_gain:
            cur_stop, locked = max(cur_stop, entry), True
        if trail_pct is not None and gain > 0:
            cur_stop = max(cur_stop, peak * (1 - trail_pct / 100.0))
    return (bars[-1][3] - entry) / entry * 100.0, 'horizon'


def summarise(name, results):
    pls = [r[0] for r in results]
    n = len(pls)
    if not n:
        return None
    wins = sum(1 for x in pls if x > 0)
    reasons = {}
    for _p, why in results:
        reasons[why] = reasons.get(why, 0) + 1
    return (name, n, sum(pls) / n, sorted(pls)[n // 2], 100.0 * wins / n,
            reasons, sum(pls))


def main():
    sigs = load_signals()
    print('LONG labelled signals with a usable stop: %d' % len(sigs))
    paths = fetch_paths(sigs)
    usable = []
    for s in sigs:
        bars = paths.get((s['ticker'], s['date']))
        if not bars:
            continue
        w = session_window(bars, s['date'])
        if len(w) < 50:
            continue
        usable.append((s, w))
    print('with a replayable minute path: %d' % len(usable))
    if not usable:
        return

    rows = []
    rows.append(summarise('as-traded (fixed stop + T1)',
                          [replay(w, s['entry'], s['stop'], s['target1'])
                           for s, w in usable]))
    rows.append(summarise('no target, stop only',
                          [replay(w, s['entry'], s['stop'], 0)
                           for s, w in usable]))
    for g in (2, 3, 4, 5, 6, 8, 10, 15):
        rows.append(summarise('breakeven lock at +%d%%' % g,
                              [replay(w, s['entry'], s['stop'], s['target1'],
                                      lock_gain=g) for s, w in usable]))
    for t in (2, 3, 4, 5, 6, 8, 10):
        rows.append(summarise('trail %d%% from peak' % t,
                              [replay(w, s['entry'], s['stop'], s['target1'],
                                      trail_pct=t) for s, w in usable]))
    for g, t in ((3, 4), (3, 6), (5, 4), (5, 6)):
        rows.append(summarise('lock +%d%% then trail %d%%' % (g, t),
                              [replay(w, s['entry'], s['stop'], s['target1'],
                                      lock_gain=g, trail_pct=t)
                               for s, w in usable]))

    print()
    print('%-30s %4s %8s %8s %6s  %s' % ('policy', 'n', 'mean', 'median', 'win', 'exits'))
    for r in rows:
        if not r:
            continue
        name, n, mean, med, win, reasons, tot = r
        rs = ' '.join('%s=%d' % (k, v) for k, v in sorted(reasons.items()))
        print('%-30s %4d %+7.2f%% %+7.2f%% %5.0f%%  %s' % (name, n, mean, med, win, rs))

    base = rows[0]
    print()
    print('as-traded mean %+.2f%% over n=%d; deltas vs that:' % (base[2], base[1]))
    for r in sorted([x for x in rows[1:] if x], key=lambda x: -x[2])[:6]:
        print('   %-30s %+.2f pts' % (r[0], r[2] - base[2]))


main()
