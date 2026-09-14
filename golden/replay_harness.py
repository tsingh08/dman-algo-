"""Record/replay characterization harness for large orchestration functions.

capture: runs the target ONCE for real, with every order/alert/state-writing
         callee stubbed and every state file redirected to a temp copy, and
         tapes the return value of each DIRECT module-level callee in call order.
record / check: runs the target with each direct callee replaced by its tape
         (network off). Only the target's own body executes, so every call it
         makes (name + args), every printed line and its return value are
         compared to the baseline.

    python golden/replay_harness.py scanner capture|record|check

The tape (golden/fixtures/<target>.tape.pkl) holds live account/market data and
stays local; this harness is not wired into CI.
"""
import sys, io, os, re, ast, json, pickle, shutil, socket, tempfile, contextlib
import datetime as _dt, zoneinfo as _zi
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unittest.mock import patch, MagicMock
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
os.environ.pop("ACCOUNT_SIZE", None)

_ET = _zi.ZoneInfo("America/New_York")
_FIXED = _dt.datetime(2026, 9, 11, 11, 0, tzinfo=_ET)   # Friday, mid-session

TARGETS = {
    "scanner": ("run_pro_scanner", (), {}),
    "premarket_briefing": ("run_premarket_briefing", (), {}, (8, 45)),
    "premarket_early": ("run_premarket_early_scan", (), {}, (6, 0)),
    "momentum_watch": ("run_momentum_watch", (), {}, (10, 30)),
    # scan mode at 3:55 PM Friday: EOD P&L + Friday close-out branches run too
    "main_scan": ("_main_mode_scan", (__import__("types").SimpleNamespace(
        ai=False, export=False, score=None, submit=True, universe="curated"), list(a.WATCHLIST)), {}, (15, 55)),
    # same, with the scanner's own (empty) result: the no-signal heartbeat path
    "main_scan_quiet": ("_main_mode_scan", (__import__("types").SimpleNamespace(
        ai=False, export=False, score=None, submit=True, universe="curated"), list(a.WATCHLIST)), {}, (15, 55)),
}

# Never run for real during capture: orders, alerts, broker-side stops, state
# writes, restarts. Value = what the stub returns.
SIDE = {
    "send_telegram": None, "_finalize_and_alert_signals": None, "_save_last_alert": None,
    "_append_scan_log": None, "_log_news_event": None, "_log_scan_halt": None,
    "resolve_live_outcomes": 0, "_check_stop_coverage": None, "_check_open_position_risk": None,
    "_submit_signals_to_alpaca": None, "sync_alpaca_fills": 0, "_force_close_day_only_positions": None,
    "send_account_pnl_telegram": None, "_check_and_heal_watchdog": None, "_write_json_atomic": None,
    "_mark_alerted": None, "_save_momentum_pending": None, "_monitor_option_position": None,
    "_monitor_earnings_spread_position": None, "_check_equity_position_target": None,
    "generate_strangle_advisory": None,
}


# Capture-time return overrides, to steer past account-state halts into the
# code worth characterizing (taped like any other return).
OVERRIDE = {"get_todays_loss": 0.0, "get_this_month_loss": 0.0}

def _scanner_signals():
    """Real signals: the scanner target's own replayed return value."""
    global _FIXED
    saved = _FIXED
    run("scanner", "record", pickle.load(open(os.path.join(HERE, "fixtures", "scanner.tape.pkl"), "rb")))
    _FIXED = saved
    return {"run_pro_scanner": _LAST_RET[0]}


TARGET_OVERRIDES = {"main_scan": _scanner_signals}

# Local-state classes that run for real in replay too (read temp copies only).
REAL = {"WinRateTracker", "PositionTracker", "ProSignal", "OpenPosition", "TradeRecord"}


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


def direct_callees(func):
    t = ast.parse(open(os.path.join(ROOT, "dman_algo.py"), encoding="utf-8").read())
    mod = {n.name for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    fn = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == func][0]
    return sorted({c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Name) and c.func.id in mod and c.func.id != func})


def norm(x):
    s = x if isinstance(x, str) else repr(x)
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"id='\d+'", "id=ID", s)
    s = re.sub(r"dman_(replay|harness)_[A-Za-z0-9_]+", "TMP", s)
    return s


_LAST_RET = [None]
_DEPTH = [0]   # >0 while inside a taped callee (its internals are not taped)


class _Rec:
    """Records (capture) or replays (record/check) every call chain made on a
    module object such as yfinance, keyed by the chain text, e.g.
    yf.Ticker('APLD').history(period='2d'). Picklable results are taped;
    anything else (a Ticker object) stays a proxy so its own calls are taped."""
    PROXY = "__proxy__"

    def __init__(self, obj, path, tape, mode, pos):
        self._o, self._p, self._t, self._m, self._pos = obj, path, tape, mode, pos

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        obj = getattr(self._o, name) if self._m == "capture" else None
        return _Rec(obj, f"{self._p}.{name}", self._t, self._m, self._pos)

    def _step(self, key, thunk):
        if self._m == "capture" and _DEPTH[0] > 0:
            val = thunk()
            try:
                pickle.dumps(val)
                return val
            except Exception:
                return _Rec(val, key, self._t, self._m, self._pos)
        seq = self._t.setdefault("chains", {}).setdefault(key, []) if self._m == "capture" else None
        if self._m == "capture":
            try:
                val, kind = thunk(), "ok"
            except Exception as e:
                val, kind = e, "err"
            try:
                blob = pickle.dumps(val)
            except Exception:
                blob = None
            seq.append((kind, blob if blob is not None else self.PROXY))
            if kind == "err":
                raise val
            return val if blob is not None else _Rec(val, key, self._t, self._m, self._pos)
        i = self._pos.get(key, 0)
        self._pos[key] = i + 1
        chain = self._t.get("chains", {}).get(key, [])
        if i >= len(chain):
            raise OSError(f"harness: tape exhausted for {key[:80]}")
        kind, blob = chain[i]
        if blob == self.PROXY:
            return _Rec(None, key, self._t, self._m, self._pos)
        val = pickle.loads(blob)
        if kind == "err":
            raise val
        return val

    def __call__(self, *ar, **kw):
        key = f"{self._p}({norm(ar)[:120]}, {norm(sorted(kw.items()))[:120]})"
        return self._step(key, lambda: self._o(*ar, **kw))

    def __getitem__(self, k):
        key = f"{self._p}[{norm(k)[:80]}]"
        return self._step(key, lambda: self._o[k])


def file_consts():
    return [k for k, v in vars(a).items()
            if isinstance(v, str) and re.search(r"_(FILE|PATH|LOG)$", k) and v.endswith((".json", ".csv", ".txt", ".log"))]


def run(target, mode, tape=None):
    global _FIXED
    func, args, kwargs = TARGETS[target][:3]
    if len(TARGETS[target]) > 3:
        hh, mm = TARGETS[target][3]
        _FIXED = _FIXED.replace(hour=hh, minute=mm)
    if mode == "capture" and target in TARGET_OVERRIDES:
        OVERRIDE.update(TARGET_OVERRIDES[target]())
    names = [n for n in (direct_callees(func) if mode == "capture" else tape["names"]) if n not in REAL]
    tmp = tempfile.mkdtemp(prefix="dman_replay_")
    consts = file_consts()
    if mode == "capture":
        files = {}
        for k in consts:
            src = getattr(a, k)
            src = src if os.path.isabs(src) else os.path.join(ROOT, src)
            if os.path.isfile(src):
                files[os.path.basename(src)] = open(src, "rb").read()
        tape = {"names": names, "files": files, "calls": {}}
    for fname, data in tape["files"].items():
        open(os.path.join(tmp, fname), "wb").write(data)

    calls, pos, depth = [], {}, _DEPTH
    depth[0] = 0

    def wrap(name, real):
        def _w(*ar, **kw):
            if depth[0] > 0:
                # nested inside another taped callee: never taped (replay
                # never reaches it), but side-effect stubs still apply
                if name in SIDE:
                    return SIDE[name]
                if name in OVERRIDE:
                    return OVERRIDE[name]
                return real(*ar, **kw)
            calls.append([name, norm(ar)[:200], norm(sorted(kw.items()))[:200]])
            if mode == "capture":
                depth[0] += 1
                try:
                    val = (SIDE[name] if name in SIDE else OVERRIDE[name] if name in OVERRIDE
                           else real(*ar, **kw))
                    kind = "ok"
                except Exception as e:
                    val, kind = e, "err"
                finally:
                    depth[0] -= 1
                try:
                    blob = pickle.dumps(val)
                except Exception:
                    blob = None
                tape["calls"].setdefault(name, []).append((kind, blob))
                if kind == "err":
                    raise val
                return val
            i = pos.get(name, 0)
            pos[name] = i + 1
            seq = tape["calls"].get(name, [])
            if i >= len(seq):
                raise OSError(f"harness: tape exhausted for {name}")
            kind, blob = seq[i]
            val = pickle.loads(blob) if blob is not None else MagicMock(name=f"{name}#untapeable")
            if kind == "err":
                raise val if isinstance(val, BaseException) else OSError(f"{name} raised")
            return val
        return _w

    def _no_net(*_x, **_k):
        raise OSError("network disabled in harness")

    ps = [patch.object(a, k, os.path.join(tmp, os.path.basename(getattr(a, k)))) for k in consts]
    # side-effect stubs apply everywhere, including callees of callees
    ps += [patch.object(a, n, (lambda _v: (lambda *_x, **_k: _v))(v)) for n, v in SIDE.items()
           if hasattr(a, n) and n not in names]
    if mode == "capture":
        ps += [patch.object(a, n, (lambda _v: (lambda *_x, **_k: _v))(v)) for n, v in OVERRIDE.items()
               if hasattr(a, n) and n not in names]
    if mode == "capture":
        from alpaca.trading.client import TradingClient as _TC

        def _refuse(*_x, **_k):
            raise RuntimeError("harness capture: order-mutating broker call refused")
        ps += [patch.object(_TC, m, _refuse) for m in ("cancel_order_by_id", "cancel_orders",
               "close_all_positions", "close_position", "exercise_options_position",
               "replace_order_by_id", "submit_order")]
    ps += [patch.object(a, n, wrap(n, getattr(a, n))) for n in names if hasattr(a, n)]
    ps += [patch.object(a, "yf", _Rec(a.yf, "yf", tape, "capture" if mode == "capture" else "replay", {}))]
    ps += [patch.object(a, "datetime", _FixedDatetime), patch.object(a, "date", _FixedDate),
           patch.object(a.time, "sleep", lambda *_: None)]
    if mode != "capture":
        ps += [patch.object(a.time, "time", lambda: _FIXED.timestamp()),
               patch.object(a, "requests", MagicMock(name="requests")),
               patch.object(socket.socket, "connect", _no_net)]
    cwd = os.getcwd()
    for p in ps:
        p.start()
    os.chdir(tmp)
    buf, how = io.StringIO(), "returned"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                r = getattr(a, func)(*args, **kwargs)
                _LAST_RET[0] = r
                how = "returned " + norm(r)[:300]
            except BaseException as e:
                how = f"{type(e).__name__}: {norm(str(e))[:200]}"
    finally:
        os.chdir(cwd)
        for p in reversed(ps):
            p.stop()
        shutil.rmtree(tmp, ignore_errors=True)
    out = {"how": how, "calls": calls,
           "stdout": [norm(l.rstrip())[:240] for l in buf.getvalue().splitlines() if l.strip()]}
    return out, tape


def main():
    target, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "check")
    fix = os.path.join(HERE, "fixtures", f"{target}.tape.pkl")
    gold_p = os.path.join(HERE, f"{target}.golden.json")
    if mode == "capture":
        out, tape = run(target, "capture")
        os.makedirs(os.path.dirname(fix), exist_ok=True)
        pickle.dump(tape, open(fix, "wb"))
        n = sum(len(v) for v in tape["calls"].values())
        unt = sum(1 for v in tape["calls"].values() for k, b in v if b is None)
        print(f"captured {n} callee returns ({unt} untapeable) -> {fix}; {out['how'][:120]}")
        return
    tape = pickle.load(open(fix, "rb"))
    out, _ = run(target, mode, tape)
    print(f"calls={len(out['calls'])} stdout={len(out['stdout'])} how={out['how'][:100]}")
    if mode == "record":
        json.dump(out, open(gold_p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("recorded", gold_p)
        return
    gold = json.load(open(gold_p, encoding="utf-8"))
    bad = [k for k in out if gold.get(k) != out[k]]
    import difflib
    for k in bad:
        g = [json.dumps(x, ensure_ascii=False) for x in gold[k]] if isinstance(gold[k], list) else [str(gold[k])]
        r = [json.dumps(x, ensure_ascii=False) for x in out[k]] if isinstance(out[k], list) else [str(out[k])]
        print("DIFF", k)
        for d in list(difflib.unified_diff(g, r, lineterm="", n=0))[2:10]:
            print("    ", d[:220])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")


main()
