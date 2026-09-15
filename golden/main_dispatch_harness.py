"""Characterization harness for main() -- the CLI dispatcher.

Runs main() once per --mode (plus a few flag variants) with EVERY module-level
function main calls replaced by a recorder, and records the exact call
sequence, stdout, and how main exited. The recorded-function list is frozen in
the golden file at record time, so helpers introduced by a refactor run for
real and must reproduce the same call sequence.

    python golden/main_dispatch_harness.py record
    python golden/main_dispatch_harness.py check
"""
import sys, io, os, re, ast, json, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unittest.mock import patch, MagicMock
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a

OUT = os.path.join(HERE, "main_dispatch.golden.json")
os.environ.pop("ACCOUNT_SIZE", None)


import datetime as _dt, zoneinfo as _zi
_ET = _zi.ZoneInfo("America/New_York")
_FIXED = _dt.datetime(2026, 9, 14, 8, 30, tzinfo=_ET)   # a Monday, pre-market


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


class _StopLoop(BaseException):
    """Raised by the patched sleep to end long-running modes deterministically."""


def main_called_functions():
    t = ast.parse(open(os.path.join(ROOT, "dman_algo.py"), encoding="utf-8").read())
    mod_fns = {n.name for n in t.body if isinstance(n, ast.FunctionDef)}
    m = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    return sorted({c.func.id for c in ast.walk(m) if isinstance(c, ast.Call)
                   and isinstance(c.func, ast.Name) and c.func.id in mod_fns and c.func.id != "main"})


def modes():
    t = ast.parse(open(os.path.join(ROOT, "dman_algo.py"), encoding="utf-8").read())
    m = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "main"][0]
    for c in ast.walk(m):
        if (isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "add_argument" and c.args
                and isinstance(c.args[0], ast.Constant) and c.args[0].value == "--mode"):
            for kw in c.keywords:
                if kw.arg == "choices":
                    return ast.literal_eval(kw.value)


def norm(x):
    s = str(x)
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"\b1[6-9]\d{8}(\.\d+)?\b", "EPOCH", s)          # time.time() values
    s = re.sub(r"\b\d+(\.\d+)?\s*(s|sec|secs|seconds|min|mins|minutes) ago\b", "N ago", s)
    s = re.sub(r"id='\d+'", "id=ID", s)
    s = re.sub(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([+-]\d{2}:\d{2})?)?", "DATE", s)
    s = re.sub(r"\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?( ET| MT)?", "HH:MM", s)
    return s


def run(argv, patch_names):
    calls = []

    def recorder(name):
        def _r(*args, **kw):
            calls.append([name, norm(args)[:160], norm(sorted(kw.items()))[:160]])
            return MagicMock(name=name)
        return _r

    sleeps = {"n": 0}

    def _sleep(*_a, **_k):
        sleeps["n"] += 1
        calls.append(["time.sleep", "", ""])
        if sleeps["n"] >= 3:
            raise _StopLoop()

    ps = [patch.object(a, n, side_effect=recorder(n)) for n in patch_names if hasattr(a, n)]
    ps += [patch.object(a.time, "sleep", side_effect=_sleep),
           patch.object(a, "requests", MagicMock(name="requests")),
           patch.object(a, "yf", MagicMock(name="yf")),
           patch.object(sys, "argv", argv),
           patch.object(a, "datetime", _FixedDatetime),
           patch.object(a, "date", _FixedDate),
           patch.object(a.time, "time", lambda: _FIXED.timestamp())]
    for p in ps:
        p.start()
    buf, how = io.StringIO(), "returned"
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                a.main()
            except SystemExit as e:
                how = f"SystemExit({e.code})"
            except _StopLoop:
                how = "stopped-after-3-sleeps"
            except Exception as e:
                how = f"{type(e).__name__}: {norm(e)[:120]}"
    finally:
        for p in reversed(ps):
            p.stop()
    return {"how": how, "calls": calls,
            "stdout": [norm(l)[:200] for l in buf.getvalue().splitlines() if l.strip()]}


def scenarios():
    out = {f"mode={m}": ["dman_algo.py", "--mode", m] for m in modes()}
    out["scan+tickers"] = ["dman_algo.py", "--mode", "scan", "--tickers", "NVDA", "PLTR"]
    out["backtest+years"] = ["dman_algo.py", "--mode", "backtest", "--years", "3"]
    out["no-args"] = ["dman_algo.py"]
    return out


mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "record":
    names = main_called_functions()
    res = {k: run(v, names) for k, v in scenarios().items()}
    json.dump({"patch_names": names, "results": res}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    for k, r in res.items():
        print(f"{k:<24} calls={len(r['calls']):<3} stdout={len(r['stdout']):<3} {r['how'][:60]}")
else:
    gold = json.load(open(OUT, encoding="utf-8"))
    res = {k: run(v, gold["patch_names"]) for k, v in scenarios().items()}
    bad = [f"{k}.{f}" for k, r in res.items() for f in r if gold["results"].get(k, {}).get(f) != r[f]]
    import difflib
    for b in bad[:20]:
        print("DIFF", b)
        k, f = b.rsplit(".", 1)
        g = [json.dumps(x, ensure_ascii=False) for x in gold["results"][k][f]] if isinstance(res[k][f], list) else [str(gold["results"][k][f])]
        r = [json.dumps(x, ensure_ascii=False) for x in res[k][f]] if isinstance(res[k][f], list) else [str(res[k][f])]
        for d in list(difflib.unified_diff(g, r, lineterm="", n=0))[2:6]:
            print("    ", d[:180])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")
