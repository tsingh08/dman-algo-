"""Generic characterization harness for dispatcher-style functions.

Replaces EVERY module-level function/class the target calls with a recorder,
then records the call sequence, stdout, and how the target exited, per
scenario. Each scenario runs twice: recorders returning a MagicMock (truthy
paths) and returning None (falsy paths). The recorded-name list is frozen in
the golden file, so helpers introduced by a refactor run for real and must
reproduce the same sequence.

    python golden/callee_harness.py telegram record|check
"""
import sys, io, os, re, ast, json, contextlib, socket
import datetime as _dt, zoneinfo as _zi
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from unittest.mock import patch, MagicMock
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
os.environ.pop("ACCOUNT_SIZE", None)

_ET = _zi.ZoneInfo("America/New_York")
_FIXED = _dt.datetime(2026, 9, 14, 10, 30, tzinfo=_ET)   # Monday, market open


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


_TG = ["flags", "flags shares", "flags shares on", "flags shares off", "flags bogus on",
       "halt", "resume", "probation", "probation NVDA", "endprobation", "endprobation NVDA",
       "setupprobation", "setupprobation Gap", "endsetupprobation", "endsetupprobation Gap",
       "status", "positions", "pnl", "restart", "reboot", "scan", "scan NVDA", "review",
       "close", "close NVDA", "why", "why NVDA", "options", "buy", "buy NVDA", "buy NVDA 2",
       "/status@DManBot", "nonsense", "HALT"]

TARGETS = {
    # name: (value in "mock" mode, value in "none" mode) for callees whose
    # result is unpacked or formatted, so both modes reach deeper paths
    "telegram": ("_handle_telegram_command", [[t] for t in _TG], {
        "is_on_probation": ((True, 0.5), (False, 1.0)),
        "_trigger_workflow_restart": ((True, "dispatched"), (False, "HTTP 404")),
        "_close_earnings_spread": (("filled", "oid-1"), ("no_quote", None)),
        "_submit_options_close": (("filled", "oid-2"), ("no_quote", None)),
        "get_todays_loss": (123.45, 0.0),
        "get_this_month_loss": (-2.5, 0.0),
        "PositionTracker": (MagicMock(name="PositionTracker"), __import__("types").SimpleNamespace(positions=[])),
    }),
}


def called_names(func):
    t = ast.parse(open(os.path.join(ROOT, "dman_algo.py"), encoding="utf-8").read())
    mod = {n.name for n in t.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    fn = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == func][0]
    direct = {c.func.id for c in ast.walk(fn) if isinstance(c, ast.Call)
              and isinstance(c.func, ast.Name) and c.func.id in mod and c.func.id != func}
    # Helpers extracted FROM the target are part of it: patching them would
    # leave the harness covering dispatch only. Run them for real and record
    # what they call instead (found 2026-09-16: adding two /flags entries
    # changed nothing in the golden, because the body never ran).
    inner = {n for n in direct if n.startswith(("_tg_cmd_", "_scan_", "_main_mode_",
                                                "_pmb_", "_mw_", "_rs_", "_ss_", "_regime_"))}
    out = set(direct) - inner
    for name in inner:
        h = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == name]
        if h:
            out |= {c.func.id for c in ast.walk(h[0]) if isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Name) and c.func.id in mod and c.func.id not in inner}
    return sorted(out)


def norm(x):
    s = str(x)
    s = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", s)
    s = re.sub(r"id='\d+'", "id=ID", s)
    return s


def run(func, args, names, ret_none, ov):
    calls = []

    def recorder(name):
        def _r(*ar, **kw):
            calls.append([name, norm(ar).replace(tmp, "TMP").replace(tmp.replace(chr(92), chr(92)*2), "TMP")[:160],
                          norm(sorted(kw.items())).replace(tmp, "TMP")[:160]])
            if name in ov:
                return ov[name][1 if ret_none else 0]
            return None if ret_none else MagicMock(name=name)
        return _r

    def _no_net(*_a, **_k):
        raise OSError("network disabled in harness")

    import tempfile
    tmp = tempfile.mkdtemp(prefix="dman_harness_")
    file_consts = [k for k, v in vars(a).items()
                   if isinstance(v, str) and re.search(r"_(FILE|PATH|LOG|DIR)$", k)]
    ps = [patch.object(a, k, os.path.join(tmp, os.path.basename(getattr(a, k)) or k))
          for k in file_consts]
    ps += [patch.object(a, n, side_effect=recorder(n)) for n in names if hasattr(a, n)]
    ps += [patch.object(a.time, "sleep", lambda *_: calls.append(["time.sleep", "", ""])),
           patch.object(a, "requests", MagicMock(name="requests")),
           patch.object(a, "yf", MagicMock(name="yf")),
           patch.object(a, "datetime", _FixedDatetime),
           patch.object(a, "date", _FixedDate),
           patch.object(a.time, "time", lambda: _FIXED.timestamp()),
           patch.object(socket.socket, "connect", _no_net),
           patch.object(a.subprocess if hasattr(a, "subprocess") else os, "system", MagicMock(name="system"))]
    for p in ps:
        p.start()
    buf, how = io.StringIO(), "returned"
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                r = getattr(a, func)(*args)
                how = f"returned {norm(r)[:80]}"
            except SystemExit as e:
                how = f"SystemExit({e.code})"
            except Exception as e:
                how = f"{type(e).__name__}: {norm(e)[:120]}"
    finally:
        os.chdir(cwd)
        for p in reversed(ps):
            p.stop()
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    return {"how": how, "calls": calls,
            "stdout": [norm(l.replace(tmp, "TMP"))[:200] for l in buf.getvalue().splitlines() if l.strip()]}


target, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "check")
func, scen, ov = TARGETS[target]
OUT = os.path.join(HERE, f"{target}.golden.json")
if mode == "record":
    names = called_names(func)
    res = {f"{rn}|{' '.join(map(str, s))}": run(func, s, names, rn == "none", ov)
           for s in scen for rn in ("mock", "none")}
    json.dump({"patch_names": names, "results": res}, open(OUT, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    for k, r in res.items():
        print(f"{k:<32} calls={len(r['calls']):<3} out={len(r['stdout']):<3} {r['how'][:50]}")
else:
    gold = json.load(open(OUT, encoding="utf-8"))
    res = {f"{rn}|{' '.join(map(str, s))}": run(func, s, gold["patch_names"], rn == "none", ov)
           for s in scen for rn in ("mock", "none")}
    bad = [f"{k}.{f}" for k, r in res.items() for f in r if gold["results"].get(k, {}).get(f) != r[f]]
    for b in bad[:20]:
        k, f = b.rsplit(".", 1)
        print("DIFF", b, "\n   gold:", str(gold["results"].get(k, {}).get(f))[:200], "\n   now: ", str(res[k][f])[:200])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")
