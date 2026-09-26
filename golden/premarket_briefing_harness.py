"""Characterization harness for run_premarket_briefing.

Like scanner.golden.json, premarket_briefing.golden.json was an orphan with no
harness behind it. This is the run that now PUSHES state -- it pre-builds the
scan universe the 9:45 gate depends on, re-anchors tracked entry prices to real
fills, and dispatches the daemon behind it -- so a silent behaviour change here
costs a whole session.

What it pins: that the universe cache is written with TODAY's date, that the
fill re-anchor is called before anything else touches positions, the
scanner-health verdict at each staleness band, and that the briefing never
submits an order.

Hermetic including the clock, so the recording does not move with the calendar.
"""
import sys, io, os, json, re, contextlib, tempfile, datetime as _dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a

OUT = os.path.join(HERE, "premarket_briefing.golden.json")

FIXED = _dt.datetime(2026, 10, 14, 8, 5, tzinfo=a.ET)     # a Wednesday, 8:05 ET
TODAY = FIXED.date()

BASE = dict(
    scan_log_age_h=0.5,        # scanner healthy
    scan_log_missing=False,
    month_loss=-2.0,
    lifted=False,
    universe=["AAPL", "MSFT", "NVDA"],
    universe_raises=False,
    reanchored=["AAPL"],
    broker_up=True,
    positions=[],
    remote=[],
    orders=[],
)

SCENARIOS = {
    "P01_normal":                     {},
    "P02_scanner_stale_hours":        dict(scan_log_age_h=8.0),
    "P03_scanner_down_days":          dict(scan_log_age_h=80.0),
    "P04_scan_log_missing":           dict(scan_log_missing=True),
    "P05_monthly_limit_lifted":       dict(month_loss=-11.0, lifted=True),
    "P06_monthly_limit_not_lifted":   dict(month_loss=-11.0, lifted=False),
    "P07_universe_build_fails":       dict(universe_raises=True),
    "P08_nothing_to_reanchor":        dict(reanchored=[]),
    "P09_broker_unavailable":         dict(broker_up=False, reanchored=[]),
    # reconciliation finding something: a breakout zone carrying a stop it must
    # not have, which is the RSKD case that cost a session to notice by hand
    "P10_reconciliation_finds_a_stop": dict(
        positions=[SimpleNamespace(ticker="RSKD", setup="Breakout Zone +120% off low",
                                   entry=8.04, shares=31, stop=0.01)],
        remote=[SimpleNamespace(symbol="RSKD", qty="31", avg_entry_price="8.04")],
        orders=[SimpleNamespace(symbol="RSKD", side="sell", order_type="stop",
                                stop_price=7.39)]),
}


def norm(txt):
    txt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(txt))
    txt = re.sub(r"\d+\.\d+s", "Ns", txt)
    return re.sub(r"\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?( ET| MT)?", "HH:MM", txt)


def run(name, over):
    cfg = {**BASE, **over}
    tg, reanchor_calls, orders = [], [], []
    tmp = tempfile.mkdtemp()
    cwd = os.getcwd()

    # scan log with a controllable age
    scan_log = os.path.join(tmp, "scan_log.json")
    if not cfg["scan_log_missing"]:
        ts = (FIXED - _dt.timedelta(hours=cfg["scan_log_age_h"])).isoformat()
        io.open(scan_log, "w", encoding="utf-8").write(json.dumps([{"ts": ts}]))

    dt = MagicMock(wraps=_dt.datetime)
    dt.now.side_effect = lambda tz=None: FIXED if tz else FIXED.replace(tzinfo=None)
    dt.today.return_value = FIXED.replace(tzinfo=None)
    dt.strptime = _dt.datetime.strptime
    dt.fromisoformat = _dt.datetime.fromisoformat

    client = MagicMock()
    client.get_all_positions.return_value = list(cfg["remote"])
    client.get_orders.return_value = list(cfg["orders"])
    client.submit_order.side_effect = lambda *x, **k: orders.append("ORDER") or MagicMock(id="x")

    def _universe():
        if cfg["universe_raises"]:
            raise RuntimeError("screener unavailable")
        return list(cfg["universe"])

    ps = [
        patch.object(a, "datetime", dt),
        patch.object(a, "_et_today", return_value=TODAY),
        patch.object(a, "SCAN_LOG_FILE", scan_log),
        patch.object(a, "get_alpaca_client", return_value=(client if cfg["broker_up"] else None)),
        patch.object(a, "PositionTracker", return_value=MagicMock(positions=cfg["positions"])),
        patch.object(a, "_reanchor_entries_to_fills",
                     side_effect=lambda pt, alp: reanchor_calls.append(sorted(alp.keys()))
                     or list(cfg["reanchored"])),
        patch.object(a, "get_this_month_loss", return_value=cfg["month_loss"]),
        patch.object(a, "_monthly_halt_lifted", return_value=cfg["lifted"]),
        patch.object(a, "build_scan_universe", side_effect=_universe),
        patch.object(a, "pre_gap_catalyst_pass", return_value=[]),
        patch.object(a, "_fetch_global_context",
                     return_value={"score": 1, "risk_mult": 1.0, "tone": "NEUTRAL",
                                   "summary": "ctx", "components": {}}),
        patch.object(a, "_fetch_breaking_news_rss", return_value=[]),
        patch.object(a, "_premarket_gaps", return_value={}),
        patch.object(a, "get_upcoming_earnings", return_value=[]),
        patch.object(a, "_get_short_float_data", return_value=(1.0, 10.0, 0, 0)),
        patch.object(a, "send_telegram", side_effect=lambda m, *x, **k: tg.append(norm(m)) or True),
        patch.object(a, "_is_duplicate_alert", return_value=False),
        patch.object(a, "_save_last_alert"), patch.object(a, "_mark_alerted"),
        patch.object(a.time, "sleep"),
    ]
    # The _pmb_* section builders: their prose is not what this pins, but their
    # RETURN SHAPES are load-bearing -- several are unpacked by the caller, so a
    # blanket None stub would only ever record a TypeError.
    _SECTION_RETURNS = {
        "_pmb_market_snapshot": ("macro", "regime", "regime2", 15.0, "warn"),
        "_pmb_milestones":      ("milestones", "pdt"),
        "_pmb_sector_health":   "sectors",
        "_pmb_macro_calendar":  "macro-cal",
        "_pmb_gap_watch":       [],
        "_pmb_overseas_section": "overseas",
        "_pmb_live_suggestion": "suggestion",
        "_pmb_weekend_section": "weekend",
        "_pmb_breakout_zone_section": "bzone",
        "_pmb_strangle_advisory": None,
    }
    for _n, _ret in _SECTION_RETURNS.items():
        if hasattr(a, _n):
            ps.append(patch.object(a, _n, return_value=_ret))

    buf = io.StringIO()
    os.chdir(tmp)                      # the universe cache is written to CWD
    for p in ps:
        p.start()
    try:
        err = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                a.run_premarket_briefing()
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
    finally:
        for p in reversed(ps):
            p.stop()
        cache = None
        cpath = os.path.join(tmp, "dman_universe_cache.json")
        if os.path.exists(cpath):
            cache = json.load(io.open(cpath, encoding="utf-8"))
        os.chdir(cwd)
    return {"error": err, "orders": orders,
            "cache_date": (cache or {}).get("date"),
            "cache_n": len((cache or {}).get("tickers", [])),
            "reanchor_calls": reanchor_calls,
            "telegram": tg,
            "stdout": [l for l in norm(buf.getvalue()).splitlines() if l.strip()]}


results = {n: run(n, o) for n, o in SCENARIOS.items()}
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
KEYS = ("error", "orders", "cache_date", "cache_n", "reanchor_calls", "telegram", "stdout")
if mode == "record":
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for n, r in results.items():
        print(f"{n:<32} cache={r['cache_date']}/{r['cache_n']} reanchor={len(r['reanchor_calls'])} "
              f"orders={len(r['orders'])} tg={len(r['telegram'])} err={r['error']}")
else:
    gold = json.load(open(OUT, encoding="utf-8"))
    bad = 0
    for n, r in results.items():
        g = gold.get(n)
        for k in KEYS:
            if g is None or g.get(k) != r[k]:
                bad += 1
                print(f"DIFF {n}.{k}")
                if g is not None and isinstance(r[k], list):
                    import difflib
                    for d in list(difflib.unified_diff(
                            [json.dumps(x, ensure_ascii=False) for x in g.get(k, [])],
                            [json.dumps(x, ensure_ascii=False) for x in r[k]],
                            lineterm="", n=0))[2:8]:
                        print("   ", d[:190])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{bad} differences")
