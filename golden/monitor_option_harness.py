"""Characterization harness for _monitor_option_position (options exit logic).

Drives every exit branch -- expiry backstop, stop, trailing giveback, T1 half,
DTE warning, theta warning, no-quote -- across every close status, with the
date pinned. Records the return value, every close request, every field
update, every Telegram message and normalized stdout.

    python golden/monitor_option_harness.py record   # save baseline
    python golden/monitor_option_harness.py check    # must print IDENTICAL
"""
import sys, io, os, json, re, contextlib
from datetime import date, timedelta
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from unittest.mock import patch
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a

OUT = os.path.join(HERE, "monitor_option.golden.json")
OCC = "TE260925C00004000"
EXPIRY = date(2026, 9, 25)


def pos(**kw):
    p = {"ticker": "TE", "setup": f"Options Call {OCC}", "entry": 0.82, "stop": 0.41,
         "target1": 1.23, "shares": 600, "atr": 0.0, "peak_premium": 0.82}
    p.update(kw)
    return p


def snap(mid, bid, theta=-0.005, bs=10, as_=10):
    return {"mid": mid, "bid": bid, "ask": round(mid * 2 - bid, 2), "theta": theta,
            "delta": 0.6, "bid_size": bs, "ask_size": as_}


FAR = EXPIRY - timedelta(days=11)                              # nothing date-driven fires
FORCE = EXPIRY - timedelta(days=max(0, a.OPTIONS_FORCE_CLOSE_DTE))
WARN = EXPIRY - timedelta(days=a.OPTIONS_CLOSE_DTE) if a.OPTIONS_CLOSE_DTE > a.OPTIONS_FORCE_CLOSE_DTE else FAR
TRAIL_PEAK = round(0.82 * (1 + a.OPTIONS_TRAIL_ACTIVATE_GAIN_PCT / 100) * 1.4, 2)

STATUSES = ["submitted", "pending", "already_closed", "pdt_blocked", "no_quote", "failed"]
SC = {
    "M01_no_quote": dict(p=pos(), s=None, today=FAR),
    "M02_quiet_hold": dict(p=pos(), s=snap(0.85, 0.80), today=FAR),
    "M04_stale_bid_below_intrinsic": dict(p=pos(stop=0.65), s=snap(0.70, 0.60), und=4.68, today=FAR),
    "M05_trailing_giveback_exit": dict(p=pos(peak_premium=TRAIL_PEAK), s=snap(0.70, 0.68), today=FAR),
    "M06_t1_half_close": dict(p=pos(), s=snap(1.30, 1.28), today=FAR),
    "M08_dte_warning": dict(p=pos(), s=snap(0.85, 0.80), today=WARN),
    "M09_theta_warning": dict(p=pos(), s=snap(0.60, 0.58, theta=-0.08), today=FAR, und=4.2),
    "M10_milestone": dict(p=pos(), s=snap(0.85, 0.80), today=FAR, milestone="🎯 milestone +25%"),
    "M11_order_flow_lean_ask_heavy": dict(p=pos(peak_premium=TRAIL_PEAK, stop=0.82), s=snap(1.20, 1.18, bs=1, as_=40), today=FAR),
    "M12_order_flow_lean_bid_heavy": dict(p=pos(peak_premium=TRAIL_PEAK, stop=0.82), s=snap(1.20, 1.18, bs=40, as_=1), today=FAR),
    "M13_trailing_no_lean_hold": dict(p=pos(peak_premium=TRAIL_PEAK, stop=0.82), s=snap(1.20, 1.18), today=FAR),
}
for st in STATUSES:
    SC[f"M03_stop_{st}"] = dict(p=pos(), s=snap(0.32, 0.30), today=FAR, status=st, und=3.9)
    SC[f"M07_expiry_backstop_{st}"] = dict(p=pos(), s=snap(0.85, 0.80), today=FORCE, status=st)


def norm(txt):
    txt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(txt))
    return re.sub(r"\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?( ET| MT)?", "HH:MM", txt)


def run(cfg):
    closes, fields, tg, marks = [], [], [], []
    ps = [
        patch.object(a, "_et_today", return_value=cfg["today"]),
        patch.object(a, "_submit_options_close",
                     side_effect=lambda occ, n, why, **k: closes.append(
                         [occ, n, why] + sorted(f"{kk}={vv}" for kk, vv in k.items())) or
                     (cfg.get("status", "submitted"), "close-123")),
        patch.object(a, "_update_option_position_field",
                     side_effect=lambda *x, **k: fields.append([str(v) for v in x] + sorted(f"{kk}={vv}" for kk, vv in k.items()))),
        patch.object(a, "send_telegram", side_effect=lambda m, *x, **k: tg.append(norm(m)) or True),
        patch.object(a, "_mark_alerted", side_effect=lambda k, *x: marks.append(k)),
        patch.object(a, "_is_alerted_today", return_value=False),
        # The expiry backstop's unresolved outcomes (no_quote/failed/pdt_blocked)
        # alert on the cooldown key rather than the once-a-day one. Unstubbed,
        # these read and WRITE the real dman_last_alerts.json: the golden would
        # mutate repo state and go flaky on its second run (a "duplicate" alert
        # suppresses the very telegram being characterised).
        patch.object(a, "_is_duplicate_alert", return_value=False),
        patch.object(a, "_save_last_alert", side_effect=lambda k, *x: marks.append(f"dup:{k}")),
        patch.object(a, "_check_options_pnl_milestone", return_value=cfg.get("milestone")),
        patch.object(a, "_cached_option_greeks", return_value={"theta": -0.005, "delta": 0.6}),
        patch.object(a, "get_live_price", return_value=cfg.get("und")),
        # The function falls back to the REAL snapshot fetch when get_snapshot_fn
        # returns nothing. Unpatched, a "no quote" scenario pulled a live market
        # quote -- the recording then depended on the market, not the code.
        patch.object(a, "_get_option_snapshot", side_effect=lambda occ: cfg["s"]),
    ]
    for p in ps:
        p.start()
    buf, err = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                ret = a._monitor_option_position(
                    dict(cfg["p"]), "CALL",
                    get_snapshot_fn=lambda occ: cfg["s"],
                    get_price_fn=(lambda t: cfg["und"]) if "und" in cfg else None)
            except Exception as e:
                ret, err = None, f"{type(e).__name__}: {e}"
    finally:
        for p in reversed(ps):
            p.stop()
    return {"ret": norm(ret), "error": err, "closes": closes, "fields": fields, "telegram": tg,
            "marks": marks, "stdout": [l for l in norm(buf.getvalue()).splitlines() if l.strip()]}


results = {k: run(v) for k, v in SC.items()}
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "record":
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for k, r in results.items():
        print(f"{k:<32} closes={len(r['closes'])} fields={len(r['fields'])} tg={len(r['telegram'])} "
              f"ret={'yes' if r['ret'] not in ('None', '') else 'None'} err={r['error']}")
else:
    gold = json.load(open(OUT, encoding="utf-8"))
    bad = [f"{k}.{f}" for k, r in results.items() for f in r if gold.get(k, {}).get(f) != r[f]]
    for b in bad[:20]:
        print("DIFF", b)
    print("GOLDEN:", "IDENTICAL" if not bad else f"{len(bad)} differences")
