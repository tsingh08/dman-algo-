"""Characterization harness for _submit_signals_to_alpaca.

Runs the function through fixed scenarios with every external boundary mocked,
and records exactly what it DID: orders requested, Telegram messages, tracker
opens, return value, and normalized stdout. `golden.py record` saves the result;
`golden.py check` re-runs and diffs. A behavior-preserving refactor must diff
to nothing.
"""
import sys, io, json, re, contextlib
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from unittest.mock import patch, MagicMock
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a

OUT = os.path.join(HERE, "submit_signals.golden.json")

# The live preflight warns when ACCOUNT_SIZE is unset, so the recording would
# depend on the machine running it (22 differences with it set). Pin it.
os.environ.pop("ACCOUNT_SIZE", None)


def sig(ticker="NVDA", setup="Gap & Hold", score=100, entry=10.0):
    return a.ProSignal(ticker=ticker, setup=setup, bias="LONG", entry=entry, stop=entry * 0.9,
                       target1=entry * 1.2, target2=entry * 1.3, rr=2.0, rsi=60.0, rvol=3.0,
                       reason="golden", confluence_score=score, not_chasing_extended_highs=True,
                       shares=10, cost=entry * 10)


BASE = dict(
    market_open=True, halted=False, probation=(False, 1.0), todays_loss=0.0, month_loss=0.0,
    pdt={"used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0},
    ctx={"risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""},
    stats={"consec_losses": 0, "consec_wins": 0, "win_rate": 0.6, "avg_win_r": 2.0,
           "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4},
    can_options=True, options_result=("opt-1", {"occ_symbol": "NVDA260925C00010000", "expiry": "2026-09-25",
                                                "strike": 10.0, "ask": 1.0, "contracts": 3, "delta": 0.6,
                                                "theta": -0.02, "gamma": 0.1, "vega": 0.05, "iv": 0.5,
                                                "oi": 500, "bid_size": 5, "ask_size": 5, "pc_ratio": 1.0,
                                                "flow_label": "", "dominant_call_strike": 10,
                                                "option_type": "CALL", "total_cost": 300.0, "mid": 1.0, "bid": 0.95, "dte": 11}),
    agg_room=None, shares_fallback=True, has_chain=True, tier_a=True, gap_liq=(2.0, 5_000_000),
    earn_safe=(True, ""), setup_killed=(False, ""), auto_allowed=(True, ""), bench=None,
    valid=(True, 10.0), equity=30_000.0,
)

SCENARIOS = {
    "S01_market_closed": dict(market_open=False),
    "S02_halted": dict(halted=True),
    "S03_daily_loss": dict(todays_loss=-5.0),
    "S04_normal_options_ok": {},
    "S05_normal_options_fail_shares": dict(options_result=(None, None)),
    "S06_pdt_zero_options_ok": dict(pdt={"used": 3, "remaining": 0, "swing_mode": True, "equity": 2800.0}),
    "S07_pdt_zero_naked_shares": dict(pdt={"used": 3, "remaining": 0, "swing_mode": True, "equity": 2800.0},
                                      options_result=(None, None), has_chain=False, equity=2800.0),
    "S08_pdt_zero_not_genuine": dict(pdt={"used": 3, "remaining": 0, "swing_mode": True, "equity": 2800.0},
                                     options_result=(None, None), has_chain=False, tier_a=False, equity=2800.0),
    "S09_swing_mode_one_left": dict(pdt={"used": 2, "remaining": 1, "swing_mode": True, "equity": 2800.0}),
    "S10_setup_killed": dict(setup_killed=(True, "0W/10L over 10 live trades")),
    "S11_not_watchlist": dict(auto_allowed=(False, "outside the backtest-validated WATCHLIST — alert only")),
    "S12_riskoff_hotstreak_probation": dict(ctx={"risk_mult": 0.35, "tone": "RISK-OFF", "score": -4, "summary": ""},
                                            stats={**BASE["stats"], "consec_wins": 4},
                                            probation=(True, 0.5)),
    "S13_monthly_loss": dict(month_loss=-6.0),
    "S14_aggregate_cap": dict(agg_room="premium at risk $600 > cap $560"),
    "S15_benched_ticker": dict(bench="3 losses in a row on this ticker"),
}


def norm(txt):
    txt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", txt)
    txt = re.sub(r"\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?( ET| MT)?", "HH:MM", txt)
    txt = re.sub(r"id=[\w-]{1,12}…?", "id=ID", txt)
    return txt


def run(name, over):
    cfg = {**BASE, **over}
    tg, orders, opens = [], [], []
    client = MagicMock()

    def _cap_order(req):
        orders.append({"type": type(req).__name__,
                       **{k: str(getattr(req, k, None)) for k in ("symbol", "qty", "side", "limit_price",
                                                                    "time_in_force", "order_class")}})
        o = MagicMock(); o.id = "order-xyz"; return o
    client.submit_order.side_effect = _cap_order

    def _cap_alp(s):
        orders.append({"type": "submit_alpaca_trade", "symbol": s.ticker, "qty": str(s.shares),
                       "no_stop_entry": str(getattr(s, "no_stop_entry", False)),
                       "swing": str(getattr(s, "swing_mode", False))})
        return "alp-1", None

    def _cap_opt(_c, ticker, cur, risk, s):
        orders.append({"type": "options_call", "symbol": ticker, "risk": str(risk)})
        return cfg["options_result"]

    pt = MagicMock(); pt.positions = []
    pt.open.side_effect = lambda p: opens.append({k: str(v) for k, v in vars(p).items()
                                                  if k in ("ticker", "setup", "shares", "entry", "stop",
                                                           "target1", "day_only")}) or True
    buf = io.StringIO()
    s = sig()
    ps = [
        patch.object(a, "ALPACA_API_KEY", "k"), patch.object(a, "ALPACA_PAPER", False),
        # Calendar guards read the real date; pin them so the recording is
        # identical on any day, including FOMC/macro blackout days.
        patch.object(a, "_FOMC_DATES", set()), patch.object(a, "_MAJOR_MACRO_EVENT_DATES", set()),
        patch.object(a, "is_market_open", return_value=cfg["market_open"]),
        patch.object(a, "is_halted", return_value=cfg["halted"]),
        patch.object(a, "is_on_probation", return_value=cfg["probation"]),
        patch.object(a, "get_todays_loss", return_value=cfg["todays_loss"]),
        patch.object(a, "get_this_month_loss", return_value=cfg["month_loss"]),
        patch.object(a, "_monthly_halt_lifted", return_value=False),
        patch.object(a, "_get_pdt_status", return_value=cfg["pdt"]),
        patch.object(a, "_fetch_global_context", return_value=cfg["ctx"]),
        patch.object(a, "WinRateTracker"),
        patch.object(a, "get_alpaca_client", return_value=client),
        patch.object(a, "_signal_can_use_options", return_value=cfg["can_options"]),
        patch.object(a, "_submit_options_call", side_effect=_cap_opt),
        patch.object(a, "_submit_options_put", return_value=(None, None)),
        patch.object(a, "_options_aggregate_room", return_value=cfg["agg_room"]),
        patch.object(a, "submit_alpaca_trade", side_effect=_cap_alp),
        patch.object(a, "_shares_fallback_allowed", return_value=cfg["shares_fallback"]),
        patch.object(a, "_has_liquid_option_chain", return_value=cfg["has_chain"]),
        patch.object(a, "_has_tier_a_catalyst", return_value=cfg["tier_a"]),
        patch.object(a, "_pdt_zero_gap_and_liquidity", return_value=cfg["gap_liq"]),
        patch.object(a, "_earnings_safe_for_naked_hold", return_value=cfg["earn_safe"]),
        patch.object(a, "_setup_is_disabled", return_value=cfg["setup_killed"]),
        patch.object(a, "_auto_trade_allowed", return_value=cfg["auto_allowed"]),
        patch.object(a, "_ticker_bench_reason", return_value=cfg["bench"]),
        patch.object(a, "validate_entry_price", return_value=cfg["valid"]),
        patch.object(a, "get_effective_account", return_value=cfg["equity"]),
        patch.object(a, "PositionTracker", return_value=pt),
        patch.object(a, "_pdt_zero_shares_active", return_value=(True, "armed")),
        patch.object(a, "_is_duplicate_alert", return_value=False),
        patch.object(a, "_mark_alerted"), patch.object(a, "_save_last_alert"),
        patch.object(a, "send_telegram", side_effect=lambda m, *x, **k: tg.append(norm(m)) or True),
        patch.object(a, "_record_day_trade", return_value=True),
        patch.object(a.time, "sleep"),
    ]
    for p in ps:
        p.start()
    try:
        a.WinRateTracker.return_value.rolling_stats.return_value = cfg["stats"]
        err = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                ret = a._submit_signals_to_alpaca([s])
            except Exception as e:
                ret, err = None, f"{type(e).__name__}: {e}"
    finally:
        for p in reversed(ps):
            p.stop()
    return {"ret": str(ret), "error": err, "orders": orders, "telegram": tg, "opens": opens,
            "stdout": [l for l in norm(buf.getvalue()).splitlines() if l.strip()]}


results = {n: run(n, o) for n, o in SCENARIOS.items()}
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "record":
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for n, r in results.items():
        print(f"{n:<34} orders={len(r['orders'])} tg={len(r['telegram'])} opens={len(r['opens'])} "
              f"stdout={len(r['stdout'])} err={r['error']}")
else:
    gold = json.load(open(OUT, encoding="utf-8"))
    bad = 0
    for n, r in results.items():
        g = gold.get(n)
        for k in ("ret", "error", "orders", "telegram", "opens", "stdout"):
            if g is None or g[k] != r[k]:
                bad += 1
                print(f"DIFF {n}.{k}")
                if g is not None and isinstance(r[k], list):
                    import difflib
                    for d in list(difflib.unified_diff([json.dumps(x, ensure_ascii=False) for x in g[k]],
                                                       [json.dumps(x, ensure_ascii=False) for x in r[k]],
                                                       lineterm="", n=0))[2:10]:
                        print("   ", d[:200])
    print("GOLDEN:", "IDENTICAL" if not bad else f"{bad} differences")
