"""Characterization harness for run_pro_scanner.

The scanner had no golden coverage at all -- its `scanner.golden.json` was an
orphan with no harness behind it -- while being the function that decides what
trades. This pins the GATING CHAIN: the VIX halt and score raise, the seasonal
bar, the daily-loss and consecutive-loss stops, probation, each hard gate, the
effective-min-score bar, the defensive-rotation penalty, the early-session
Gap & Hold hold, the portfolio-heat cap, and which signals come out the other
end.

Every external boundary is mocked, including the clock, so a recording is
identical on any machine and any day. `record` saves, `check` re-runs and
diffs; a behaviour-preserving refactor must diff to nothing.
"""
import sys, io, os, json, re, contextlib, tempfile, datetime as _dt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from unittest.mock import patch, MagicMock
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a

OUT = os.path.join(HERE, "scanner.golden.json")

# A Wednesday in October: not a seasonally weak month, not a Friday, 11:00 ET
# so the 9:30-9:44 Gap & Hold hold does not apply unless a scenario asks.
FIXED = _dt.datetime(2026, 10, 14, 11, 0, tzinfo=a.ET)


def _regime(vix=15.0, regime="BULL", shock=False, rotation=False):
    return {"regime": regime, "score": 12, "vix_shock": shock,
            "defensive_rotation": rotation,
            "details": {"VIX": vix, "VIX Shock": "none", "Def Rotation": "tech weak"}}


def sig(ticker="NVDA", setup="Gap & Hold", score=95, bias="LONG"):
    s = a.ProSignal(ticker=ticker, setup=setup, bias=bias, entry=10.0, stop=9.0,
                    target1=12.0, target2=13.0, rr=2.0, rsi=60.0, rvol=3.0,
                    reason="golden", confluence_score=score)
    s.regime_ok = s.mtf_ok = s.earnings_ok = s.macro_ok = True
    s.divergence_free = s.not_chasing_extended_highs = True
    s.risk_usd, s.shares, s.cost = 50.0, 10, 100.0
    return s


BASE = dict(
    vix=15.0, regime="BULL", shock=False, rotation=False,
    min_score=85, adaptive=85, probation=(False, 1.0), consec=(None, None),
    todays_loss=0.0, seasonal_months=set(), heat=0.0, equity=2600.0,
    tickers=["NVDA"], signal=sig(), veto=(False, ""), eff_min=(85, 85),
    hour=11, sector="Technology",
)

SCENARIOS = {
    "G01_normal_pass":            {},
    "G02_vix_extreme_halt":       dict(vix=41.0),
    "G03_vix_over_25_raises_bar": dict(vix=26.0),
    "G04_vix_shock_raises_bar":   dict(shock=True),
    "G05_daily_loss_limit":       dict(todays_loss=-5.0),
    "G06_consec_loss_gate_stops": dict(consec=("return", [])),
    "G07_probation_active":       dict(probation=(True, 0.5)),
    "G08_seasonal_month":         dict(seasonal_months={10}),
    "G09_catalyst_veto":          dict(veto=(True, "market recap only")),
    "G10_regime_blocked":         dict(signal="regime"),
    "G11_mtf_blocked":            dict(signal="mtf"),
    "G12_earnings_blackout":      dict(signal="earnings"),
    "G13_macro_blackout":         dict(signal="macro"),
    "G14_divergence":             dict(signal="divergence"),
    "G15_chasing_highs":          dict(signal="chasing"),
    "G16_score_below_bar":        dict(signal=sig(score=70)),
    "G17_bar_capped":             dict(eff_min=(95, 110)),
    "G18_def_rotation_penalty":   dict(rotation=True, signal=sig(score=88), eff_min=(85, 85)),
    "G19_gap_hold_early_session": dict(hour=9),
    "G20_heat_cap_excludes":      dict(heat=0.059),
    "G21_no_signal_at_all":       dict(signal=None),
}


def norm(txt):
    txt = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", str(txt))
    txt = re.sub(r"\d+\.\d+s", "Ns", txt)
    return re.sub(r"\d{1,2}:\d{2}(:\d{2})?( ?[AP]M)?( ET| MT)?", "HH:MM", txt)


def _mk_signal(spec):
    """spec: None, a ProSignal, or the name of a hard gate to fail."""
    if spec is None or isinstance(spec, a.ProSignal):
        return spec
    s = sig()
    setattr(s, {"regime": "regime_ok", "mtf": "mtf_ok", "earnings": "earnings_ok",
                "macro": "macro_ok", "divergence": "divergence_free",
                "chasing": "not_chasing_extended_highs"}[spec], False)
    return s


def run(name, over):
    cfg = {**BASE, **over}
    tg, halts, finalized, logged = [], [], [], []
    the_sig = _mk_signal(cfg["signal"])
    now = FIXED.replace(hour=cfg["hour"], minute=40 if cfg["hour"] == 9 else 0)

    dt = MagicMock(wraps=_dt.datetime)
    dt.now.side_effect = lambda tz=None: now if tz else now.replace(tzinfo=None)
    dt.today.return_value = now.replace(tzinfo=None)
    dt.strptime = _dt.datetime.strptime
    dt.fromisoformat = _dt.datetime.fromisoformat

    tracker = MagicMock()
    tracker.rolling_stats.return_value = {"consec_losses": 0, "consec_wins": 0,
                                          "win_rate": 0.4, "avg_win_r": 2.0,
                                          "avg_loss_r": 1.0, "total": 20,
                                          "wins": 8, "losses": 12,
                                          "consec_losses_today": 0}
    tracker.adaptive_min_score.return_value = cfg["adaptive"]

    buf = io.StringIO()
    ps = [
        patch.object(a, "datetime", dt),
        patch.object(a, "WinRateTracker", return_value=tracker),
        patch.object(a, "resolve_live_outcomes", return_value=0),
        patch.object(a, "is_market_open", return_value=True),
        patch.object(a, "label_signal_features"),
        patch.object(a, "_enforce_setup_drift_restrictions"),
        patch.object(a, "is_on_probation", return_value=cfg["probation"]),
        patch.object(a, "_scan_consecutive_loss_gate", return_value=cfg["consec"]),
        patch.object(a, "get_todays_loss", return_value=cfg["todays_loss"]),
        patch.object(a, "SEASONAL_WEAK_MONTHS", cfg["seasonal_months"]),
        patch.object(a, "get_market_regime",
                     return_value=_regime(cfg["vix"], cfg["regime"], cfg["shock"], cfg["rotation"])),
        patch.object(a, "get_top_sectors", return_value=["Technology"]),
        patch.object(a, "_check_open_position_risk"),
        patch.object(a, "_scan_prefetch_news", side_effect=lambda m, t: m),
        patch.object(a, "prewarm_alpaca_bars", return_value=1),
        patch.object(a, "fetch_df", return_value=MagicMock(__len__=lambda s: 300)),
        patch.object(a, "_compute_indicators_cached", return_value=MagicMock(__len__=lambda s: 300)),
        patch.object(a, "detect_accumulation", return_value=None),
        patch.object(a, "_raw_signals", return_value=the_sig),
        patch.object(a, "_news_boost_after_sentiment_veto", return_value=False),
        patch.object(a, "_grade_catalyst", return_value=("C", "")),
        patch.object(a, "score_signal", side_effect=lambda s, *x, **k: s),
        patch.object(a, "_catalyst_veto", return_value=cfg["veto"]),
        patch.object(a, "_log_signal_features",
                     side_effect=lambda *x, **k: logged.append(str(k.get("reject_reason", "")))),
        patch.object(a, "_effective_min_score", return_value=cfg["eff_min"]),
        patch.object(a, "TICKER_SECTOR", {"NVDA": cfg["sector"]}),
        patch.object(a, "ENABLE_SMALLCAP", False),
        patch.object(a, "_scan_portfolio_heat", return_value=cfg["heat"]),
        patch.object(a, "get_effective_account", return_value=cfg["equity"]),
        patch.object(a, "_apply_sector_concentration_cap", side_effect=lambda s: s),
        patch.object(a, "_finalize_and_alert_signals",
                     side_effect=lambda s, *x: finalized.extend(getattr(z, "ticker", "?") for z in s)),
        patch.object(a, "_report_accumulation"),
        patch.object(a, "_scan_persist_log"),
        patch.object(a, "_scan_near_miss_tier", return_value=([], [])),
        patch.object(a, "_scan_friday_closeout"),
        patch.object(a, "_mark_signals_taken"),
        patch.object(a, "_log_scan_halt", side_effect=lambda r, *x: halts.append(r)),
        patch.object(a, "_is_duplicate_alert", return_value=False),
        patch.object(a, "_save_last_alert"), patch.object(a, "_mark_alerted"),
        patch.object(a, "send_telegram", side_effect=lambda m, *x, **k: tg.append(norm(m)) or True),
    ]
    for p in ps:
        p.start()
    try:
        err = None
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            try:
                out = a.run_pro_scanner(tickers=cfg["tickers"], min_score=cfg["min_score"])
                ret = [getattr(s, "ticker", "?") for s in (out or [])]
            except Exception as e:
                ret, err = None, f"{type(e).__name__}: {e}"
    finally:
        for p in reversed(ps):
            p.stop()
    return {"ret": ret, "error": err, "halts": halts, "finalized": finalized,
            "rejects": logged, "telegram": tg,
            "stdout": [l for l in norm(buf.getvalue()).splitlines() if l.strip()]}


results = {n: run(n, o) for n, o in SCENARIOS.items()}
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
KEYS = ("ret", "error", "halts", "finalized", "rejects", "telegram", "stdout")
if mode == "record":
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for n, r in results.items():
        print(f"{n:<30} ret={r['ret']} halts={r['halts']} tg={len(r['telegram'])} "
              f"out={len(r['stdout'])} err={r['error']}")
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
