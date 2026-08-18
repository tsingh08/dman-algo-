"""
╔══════════════════════════════════════════════════════════════════════════╗
║   D🔥man Trading Algorithm v3 — PRO EDITION                            ║
║   Target: 80%+ Win Rate  |  @ProfessorDman1 Style                      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  NEW IN v3 (layers on top of v2's 8 signal detectors)                   ║
║  ├─ 01. Market Regime Filter   — SPY/QQQ/VIX trend gate                 ║
║  ├─ 02. Multi-Timeframe (MTF)  — Weekly chart must agree with daily     ║
║  ├─ 03. Relative Strength      — Stock must beat SPY on 5/20/60d basis  ║
║  ├─ 04. Sector Rotation        — Only trade top-performing sectors      ║
║  ├─ 05. Earnings Blackout      — No trades within 5d of earnings        ║
║  ├─ 06. Confluence Scorer      — 100-pt score; only fire on 75+         ║
║  ├─ 07. Fibonacci Levels       — Entry at key retracement zones         ║
║  ├─ 08. VWAP Analysis          — Intraday support/resistance layer      ║
║  ├─ 09. Volume Profile (POC)   — Point-of-Control alignment             ║
║  ├─ 10. ATR Stop Optimizer     — Stops at real support, not arbitrary   ║
║  ├─ 11. AI Setup Scorer        — Claude API scores each setup 1–10      ║
║  ├─ 12. Kelly Criterion Size   — Optimal position sizing for edge       ║
║  ├─ 13. Win Rate Tracker       — Rolling perf; auto-tightens filters    ║
║  ├─ 14. Consecutive Loss Guard — Halts after 3 consecutive losses       ║
║  └─ 15. Momentum Divergence    — RSI/MACD divergence early warning      ║
╠══════════════════════════════════════════════════════════════════════════╣
║  INSTALL                                                                 ║
║    pip install yfinance pandas numpy requests                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  USAGE                                                                   ║
║    python dman_algo_v3.py --mode scan                                    ║
║    python dman_algo_v3.py --mode scan --score 80   (stricter)           ║
║    python dman_algo_v3.py --mode backtest                                ║
║    python dman_algo_v3.py --mode performance                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import os, sys, json, time, math, re, argparse, warnings, traceback, requests, csv, tempfile
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional
import zoneinfo

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ET = zoneinfo.ZoneInfo("America/New_York")
MT = zoneinfo.ZoneInfo("America/Denver")   # display timezone — Denver, CO (MST/MDT)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

_acct_raw    = os.getenv("ACCOUNT_SIZE", "").strip()
ACCOUNT_SIZE = float(_acct_raw) if _acct_raw else 25_000.0   # set via ACCOUNT_SIZE env/secret
RISK_PER_TRADE     = 0.02        # base risk — Kelly may reduce this
MIN_RR             = 2.0
OPTIMIZE_STOP_MIN_RISK_FRACTION = 0.5   # optimize_stop() may tighten a raw
                                          # stop toward real price structure,
                                          # but never past this fraction of
                                          # the original risk-per-share — see
                                          # optimize_stop()'s docstring for
                                          # the 2026-08-16 fix this backs
                                          # (the previous clamp was dead code,
                                          # so tightening had no real cap).
DAY2_MAX_CUMULATIVE_MOVE_PCT = 15.0   # Day 2 Continuation: skip if price has
                                       # already run this far from the pre-gap
                                       # baseline by entry — confirmed live
                                       # 2026-08-06 (AMZN +20.8% cumulative,
                                       # bought the exact top). See detect_gap_and_hold().
MIN_CONFLUENCE     = 75          # 0-100 score; raise to 80 for extra caution
SETUP_MIN_CONFLUENCE = {         # per-setup overrides for historically weak setups
    "Gap & Short":    82,
    "Vol Breakdown":  85,
    "MACD Bear":      82,
    "OS Bounce":      82,
    "Morning Runner": 85,
    # Added 2026-08-14 — live setup_stats() confirms 0% win rate over 3
    # non-BE trades this past month (IOTR -8.7%, CLRO -28.1%, FGL -37.9%),
    # avg loss -24.9%, well beyond backtest-implied risk. Root cause
    # (confirmed on FGL and CLRO specifically): plain-stop fills on
    # low-float/illiquid names slip 8-18 points past the intended stop
    # price with no liquidity in between to catch it — a real, structural
    # risk this setup type carries live that the backtest didn't capture.
    # size_position_kelly() already uses per-setup win rate, but floors it
    # at 50% (see tracker.setup_stats() call site) specifically to avoid
    # over-reacting to a small sample in the SIZING math — this raises the
    # ENTRY quality bar instead (same lever VOLATILE_TICKERS already uses),
    # requiring a materially higher-conviction setup before taking this
    # specific risk again, rather than changing sizing based on 3 trades.
    "Low Float Catalyst": 90,
}

# Per-setup live performance drift monitor (WinRateTracker.setup_performance_drift)
# — a setup needs at least this many LIVE trades before its win rate means
# anything, and gets flagged once that win rate drops below the floor. See
# setup_performance_drift()'s docstring for why this is live-only.
SETUP_PERFORMANCE_ALERT_MIN_TRADES = 3
SETUP_PERFORMANCE_ALERT_WR_FLOOR   = 0.40

# OB Reversal disabled — 41% WR / avg -4.77% in backtest, consistent loser
ENABLE_OB_REVERSAL = False

# EMA Breakdown disabled — 39.5% WR / avg -0.14% in backtest, noise with no edge
ENABLE_EMA_BREAKDOWN = False

# High-beta / speculative tickers require tighter confluence to trade
VOLATILE_TICKERS = {
    "RIOT","GME","SOUN","RXRX","RKLB",
    "NIO","AFRM","HOOD","CELH","RIVN",
    "BABA","BIDU","PDD","JD","KWEB",   # Chinese ADRs — geopolitical + currency risk, tighter confluence needed
    "COIN","MSTR",                     # crypto-proxy — extreme vol, needs high conviction
    "SNDK","SKHY",                     # memory/storage supercycle — +13.7%/+7.3% single-day moves, needs high conviction
}
VOLATILE_MIN_CONFLUENCE = 88   # vs 85 default for standard tickers

# VCP disabled — avg -0.94% per trade in backtest, below breakeven
ENABLE_VCP = False

# Vol Breakout disabled — 39.1% WR / avg -1.53%, still losing after tightening
ENABLE_VOL_BREAKOUT = False

# EMA Pullback disabled — 33.3% WR constantly trips the 3-consec-loss halt; avg +0.77% not worth it
ENABLE_EMA_PULLBACK = False

# Gap & Short disabled — 40.0% WR / avg +1.51% across 5-trade backtest sample, consistent drag
ENABLE_GAP_SHORT = False

# MACD Bear disabled — 0% WR / 1-trade backtest sample, insufficient edge; short setup in BULL-dominant algo
ENABLE_MACD_BEAR = False

# MACD Cross disabled — 47.1% WR / 17 trades, below breakeven; mean-reversion setup underperforms in BULL regime
ENABLE_MACD_CROSS = False

# OS Bounce disabled — 0% WR / 1 trade (avg -0.47%); reversal setups structurally weak in BULL regime
ENABLE_OS_BOUNCE = False

# ── Small-cap / Low Float Catalyst module (Dman style) ──────────────────────
# Professor Dman primarily trades nano/micro-cap stocks with:
#   - Float < 5M shares, price $0.50-$20
#   - Catalyst-driven RVOL spike (RS plays, FDA news, mergers, short squeezes)
#   - MACD bullish "curling upside" + above 20D MA as technical confirmation
# NOTE: No reliable backtest (survivorship bias, RS price adjustments). Forward-test only.
ENABLE_SMALLCAP         = True
ENABLE_DYNAMIC_SMALLCAP = True   # Finviz screener discovery of new Dman-style plays
SMALLCAP_MAX_FLOAT_M   = 5.0      # max float in millions of shares
SMALLCAP_MAX_PRICE     = 20.0     # max stock price
SMALLCAP_MIN_PRICE     = 0.10     # min stock price — Dman buys $0.07-$0.50 regularly
SMALLCAP_MIN_RVOL      = 2.0      # minimum RVOL for dynamically discovered tickers
DMAN_WATCHLIST_MIN_RVOL = 0.5    # watchlist tickers: Dman's call IS the catalyst — 0.5x enough
CATALYST_NEWS_LOOKBACK_HOURS = 48   # detect_low_float_catalyst()'s real-news confirmation
                                      # window for dynamically-discovered (non-watchlist) tickers —
                                      # wider than the main scan's 20h news prefetch since a
                                      # low-float RVOL spike can be a delayed market reaction to
                                      # news that broke a day or two earlier, not always same-day
SMALLCAP_RISK_PCT      = 0.02     # 2% account risk per trade — aggressive sizing for small account
SMALLCAP_MAX_COST      = 2_500    # hard cap on position cost — micro-caps are high risk
SMALLCAP_MIN_SCORE          = 55   # minimum score for dynamically discovered tickers
DMAN_WATCHLIST_MIN_SCORE    = 45   # lower bar for personally curated watchlist tickers
SMALLCAP_T1_MULT       = 0.30     # T1 at +30% (Dman targets 50-200% — partial at 30%)
SMALLCAP_T2_MULT       = 0.75     # T2 at +75%
SMALLCAP_STOP_PCT      = 0.18     # 18% stop — penny stocks are volatile; wider needed
SMALLCAP_52WK_LOW_PCT  = 0.30     # "bottom chart" = within 30% of 52-week low
SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT = 12.0   # skip entry if price already faded this
                                            # far off TODAY's own high — confirmed live
                                            # 2026-08-06: CLRO entered 14.4% off a
                                            # 22-min-old intraday high, mid-drop

# Ultra-low float tier — Dman's "thin walls" plays (float < 2M): 100-200%+ potential
ULTRA_LOW_FLOAT_M      = 2.0      # threshold for ultra-low float tier
ULTRA_LOW_T1_MULT      = 0.50     # T1 at +50% (higher first target for thinner floats)
ULTRA_LOW_T2_MULT      = 1.50     # T2 at +150% — matches Dman's "$10+ from $4" targets
ULTRA_LOW_STOP_PCT     = 0.20     # 20% stop — extra room for extreme volatility

# Moon Shot tier — ultra-low float + massive gap + extreme RVOL = 2x+ potential
# Example: IOTR gapped +40.87% on 0.64M float at 17x RVOL (Jul 8 2026)
# When all three conditions are met, allocation steps up and T3 is set at 2x entry.
MOONSHOT_MIN_GAP_PCT   = 15.0    # gap ≥ 15% from prior close
MOONSHOT_MIN_RVOL      = 8.0     # RVOL ≥ 8x
MOONSHOT_RISK_MULT     = 5.0     # 5× base risk (2% → 10% of account on moon shots)
MOONSHOT_T3_MULT       = 1.0     # T3 at +100% — the "double"

# Pre-market auto-submit: when True, the 7 AM scan will place extended-hours limit
# orders on Tier A/B moon shots (gap ≥15%, float <2M) at the pre-market VWAP price.
# Captures IOTR-style moves that complete entirely before 9:45 AM.
# Extended-hours orders are single-leg only (no bracket); momentum-watch handles exits.
ENABLE_PREMARKET_SUBMIT = True

# ── Options trading (large-cap Gap & Hold only) ───────────────────────────────
# When True, the algo buys calls on WATCHLIST tickers instead of shares.
# Falls back to equity order if no liquid contract is found.
ENABLE_OPTIONS_TRADING      = True   # buy options instead of shares on WATCHLIST signals
OPTIONS_MAX_POSITION_COST   = 2000.0  # flat target budget per options trade (2026-08-09 —
                                       # direct instruction to size up further, was $1,250 set
                                       # 2026-08-07). Raises the "too expensive" ceiling
                                       # (1.5x this) from $1,875 to $3,000 — confirmed live
                                       # 2026-08-08 that AMZN's cheapest qualifying ITM
                                       # contract cost $2,093, just over the old ceiling; this
                                       # unlocks that class of higher-priced-underlying signal.
                                       # ~39.6% of account equity ($5,044.65 as of 2026-08-08)
                                       # in a SINGLE options trade — real concentration risk,
                                       # not something the portfolio heat cap compensates for
                                       # (PORTFOLIO_HEAT_LIMIT/SMALLCAP_RISK_PCT count POSITIONS
                                       # flatly, not actual dollar size — see
                                       # _submit_signals_to_alpaca's heat-cap loop). Still scaled
                                       # by _risk_off_mult so this stays adaptive to real regime/
                                       # macro conditions (e.g. derated heading into NFP/CPI),
                                       # not a flat number regardless of risk.
OPTIONS_DTE_MIN             = 5      # minimum 5 DTE — allows weekly for fast gap plays
OPTIONS_DTE_MAX             = 28     # max 4 weeks
OPTIONS_MAX_SPREAD_PCT      = 0.15   # skip contract if bid-ask spread > 15% of mid (was 20% of ask)
OPTIONS_MIN_UNDERLYING_VOL  = 5_000_000   # underlying must avg ≥5M shares/day (liquid options)
OPTIONS_ENABLE_PUTS         = True   # buy ITM puts on bearish/Bear Gap Hold signals
OPTIONS_DATA_FEED           = "opra"        # preferred feed — real OPRA tape. Entitlement has flipped
                                            # on/off before without any code change: 403'd 2026-07-29,
                                            # fixed by switching to "indicative", confirmed "opra" working
                                            # again 2026-07-30 under Algo Trader Plus, then confirmed 403
                                            # AGAIN 2026-08-06 — a full week of every single options
                                            # signal silently failing (_get_option_snapshot returning None
                                            # for every contract) and falling through to skip/equity with
                                            # zero visibility, discovered only by manually testing the raw
                                            # endpoint. _resolve_options_feed() below now detects this at
                                            # runtime instead of trusting this constant blindly, and alerts
                                            # instead of failing silently. openInterest is separately always
                                            # absent from this snapshot endpoint regardless of feed — see
                                            # _find_best_call_contract, which merges real OI in from the
                                            # contracts response.
STOCK_DATA_FEED             = "sip"         # Alpaca Algo Trader Plus — consolidated real-time SIP feed
                                             # (preferred value only — actual calls use
                                             # _resolve_stock_feed(), see SECTION 2, which
                                             # detects entitlement at runtime and alerts on a
                                             # downgrade, same as _resolve_options_feed() does
                                             # for OPRA. ATP billing voided twice before finally
                                             # activating 2026-08-08 — a future lapse would
                                             # otherwise fail silently into the yfinance fallback
                                             # with zero visibility.)

# Dman's curated small-cap watch — always scanned regardless of dollar-volume threshold.
# Lives in a data file (not source) so the StockTwits monitor can add tickers
# without modifying code. ONLY DMan's actual StockTwits/Twitter calls belong here.
SMALLCAP_WATCHLIST_FILE = "dman_smallcap_watchlist.json"
_SMALLCAP_FALLBACK = [
    "APVO", "MASK", "ONCO", "ARTL", "ELAB", "CDT", "CAST",
    "ATOS", "IMPP", "GFAI", "BFRI", "TRVI", "LABT", "YHC", "LIMN",
]


def _load_smallcap_watchlist() -> list[str]:
    """Load the curated small-cap list from its data file; fallback if missing."""
    try:
        with open(SMALLCAP_WATCHLIST_FILE) as _f:
            _data = _f.read()
        _parsed = json.loads(_data)
        _tickers = _parsed.get("tickers", []) if isinstance(_parsed, dict) else _parsed
        _clean = [str(t).upper().strip() for t in _tickers if str(t).strip()]
        if _clean:
            return _clean
    except Exception:
        pass
    return list(_SMALLCAP_FALLBACK)


DMAN_SMALLCAP_WATCHLIST = _load_smallcap_watchlist()

# ── Options layer (Dman style: ITM calls, live execution) ────────────────────
# The old dual-path (advisory vs live) system is gone — ONE live path remains:
# _find_best_call_contract / _find_best_put_contract → _submit_options_call/put,
# monitored + auto-closed by _monitor_option_position (cron hourly, daemon 60s).
ENABLE_OPTIONS          = True
OPTIONS_SETUPS          = {"Gap & Hold", "Morning Runner"}  # alert annotation only
OPTIONS_MIN_PRICE       = 10.0          # no options notes on sub-$10 stocks (illiquid chains)
OPTIONS_TARGET_DTE      = 14            # target 2-week DTE — DMan gap plays resolve in 1-5 days
OPTIONS_ITM_TARGET_PCT  = 0.04          # target 4% ITM (≈ delta 0.70) — documents strike scan intent
OPTIONS_CLOSE_DTE       = 7             # DTE ≤ 7 → close/roll warning from the monitor

# Trailing exit for options premium — added 2026-08-10, replacing the fixed
# T2 (+150%) auto-close. Confirmed live: a fixed target ignores how the
# trade actually got there — the same +150% print means something very
# different after a smooth grind up vs. a spike that's already reversing.
# Once a position has run up ACTIVATE_GAIN_PCT from entry, this tracks the
# highest premium seen (peak_premium) and closes the remaining position if
# premium gives back TRAIL_GIVEBACK_PCT off that peak — adapts to the
# trade's real trend instead of waiting for (or missing) one static number.
# Wider than the equivalent equity thresholds (15% activate / matched trail)
# because options premium is inherently more volatile than the underlying
# (leverage + theta + IV all move it) — a tight trail here would whipsaw
# out on normal option noise. The T1 half-sell and the pre-trail fixed stop
# (-50%, protects a position that never gets meaningfully profitable at
# all) are unchanged — this only replaces what happens to the runner AFTER
# real profit has accrued.
OPTIONS_TRAIL_ACTIVATE_GAIN_PCT = 25.0

# Giveback tolerance now WIDENS with how deep the position already is,
# instead of a flat percentage — confirmed live 2026-08-12: SMCI's call
# peaked at +160% (option $3.44 off a $1.32 entry), a flat 30% giveback
# stopped the remaining position out at $2.38 (+80%), and premium then
# ran on to a $4.25 high / $3.75 close the SAME DAY — a real, verified
# missed continuation, not just noise. A pullback that would threaten a
# modest +25-30% winner is normal chop once a position is already up
# 100%+; direct instruction after that trade was to scale the tolerance
# with depth of profit rather than apply one number everywhere. See
# _options_trail_giveback_pct().
OPTIONS_TRAIL_GIVEBACK_MIN_PCT    = 30.0    # tolerance right at the activation threshold
OPTIONS_TRAIL_GIVEBACK_MAX_PCT    = 50.0    # tolerance once peak gain is very large
OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT = 150.0   # peak gain (%) at which tolerance reaches the max

# Pre-event strangles (direction-neutral, buys both call + put before big catalysts)
STRANGLE_TICKERS    = ["SPY", "QQQ"]   # always liquid enough for two-legged plays
STRANGLE_OTM_PCT    = 0.04             # 4% OTM per leg (keeps premium reasonable)
STRANGLE_TARGET_DTE = 7                # weekly options — captures the move, limits theta
STRANGLE_MIN_DTE    = 5                # ≥5 DTE avoids same-week expiry (gamma/theta bleed)
STRANGLE_MAX_DTE    = 14
STRANGLE_RISK_PCT   = 0.01             # 1% of account per strangle event

# ── Earnings vertical/double debit spreads (rule-based, single-stock, WATCHLIST) ──
# Turns tonight's manually-built META spread (2026-07-29) into a permanent, tested
# feature: buy an OTM call debit spread and/or an OTM put debit spread ahead of a
# WATCHLIST ticker's own earnings, submitted as ONE atomic multi-leg (MLEG) order —
# never as separate single-leg orders, which would carry leg-imbalance risk.
ENABLE_EARNINGS_SPREADS        = True
EARNINGS_SPREAD_RISK_PCT       = 0.05    # 5% of equity per event (~$150 at $2,997.77)
EARNINGS_SPREAD_MIN_DTE        = 3
EARNINGS_SPREAD_MAX_DTE        = 14
EARNINGS_SPREAD_TARGET_DTE     = 7
EARNINGS_SPREAD_LONG_OTM_PCT   = 0.07    # starting point — narrows to fit budget, see build_earnings_spread_plan()
EARNINGS_SPREAD_SHORT_OTM_PCT  = 0.12
EARNINGS_SPREAD_MAX_WIDTH_PCT  = 0.12    # widest the short strike may sit from the long strike
EARNINGS_SPREAD_MIN_WIDTH_PCT  = 0.03    # hard floor — below this, skip rather than distort further
EARNINGS_SPREAD_MAX_SPREAD_PCT = 0.15    # per-leg bid/ask liquidity cap, mirrors OPTIONS_MAX_SPREAD_PCT
EARNINGS_SPREAD_MIN_OI         = 25      # per-leg minimum open interest
EARNINGS_SPREAD_BUDGET_SLACK   = 1.3     # skip only if min-width spread still costs > budget * this
EARNINGS_SPREAD_CLOSE_DTE      = 1       # close this many days before expiry — avoid short-leg pin/assignment risk
EARNINGS_SPREAD_TAKE_PROFIT_PCT= 0.70    # optional early close at this fraction of max gain
EARNINGS_DIRECTIONAL_MIN_MOVES = 3       # of EARNINGS_DIRECTIONAL_LOOKBACK, need this many same-sign to go single-sided
EARNINGS_DIRECTIONAL_LOOKBACK  = 4       # how many past earnings moves to look at
EARNINGS_DIRECTIONAL_MIN_AVG_PCT = 8.0   # ...and the average magnitude must be at least this
EARNINGS_IV_BACKWARDATION_MIN  = 0.05    # front-vs-back ATM IV gap required to treat "earnings today" as still pending
EARNINGS_APPROVAL_TIMEOUT_MIN  = 240     # minutes to wait for a Telegram YES before the offer expires
                                          # (permanent, always-on gate — no auto-promotion to autonomous
                                          # submission; every earnings spread requires a human YES).
                                          # Was 30 min until 2026-08-10: confirmed live that's
                                          # unworkable for a user who isn't on their phone most of
                                          # the workday — a real SMCI offer built at 11:40 AM expired
                                          # at 12:10 PM and was swept before it was ever seen,
                                          # leading to an unmanaged manual trade placed outside the
                                          # algo entirely. 4 hours gives a realistic window (a lunch
                                          # break, an end-of-day check) without leaving an offer live
                                          # into the next session. See EARNINGS_APPROVAL_MAX_PRICE_DRIFT_PCT
                                          # for the staleness guard this widened window now needs.
EARNINGS_APPROVAL_MAX_PRICE_DRIFT_PCT = 8.0   # abort a late approval instead of submitting a spread
                                               # priced off a stale snapshot — see
                                               # _handle_earnings_approval_reply's drift check
EARNINGS_SPREAD_PENDING_FILE   = "dman_earnings_pending.json"

# ── Telegram manual options browse/buy ("/options TICKER" -> "/buy N" -> YES) ──
TELEGRAM_OPTIONS_MENU_FILE        = "dman_telegram_options_menu.json"
TELEGRAM_MANUAL_BUY_FILE          = "dman_telegram_manual_buy.json"
TELEGRAM_OPTIONS_MENU_TIMEOUT_MIN = 10     # /options menu expires — a stale index must not resolve
TELEGRAM_MANUAL_BUY_TIMEOUT_MIN   = 5      # YES/NO confirmation window
MANUAL_BUY_MAX_PRICE_DRIFT_PCT    = 8.0    # abort a late YES rather than submit off a stale quote —
                                            # same staleness philosophy as EARNINGS_APPROVAL_MAX_PRICE_DRIFT_PCT
MANUAL_BUY_MAX_CONTRACTS          = 10     # /buy N [qty] [price] — qty ceiling, matches the
                                            # automated path's own flat contract cap
MANUAL_BUY_MAX_RISK_DOLLARS       = OPTIONS_MAX_POSITION_COST  # never let a manual buy's total
                                                                 # cost exceed the same per-trade
                                                                 # ceiling automated entries respect
MANUAL_BUY_MAX_PRICE_VS_ASK_MULT  = 2.0    # fat-finger guard — reject a chosen price more than
                                            # 2x the live ask outright (e.g. typing "15" meaning
                                            # "1.50"); a price BELOW the ask is never blocked here,
                                            # that's just a passive limit that may or may not fill

MONTHLY_LOSS_LIMIT = 0.04          # halt for the month when down ≥4% of account
MONTHLY_PNL_FILE   = "dman_monthly_pnl.json"

# ── VIX-adjusted position sizing ─────────────────────────────────────────────
# Scales share count down proportionally when VIX exceeds baseline.
# VIX 20 (baseline) → 1.0x  |  VIX 30 → 0.67x  |  VIX 40 → 0.50x
# Never sizes UP when VIX is below baseline (floor at 1.0 — risk control only).
VIX_SIZE_BASE = 20.0   # baseline VIX; size is 1.0x at or below this level

# Sector ETF momentum confirmation (score_signal()) uses SECTOR_ETFS below —
# a near-duplicate SECTOR_ETF (singular) dict used to live here with GICS
# full names ("Health Care", "Communication Services", "Consumer
# Discretionary") that didn't match the abbreviated labels TICKER_SECTOR
# actually assigns ("Healthcare", "Comm Services", "Consumer Disc").
# Confirmed live 2026-08-07: that mismatch silently zeroed the 8-pt sector-
# ETF-momentum score for 28 of ~84 curated watchlist tickers (GOOGL, NFLX,
# AMZN, TSLA, every Healthcare name, ...) regardless of whether their
# sector was actually hot. Removed the duplicate — SECTOR_ETFS is the one
# canonical sector-name -> ETF map now.

# Seasonal regime — backtest shows Jan(38% WR), Jul(38%), Aug(25%), Sep(29%), Dec(33%) are chronic losers
SEASONAL_WEAK_MONTHS = {1, 7, 8, 9, 12}
SEASONAL_MIN_SCORE   = 92          # raised bar during weak months

# ADX trend-strength gate — skip directionless/choppy stocks before any pattern check
ADX_TREND_MIN = 20                 # <20 = ranging market; patterns fail more often

ALLOW_SHORTS       = False   # all short setups disabled — prevents accidental live short orders
MAX_POSITIONS      = 5
DAILY_LOSS_LIMIT   = 0.03
MAX_CONSEC_LOSSES  = 3           # halt after this many consecutive losses
EARNINGS_BLACKOUT  = 5           # days before earnings to avoid
WIN_RATE_FILE      = "dman_win_rate.json"
POSITIONS_FILE     = "dman_positions.json"
DAILY_PNL_FILE     = "dman_daily_pnl.json"
PORTFOLIO_HEAT_LIMIT = 0.06      # max total account % at risk across all open positions
ATR_PCT_MIN          = 1.5       # stock must move at least 1.5% avg daily
AVG_DOLLAR_VOL_MIN   = 50_000_000  # $50M avg daily dollar volume floor
MACRO_BLACKOUT       = 1         # days before/after FOMC/NFP to avoid

# Claude API for AI scoring (optional — leave blank to skip)
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")

# GitHub Actions' own auto-provided token — no new secret needed, just
# `permissions: actions: write` on whichever workflow reads it. Powers
# /restart (phone-triggered) and the watchdog's automatic self-heal restart.
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO  = os.getenv("GITHUB_REPOSITORY", "tsingh08/dman-algo-")

# Telegram alerts (optional — set via env vars or hardcode)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Alpaca Paper Trading (optional — set via env vars or hardcode)
# Get keys at app.alpaca.markets → Paper Trading → API Keys
ALPACA_API_KEY    = os.getenv("APCA_API_KEY_ID",    "")   # standard Alpaca env var name
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
BENZINGA_API_KEY  = os.getenv("BENZINGA_API_KEY", "")     # Benzinga Basic — real-time news
# Massive.com's Benzinga-earnings proxy (api.massive.com/benzinga/v1/earnings) —
# unlike BENZINGA_API_KEY's direct calendar endpoint, its ticker filter is
# confirmed server-side accurate (tested live 2026-07-30: ticker=AAPL returns
# only AAPL, with a real date/time/date_status). Reads MASSIVE_API_KEY first
# (Massive's own documented env var name) and falls back to
# BENZINGA_EARNING_API_KEY (this account's existing var) so either works.
MASSIVE_API_KEY   = os.getenv("MASSIVE_API_KEY", "") or os.getenv("BENZINGA_EARNING_API_KEY", "")
ALPACA_PAPER      = False     # LIVE — real brokerage, real money
ENTRY_DRIFT_MAX   = 0.02      # reject signal if price drifted >2% from computed entry
ALPACA_SYNC_FILE   = "dman_alpaca_sync.json"
LAST_ALERTS_FILE   = "dman_last_alerts.json"
ALERT_COOLDOWN_MIN = 30          # suppress duplicate Telegram alert for same ticker within N min
TELEGRAM_STATE_FILE = "dman_telegram_state.json"  # getUpdates offset for two-way bot commands
HALT_FILE           = "dman_halt.json"            # exists = /halt active: no new entries (exits still run)
LIVE_SIGNALS_FILE  = "dman_live_signals.json"   # pending live signals awaiting outcome
LIVE_OUTCOMES_FILE = "dman_live_outcomes.csv"    # ground-truth live trade log
SCAN_LOG_FILE      = "dman_scan_log.json"        # rolling log of each scan run (last 20)
NEWS_LOG_FILE      = "dman_news_log.json"        # rolling background news log — see _log_news_event()
NEWS_LOG_MAX_ENTRIES = 500

# Shared scan metadata written by run_pro_scanner(), read by the heartbeat in main()
_last_scan_meta: dict = {}

SECTOR_ETFS = {
    "Technology":    "XLK",
    "Financials":    "XLF",
    "Healthcare":    "XLV",
    "Energy":        "XLE",
    "Consumer Disc": "XLY",
    "Industrials":   "XLI",
    "Materials":     "XLB",
    "Utilities":     "XLU",
    "Real Estate":   "XLRE",
    "Comm Services": "XLC",
    "Consumer Stap": "XLP",
}

# AI theme bonus (2026-08-07, direct instruction) — AI isn't a GICS/SPDR
# sector, so these names only ever got generic "Technology"/XLK momentum
# scoring even when AI-specific moves were meaningfully sharper than
# broad tech. AIQ (Global X AI & Technology ETF) tracks the AI theme
# specifically — confirmed live data available. This is an ADDITIVE bonus
# on top of the existing sector score, not a replacement — see
# score_signal()'s "4.6 AI Theme Momentum" block.
AI_THEME_ETF = "AIQ"
AI_THEME_TICKERS = {
    "NVDA", "AMD", "AVGO", "SMCI", "MRVL", "ARM", "MU",   # AI infrastructure/chips
    "PLTR", "META", "MSFT", "CRWD", "SNOW", "APP",         # AI software/platforms
    "SOUN", "IONQ",                                        # pure-play AI/quantum
}

# SEC EDGAR Form 4 insider-buying bonus (2026-08-08, "let's do the free
# upgrades that's possible out there completely" — explicitly in place of
# any further paid Massive/Benzinga tiers). Fully public, free, no API key —
# SEC just requires a compliant User-Agent identifying the requester (a
# generic/missing UA gets 403). See check_insider_activity() in the
# "FILTER 05b: INSIDER BUYING" section for the actual signal logic.
SEC_EDGAR_USER_AGENT = "DManAlgo research singh.tanveer2081@gmail.com"
_SEC_CIK_MAP_FILE    = "dman_sec_cik_map.json"
_SEC_CIK_MAP_TTL_S   = 7 * 24 * 3600   # 1 week — the ticker/CIK map barely changes day to day

TICKER_SECTOR = {
    # Mega-cap tech
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "META":"Technology","GOOGL":"Comm Services","AMZN":"Consumer Disc","TSLA":"Consumer Disc",
    "NFLX":"Comm Services","CRM":"Technology","SNOW":"Technology","PLTR":"Technology",
    "SMCI":"Technology","AVGO":"Technology",
    "AMAT":"Technology","MU":"Technology",
    # Memory/storage supercycle names (added 2026-08-13, direct request) —
    # SNDK +13.7%/SKHY +7.3% same-day as the broader rally, both flagged
    # VOLATILE_TICKERS below given the size of that single-day move.
    "SNDK":"Technology","SKHY":"Technology",
    # Semis
    "QCOM":"Technology","MRVL":"Technology","KLAC":"Technology","ON":"Technology",
    # High-beta fintech (ARM, COIN, MARA, PYPL, UPST, SOFI, SHOP removed — 0% WR or <30% WR in backtest)
    "RIOT":"Technology","HOOD":"Financials",
    "AFRM":"Financials",
    # EV / mobility (UBER removed — 0% WR)
    "RIVN":"Consumer Disc","NIO":"Consumer Disc",
    # Clean energy
    "ENPH":"Technology","FSLR":"Technology",
    # Healthcare / biotech
    "MRNA":"Healthcare","ABBV":"Healthcare","LLY":"Healthcare",
    "BIIB":"Healthcare","GILD":"Healthcare","VRTX":"Healthcare","HIMS":"Healthcare",
    "RXRX":"Healthcare",
    # Consumer / social (DKNG, ABNB removed — 0% WR)
    "RBLX":"Comm Services","SPOT":"Comm Services",
    "DUOL":"Technology","CELH":"Consumer Disc","GME":"Consumer Disc",
    # Energy (OXY removed — 0% WR)
    "XOM":"Energy",
    # AI / high-vol growth
    "RKLB":"Industrials",
    "ARM":"Technology","APP":"Technology","ANET":"Technology",  # AI momentum
    # Cybersecurity (high-conviction gap plays, missing from original list)
    "CRWD":"Technology","PANW":"Technology","NET":"Technology",
    # Cloud/SaaS earnings gappers
    "DDOG":"Technology",
    # Crypto-proxy (volatile — min score 88 via VOLATILE_TICKERS)
    "COIN":"Financials",
    # BTC miners / crypto-proxy (massive gap days on BTC moves + earnings)
    "MSTR":"Technology","MARA":"Technology","CLSK":"Technology",
    # Consumer momentum (gap & hold compatible)
    "UBER":"Consumer Disc","ELF":"Consumer Disc","DECK":"Consumer Disc",
    # AI infrastructure / power
    "VST":"Utilities",
    # Additional cybersecurity / cloud earnings gappers
    "FTNT":"Technology","ZS":"Technology","MDB":"Technology",
    # Consumer momentum gappers (affordable options)
    "ONON":"Consumer Disc","ABNB":"Consumer Disc","DKNG":"Consumer Disc",
    # Chinese AI / tech ADRs (Dman watches FXI + DeepSeek rally names)
    "BABA":"Comm Services","BIDU":"Comm Services","PDD":"Consumer Disc",
    "JD":"Consumer Disc","KWEB":"Technology",
    # Quantum computing / AI small-mid cap (20-50% catalyst gaps)
    "IONQ":"Technology","SOUN":"Technology","BBAI":"Technology",
    # Power semis — EV/AI chip plays with large earnings gaps
    "WOLF":"Technology",
    # Space / eVTOL — high-catalyst sectors (contract wins, FAA approvals)
    "LUNR":"Industrials","ACHR":"Industrials","JOBY":"Industrials",
    # Defense / drone tech
    "RCAT":"Industrials",
    # Market regime ETFs
    "SPY":"","QQQ":"","IWM":"",
}

WATCHLIST = list(TICKER_SECTOR.keys())

# Extended universe — scanned with --universe all when FTP/API fails.
# These are strong gap-and-hold candidates not in the curated list.
# RVOL filter still applies — only high-volume days make it through.
EXTENDED_UNIVERSE = [
    # Cybersecurity
    "ZS","FTNT","OKTA","CYBR","S",
    # Cloud / SaaS
    "MDB","GTLB","BILL","SMAR","CELH",
    # Biotech gappers
    "ALNY","BMRN","SRPT","IONS","RCKT",
    # Consumer / lifestyle growth
    "ONON","LULU","CAVA","EXAS","RH",
    # Fintech / payments
    "SQ","NU","MELI","ADYEY",
    # Power / AI infrastructure
    "CEG","NRG","AES","TLN",
    # Other high-momentum
    "AXON","TMDX","CELH","DKNG","ABNB",
]

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1.5 — DYNAMIC UNIVERSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def is_market_open() -> bool:
    """
    Return True if US market is currently open (9:30am–4:00pm ET, Mon–Fri,
    excluding NYSE holidays).

    Confirmed live 2026-08-13 audit: this never checked _MARKET_HOLIDAYS
    (defined later in this file but referenced fine here — function bodies
    only evaluate at call time, long after the whole module has loaded) —
    a pure weekday+time check would report the market open on Thanksgiving,
    July 4th, etc. whenever they land on a weekday. The one caller that
    matters most is _submit_signals_to_alpaca()'s belt-and-suspenders gate
    right before real order submission, specifically for the case its own
    docstring calls out — "in case this function is called directly (e.g.
    --mode alpaca, manual workflow_dispatch after close)" — exactly the
    kind of off-schedule invocation that bypasses whatever normally keeps
    the cron schedule from running on a holiday in the first place.
    """
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    if now.date() in _MARKET_HOLIDAYS:
        return False
    t = now.hour * 100 + now.minute
    return 930 <= t <= 1600

def build_scan_universe(min_price: float = 2.0,
                        min_avg_vol: int  = 200_000,
                        min_dollar_vol: float = 1_000_000,
                        min_rvol: float   = 1.5,
                        hard_cap: int     = 1_200) -> list[str]:
    """
    Fetch all NASDAQ + NYSE listed symbols, filter by price/volume,
    return every ticker with RVOL >= min_rvol (vs a fixed top-N cut).
    Quiet days yield ~200-300 names; active days (NFP, earnings wave) may
    yield 800+. hard_cap guards the scan budget on extreme days.
    Falls back to WATCHLIST if any step fails.
    """
    import io
    print("  [universe] Fetching NASDAQ + NYSE symbol list...", flush=True)
    try:
        # NASDAQ provides the symbol directory via their web interface (more reliable than FTP)
        def _fetch_syms(url: str, sym_col: str) -> list[str]:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            raw = df[sym_col].dropna().astype(str).tolist()
            return [s.strip() for s in raw if s.strip().isalpha() and 1 <= len(s.strip()) <= 5]

        nasdaq_syms = _fetch_syms(
            "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "Symbol")
        other_syms  = _fetch_syms(
            "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "ACT Symbol")
        all_syms = sorted(set(nasdaq_syms + other_syms) - set(WATCHLIST))
    except Exception as e:
        print(f"  [universe] Symbol fetch failed ({e}), using curated + extended list.", flush=True)
        # Fallback: curated watchlist + extended universe (RVOL filter still applied below)
        all_syms = sorted(set(EXTENDED_UNIVERSE) - set(WATCHLIST))

    print(f"  [universe] {len(all_syms):,} symbols → filtering by price/volume...", flush=True)

    # Batch-download 5-day snapshot to score RVOL cheaply
    active: list[tuple[str, float]] = []
    batch_size = 400
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]
    # Confirmed live 2026-08-10: the 7-min budget below only ever gets through
    # ~56% of the batches (18/32 on a real run) — combined with `set()`'s
    # hash-randomized iteration order (a fresh, unseeded value every process,
    # confirmed: the same 7 tickers printed in a different order on every
    # run), this meant a different ~44% of the whole market was silently
    # skipped every single scan with no guarantee any given ticker was ever
    # actually checked. Sorting (above) makes the order deterministic, and
    # rotating the starting batch by day-of-year means each day's ~56% is a
    # DIFFERENT slice that cycles through the full universe over a few days,
    # instead of leaving coverage to chance every run.
    if batches:
        _rotate = date.today().toordinal() % len(batches)
        batches = batches[_rotate:] + batches[:_rotate]
    _univ_start = time.monotonic()
    _univ_budget = 7 * 60  # 7-min cap so universe build never blows the 25-min job timeout
    for idx, batch in enumerate(batches, 1):
        if time.monotonic() - _univ_start > _univ_budget:
            print(f"  [universe] Time budget reached after batch {idx-1} — "
                  f"continuing with {len(active)} candidates", flush=True)
            break
        try:
            snap = yf.download(
                batch, period="5d", progress=False,
                group_by="ticker", auto_adjust=True, threads=True
            )
            for sym in batch:
                try:
                    closes = snap[sym]["Close"].dropna()
                    vols   = snap[sym]["Volume"].dropna()
                    if len(closes) < 2 or len(vols) < 2:
                        continue
                    price    = float(closes.iloc[-1])
                    avg_vol  = float(vols.iloc[:-1].mean())
                    today_vol = float(vols.iloc[-1])
                    if price < min_price or avg_vol < min_avg_vol or price * avg_vol < min_dollar_vol:
                        continue
                    rvol = today_vol / avg_vol if avg_vol > 0 else 0
                    active.append((sym, rvol))
                except Exception:
                    continue
        except Exception:
            continue
        print(f"  [universe] batch {idx}/{len(batches)} done ({len(active)} candidates so far)", flush=True)

    # Every stock showing unusual volume passes — no hard top-N cut.
    # Sort descending so the highest movers are analyzed first (budget hits
    # the tail, not the best candidates).
    active.sort(key=lambda x: x[1], reverse=True)
    dynamic = [sym for sym, rvol in active if rvol >= min_rvol]
    if len(dynamic) > hard_cap:
        print(f"  [universe] {len(dynamic)} candidates exceed hard cap {hard_cap} "
              f"— trimming to top {hard_cap} by RVOL", flush=True)
        dynamic = dynamic[:hard_cap]
    combined = list(dict.fromkeys(WATCHLIST + dynamic))  # curated first, then movers, deduped
    print(f"  [universe] Final universe: {len(combined)} tickers "
          f"({len(WATCHLIST)} curated + {len(dynamic)} RVOL≥{min_rvol}x active)", flush=True)
    return combined


def fetch_premarket_gap_universe(min_price: float = 0.30, max_price: float = 20.0,
                                 min_avg_vol: int = 100_000, min_gap_pct: float = 5.0,
                                 max_tickers: int = 60) -> list[str]:
    """
    Broad pre-market gap sweep for run_premarket_early_scan(), which otherwise only
    checks the curated DMAN_SMALLCAP_WATCHLIST. Confirmed gap: BIYA gapped +39% by
    7 AM ET on 2026-07-27 and was never scanned because it wasn't yet a "known"
    DMan ticker — by the time the hourly cron's dynamic-mover discovery found it
    (10:30 AM), it had already run from ~$2.90 to $4.26.

    Two-stage, since RVOL isn't available pre-market (today's session hasn't
    started yet):
      1. Cheap liquidity/price filter using YESTERDAY's daily bar (same
         batched-download technique as build_scan_universe()) — a floor on
         average volume substitutes for the RVOL check that isn't possible yet.
      2. Batch-check current pre-market price against yesterday's close for a
         real gap; only tickers already gapping ≥min_gap_pct get returned.

    Everything downstream in run_premarket_early_scan() — catalyst tiering,
    EDGAR 8-K check, PDT guard, auto-submit thresholds — is untouched; this
    only widens which tickers reach that existing, already-vetted logic.
    """
    import io
    try:
        def _fetch_syms(url: str, sym_col: str) -> list[str]:
            r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            raw = df[sym_col].dropna().astype(str).tolist()
            return [s.strip() for s in raw if s.strip().isalpha() and 1 <= len(s.strip()) <= 5]

        nasdaq_syms = _fetch_syms("https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt", "Symbol")
        other_syms  = _fetch_syms("https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt", "ACT Symbol")
        all_syms = list(set(nasdaq_syms + other_syms) - set(WATCHLIST) - set(DMAN_SMALLCAP_WATCHLIST))
    except Exception as e:
        print(f"  [pm-universe] Symbol fetch failed ({e}) — skipping broad pre-market sweep", file=sys.stderr)
        return []

    print(f"  [pm-universe] {len(all_syms):,} symbols → filtering by yesterday's liquidity...", flush=True)

    # Stage 1: liquidity/price floor from yesterday's daily bar.
    liquid: dict[str, float] = {}   # ticker -> prev_close
    batch_size = 400
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]
    _start  = time.monotonic()
    # 5-min cap, leaving room for the per-ticker deep dive in run_premarket_early_scan()
    # inside the 35-min workflow timeout. Measured against the real ~12,400-symbol
    # NASDAQ+NYSE list, this covers ~12 of 31 batches (~40%) before time runs out —
    # a real, honest limitation, not full-market coverage. Python's per-process
    # hash randomization means set() ordering (and so which ~40% gets checked)
    # varies run to run, so coverage isn't the same 60% left out every single day.
    _budget = 5 * 60
    for idx, batch in enumerate(batches, 1):
        if time.monotonic() - _start > _budget:
            print(f"  [pm-universe] time budget reached after batch {idx-1} of "
                  f"{len(batches)} — continuing with {len(liquid)} candidates", flush=True)
            break
        try:
            snap = yf.download(batch, period="5d", progress=False,
                               group_by="ticker", auto_adjust=True, threads=True)
            for sym in batch:
                try:
                    closes = snap[sym]["Close"].dropna()
                    vols   = snap[sym]["Volume"].dropna()
                    if len(closes) < 2 or len(vols) < 2:
                        continue
                    price   = float(closes.iloc[-1])
                    avg_vol = float(vols.mean())
                    if min_price <= price <= max_price and avg_vol >= min_avg_vol:
                        liquid[sym] = price
                except Exception:
                    continue
        except Exception:
            continue
    print(f"  [pm-universe] {len(liquid)} liquid candidate(s) → checking pre-market gap...", flush=True)
    if not liquid:
        return []

    # Stage 2: current pre/post-market price vs yesterday's close.
    gappers: list[tuple[str, float]] = []   # (ticker, abs gap %)
    cand_syms    = list(liquid.keys())
    cand_batches = [cand_syms[i:i+batch_size] for i in range(0, len(cand_syms), batch_size)]
    for batch in cand_batches:
        try:
            snap = yf.download(batch, period="1d", interval="1m", prepost=True,
                               progress=False, group_by="ticker", threads=True)
            for sym in batch:
                try:
                    closes = snap[sym]["Close"].dropna()
                    if closes.empty:
                        continue
                    cur        = float(closes.iloc[-1])
                    prev_close = liquid[sym]
                    if prev_close <= 0:
                        continue
                    gap_pct = abs((cur - prev_close) / prev_close * 100)
                    if gap_pct >= min_gap_pct:
                        gappers.append((sym, gap_pct))
                except Exception:
                    continue
        except Exception:
            continue

    gappers.sort(key=lambda x: x[1], reverse=True)
    result = [sym for sym, _ in gappers[:max_tickers]]
    print(f"  [pm-universe] {len(result)} pre-market gapper(s) ≥{min_gap_pct}% found beyond "
          f"curated watchlist: {', '.join(result[:15])}{'...' if len(result) > 15 else ''}", flush=True)
    return result


def fetch_dman_dynamic_tickers(max_tickers: int = 80) -> list[str]:
    """
    Fetch today's actual movers from Yahoo Finance (day gainers + most actives).
    Replaces Finviz, which is consistently blocked in GitHub Actions.
    Returns tickers that are genuinely moving with volume RIGHT NOW — not a
    static screener but a live snapshot of what has institutional interest today.

    Widened 2026-08-07 (direct instruction to cast a wider smallcap net):
    price floor was $2.00, which excluded every sub-$2 penny name from
    organic discovery entirely — DMan's own actual style explicitly
    includes sub-$1 plays (e.g. "$CISS .08s/.09s", confirmed from his real
    posts). Dropped to $0.20 (still above true sub-penny/halted-risk
    territory) so those are actually reachable here instead of only ever
    surfacing via his own curated watchlist. Also doubled the raw
    candidate pool per screener (25 -> 50) so more than just the very top
    movers get a chance to pass the RVOL/volume filters below.
    """
    if not ENABLE_DYNAMIC_SMALLCAP:
        return []
    found: list[tuple[str, float]] = []   # (ticker, rvol)
    _headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    for scr_id in ("day_gainers", "most_actives"):
        url = (
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            f"?formatted=false&lang=en-US&region=US&scrIds={scr_id}&count=50"
            "&fields=symbol,regularMarketPrice,regularMarketVolume,"
            "averageDailyVolume10Day,regularMarketChangePercent"
        )
        try:
            resp = requests.get(url, headers=_headers, timeout=10)
            if resp.status_code != 200:
                continue
            quotes = (resp.json()
                      .get("finance", {})
                      .get("result", [{}])[0]
                      .get("quotes", []))
            for q in quotes:
                sym     = q.get("symbol", "")
                price   = float(q.get("regularMarketPrice", 0) or 0)
                chg_pct = float(q.get("regularMarketChangePercent", 0) or 0)
                vol     = float(q.get("regularMarketVolume", 0) or 0)
                avg_vol = float(q.get("averageDailyVolume10Day", 1) or 1)
                rvol    = vol / avg_vol if avg_vol > 0 else 0
                # Price $0.20–$100, gap ≥ 2%, RVOL ≥ 1.5x, avg vol ≥ 100K
                if (sym and sym.isalpha() and 1 <= len(sym) <= 5
                        and 0.20 <= price <= 100.0
                        and chg_pct >= 2.0
                        and rvol >= 1.5
                        and avg_vol >= 100_000):
                    found.append((sym, rvol))
        except Exception:
            continue
    # Sort by RVOL descending — highest conviction movers first
    found.sort(key=lambda x: x[1], reverse=True)
    unique = list(dict.fromkeys(s for s, _ in found))[:max_tickers]
    return unique


ENABLE_EARNINGS_MOVER_SCAN   = os.getenv("ENABLE_EARNINGS_MOVER_SCAN", "true").strip().lower() == "true"
EARNINGS_MOVER_MIN_IMPORTANCE  = 3        # Benzinga/Massive importance scale, 1-5
EARNINGS_MOVER_MIN_SURPRISE_PCT = 0.05    # at least a 5% EPS or revenue beat
EARNINGS_MOVER_MIN_GAP_PCT     = 5.0      # live price must actually be reacting, not just a beat on paper
EARNINGS_MOVER_MIN_AVG_VOLUME  = 300_000  # liquid enough to trade on a small account without brutal spreads
EARNINGS_MOVER_MAX_PRICE       = 500.0    # keep position sizing realistic on a small account


def _alert_massive_api_failure(source: str, detail: str) -> None:
    """
    One Telegram alert per source per day on a Massive/Benzinga API
    failure. Added 2026-08-13 after an API audit found both
    fetch_earnings_mover_tickers() and _fetch_massive_reference_news()
    fail open (return []) on any non-200/exception with zero logging or
    alerting — the exact silent-failure shape that cost a full week of
    missed options signals during the 2026-08-06 OPRA entitlement 403
    before anyone noticed. Those two functions are discovery/gating nets
    rather than critical-path order logic, so this still fails open
    (never blocks a scan), it just makes a persistent outage visible
    instead of silently degrading earnings-mover discovery or the
    catalyst sentiment gate for days. _is_alerted_today's existing daily
    dedup naturally rate-limits this to one ping per source per day even
    if the underlying call fails on every single invocation.
    """
    key = f"MASSIVE_API_FAIL_{source}_{date.today().isoformat()}"
    if _is_alerted_today(key):
        return
    try:
        send_telegram(
            f"⚠️ <b>Massive API failure</b> — {source}\n{detail}\n"
            f"Failing open (returning empty) — check MASSIVE_API_KEY / "
            f"api.massive.com status if this repeats."
        )
    except Exception:
        pass
    _mark_alerted(key)


def fetch_earnings_mover_tickers(max_tickers: int = 15) -> list[str]:
    """
    Broad, watchlist-independent earnings-reaction scan. Every other earnings
    lookup in this file (check_earnings_safe, _recent_earnings_surprise,
    get_upcoming_earnings, the premarket briefing's earnings section) only
    ever queries ONE ticker at a time, and only for tickers already on
    WATCHLIST/DMAN_SMALLCAP_WATCHLIST — so a name that isn't already on
    those fixed lists is structurally invisible no matter how big its move.

    Confirmed live 2026-08-12: CRWV beat EPS by +30.9% and gapped +15.7% the
    next morning on 24M avg daily volume — about as clean a catalyst as
    exists — and the algo never once considered it, because CRWV was never
    on WATCHLIST. Also confirmed live the same day: Massive's
    /benzinga/v1/earnings endpoint (see _fetch_massive_earnings) actually
    supports a plain date-range query with NO ticker filter, returning every
    company reporting in that window — nothing in this file used it that
    way before. This closes the blind spot the same way
    fetch_dman_dynamic_tickers() already closes it for pure price/volume
    movers: discover new tickers and feed them into the SAME
    scan_signal()/score_signal() pipeline everything else goes through
    (MTF, sector, RS, macro-safe, earnings-safe blackout, heat cap,
    confluence score, min-score gate) rather than building a separate entry
    path with its own risk surface.

    Window is yesterday through today so both last night's AMC reports
    (reacting at today's open) and this morning's BMO reports are covered.
    Requires a real beat on EPS or revenue AND a live gap actually
    confirming the market cares — ALLOW_SHORTS is False, so a miss (even a
    big one) isn't tradeable here regardless of how dramatic the gap down
    is. Fails open to an empty list on any error — this is a discovery net,
    not a critical-path dependency the rest of the scan needs to function.
    """
    if not ENABLE_EARNINGS_MOVER_SCAN or not MASSIVE_API_KEY:
        return []
    try:
        yesterday = date.today() - timedelta(days=1)
        resp = requests.get(
            "https://api.massive.com/benzinga/v1/earnings",
            params={
                "date.gte": yesterday.isoformat(),
                "date.lte": date.today().isoformat(),
                "limit":    200,
                "apiKey":   MASSIVE_API_KEY,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            _alert_massive_api_failure(
                "earnings-mover", f"HTTP {resp.status_code} from /benzinga/v1/earnings")
            return []
        results = resp.json().get("results", []) or []
    except Exception as exc:
        _alert_massive_api_failure("earnings-mover", str(exc))
        return []

    found: list[tuple[str, float]] = []   # (ticker, gap_pct)
    for r in results:
        ticker = str(r.get("ticker", "")).upper().strip()
        if not ticker or not ticker.isalpha() or not (1 <= len(ticker) <= 5):
            continue
        if ticker in WATCHLIST:
            continue   # already covered by the normal watchlist path
        if (r.get("importance") or 0) < EARNINGS_MOVER_MIN_IMPORTANCE:
            continue
        eps_pct = r.get("eps_surprise_percent")
        rev_pct = r.get("revenue_surprise_percent")
        if eps_pct is None and rev_pct is None:
            continue
        beat = ((eps_pct is not None and eps_pct >= EARNINGS_MOVER_MIN_SURPRISE_PCT) or
                (rev_pct is not None and rev_pct >= EARNINGS_MOVER_MIN_SURPRISE_PCT))
        if not beat:
            continue
        try:
            px = get_live_price(ticker)
            if px is None or px <= 0 or px > EARNINGS_MOVER_MAX_PRICE:
                continue
            hist = fetch_df(ticker, period_days=30)
            if hist is None or len(hist) < 5:
                continue
            # fetch_df's last daily bar is TODAY's still-forming bar once
            # the market has opened — this function runs from every hourly
            # scan throughout the day, not just pre-market. Comparing the
            # live quote against that same still-forming bar's close made
            # gap_pct ≈ 0 on every call, so this discovery feature (built
            # specifically to catch a CRWV-style mover) never once actually
            # returned a candidate. Use the last COMPLETED bar (yesterday's
            # close) whenever today's bar is already present in the
            # history; fall back to the last bar itself on the rare
            # pre-market call where today's bar doesn't exist yet.
            try:
                _last_bar_is_today = hist.index[-1].date() == date.today()
            except Exception:
                _last_bar_is_today = True   # safest assumption during market hours
            prev_close = float(hist["Close"].iloc[-2] if _last_bar_is_today else hist["Close"].iloc[-1])
            avg_vol    = float(hist["Volume"].tail(10).mean())
            if avg_vol < EARNINGS_MOVER_MIN_AVG_VOLUME or prev_close <= 0:
                continue
            gap_pct = (px - prev_close) / prev_close * 100
            if gap_pct < EARNINGS_MOVER_MIN_GAP_PCT:
                continue   # beat on paper, but the market isn't actually reacting
        except Exception:
            continue
        found.append((ticker, gap_pct))

    found.sort(key=lambda x: x[1], reverse=True)
    unique = list(dict.fromkeys(t for t, _ in found))[:max_tickers]
    return unique


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

_cache: dict[str, pd.DataFrame] = {}
_indicator_cache: dict[str, pd.DataFrame] = {}   # see _compute_indicators_cached()

def _bars_to_df(bars: list, min_bars: int = 20) -> Optional[pd.DataFrame]:
    """Convert a list of alpaca-py Bar objects to the OHLCV DataFrame shape
    used everywhere else in this file. Applies the same staleness check
    (last bar within 3 calendar days) as the yfinance path. Never raises —
    any malformed/unexpected bar data returns None (caller falls back to
    yfinance) rather than propagating an exception into the scan loop."""
    try:
        if len(bars) < min_bars:
            return None
        _records = [
            {"Open": b.open, "High": b.high, "Low": b.low,
             "Close": b.close, "Volume": b.volume, "Date": b.timestamp}
            for b in bars
        ]
        df = pd.DataFrame(_records).set_index("Date")
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        _last_bar_date = df.index[-1].date()
        if (date.today() - _last_bar_date).days > 3:
            return None
        return df
    except Exception:
        return None


def _load_feed_state(state: dict, filepath: str) -> None:
    """
    Shared load logic for _load_stock_feed_state()/_load_options_feed_state().

    Confirmed live 2026-08-08 (options side): the in-memory feed-state
    dict resets to {"feed": None, "checked_at": 0.0} on every fresh
    process — GitHub Actions spins up a brand-new process for every scan/
    momentum-watch/daemon-session run, so from each process's perspective
    "feed" always starts at None. The "only alert on a real state change"
    check then reads every single re-probe as a fresh None -> downgraded
    transition and re-sends the Telegram alert, and the hourly recheck
    throttle never actually throttles across processes either (checked_at
    resets to 0.0 every time too, so every process re-probes immediately).
    Persisting to disk makes both the recheck throttle and the alert-on-
    change check work across process boundaries, not just within one.

    Mutates `state` in place (.clear()+.update(), never reassigns it) so
    the caller's module-level dict object identity is preserved — no
    `global` needed here, and _resolve_stock_feed()/_resolve_options_feed()
    can keep mutating the same dict via subscript afterward.
    """
    if state["checked_at"] != 0.0:
        return   # already loaded (or updated) this process — don't clobber
    try:
        if os.path.exists(filepath):
            with open(filepath) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and "feed" in loaded and "checked_at" in loaded:
                state.clear()
                state.update(loaded)
    except Exception:
        pass


def _save_feed_state(state: dict, filepath: str) -> None:
    """Shared save logic for _save_stock_feed_state()/_save_options_feed_state()."""
    try:
        _write_json_atomic(filepath, state)
    except Exception:
        pass


_stock_feed_state: dict = {"feed": None, "checked_at": 0.0}
_STOCK_FEED_RECHECK_S = 3600   # re-probe hourly — mirrors _resolve_options_feed()
_STOCK_FEED_STATE_FILE = "dman_stock_feed_state.json"

def _load_stock_feed_state() -> None:
    _load_feed_state(_stock_feed_state, _STOCK_FEED_STATE_FILE)


def _save_stock_feed_state() -> None:
    _save_feed_state(_stock_feed_state, _STOCK_FEED_STATE_FILE)


def _resolve_stock_feed() -> str:
    """
    Mirrors _resolve_options_feed() for the stock-side SIP entitlement.
    Algo Trader Plus billing voided twice (invoices 2RAEKK6H-0001,
    2RAEKK6H-0002) before finally activating 2026-08-08 -- if the
    subscription ever lapses again (payment failure, etc.), the existing
    per-call try/except in _fetch_alpaca_daily / prewarm_alpaca_bars /
    _fetch_intraday_bars would silently swallow the resulting 403 and fall
    through to the yfinance fallback with ZERO visibility that anything
    changed — exactly the gap that let OPRA's entitlement flip go unnoticed
    for up to a week before _resolve_options_feed() existed. This probes
    once (cached for _STOCK_FEED_RECHECK_S, persisted across processes) and
    actually alerts on a downgrade or a restoration.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return STOCK_DATA_FEED
    _load_stock_feed_state()
    now = time.time()
    if _stock_feed_state["feed"] is not None and \
       (now - _stock_feed_state["checked_at"]) < _STOCK_FEED_RECHECK_S:
        return _stock_feed_state["feed"]

    _stock_feed_state["checked_at"] = now
    # Default to whatever was cached before THIS probe, not the preferred
    # feed. Found 2026-08-16 review: this used to always start as
    # STOCK_DATA_FEED (the preferred feed), and the except-branch never
    # touched it — so any transient error (a timeout/connect failure, not
    # just a clean 403 response) silently reverted a real, still-active
    # downgrade (e.g. cached "iex") back to the preferred feed for the
    # next _STOCK_FEED_RECHECK_S (1hr) window. Every call in that window
    # then hit real 403s and fell through to the slower yfinance path —
    # exactly the silent-degradation failure mode this mechanism exists to
    # prevent, just for up to an hour instead of indefinitely. The
    # comment here already said "keep whatever we resolved last" — this
    # makes the code actually do that.
    resolved = _stock_feed_state["feed"] or STOCK_DATA_FEED
    try:
        r = requests.get(
            "https://data.alpaca.markets/v2/stocks/AAPL/quotes/latest",
            headers={"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
            params={"feed": STOCK_DATA_FEED},
            timeout=8,
        )
        if r.status_code == 403:
            resolved = "iex"
            if _stock_feed_state["feed"] != resolved:   # only alert on a real state change
                print(f"  ⚠️  {STOCK_DATA_FEED} stock feed not entitled — falling back to iex")
                send_telegram(
                    f"⚠️ <b>Stock data feed downgraded</b>\n"
                    f"{STOCK_DATA_FEED} returned 403 (not entitled) — falling back to iex "
                    f"(free tier, single-exchange quotes, no consolidated tape). "
                    f"Check the Algo Trader Plus subscription/billing if this is unexpected."
                )
        elif r.status_code == 200:
            resolved = STOCK_DATA_FEED
            if _stock_feed_state["feed"] == "iex":
                # Was on fallback, preferred feed just came back — worth knowing.
                send_telegram(f"✅ <b>{STOCK_DATA_FEED} stock feed entitlement restored</b> — back to real-time SIP quotes.")
    except Exception:
        pass   # network hiccup — keep whatever was cached (resolved already defaults to it)

    _stock_feed_state["feed"] = resolved
    _save_stock_feed_state()
    return resolved


def _fetch_alpaca_daily(ticker: str, period_days: int) -> Optional[pd.DataFrame]:
    """Single-ticker Alpaca SIP daily bars (Algo Trader Plus real-time feed)."""
    if not ALPACA_AVAILABLE:
        return None
    try:
        dc = get_alpaca_data_client()
        if dc is None:
            return None
        from datetime import timezone as _tz
        _alp_req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=(datetime.today() - timedelta(days=period_days + 5)).replace(tzinfo=_tz.utc),
            feed=_resolve_stock_feed(),
        )
        _resp = dc.get_stock_bars(_alp_req)
        # BarSet doesn't support `in` the way a dict does — `ticker in _resp` is
        # always False even when _resp.data[ticker] has real bars (confirmed
        # live 2026-07-29), so this always silently returned [] before. .data
        # is a real dict; use it directly.
        _bars = _resp.data.get(ticker, [])
        return _bars_to_df(_bars)
    except Exception:
        return None


def prewarm_alpaca_bars(tickers: list[str], period_days: int = 430,
                        chunk_size: int = 100) -> int:
    """
    Batch-fetch daily bars for many tickers in a handful of Alpaca SIP calls
    (Algo Trader Plus — paid for exactly this: fast, reliable, real-time
    consolidated-tape data) instead of one yfinance call per ticker. Warms
    the shared `_cache` so every subsequent fetch_df(ticker) call in the
    scan hits the cache immediately — this is what actually uses the
    subscription for the thing that matters most (scan speed/reliability),
    rather than only as an emergency fallback when yfinance fails.

    Fail-safe by design: any error just means fewer tickers got pre-warmed
    this run — fetch_df() always has its own per-ticker Alpaca-then-
    yfinance path as a complete backstop. Returns count of tickers warmed.
    The entire function body is one outer try/except (belt-and-suspenders
    on top of the per-chunk try/except below) — this function must NEVER
    be able to take down the scan that calls it, whatever goes wrong.

    chunk_size loosened 50→100 and the inter-chunk pause 0.2s→0.1s
    (2026-08-13, API-usage audit): the original chunk_size=50 was picked
    before this account had ever actually run under real ATP credentials,
    purely as an untested guess against undocumented multi-symbol batch
    limits. It has since run this exact path across every live session
    since ATP was confirmed active with zero recorded 429s
    (dman_rate_limit_events.json has never had a prewarm-related entry) —
    still keeping a real, non-zero pause rather than removing it outright,
    since that history doesn't prove there's no limit, only that this
    account's actual daily universe (WATCHLIST + smallcap watchlist,
    ~200 tickers, 2 chunks at the new size) has never come close to it.
    """
    try:
        if not ALPACA_AVAILABLE:
            return 0
        dc = get_alpaca_data_client()
        if dc is None:
            return 0
        from datetime import timezone as _tz
        warmed = 0
        _start = (datetime.today() - timedelta(days=period_days + 5)).replace(tzinfo=_tz.utc)
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            try:
                _req = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=_start,
                    feed=_resolve_stock_feed(),
                )
                _resp = dc.get_stock_bars(_req)
                for ticker in chunk:
                    key = f"{ticker}_1d"
                    if key in _cache:
                        continue
                    try:
                        # see _fetch_alpaca_daily — BarSet's `in` always returns
                        # False, so this always silently returned [] before.
                        _bars = _resp.data.get(ticker, [])
                        df = _bars_to_df(_bars)
                    except Exception:
                        continue   # one malformed ticker's bars must not skip the rest of the chunk
                    if df is not None:
                        _cache[key] = df
                        warmed += 1
                if i + chunk_size < len(tickers):
                    time.sleep(0.1)   # brief pause between chunks — avoid rate-limit bursts
            except Exception as exc:
                print(f"  [prewarm_alpaca_bars] chunk {i}-{i+len(chunk)} failed "
                      f"(non-fatal, falls back to per-ticker fetch): {exc}", file=sys.stderr)
                continue
        return warmed
    except Exception as exc:
        print(f"  [prewarm_alpaca_bars] aborted (non-fatal, falls back to "
              f"per-ticker fetch): {exc}", file=sys.stderr)
        return 0


def fetch_df(ticker: str, period_days: int = 430,
             interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    OHLCV data with in-memory caching. Alpaca SIP (Algo Trader Plus —
    real-time consolidated tape, paid subscription) is tried first for
    daily bars since it's faster and more reliable than yfinance's
    unofficial, rate-limited endpoint; yfinance remains the fallback for
    tickers Alpaca doesn't cover (thin OTC/penny names) or if Alpaca is
    unavailable. Usually a no-op here anyway — prewarm_alpaca_bars()
    already populates the cache in one batch call before the scan loop
    starts, so this mostly just returns the cached result.
    """
    key = f"{ticker}_{interval}"
    if key in _cache:
        return _cache[key]

    if interval == "1d":
        df = _fetch_alpaca_daily(ticker, period_days)
        if df is not None:
            _cache[key] = df
            return df

    for attempt in range(3):
        try:
            end   = datetime.today()
            start = end - timedelta(days=period_days)
            raw   = yf.download(ticker, start=start, end=end,
                                interval=interval, progress=False, auto_adjust=True)
            if raw is None or len(raw) < 20:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            # Staleness check — last bar must be within 3 calendar days (handles weekends/holidays)
            # Protects against yfinance returning cached prior-session data as "today"
            _last_bar_date = raw.index[-1].date() if hasattr(raw.index[-1], "date") else raw.index[-1]
            if (date.today() - _last_bar_date).days > 3:
                print(f"  [fetch_df] {ticker} data stale (last bar {_last_bar_date}) — skipping", file=sys.stderr)
                return None
            _cache[key] = raw
            return raw
        except Exception as exc:
            if attempt < 2:
                time.sleep(1 << attempt)  # 1s then 2s backoff
            else:
                print(f"  [fetch_df] {ticker} failed after 3 attempts: {exc}", file=sys.stderr)
    return None


def fetch_weekly(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch weekly OHLCV (1wk interval) for multi-timeframe analysis."""
    return fetch_df(ticker, period_days=730, interval="1wk")


def get_current_price(ticker: str) -> Optional[float]:
    """Get the most recent close price."""
    df = fetch_df(ticker)
    if df is None or df.empty:
        return None
    return float(df["Close"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — INDICATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def is_opex_week() -> bool:
    """True if this week contains the third Friday of the month (monthly options expiration)."""
    today = date.today()
    first = today.replace(day=1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    week_start = third_friday - timedelta(days=third_friday.weekday())
    return week_start <= today <= week_start + timedelta(days=4)


def _get_third_friday(yr: int, mo: int) -> date:
    """Return the third Friday of the given year/month."""
    first = date(yr, mo, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(weeks=2)


def is_opex_day() -> bool:
    """True only on the actual OPEX Friday (3rd Friday of the month)."""
    today = date.today()
    return today.weekday() == 4 and today == _get_third_friday(today.year, today.month)


def is_opex_eve() -> bool:
    """True on the Thursday immediately before OPEX Friday."""
    tomorrow = date.today() + timedelta(days=1)
    return tomorrow.weekday() == 4 and tomorrow == _get_third_friday(tomorrow.year, tomorrow.month)


_beta_cache: dict[str, float] = {}

def get_beta(ticker: str) -> float:
    """Fetch stock beta from yfinance info dict. Cached per session. Returns 1.0 on failure."""
    if ticker in _beta_cache:
        return _beta_cache[ticker]
    try:
        raw = yf.Ticker(ticker).info.get("beta") or 1.0
        beta = float(raw)
        beta = max(0.1, min(5.0, beta))
    except Exception:
        beta = 1.0
    _beta_cache[ticker] = beta
    return beta


def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    # Telegram hard limit is 4096 chars per message. Truncate rather than drop silently.
    if len(message) > 4000:
        message = message[:3970] + "\n… <i>[truncated]</i>"
    for attempt in range(2):
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code == 200:
                return True
            # Non-200 — log so GitHub Actions captures it
            print(f"  [Telegram] HTTP {resp.status_code}: {resp.text[:120]}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)   # brief pause before retry
        except Exception as exc:
            print(f"  [Telegram] send failed (attempt {attempt+1}): {exc}", file=sys.stderr)
            if attempt == 0:
                time.sleep(3)
    return False


# The full command list, kept in exactly one place — the unknown-command
# help text in _handle_telegram_command() and _register_telegram_commands()
# both source from this so the two can't silently drift apart the way the
# git-sync logic across workflow files did.
TELEGRAM_COMMANDS: list[tuple[str, str]] = [
    ("status",    "Account, halt state, positions count"),
    ("positions", "List tracked positions"),
    ("pnl",       "Today + month P&L"),
    ("halt",      "Block new entries (exits still run)"),
    ("resume",    "Re-enable entries"),
    ("close",     "Close a position now — /close TICKER"),
    ("why",       "Explain a ticker's current signal — /why TICKER"),
    ("options",   "Browse live calls/puts — /options TICKER [E]"),
    ("buy",       "Confirm a buy from /options — /buy N [qty] [price]"),
    ("restart",   "Force a fresh daemon session (last resort)"),
]


def _register_telegram_commands() -> bool:
    """
    Registers TELEGRAM_COMMANDS with Telegram's setMyCommands so they show
    up in the client's command autocomplete/menu button instead of only
    working if typed out from memory. Confirmed live 2026-08-13 API audit:
    only sendMessage and getUpdates were ever used anywhere in this file —
    setMyCommands costs one call and is idempotent (Telegram just overwrites
    the registered list), so it's safe to call on every daemon startup
    rather than needing one-time bookkeeping. Never raises — a failure here
    only means the command menu doesn't populate, not that commands stop
    working (they're still handled by _handle_telegram_command() either way).
    """
    if not TELEGRAM_TOKEN:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setMyCommands",
            json={"commands": [{"command": c, "description": d} for c, d in TELEGRAM_COMMANDS]},
            timeout=10,
        )
        return resp.status_code == 200 and resp.json().get("ok", False)
    except Exception as exc:
        print(f"  [Telegram] setMyCommands failed: {exc}", file=sys.stderr)
        return False


def _send_signal_alert_batch(messages: list[str]) -> None:
    """
    Sends one or more formatted signal alerts as digest message(s) instead
    of one Telegram message per signal. Confirmed live 2026-08-13 API
    audit: send_telegram() had zero batching/combining logic anywhere — a
    single scan producing several signals fired that many separate
    notifications, each repeating its own header/formatting overhead.

    A single signal is sent exactly as before (no behavior change for the
    common case). Multiple signals are combined under one shared header,
    separated by a divider, splitting into more than one Telegram message
    only if the combined length would exceed a safe margin below
    Telegram's ~4096-char hard limit — send_telegram()'s own truncation
    is a last-resort safety net, not something this should lean on to
    silently cut a real alert's content.
    """
    if not messages:
        return
    if len(messages) == 1:
        send_telegram(messages[0])
        return
    _header = f"🔥 <b>{len(messages)} new plays found</b>\n{'─'*20}"
    _chunk_parts = [_header]
    _chunk_len   = len(_header)
    for msg in messages:
        _piece = "\n\n" + msg
        if _chunk_len + len(_piece) > 3800 and len(_chunk_parts) > 1:
            send_telegram("".join(_chunk_parts))
            _chunk_parts = [_header]
            _chunk_len   = len(_header)
        _chunk_parts.append(_piece)
        _chunk_len += len(_piece)
    if len(_chunk_parts) > 1:
        send_telegram("".join(_chunk_parts))


def format_signal_telegram(s: "ProSignal", regime: dict) -> str:
    """Format a ProSignal as a Telegram-ready HTML message."""
    arrow = "🟢 LONG" if s.bias == "LONG" else "🔴 SHORT"
    opex  = " ⚠️ OpEx week" if is_opex_week() else ""

    # For stocks >$10 with options enabled: flag it so user knows options alert follows
    if ENABLE_OPTIONS and s.entry > OPTIONS_MIN_PRICE and s.setup in OPTIONS_SETUPS:
        trade_note = "\n🎯 <b>OPTIONS play</b> — trade the ITM call below, not the stock directly."
    elif s.entry <= OPTIONS_MIN_PRICE:
        trade_note = "\n📈 <b>STOCK play</b> — price under $10, buy shares directly."
    else:
        trade_note = ""

    # FOMC proximity warning — if FOMC is within 5 calendar days, flag it in the alert
    _today_sig = date.today()
    _fomc_note = ""
    for _ev in sorted(_FOMC_DATES):
        _d = (_ev - _today_sig).days
        if 1 <= _d <= 5:
            if _d == 2 and _today_sig.weekday() == 0:  # Monday, FOMC exactly Wednesday
                _fomc_note = (f"\n⚠️ <b>FOMC {_ev.strftime('%a %b %d')} in {_d}d — LAST CLEAR DAY</b> "
                              f"— size at 75%, target T1, exits by Friday preferred")
            else:
                _fomc_note = (f"\n⚠️ <b>FOMC {_ev.strftime('%a %b %d')} in {_d}d</b> "
                              f"— size at 75%, target T1 only")
            break
        if _d > 5:
            break

    # Friday warning — positions opened today carry weekend risk
    _friday_note = ""
    if _today_sig.weekday() == 4:
        _friday_note = "\n📅 <b>Friday signal</b> — plan your exit before close (weekend risk)"

    # Earnings beat/miss context — real numbers, not narrative. Explains
    # WHY a Gap & Hold/Morning Runner fired if the gap is earnings-driven,
    # instead of leaving the trader to guess. See _recent_earnings_surprise.
    _earn_note = ""
    try:
        _surprise = _recent_earnings_surprise(s.ticker)
        if _surprise:
            _earn_note = _format_earnings_surprise_note(_surprise)
    except Exception:
        pass

    return (
        f"<b>D🔥man Signal{opex}</b>\n"
        f"{arrow} <b>{s.ticker}</b> — {s.setup}\n"
        f"Entry: <b>${s.entry}</b>  Stop: ${s.stop}\n"
        f"T1 (2R): ${s.target1}   T2 (3R): ${s.target2}\n"
        f"R/R: {s.rr}:1   RSI: {s.rsi}   RVOL: {s.rvol}x\n"
        f"Score: {s.confluence_score}/100"
        + (f"  AI: {s.ai_score}/10" if s.ai_score else "")
        + f"\nMarket: {regime.get('regime','?')}\n{s.reason}"
        + trade_note
        + _earn_note
        + _fomc_note
        + _friday_note
        + f"\n💬 <code>/options {s.ticker}</code> to browse strikes and buy"
    )


def _write_json_atomic(path: str, data, **dump_kwargs) -> None:
    """
    Write JSON to `path` crash-safely: serialize to a temp file in the same
    directory, then atomically swap it into place via os.replace().

    Found 2026-08-16 review: every JSON write in this file used to be a
    plain `open(path, "w")` + `json.dump()` — a process killed mid-write
    (OOM-kill, a VPS reboot, a broken pipe) leaves a truncated file, and
    every reader's `except (FileNotFoundError, json.JSONDecodeError)`
    handler treats that identically to "this file never existed." For
    PositionTracker specifically, that means silently reporting ZERO open
    positions to a live-money guard loop instead of surfacing real
    corruption. os.replace() is atomic on both POSIX and Windows (unlike
    os.rename(), which raises on Windows if the destination already
    exists) — a reader can never observe a partially-written file, only
    the fully-old or fully-new one.
    """
    _dir = os.path.dirname(os.path.abspath(path)) or "."
    _fd, _tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=_dir)
    try:
        with os.fdopen(_fd, "w") as f:
            json.dump(data, f, **dump_kwargs)
        os.replace(_tmp_path, path)
    except Exception:
        try:
            os.unlink(_tmp_path)
        except Exception:
            pass
        raise


def _load_last_alerts() -> dict:
    try:
        with open(LAST_ALERTS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_last_alert(ticker: str) -> None:
    alerts = _load_last_alerts()
    alerts[ticker] = datetime.now(ET).isoformat()
    try:
        _write_json_atomic(LAST_ALERTS_FILE, alerts)
    except Exception:
        pass

def _is_duplicate_alert(ticker: str) -> bool:
    alerts = _load_last_alerts()
    if ticker not in alerts:
        return False
    try:
        last = datetime.fromisoformat(alerts[ticker])
        return (datetime.now(ET) - last).total_seconds() < ALERT_COOLDOWN_MIN * 60
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  LIVE OUTCOME LOGGER — ground-truth WR tracker (fills the daily-bar gap)
# ═══════════════════════════════════════════════════════════════════════════

def _log_live_signal(sig: "ProSignal") -> None:
    """Write a fired alert to the pending live signals file."""
    try:
        data = json.load(open(LIVE_SIGNALS_FILE)) if os.path.exists(LIVE_SIGNALS_FILE) else {}
    except Exception:
        data = {}
    pending = data.get("pending", [])
    today = datetime.now(ET).strftime("%Y-%m-%d")
    # Skip if same ticker+date already logged
    if any(p["ticker"] == sig.ticker and p["date"] == today for p in pending):
        return
    pending.append({
        "ticker":   sig.ticker,
        "setup":    sig.setup,
        "bias":     sig.bias,
        "entry":    sig.entry,
        "stop":     sig.stop,
        "target1":  sig.target1,
        "target2":  sig.target2,
        "score":    sig.confluence_score,
        "date":     today,
        "timestamp": datetime.now(ET).isoformat(),
    })
    data["pending"] = pending
    try:
        _write_json_atomic(LIVE_SIGNALS_FILE, data, indent=2)
    except Exception:
        pass


def _simulate_trade_outcome(ticker: str, entry: float, stop: float,
                             target1: float, target2: float, bias: str,
                             start_date: str) -> dict:
    """
    Simulate trade outcome using daily bars from start_date onward.
    Returns dict with exit_date, exit_px, exit_reason, outcome, pnl_pct, hold_bars.
    Returns None if trade is still open (not enough bars yet).
    """
    try:
        start = date.fromisoformat(start_date)
        # Fetch enough bars (add buffer for weekends/holidays)
        end = datetime.now(ET).date() + timedelta(days=1)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or len(df) < 2:
            return None
        # Newer yfinance versions return MultiIndex columns even for a single
        # ticker (e.g. ('High', 'AAPL') instead of 'High') — without this,
        # bar["High"] on a row returns a Series instead of a scalar, and
        # float(bar["High"]) throws. fetch_df() already does this same
        # normalization for its own yf.download() call; this function has
        # its own separate call that was missing it. This was the actual
        # cause of both DMan PRO Scanner failures on 2026-07-24: MBLY's
        # pending signal became eligible for resolution that day (its
        # start_date's next calendar day) and every run_pro_scanner() call
        # crashed here before reaching the scan itself.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        # Skip the entry day — we enter at the alert price; simulation starts next bar
        df = df.iloc[1:]
        if len(df) == 0:
            return None
    except Exception:
        return None

    is_long   = bias == "LONG"
    trail_stop = stop
    be1r_px   = (entry + (entry - stop)) if is_long else (entry - (stop - entry))
    be1r_set  = False
    t1_hit    = False
    hold      = 0

    for i in range(len(df)):
        bar  = df.iloc[i]
        H    = float(bar["High"])
        L    = float(bar["Low"])
        C    = float(bar["Close"])
        hold += 1
        exit_bar_date = str(df.index[i].date())

        # Stop check FIRST, against trail_stop as it stood at the START of
        # this bar -- found in the 2026-08-16 review: this used to run
        # AFTER the BE@1R block below, so a bar whose High reached +1R
        # AND whose Low also breached the ORIGINAL stop got the stop
        # check evaluated against the JUST-PROMOTED breakeven stop
        # instead, misclassifying it as a breakeven "win" (~0%). A daily
        # bar can't tell you which extreme happened first intraday — the
        # Low could just as easily have breached the original stop
        # BEFORE the High ever reached +1R later that same day, which
        # would be a real loss, not breakeven. Checking the stop against
        # the pre-bar trail_stop (conservative: assume the worse-case
        # ordering, same convention already used correctly for the T1
        # block further below) closes that look-ahead gap, which could
        # otherwise inflate the reported win rate.
        stopped = (is_long and L <= trail_stop) or (not is_long and H >= trail_stop)
        if stopped:
            exit_px     = trail_stop
            exit_reason = "STOP(BE)" if be1r_set else "STOP"
            pnl_pct     = ((exit_px - entry) / entry * 100) if is_long else ((entry - exit_px) / entry * 100)
            return {"exit_date": exit_bar_date, "exit_px": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "outcome": "WIN" if pnl_pct >= 0 else "LOSS",
                    "pnl_pct": round(pnl_pct, 2), "hold_bars": hold}

        # BE@1R: move stop to entry once 1R profit is reached (before T1) --
        # applies only to FUTURE bars now, same as T1's own trail_stop move.
        if not be1r_set and not t1_hit:
            if (is_long and H >= be1r_px) or (not is_long and L <= be1r_px):
                trail_stop = entry
                be1r_set   = True

        # T1 hit — scale out, move stop to breakeven
        if not t1_hit:
            if (is_long and H >= target1) or (not is_long and L <= target1):
                t1_hit     = True
                trail_stop = entry
                continue

        # T2 hit (only after T1)
        if t1_hit:
            if (is_long and H >= target2) or (not is_long and L <= target2):
                exit_px  = target2 * (0.999 if is_long else 1.001)
                pnl_pct  = ((exit_px - entry) / entry * 100) if is_long else ((entry - exit_px) / entry * 100)
                return {"exit_date": exit_bar_date, "exit_px": round(exit_px, 2),
                        "exit_reason": "T2", "outcome": "WIN",
                        "pnl_pct": round(pnl_pct, 2), "hold_bars": hold}

        # Stall exit: <0.5% move after 3 bars and T1 not yet hit
        if hold >= 3 and not t1_hit and abs(C - entry) / entry * 100 < 0.5:
            pnl_pct = ((C - entry) / entry * 100) if is_long else ((entry - C) / entry * 100)
            return {"exit_date": exit_bar_date, "exit_px": round(C, 2),
                    "exit_reason": "STALL",
                    "outcome": "WIN" if pnl_pct >= 0 else "LOSS",
                    "pnl_pct": round(pnl_pct, 2), "hold_bars": hold}

        # Time exit: 15 bars
        if hold >= 15:
            pnl_pct = ((C - entry) / entry * 100) if is_long else ((entry - C) / entry * 100)
            return {"exit_date": exit_bar_date, "exit_px": round(C, 2),
                    "exit_reason": "TIME",
                    "outcome": "WIN" if pnl_pct >= 0 else "LOSS",
                    "pnl_pct": round(pnl_pct, 2), "hold_bars": hold}

    return None  # still open


def resolve_live_outcomes(verbose: bool = True) -> int:
    """
    Check all pending live signals and resolve completed ones to the CSV log.
    Returns the number of trades resolved this run.

    Dedup against the CSV by (ticker, entry_date) added 2026-08-01 — this
    function is not the only thing that can cause a re-resolve:
    sync_live_signals_with_remote()'s "pending" list uses a deliberate
    union-merge across git syncs specifically so a real signal can never be
    silently dropped (see its docstring), which means an already-resolved
    entry CAN legitimately reappear in "pending" after a merge. That was
    accepted on the assumption a resurrected-then-re-resolved entry
    "self-heals" — true for the pending list itself, but resolve_live_outcomes
    had no memory of what it had already written, so every resurrection
    across every process/cycle (daemon scan_loop, hourly cron, guard_loop
    sync) appended ANOTHER row for the same trade. Confirmed live: one real
    LGHL trade (entered 2026-07-27) ended up logged 33 times in
    dman_live_outcomes.csv. The union-merge behavior on the pending list
    stays as-is (still the right call — never lose a real signal); this
    just makes the CSV write itself idempotent so a resurrection is a no-op
    instead of a duplicate.
    """
    if not os.path.exists(LIVE_SIGNALS_FILE):
        if verbose:
            print("  No live signals file found — nothing to resolve.")
        return 0

    try:
        with open(LIVE_SIGNALS_FILE) as f:
            data = json.load(f)
    except Exception:
        return 0

    pending   = data.get("pending", [])
    today_str = datetime.now(ET).strftime("%Y-%m-%d")
    still_open: list[dict] = []
    resolved_count = 0

    already_logged: set[tuple[str, str]] = set()
    if os.path.exists(LIVE_OUTCOMES_FILE):
        try:
            with open(LIVE_OUTCOMES_FILE) as f:
                for row in csv.DictReader(f):
                    already_logged.add((row.get("ticker", ""), row.get("entry_date", "")))
        except Exception:
            pass   # fail-open on a malformed CSV — worst case, a duplicate slips through once

    # Write CSV header if file doesn't exist
    write_header = not os.path.exists(LIVE_OUTCOMES_FILE)
    csv_rows: list[str] = []
    if write_header:
        csv_rows.append("ticker,setup,bias,entry_date,exit_date,entry,stop,"
                        "target1,target2,exit_px,exit_reason,pnl_pct,outcome,score,hold_bars")

    for p in pending:
        if (p["ticker"], p["date"]) in already_logged:
            # Already resolved and logged in a prior cycle — the pending
            # list resurrected it via the union-merge, but there's nothing
            # left to do. Drop it from pending (don't re-add to still_open)
            # rather than looping on it forever.
            continue
        if p["date"] >= today_str:
            # Entered today — needs at least one full day's bar to evaluate
            still_open.append(p)
            continue

        result = _simulate_trade_outcome(
            ticker=p["ticker"], entry=p["entry"], stop=p["stop"],
            target1=p["target1"], target2=p["target2"],
            bias=p["bias"], start_date=p["date"]
        )
        if result is None:
            # Still open / not enough data
            still_open.append(p)
            continue

        # Resolved — append to CSV
        row = (f"{p['ticker']},{p['setup']},{p['bias']},"
               f"{p['date']},{result['exit_date']},"
               f"{p['entry']},{p['stop']},{p['target1']},{p['target2']},"
               f"{result['exit_px']},{result['exit_reason']},"
               f"{result['pnl_pct']},{result['outcome']},"
               f"{p.get('score', 0)},{result['hold_bars']}")
        csv_rows.append(row)
        resolved_count += 1

        if verbose:
            icon = "✅" if result["outcome"] == "WIN" else ("⚪" if result["pnl_pct"] == 0 else "❌")
            print(f"  {icon} {p['ticker']:6} {p['setup']:<14}  "
                  f"{result['exit_reason']:<8}  {result['pnl_pct']:+.1f}%  "
                  f"({p['date']} → {result['exit_date']}, {result['hold_bars']}d)")

    if csv_rows:
        with open(LIVE_OUTCOMES_FILE, "a") as f:
            f.write("\n".join(csv_rows) + "\n")

    data["pending"] = still_open
    _write_json_atomic(LIVE_SIGNALS_FILE, data, indent=2)

    if verbose and resolved_count:
        print(f"\n  {resolved_count} trade(s) resolved → {LIVE_OUTCOMES_FILE}")
    return resolved_count


def run_readiness_scan() -> None:
    """
    Sunday evening readiness check — scans the watchlist for tickers primed
    for Monday gap setups: MACD > 0, last session green, sector ETF above EMA50.
    Sends a compact Telegram summary so Monday's opportunities are known before open.
    """
    now_et   = datetime.now(ET)
    date_str = now_et.strftime("%A %b %d, %Y")
    print(f"  Running Sunday readiness scan ({date_str})...")

    ready:   list[str] = []   # all 3 filters pass
    partial: list[tuple[str, str]] = []  # 2/3 pass — show what's missing

    for ticker in WATCHLIST:
        try:
            df = fetch_df(ticker)
            if df is None or len(df) < 30:
                continue
            df = compute_indicators(df.copy())
            r  = df.iloc[-1]
            p  = df.iloc[-2]

            macd_ok   = float(r.get("MACD", 0) or 0) > 0
            bar_green = float(r["Close"]) > float(r["Open"])   # last session green
            sec_ok    = _sector_etf_above_ema50(ticker)

            passes = sum([macd_ok, bar_green, sec_ok])
            if passes == 3:
                ready.append(ticker)
            elif passes == 2:
                miss = ("MACD-" if not macd_ok else
                        "red close" if not bar_green else "sector⚠️")
                partial.append((ticker, miss))
        except Exception:
            continue

    # Build Telegram message
    if ready:
        ready_str = "  ".join(f"✅ <b>{t}</b>" for t in ready[:12])
    else:
        ready_str = "  None fully primed this week"

    partial_str = ""
    if partial:
        partial_lines = [f"⚠️ {t} ({m})" for t, m in partial[:6]]
        partial_str = "\n<b>PARTIAL (2/3):</b> " + "  ".join(partial_lines)

    # FOMC / macro note for the coming week
    _td  = now_et.date()
    _macro_notes = []
    for offset in range(1, 6):   # Mon–Fri
        d = _td + timedelta(days=offset)
        if d in _FOMC_DATES:
            _macro_notes.append(f"⛔ FOMC {d.strftime('%a %b %d')} — blackout ±1 day")
        if d in _CPI_DATES:
            _macro_notes.append(f"📊 CPI {d.strftime('%a %b %d')}")
        if d in _PPI_DATES:
            _macro_notes.append(f"📊 PPI {d.strftime('%a %b %d')}")
        if d.weekday() == 4 and d == _get_third_friday(d.year, d.month):
            _macro_notes.append(f"⚡ OPEX {d.strftime('%a %b %d')} — gamma event")
    macro_note = ("\n" + "\n".join(_macro_notes)) if _macro_notes else ""

    msg = (
        f"📋 <b>DMan Sunday Readiness — Week of {(_td + timedelta(days=1)).strftime('%b %d')}</b>\n\n"
        f"<b>READY FOR MONDAY GAPS ({len(ready)} tickers — MACD+ ✓ green ✓ sector ✓):</b>\n"
        f"{ready_str}"
        f"{partial_str}"
        f"{macro_note}\n\n"
        f"<i>These pass all pre-gap filters. Watch for ≥1.5% gaps at open Monday.</i>"
    )
    sent = send_telegram(msg)
    print(f"  {'✅ Sent' if sent else '⚠️  Telegram not configured'} — "
          f"{len(ready)} ready, {len(partial)} partial")


# ── Telegram two-way commands ────────────────────────────────────────────────
# The bot is no longer one-way: /halt /resume /close /status /positions /pnl
# /restart work from your phone. Cron runs poll once per run; the daemon
# polls live.

def _trigger_workflow_restart(workflow_file: str, ref: str = "main") -> tuple[bool, str]:
    """
    Dispatch a fresh run of a GitHub Actions workflow via the REST API —
    the mechanism behind both /restart (phone-triggered, last resort) and
    the watchdog's automatic self-heal. GITHUB_TOKEN is the token GitHub
    auto-provides to every Actions run; no new secret to create, just
    `permissions: actions: write` on whichever workflow calls this.

    Critically, this doesn't need the daemon itself to be alive to work:
    _process_telegram_commands() runs at the top of every single mode
    dispatch (scan, watchdog, everything — see main()), and the watchdog
    runs on its OWN 30-min schedule completely independent of daemon
    health, so a stuck/crashed/hung daemon still gets picked up and
    restarted without anyone needing to touch a computer.

    dman_daemon.yml's concurrency group (group: dman-daemon,
    cancel-in-progress: true) applies to every trigger type including
    workflow_dispatch, so dispatching a fresh run automatically supersedes
    a stuck one — no separate cancel-then-restart dance needed.
    """
    if not GITHUB_TOKEN:
        return False, "GITHUB_TOKEN not available in this run"
    try:
        resp = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{workflow_file}/dispatches",
            headers={
                "Authorization":        f"Bearer {GITHUB_TOKEN}",
                "Accept":               "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": ref},
            timeout=15,
        )
        if resp.status_code == 204:
            return True, "dispatched"
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, str(exc)


def _check_and_heal_watchdog(now_et: Optional[datetime] = None) -> None:
    """
    Meta-watchdog: checks whether DMan Watchdog's OWN scheduled trigger has
    actually fired recently, and dispatches a fresh run if not.

    Found live 2026-08-17: both dman_daemon.yml's and dman_watchdog.yml's
    scheduled cron triggers silently failed to fire for 2+ hours during a
    live trading session with real open positions — a known, GitHub-
    platform-level scheduling reliability issue (see both workflow files'
    own docstrings for the near-identical 2026-08-03 precedent, already
    partially mitigated once by offsetting trigger minutes off the
    hour/half-hour) — while Scanner, StockTwits, and Pre-Market Briefing
    all fired normally that same morning. run_watchdog() already detects
    and auto-restarts a stale DAEMON — but only if the watchdog itself
    actually runs. Nothing previously checked whether the watchdog's own
    trigger fired, so when GitHub dropped both crons on the same morning,
    nobody — human or machine — noticed until a human happened to check
    by hand, hours into the session.

    Deliberately minimal: this only checks watchdog freshness and, if
    stale, dispatches dman_watchdog.yml — it does NOT duplicate
    run_watchdog()'s own daemon-staleness/auto-restart logic (single
    source of truth stays there). Once the watchdog actually runs again,
    its own existing checks take it from there.

    Called from two independently-scheduled, empirically-reliable-that-
    morning workflows (scan mode + stocktwits mode) rather than added as
    a THIRD single point of failure — the whole point is that no one
    trigger being dropped can recreate this exact silent gap again.
    Fails quiet on any error (network, missing token): this is a
    redundant safety net layered on top of the primary watchdog, not
    something that should ever block or alarm on its own account.
    """
    now_et = now_et or datetime.now(ET)
    if now_et.weekday() >= 5:
        return
    t = now_et.hour * 100 + now_et.minute
    if not (935 <= t <= 1600):   # 5 min past open, so the 9:07/9:37 AM slots have had a chance to fire
        return
    if not GITHUB_TOKEN:
        return
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/dman_watchdog.yml/runs",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
            params={"per_page": 1}, timeout=10,
        )
        runs = resp.json().get("workflow_runs", [])
        if not runs:
            return
        from datetime import timezone as _tz
        _last_run_at = datetime.fromisoformat(runs[0]["created_at"].replace("Z", "+00:00"))
        _stale_min = (datetime.now(_tz.utc) - _last_run_at).total_seconds() / 60
    except Exception:
        return

    if _stale_min <= 40:   # watchdog's own cadence is every 30 min -- 40 min is a real gap, not a blip
        return
    _key = "__META_WATCHDOG_RESTART__"
    if _is_duplicate_alert(_key):
        return
    print(f"  🐕‍🦺 Meta-watchdog: DMan Watchdog hasn't run in {_stale_min:.0f} min — dispatching a fresh run")
    _ok, _msg = _trigger_workflow_restart("dman_watchdog.yml")
    if _ok:
        send_telegram(f"🐕‍🦺 <b>Meta-watchdog</b>: DMan Watchdog hadn't run in {_stale_min:.0f} min "
                      "(its own schedule may have been dropped by GitHub) — dispatched a fresh run.")
    else:
        send_telegram(f"🐕‍🦺 <b>Meta-watchdog</b>: DMan Watchdog appears stale ({_stale_min:.0f} min) "
                      f"but auto-restart failed ({_msg}). Reply <b>/restart</b> manually, or "
                      f"GitHub app → Actions → DMan Watchdog → Run workflow.")
    _save_last_alert(_key)


def is_halted() -> bool:
    """True when a manual /halt is active — blocks NEW entries only."""
    return os.path.exists(HALT_FILE)


def _entry_circuit_breakers_ok() -> tuple[bool, str]:
    """
    Checks the same four entry-blocking conditions _submit_signals_to_alpaca()
    already enforces (halt, consecutive-loss guard, daily loss limit, monthly
    loss limit) — factored out so every path that can place a REAL order on a
    human's approval (earnings-spread YES, manual /buy YES) enforces the same
    set, not a hand-copied subset that's easy to let drift out of sync.

    Added 2026-08-16 review: _handle_earnings_approval_reply() previously
    checked none of these four (only offer-matching, Alpaca reachability, and
    price drift) — a real multi-leg spread could submit while the bot was
    halted or a loss limit had already tripped. _handle_manual_options_buy_reply()
    checked halt and macro but not the three loss/consec-loss guards.

    Returns (ok, reason) — reason is a human-readable string for the Telegram
    reply when ok is False, empty otherwise.
    """
    if is_halted():
        return False, "bot is halted (/resume first)"
    _stats = WinRateTracker().rolling_stats()
    if _stats["consec_losses"] >= MAX_CONSEC_LOSSES:
        return False, f"consecutive-loss guard active ({_stats['consec_losses']} losses)"
    if get_todays_loss() <= -(DAILY_LOSS_LIMIT * 100):
        return False, "daily loss limit active"
    if get_this_month_loss() <= -(MONTHLY_LOSS_LIMIT * 100):
        return False, "monthly loss limit active"
    return True, ""


def _handle_telegram_command(text: str) -> None:
    """Execute one bot command and reply via Telegram."""
    _parts = text.split()
    _cmd   = _parts[0].lower().lstrip("/").split("@")[0]
    _arg   = _parts[1].upper().strip() if len(_parts) > 1 else ""

    if _cmd == "halt":
        _reason = " ".join(_parts[1:]) or "manual"
        try:
            with open(HALT_FILE, "w") as _f:
                json.dump({"halted_at": datetime.now(ET).isoformat(),
                           "reason": _reason}, _f)
            send_telegram(f"🛑 <b>HALTED</b> — no new entries will be submitted "
                          f"(reason: {_reason}). Exits/stops still enforced. "
                          f"Send /resume to re-enable.")
        except Exception as _e:
            send_telegram(f"❌ /halt failed: {_e}")

    elif _cmd == "resume":
        try:
            if os.path.exists(HALT_FILE):
                os.remove(HALT_FILE)
                send_telegram("🟢 <b>RESUMED</b> — entries re-enabled.")
            else:
                send_telegram("🟢 Not halted — nothing to resume.")
        except Exception as _e:
            send_telegram(f"❌ /resume failed: {_e}")

    elif _cmd == "status":
        _h = "🛑 HALTED" if is_halted() else "🟢 active"
        try:
            _acct = get_alpaca_client().get_account()
            _eq   = float(_acct.equity)
            _dt   = int(getattr(_acct, "daytrade_count", 0) or 0)
            _acct_line = (f"Equity <b>${_eq:,.2f}</b>  "
                          f"BP ${float(_acct.buying_power):,.2f}  "
                          f"Day trades {_dt}/3")
        except Exception:
            _acct_line = "Alpaca unreachable"
        _n_pos = len(PositionTracker().positions)
        send_telegram(f"📊 <b>DMan status</b> — {_h}\n{_acct_line}\n"
                      f"Tracked positions: {_n_pos}\n"
                      f"Today P&L: {get_todays_loss():+.2f}%  "
                      f"Month: {get_this_month_loss():+.2f}%")

    elif _cmd == "positions":
        _pt = PositionTracker()
        if not _pt.positions:
            send_telegram("📭 No tracked positions.")
        else:
            _lines = []
            for _p in _pt.positions:
                if _p.setup.startswith("Earnings "):
                    _lines.append(f"<b>{_p.ticker}</b> [SPREAD] {_p.setup}  "
                                  f"cost ${_p.entry:.0f}  max loss ${_p.max_loss:.0f}  "
                                  f"max gain ${_p.max_gain:.0f}")
                    continue
                _tag = "OPT" if _p.setup.startswith("Options ") else _p.bias
                _lines.append(f"<b>{_p.ticker}</b> [{_tag}] entry ${_p.entry}  "
                              f"stop ${_p.stop}  T1 ${_p.target1}")
            send_telegram("📋 <b>Open positions</b>\n" + "\n".join(_lines))

    elif _cmd == "pnl":
        send_telegram(f"💰 <b>P&L</b>\nToday: {get_todays_loss():+.2f}%\n"
                      f"Month: {get_this_month_loss():+.2f}%")

    elif _cmd in ("restart", "reboot"):
        send_telegram("🔄 <b>Restart requested</b> — dispatching a fresh daemon session "
                      "(automatically cancels any stuck/hung session first, same "
                      "concurrency group). This works even if the current daemon is "
                      "completely frozen.")
        _ok, _msg = _trigger_workflow_restart("dman_daemon.yml")
        if _ok:
            send_telegram("✅ Fresh daemon session dispatched — should be live within a "
                          "few minutes. You'll get the usual \"daemon ONLINE\" message "
                          "once it starts.")
        else:
            send_telegram(f"❌ Restart dispatch failed: {_msg}\n"
                          f"Fallback: GitHub app → Actions → DMan Cloud Daemon → Run workflow.")

    elif _cmd == "close" and _arg:
        _pt  = PositionTracker()
        _pos = next((p for p in _pt.positions if p.ticker == _arg), None)
        if _pos is None:
            send_telegram(f"❓ /close: no tracked position for {_arg}")
        elif _pos.setup.startswith("Earnings "):
            _st, _oid = _close_earnings_spread(asdict(_pos), f"manual /close {_arg}")
            send_telegram(f"📤 /close {_arg}: {_st}"
                          + (f" (id {_oid[:8]}…)" if _oid else ""))
        elif _pos.setup.startswith("Options "):
            _occ_c  = _pos.setup.split()[2]
            _ctrs_c = max(1, int(_pos.shares) // 100)
            _st, _oid = _submit_options_close(_occ_c, _ctrs_c, f"manual /close {_arg}")
            send_telegram(f"📤 /close {_arg}: {_st}"
                          + (f" (id {_oid[:8]}…)" if _oid else ""))
        else:
            try:
                get_alpaca_client().close_position(_arg)
                send_telegram(f"📤 /close {_arg}: equity close submitted")
            except Exception as _e:
                send_telegram(f"❌ /close {_arg} failed: {_e}")

    elif _cmd == "why" and _arg:
        try:
            send_telegram(explain_ticker(_arg))
        except Exception as _e:
            send_telegram(f"❌ /why {_arg} failed: {_e}")

    elif _cmd == "options":
        _handle_options_command(_parts)

    elif _cmd == "buy":
        _handle_buy_command(_parts)

    else:
        send_telegram(
            "🤖 <b>DMan commands</b>\n" +
            "\n".join(f"/{c} — {d}" for c, d in TELEGRAM_COMMANDS)
        )


def _handle_options_command(parts: list[str]) -> None:
    """
    /options TICKER [E] — browse OPTIONS_CHAIN_STRIKES_PER_SIDE calls and
    the same number of puts around ATM, numbered so /buy N can reference
    one. Defaults to the nearest OPTIONS_TARGET_DTE expiry (same one the
    automated path would pick); E selects a different expiry by its index
    in the "other expiries" list this command always shows, so
    "/options SMCI" then "/options SMCI 3" browses out the real listed
    ladder instead of only ever seeing one date. Read-only: never places
    an order by itself, and works regardless of /halt state (browsing
    isn't an entry — /buy's final YES confirmation is where
    halt/macro-blackout actually get enforced).
    """
    ticker = parts[1].upper().strip() if len(parts) > 1 else ""
    if not ticker:
        send_telegram("Usage: <code>/options TICKER [E]</code> — E picks an expiry "
                       "from the list shown at the bottom of the last /options reply.")
        return
    _expiry_idx = None
    if len(parts) > 2:
        if not parts[2].isdigit():
            send_telegram(f"❌ Couldn't parse expiry number '{parts[2]}' — usage: /options TICKER [E]")
            return
        _expiry_idx = int(parts[2])

    client = get_alpaca_client()
    if client is None:
        send_telegram("❌ Alpaca unavailable — can't fetch options chain.")
        return
    px = get_live_price(ticker)
    if px is None or px <= 0:
        send_telegram(f"❌ Couldn't get a live price for {ticker}.")
        return

    _expiries = _fetch_available_expiries(client, ticker)
    _target_expiry = None
    if _expiry_idx is not None:
        if not _expiries or not (1 <= _expiry_idx <= len(_expiries)):
            send_telegram(f"❌ No expiry #{_expiry_idx} for {ticker} "
                           f"({len(_expiries)} available) — run /options {ticker} to see the list.")
            return
        _target_expiry = _expiries[_expiry_idx - 1]

    chain = _fetch_option_chain_for_display(client, ticker, px, expiry=_target_expiry)
    if not chain or not chain["items"]:
        send_telegram(f"❌ No liquid options chain found for {ticker} "
                       f"(below the {OPTIONS_MIN_UNDERLYING_VOL/1e6:.0f}M underlying volume "
                       f"floor, or no listed contracts near the money).")
        return

    calls = sorted((i for i in chain["items"] if i["type"] == "CALL"), key=lambda x: x["strike"])
    puts  = sorted((i for i in chain["items"] if i["type"] == "PUT"),  key=lambda x: x["strike"])
    ordered = calls + puts
    for _i, _item in enumerate(ordered, start=1):
        _item["idx"] = _i

    _now = datetime.now(ET)
    menu = {
        "ticker": ticker, "created_at": _now.isoformat(),
        "expires_at": (_now + timedelta(minutes=TELEGRAM_OPTIONS_MENU_TIMEOUT_MIN)).isoformat(),
        "expiry": chain["expiry"], "items": ordered,
    }
    try:
        _write_json_atomic(TELEGRAM_OPTIONS_MENU_FILE, menu, indent=2)
    except Exception as _e:
        send_telegram(f"❌ Couldn't save options menu: {_e}")
        return

    _lines = [f"📊 <b>{ticker}</b>  ${px:.2f}   exp {chain['expiry']} ({chain['dte']}d)"]
    if calls:
        _lines.append("\n<b>CALLS</b>")
        for _i in calls:
            _dtag = "~" if _i["delta_estimated"] else ""
            _lines.append(f"{_i['idx']}) ${_i['strike']:g}C  bid/ask "
                          f"{_i['bid']:.2f}/{_i['ask']:.2f}  Δ{_dtag}{_i['delta']:.2f}")
    if puts:
        _lines.append("\n<b>PUTS</b>")
        for _i in puts:
            _dtag = "~" if _i["delta_estimated"] else ""
            _lines.append(f"{_i['idx']}) ${_i['strike']:g}P  bid/ask "
                          f"{_i['bid']:.2f}/{_i['ask']:.2f}  Δ{_dtag}{_i['delta']:.2f}")
    if _expiries:
        _today = date.today()
        _exp_bits = [f"{_n}) {_e.strftime('%a %b %d')} ({(_e-_today).days}d)"
                     for _n, _e in enumerate(_expiries, start=1)]
        _lines.append("\n<b>Other expiries</b>  (/options " + ticker + " E)")
        _lines.append("  " + "   ".join(_exp_bits))
    _lines.append(f"\nReply <code>/buy N [qty] [price]</code> to confirm "
                  f"(e.g. <code>/buy {ordered[0]['idx']} 3 1.50</code> = 3 contracts @ $1.50 — "
                  f"qty defaults to 1, price defaults to ask if omitted). "
                  f"Menu expires in {TELEGRAM_OPTIONS_MENU_TIMEOUT_MIN} min.")
    send_telegram("\n".join(_lines))


def _handle_buy_command(parts: list[str]) -> None:
    """
    /buy N [qty] [price] — stage a confirmation for item N from the most
    recent /options menu. qty defaults to 1 contract if omitted; price
    defaults to the item's live ask if omitted — give one explicitly
    (e.g. "1.50") to set your own limit instead of paying the ask, exactly
    like a normal limit order. This is the price that actually gets
    submitted (see _submit_manual_options_buy) — it is NOT silently
    re-priced to ask+3% the way automated entries are, since the whole
    point of specifying qty/price directly is control over what you pay.

    Does NOT place the order — sends a confirmation message the user must
    reply YES/NO to (same pattern as the earnings-spread approval flow),
    so a single mistyped index/qty/price can never directly become a live
    order. halt/macro-blackout checks happen at the YES step
    (_handle_manual_options_buy_reply), not here — see that function's
    docstring for why.
    """
    if len(parts) < 2 or not parts[1].isdigit():
        send_telegram("Usage: <code>/buy N [qty] [price]</code> — N from the last /options "
                       "menu (e.g. <code>/buy 5 3 1.50</code> = item 5, 3 contracts, $1.50 limit).")
        return
    idx = int(parts[1])

    qty = 1
    if len(parts) >= 3:
        if not parts[2].isdigit() or int(parts[2]) < 1:
            send_telegram(f"❌ Couldn't parse quantity '{parts[2]}' — usage: /buy N [qty] [price]")
            return
        qty = min(int(parts[2]), MANUAL_BUY_MAX_CONTRACTS)

    try:
        with open(TELEGRAM_OPTIONS_MENU_FILE) as f:
            menu = json.load(f)
    except Exception:
        send_telegram("❌ No active /options menu — run /options TICKER first.")
        return

    if datetime.now(ET) >= datetime.fromisoformat(menu["expires_at"]):
        send_telegram("⏰ That /options menu expired — run /options TICKER again.")
        return

    item = next((i for i in menu["items"] if i["idx"] == idx), None)
    if item is None:
        send_telegram(f"❌ No item #{idx} on the current menu ({len(menu['items'])} items). "
                       f"Run /options {menu['ticker']} again to see it.")
        return
    if item["ask"] <= 0:
        send_telegram(f"❌ {item['occ_symbol']} has no usable quote right now.")
        return

    limit_price = item["ask"]
    if len(parts) >= 4:
        try:
            limit_price = float(parts[3])
        except ValueError:
            send_telegram(f"❌ Couldn't parse price '{parts[3]}' — usage: /buy N [qty] [price]")
            return
        if limit_price <= 0:
            send_telegram("❌ Price must be positive.")
            return
        if limit_price > item["ask"] * MANUAL_BUY_MAX_PRICE_VS_ASK_MULT:
            send_telegram(f"⚠️ ${limit_price:.2f} is more than {MANUAL_BUY_MAX_PRICE_VS_ASK_MULT:.0f}x "
                           f"the live ask (${item['ask']:.2f}) for {item['occ_symbol']} — this looks "
                           f"like a typo, not submitted. Use /buy {idx} {qty} to confirm you really "
                           f"meant a price that far above ask.")
            return

    total_cost = round(qty * limit_price * 100, 2)
    if total_cost > MANUAL_BUY_MAX_RISK_DOLLARS:
        send_telegram(f"⚠️ {qty}x {item['occ_symbol']} @ ${limit_price:.2f} = ~${total_cost:.0f}, "
                       f"over the ${MANUAL_BUY_MAX_RISK_DOLLARS:.0f} per-trade cap — "
                       f"reduce the quantity or price.")
        return

    _now = datetime.now(ET)
    pending = {
        "ticker": menu["ticker"], "occ_symbol": item["occ_symbol"],
        "option_type": item["type"], "strike": item["strike"], "expiry": menu["expiry"],
        "contracts": qty, "limit_price": limit_price, "ask_at_confirm": item["ask"],
        "total_cost": total_cost,
        "created_at": _now.isoformat(),
        "expires_at": (_now + timedelta(minutes=TELEGRAM_MANUAL_BUY_TIMEOUT_MIN)).isoformat(),
    }
    try:
        _write_json_atomic(TELEGRAM_MANUAL_BUY_FILE, pending, indent=2)
    except Exception as _e:
        send_telegram(f"❌ Couldn't stage confirmation: {_e}")
        return

    _word = item["type"].lower() + ("s" if qty != 1 else "")   # "call"/"calls", "put"/"puts"
    _px_note = "the asking price of" if limit_price == item["ask"] else "your limit price of"
    send_telegram(
        f"🎯 <b>Confirm</b>: Buy {qty} {_word} of {menu['ticker']} at {_px_note} "
        f"${limit_price:.2f}\n"
        f"(${item['strike']:g} strike, exp {menu['expiry']}, ~${total_cost:.0f} total, "
        f"Δ{item['delta']:.2f})\n"
        f"Reply <b>YES</b> to place, <b>NO</b> to cancel. Expires in "
        f"{TELEGRAM_MANUAL_BUY_TIMEOUT_MIN} min."
    )


def _handle_manual_options_buy_reply(text: str) -> bool:
    """
    Matches a bare (non-"/"-prefixed) YES/NO reply against a pending manual
    options buy staged by /buy. Mirrors _handle_earnings_approval_reply's
    shape (expiration check, price-staleness re-check before submitting)
    but for a single pending buy at a time rather than a list — the
    /options -> /buy -> YES/NO flow is a one-at-a-time interactive
    sequence, not concurrent offers. Halt, the loss/consec-loss circuit
    breakers (via _entry_circuit_breakers_ok(), added 2026-08-16 — this
    used to only check halt), and macro-blackout are all enforced HERE,
    not at /buy time, so staging a confirmation always reflects the
    real-time gate state at the moment money would actually move. Returns
    True if the message was YES/NO-shaped (whether or not a buy was
    actually pending), so the caller doesn't also try to process it as an
    earnings-approval reply.
    """
    import re
    m = re.match(r"^(yes|y|no|n)\b\s*$", text.strip(), re.IGNORECASE)
    if not m:
        return False

    try:
        with open(TELEGRAM_MANUAL_BUY_FILE) as f:
            pending = json.load(f)
    except Exception:
        return False   # nothing pending — not our message to consume

    try:
        os.remove(TELEGRAM_MANUAL_BUY_FILE)
    except Exception:
        pass

    if datetime.now(ET) >= datetime.fromisoformat(pending["expires_at"]):
        send_telegram(f"⏰ {pending['ticker']} buy confirmation expired — no order placed.")
        return True

    is_yes = m.group(1).lower() in ("yes", "y")
    if not is_yes:
        send_telegram(f"👍 {pending['ticker']} buy cancelled — no order placed.")
        return True

    _cb_ok, _cb_reason = _entry_circuit_breakers_ok()
    if not _cb_ok:
        send_telegram(f"🛑 {pending['ticker']} buy NOT placed — {_cb_reason}.")
        return True

    _macro_ok, _ = check_macro_safe()
    if not _macro_ok:
        send_telegram(f"⛔ {pending['ticker']} buy NOT placed — macro blackout active "
                       f"(FOMC/CPI/NFP window, stops unreliable right now). Try again "
                       f"after the blackout lifts.")
        return True

    client = get_alpaca_client()
    if client is None:
        send_telegram(f"❌ {pending['ticker']} buy approved but Alpaca is unavailable — not submitted.")
        return True

    order_id, err = _submit_manual_options_buy(client, pending)
    if err:
        send_telegram(f"❌ <b>{pending['ticker']} buy FAILED</b>\n{err}")
        return True

    _word = pending["option_type"].lower() + ("s" if pending["contracts"] != 1 else "")
    send_telegram(
        f"📤 <b>Submitted</b>: {pending['contracts']} {_word} of {pending['ticker']}  "
        f"id {order_id[:8]}…\n"
        f"(${pending['strike']:g} strike, exp {pending['expiry']}, ~${pending['total_cost']:.0f} total)"
    )
    return True


_EARNINGS_OFFER_TOMBSTONE_S = 6 * 3600   # same margin as _CLOSED_IDENTITY_TOMBSTONE_S

def _earnings_offer_identity(entry: dict) -> str:
    return f"{entry.get('ticker', '')}_{entry.get('earn_date', '')}"


def _load_earnings_state() -> tuple[list[dict], dict]:
    try:
        with open(EARNINGS_SPREAD_PENDING_FILE) as f:
            data = json.load(f)
        if isinstance(data, list):   # pre-migration legacy shape: bare list, no tombstones yet
            return data, {}
        return data.get("pending", []), data.get("consumed", {})
    except Exception:
        return [], {}


def _load_earnings_pending() -> list[dict]:
    return _load_earnings_state()[0]


def _save_earnings_state(pending: list[dict], consumed: dict) -> None:
    now = time.time()
    consumed = {k: v for k, v in consumed.items() if now - float(v) < _EARNINGS_OFFER_TOMBSTONE_S}
    _write_json_atomic(EARNINGS_SPREAD_PENDING_FILE,
                       {"pending": pending, "consumed": consumed}, indent=2, default=str)


def _save_earnings_pending(entries: list[dict]) -> None:
    """
    Back-compat wrapper for callers that aren't consuming (finalizing) any
    specific offer -- preserves whatever consumed-identity tombstones are
    already on disk. A caller that IS finalizing a specific offer (a YES/
    NO reply reaching a terminal outcome) must use
    _consume_earnings_offer_save() instead, so the removal is tombstoned
    against resurrection by a stale remote copy of this file -- see
    sync_earnings_pending_with_remote().
    """
    _, consumed = _load_earnings_state()
    _save_earnings_state(entries, consumed)


def _consume_earnings_offer_save(pending: list[dict], entry: dict) -> None:
    """
    Persist `pending` (with `entry` already removed by the caller) and
    tombstone entry's identity so a later git merge against a stale
    remote copy of dman_earnings_pending.json -- one that still shows
    this offer as "awaiting_approval" -- can't resurrect it. Found in the
    2026-08-16 review: unlike every sibling multi-writer state file, this
    one had no semantic merge at all, just git's default whole-file
    last-writer-wins. A resurrected offer here is a live-money risk, not
    a cosmetic one: this daemon and the hourly cron scanner both handle
    Telegram replies from separate checkouts, and these offers gate real
    earnings-spread orders -- a resurrected APPROVED-and-submitted offer
    could let a later stray/ambiguous YES re-submit the same real spread
    a second time; a resurrected REJECTED offer carries the identical
    risk if the user later approves what they believe is a different
    pending offer.
    """
    _, consumed = _load_earnings_state()
    consumed[_earnings_offer_identity(entry)] = time.time()
    _save_earnings_state(pending, consumed)


def format_earnings_spread_telegram(plan: dict) -> str:
    """Telegram approval-request message for a pending earnings spread."""
    lines = [f"⚡ <b>DMan EARNINGS SPREAD</b> — {plan['ticker']}  [{plan.get('timing', '?')}]"]

    moves = plan.get("last_moves_pct", [])
    kind_label = f"Single-sided {plan['directional']} spread" if plan.get("directional") \
        else "Non-directional double spread"
    if moves:
        moves_str = "/".join(f"{m:+.1f}" for m in moves)
        lines.append(f"{kind_label} (history: {moves_str}% last {len(moves)} qtrs)")
    else:
        lines.append(f"{kind_label} (no reliable history found — defaulting safe)")

    if plan.get("call"):
        c = plan["call"]
        lines.append(f"📈 CALL debit spread   buy {c['long_strike']:.0f}C / sell {c['short_strike']:.0f}C   "
                     f"exp {date.fromisoformat(c['expiry']).strftime('%b %d')} ({c['dte']}d)")
    if plan.get("put"):
        p = plan["put"]
        lines.append(f"📉 PUT debit spread    buy {p['long_strike']:.0f}P / sell {p['short_strike']:.0f}P   "
                     f"exp {date.fromisoformat(p['expiry']).strftime('%b %d')} ({p['dte']}d)")

    eff_acct = get_effective_account()
    pct = (plan["total_cost"] / eff_acct * 100) if eff_acct > 0 else 0
    lines.append(f"Net debit: ${plan['total_cost']:.0f}  ({plan['sets']} set"
                 f"{'s' if plan['sets'] != 1 else ''})  =  {pct:.1f}% of ${eff_acct:,.2f} equity")

    gains = [f"${plan[side]['max_gain']:.0f}" for side in ("call", "put") if plan.get(side)]
    lines.append(f"Max loss: ${plan['max_loss']:.0f}   Max gain: {' / '.join(gains)}")

    bes = []
    if plan.get("call"):
        bes.append(f"${plan['call']['breakeven']:.2f} ↑")
    if plan.get("put"):
        bes.append(f"${plan['put']['breakeven']:.2f} ↓")
    lines.append(f"Breakevens: {'  '.join(bes)}")

    if plan.get("ai_analysis"):
        lines.append("")
        lines.append(f"🧠 <i>{plan['ai_analysis']}</i>")

    _n_sides = int(bool(plan.get("call"))) + int(bool(plan.get("put")))
    lines.append(f"Reply <b>YES {plan['ticker']}</b> to approve (1 atomic order, {_n_sides} side(s)) "
                 f"· <b>NO {plan['ticker']}</b> to reject · expires in {EARNINGS_APPROVAL_TIMEOUT_MIN} min")
    return "\n".join(lines)


def _open_earnings_spread_position(plan: dict) -> OpenPosition:
    """
    Builds the OpenPosition record for a filled earnings spread and persists
    it via PositionTracker. Only one caller exists today —
    _handle_earnings_approval_reply(), after an explicit human YES. The
    Telegram approval gate is PERMANENT (see dman_daemon.py:earnings_loop
    docstring): no autonomous submission path is planned.
    legs are stored in submission order (long, short[, long, short]) —
    _close_earnings_spread()/_monitor_earnings_spread_position() rely on
    even/odd index meaning long/short.
    """
    legs = []
    if plan.get("call"):
        legs += [plan["call"]["long_occ"], plan["call"]["short_occ"]]
    if plan.get("put"):
        legs += [plan["put"]["long_occ"], plan["put"]["short_occ"]]
    setup_tag = ("Earnings Double Spread" if (plan.get("call") and plan.get("put"))
                 else ("Earnings Call Spread" if plan.get("call") else "Earnings Put Spread"))
    max_gain = max(plan.get("call", {}).get("max_gain", 0), plan.get("put", {}).get("max_gain", 0))

    pos = OpenPosition(
        ticker=plan["ticker"],
        bias=(plan["directional"] if plan.get("directional") else "NEUTRAL"),
        setup=setup_tag, entry=plan["total_cost"], stop=0.0, target1=0.0, target2=0.0,
        shares=0, entry_date=date.today().isoformat(),
        legs=legs, spread_qty=plan["sets"], max_loss=plan["max_loss"],
        max_gain=max_gain, earn_date=plan["earn_date"],
    )
    PositionTracker().open(pos)
    return pos


def _handle_earnings_approval_reply(text: str) -> bool:
    """
    Matches a bare (non-"/"-prefixed) YES/NO reply against pending earnings-
    spread approval offers. Ticker is optional if exactly one offer is
    pending; with 2+ pending and no ticker given, the reply is logged and
    ignored rather than silently guessed which offer it's for — an
    ambiguous approval submitting the WRONG trade would be worse than one
    that expires unused. Returns True if the message was earnings-approval-
    shaped (whether or not it matched a live offer), so the caller doesn't
    also try to process it as some other kind of message.
    """
    import re
    m = re.match(r"^(yes|y|no|n)\b\s*([A-Za-z]*)\s*$", text.strip(), re.IGNORECASE)
    if not m:
        return False
    is_yes = m.group(1).lower() in ("yes", "y")
    ticker_hint = m.group(2).upper().strip()

    pending = _load_earnings_pending()
    if not pending:
        return False

    now = datetime.now(ET)
    still_pending = []
    for entry in pending:
        if entry.get("status") == "awaiting_approval":
            try:
                if now >= datetime.fromisoformat(entry["expires_at"]):
                    send_telegram(f"⏰ {entry['ticker']} earnings spread offer expired — "
                                 f"no reply within {EARNINGS_APPROVAL_TIMEOUT_MIN} min, no order placed.")
                    continue
            except Exception:
                pass
        still_pending.append(entry)
    pending = still_pending

    awaiting = [e for e in pending if e.get("status") == "awaiting_approval"]
    if not awaiting:
        _save_earnings_pending(pending)
        return False   # nothing pending — not our message to consume

    if ticker_hint:
        matches = [e for e in awaiting if e["ticker"] == ticker_hint]
    elif len(awaiting) == 1:
        matches = awaiting
    else:
        print(f"  ⚠️  Ambiguous YES/NO reply with {len(awaiting)} pending earnings-spread "
              f"offers and no ticker given — ignoring (reply 'YES TICKER' explicitly)")
        _save_earnings_pending(pending)
        return True

    if not matches:
        _save_earnings_pending(pending)
        return False   # ticker given but doesn't match any pending offer

    entry = matches[0]
    pending = [e for e in pending if e is not entry]

    if not is_yes:
        send_telegram(f"👍 {entry['ticker']} earnings spread rejected — no order placed.")
        _consume_earnings_offer_save(pending, entry)
        return True

    _cb_ok, _cb_reason = _entry_circuit_breakers_ok()
    if not _cb_ok:
        send_telegram(f"🛑 {entry['ticker']} earnings spread approved but NOT submitted — {_cb_reason}.")
        _consume_earnings_offer_save(pending, entry)
        return True

    _macro_ok, _ = check_macro_safe()
    if not _macro_ok:
        send_telegram(f"⛔ {entry['ticker']} earnings spread NOT submitted — macro blackout active "
                       f"(FOMC/CPI/NFP window, stops unreliable right now).")
        _consume_earnings_offer_save(pending, entry)
        return True

    client = get_alpaca_client()
    if client is None:
        send_telegram(f"❌ {entry['ticker']} earnings spread approved but Alpaca is unavailable — not submitted.")
        _consume_earnings_offer_save(pending, entry)
        return True

    plan = entry["plan"]

    # Added 2026-08-10 alongside widening EARNINGS_APPROVAL_TIMEOUT_MIN to
    # 4 hours: the plan's strikes/net_debit/AI analysis are all computed
    # from a price SNAPSHOT at offer-build time (plan["current_price"]),
    # never refreshed. A 30-min window made that an acceptable risk; a
    # 4-hour window does not — a real move in either direction could make
    # the debit spread's math (breakeven, max gain, even which strikes are
    # still sensibly ITM/OTM) stale enough that blindly submitting it no
    # longer reflects the setup that was actually approved. Abort and ask
    # for a fresh scan rather than submit off outdated pricing.
    _snapshot_px = float(plan.get("current_price", 0) or 0)
    if _snapshot_px > 0:
        _now_px = get_live_price(entry["ticker"])
        if _now_px is not None:
            _drift_pct = abs(_now_px - _snapshot_px) / _snapshot_px * 100
            if _drift_pct > EARNINGS_APPROVAL_MAX_PRICE_DRIFT_PCT:
                send_telegram(
                    f"⚠️ <b>{entry['ticker']} earnings spread NOT submitted</b> — price moved "
                    f"{_drift_pct:.1f}% since this offer was built (${_snapshot_px:.2f} → "
                    f"${_now_px:.2f}), past the {EARNINGS_APPROVAL_MAX_PRICE_DRIFT_PCT:.0f}% "
                    f"staleness limit. The strikes/pricing no longer reflect the current setup — "
                    f"wait for the next scan to generate a fresh offer."
                )
                _consume_earnings_offer_save(pending, entry)
                return True

    order_id, err = _submit_earnings_spread(client, plan)
    if err:
        send_telegram(f"❌ <b>{entry['ticker']} earnings spread FAILED</b>\n{err}")
        _consume_earnings_offer_save(pending, entry)
        return True

    _open_earnings_spread_position(plan)
    send_telegram(f"📤 <b>{entry['ticker']} EARNINGS SPREAD SUBMITTED</b>  id {order_id[:8]}…\n"
                 f"Cost ${plan['total_cost']:.0f}  Max loss ${plan['max_loss']:.0f}")
    _consume_earnings_offer_save(pending, entry)
    return True


def _process_telegram_commands(timeout: int = 0) -> int:
    """
    Poll Telegram getUpdates and execute pending bot commands.
    Returns count processed. Only TELEGRAM_CHAT_ID messages are honored.
    Cron runs call this once per run (timeout=0); the daemon long-polls
    (timeout=25) for real-time response.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return 0
    try:
        with open(TELEGRAM_STATE_FILE) as _f:
            _offset = int(json.load(_f).get("offset", 0))
    except Exception:
        _offset = 0
    try:
        _r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
            params={"offset": _offset + 1, "timeout": timeout,
                    "allowed_updates": '["message"]'},
            timeout=timeout + 15,
        )
        _updates = _r.json().get("result", [])
    except Exception:
        return 0

    _handled = 0
    for _upd in _updates:
        _offset = max(_offset, int(_upd.get("update_id", 0)))
        _m = _upd.get("message") or {}
        if str(_m.get("chat", {}).get("id", "")) != str(TELEGRAM_CHAT_ID):
            continue   # ignore anyone who isn't the account owner
        _text = (_m.get("text") or "").strip()
        if not _text.startswith("/"):
            # Plain (non-"/") replies used to be silently dropped here —
            # that's the only way a human can answer an earnings-spread
            # approval request (see format_earnings_spread_telegram()) or a
            # manual /options -> /buy confirmation. The manual-buy check
            # runs first since it's a narrower, one-at-a-time flow — if
            # nothing's pending there it returns False immediately and
            # falls through to the earnings check unchanged.
            try:
                if _handle_manual_options_buy_reply(_text):
                    _handled += 1
                elif _handle_earnings_approval_reply(_text):
                    _handled += 1
            except Exception as _e:
                print(f"  ⚠️  Reply handling error ({_text}): {_e}")
            continue
        try:
            _handle_telegram_command(_text)
            _handled += 1
        except Exception as _e:
            print(f"  ⚠️  Telegram command error ({_text}): {_e}")
    if _updates:
        try:
            with open(TELEGRAM_STATE_FILE, "w") as _f:
                json.dump({"offset": _offset}, _f)
        except Exception:
            pass
    return _handled


def _stocktwits_inject_tickers(tickers_to_add: list[str]) -> list[str]:
    """
    Add new tickers to the small-cap watchlist data file
    (dman_smallcap_watchlist.json). Returns the tickers actually added
    (skips ones already present). The workflow commits the JSON so the
    next scan picks them up — no source-code modification involved.
    """
    try:
        with open(SMALLCAP_WATCHLIST_FILE) as _f:
            _data = json.load(_f)
        if not isinstance(_data, dict):
            _data = {"tickers": list(_data)}
    except (FileNotFoundError, json.JSONDecodeError):
        _data = {"tickers": list(DMAN_SMALLCAP_WATCHLIST)}

    _current = {str(t).upper() for t in _data.get("tickers", [])}
    _added: list[str] = []
    for _t in tickers_to_add:
        _t = _t.upper().strip()
        if not _t or _t in _current:
            print(f"  📡 {_t} already in watchlist — skipping")
            continue
        _data.setdefault("tickers", []).append(_t)
        _current.add(_t)
        _added.append(_t)

    if _added:
        try:
            with open(SMALLCAP_WATCHLIST_FILE, "w") as _f:
                json.dump(_data, _f, indent=2)
            # Update the in-memory list so this run's scans see them too
            DMAN_SMALLCAP_WATCHLIST.extend(_added)
        except Exception as _e:
            print(f"  ⚠️  Could not write {SMALLCAP_WATCHLIST_FILE}: {_e}")
            return []
    return _added


def run_watchdog() -> None:
    """
    Independent health check for the whole system — deliberately decoupled
    from both the scanner and the daemon so a failure in either doesn't also
    take down the thing watching them. Runs on its own GitHub Actions
    schedule and checks two different failure classes:

    1. Hard failures — the most recent Scanner/Daemon workflow run reported
       "failure". Straightforward, and the existing per-workflow Telegram
       notifications already cover this somewhat, but this is a second,
       independent confirmation.
    2. SILENT failures — a run reports "success" but produced no real work.
       This is the class that actually cost hours today: the daemon's
       missing `import json` made every state sync fail internally while
       the workflow itself kept reporting green for two full sessions.
       A conclusion check alone would have missed it entirely. Caught here
       by checking DATA FRESHNESS directly — has the daemon's own sync
       timestamp moved recently, has at least one scan logged today by a
       reasonable point in the session — regardless of what the workflow
       run status claims.
    """
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        print("  🐕 Watchdog: weekend, skipping.")
        return
    t = now_et.hour * 100 + now_et.minute
    issues: list[str] = []

    # 1. Hard failures — most recent run of each critical workflow.
    for _wf_name, _wf_file in [("DMan PRO Scanner", "dman_scanner.yml"),
                                ("DMan Cloud Daemon", "dman_daemon.yml")]:
        try:
            resp = requests.get(
                f"https://api.github.com/repos/tsingh08/dman-algo-/actions/workflows/{_wf_file}/runs",
                params={"per_page": 1}, timeout=10,
            )
            runs = resp.json().get("workflow_runs", [])
            if runs and runs[0].get("conclusion") == "failure":
                issues.append(f"❌ {_wf_name}: most recent run FAILED "
                              f"({runs[0]['created_at'][:16]})")
        except Exception as exc:
            print(f"  [watchdog] GitHub check failed for {_wf_name}: {exc}")

    # 2. Silent failures — only meaningful once the session has been running
    # a while, so a check that fires right at the open doesn't false-alarm
    # on things that legitimately haven't happened yet.
    daemon_likely_down = False
    if 930 <= t <= 1600:
        try:
            with open(ALPACA_SYNC_FILE) as f:
                _last_sync = datetime.fromisoformat(json.load(f).get("last_sync", ""))
            _stale_min = (datetime.now() - _last_sync).total_seconds() / 60
            if _stale_min > 30:
                issues.append(f"⚠️ Daemon sync stale — last synced {_stale_min:.0f} min ago")
            # The daemon's own sync cadence is every 5 min (SYNC_EVERY_S) —
            # 30 min stale is already 6x that, so it's flagged above as an
            # issue regardless. 45 min is a second, more severe threshold:
            # a blip that self-resolves within 30-45 min doesn't need
            # intervention, but this is well past "having a bad minute" and
            # into "the daemon isn't coming back on its own" territory —
            # worth auto-restarting rather than waiting for a human to
            # notice a Telegram message during a busy day.
            if _stale_min > 45:
                daemon_likely_down = True
        except Exception:
            issues.append("⚠️ dman_alpaca_sync.json unreadable/missing — daemon may never have synced today")
            if t >= 1000:   # give it 30 min after open before assuming it never started
                daemon_likely_down = True

        if t >= 1030:
            try:
                with open(SCAN_LOG_FILE) as f:
                    _log = json.load(f)
                _today_str = now_et.strftime("%Y-%m-%d")
                if not any(e.get("ts", "").startswith(_today_str) for e in _log):
                    issues.append("⚠️ No scan_log entry yet today (past 10:30 AM ET)")
            except Exception:
                issues.append("⚠️ dman_scan_log.json unreadable/missing")

        # 3. Alpaca 429s today — a signal the sync/scan freshness checks above
        # can't provide, since the daemon can keep running/syncing normally
        # while options data specifically is rate-limited underneath it
        # (confirmed live 2026-07-30 — see _record_alpaca_429 docstring).
        try:
            with open(_RATE_LIMIT_EVENTS_FILE) as f:
                _rl = json.load(f).get(now_et.strftime("%Y-%m-%d"), {})
            _rl_total = sum(_rl.values())
            if _rl_total >= 10:
                _detail = ", ".join(f"{k}={v}" for k, v in _rl.items())
                issues.append(f"⚠️ {_rl_total} Alpaca 429s today ({_detail}) — "
                              f"options data may be degraded/rate-limited")
        except Exception:
            pass   # no file yet = no 429s recorded today, not an issue

        # Auto-heal: the whole point of this daemon-independent watchdog is
        # that it keeps running even when the daemon doesn't, so it's the
        # one thing that CAN restart it without a human noticing a Telegram
        # message and doing it manually. Own cooldown key (not the shared
        # issue-alert one below) — a restart that doesn't fix things within
        # one cooldown window shouldn't be re-dispatched every single 30-min
        # cycle forever; ALERT_COOLDOWN_MIN gives it room to actually come
        # up before trying again.
        if daemon_likely_down:
            _restart_key = "__WATCHDOG_AUTO_RESTART__"
            if not _is_duplicate_alert(_restart_key):
                print("  🐕 Daemon appears down — auto-restarting")
                _r_ok, _r_msg = _trigger_workflow_restart("dman_daemon.yml")
                if _r_ok:
                    issues.append("🔄 Daemon appeared down — auto-restarted "
                                  "(fresh session dispatched, no action needed)")
                else:
                    issues.append(f"❌ Daemon appeared down — auto-restart FAILED "
                                  f"({_r_msg}). Manual action needed: reply "
                                  f"<b>/restart</b> or GitHub app → Actions → "
                                  f"DMan Cloud Daemon → Run workflow.")
                _save_last_alert(_restart_key)
            else:
                issues.append("🔄 Daemon still appears down — already auto-restarted "
                              "recently, waiting to see if it recovers before trying again")

    if not issues:
        print("  🐕 Watchdog: all checks passed")
        return

    print(f"  🐕 Watchdog: {len(issues)} issue(s) found")
    for i in issues:
        print(f"     {i}")
    _key = "__WATCHDOG__"
    if not _is_duplicate_alert(_key):
        send_telegram("🐕 <b>DMan Watchdog</b> — possible issue(s) detected:\n\n"
                      + "\n".join(issues))
        _save_last_alert(_key)
    else:
        print("  🐕 (alert suppressed — sent within the last "
              f"{ALERT_COOLDOWN_MIN} min already)")


def run_fallback_guard() -> None:
    """
    Local, GitHub-Actions-INDEPENDENT contingency. Confirmed necessary live
    2026-08-06: a multi-hour GitHub platform outage left every scheduled
    scan/daemon/watchdog run failing identically at ~15 minutes — GitHub's
    own runner-queue timeout (platform-level, distinct from and shorter
    than every one of our workflows' own configured timeout-minutes, which
    never even started counting since the job never began running). Real
    market hours, zero scan coverage, no recovery short of a human noticing
    and running the scan by hand — which is exactly what happened. GitHub's
    own status page confirmed BOTH hosted and self-hosted runners were
    affected, so a self-hosted runner would not have helped here — the
    fallback has to not touch GitHub's runner system at all.

    Designed to run OUTSIDE GitHub Actions entirely (e.g. Windows Task
    Scheduler on the account owner's own machine, see
    scripts/register_fallback_task.ps1) so a GitHub-side outage can't also
    take down the thing meant to work around it. Checks the scanner
    workflow's most recent run via the GitHub API; if it shows the ~15-min
    "never acquired a runner" duration signature, is still queued/running
    long past normal, or simply never started in the current window, runs
    the scan locally instead and alerts either way so there's no silent
    gap in visibility.
    """
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        print("  🆘 Fallback guard: weekend, skipping.")
        return
    t = now_et.hour * 100 + now_et.minute
    if not (930 <= t <= 1600):
        print("  🆘 Fallback guard: outside market hours, skipping.")
        return
    if not GITHUB_TOKEN:
        print("  🆘 Fallback guard: no GITHUB_TOKEN configured — cannot check workflow health, skipping.")
        return

    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/dman_scanner.yml/runs",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"},
            params={"per_page": 1}, timeout=10,
        )
        runs = resp.json().get("workflow_runs", [])
    except Exception as exc:
        print(f"  🆘 Fallback guard: GitHub health check itself failed ({exc}) — can't verify, skipping to be safe.")
        return

    unhealthy, reason = _diagnose_github_health(runs, now_et)
    if not unhealthy:
        print("  🆘 Fallback guard: GitHub Actions looks healthy, nothing to do.")
        return

    _key = "__FALLBACK_GUARD_ACTIVATED__"
    print(f"  🆘 Fallback guard: {reason} — running today's scan locally.")
    if not _is_duplicate_alert(_key):
        send_telegram(
            f"🆘 <b>Local fallback activated</b>\n{reason}.\n"
            f"Running today's scan on your machine instead — coverage continues."
        )
        _save_last_alert(_key)

    import subprocess
    universe = "curated" if t < 1000 else "all"
    result = subprocess.run(
        [sys.executable, __file__, "--mode", "scan", "--universe", universe, "--ai", "--submit"],
        capture_output=False,
    )
    if result.returncode != 0:
        send_telegram("🆘 Local fallback scan itself failed — check your machine directly, this isn't a GitHub problem anymore.")


def _diagnose_github_health(runs: list[dict], now_et: datetime) -> tuple[bool, str]:
    """
    Pulled out of run_fallback_guard() for direct testability. runs is the
    workflow_runs list from GitHub's API (newest first); returns
    (is_unhealthy, human_reason).
    """
    if not runs:
        return True, "no scanner workflow runs found at all"

    latest = runs[0]
    created = datetime.fromisoformat(latest["created_at"].replace("Z", "+00:00"))
    age_min = (datetime.now(created.tzinfo) - created).total_seconds() / 60
    status = latest.get("status")
    conclusion = latest.get("conclusion")

    if status == "completed" and conclusion == "failure":
        updated = datetime.fromisoformat(latest["updated_at"].replace("Z", "+00:00"))
        duration_min = (updated - created).total_seconds() / 60
        # Confirmed live 2026-08-06 across 3 different workflows with 3
        # different configured timeout-minutes, all failing at the same
        # ~15 min mark — that's GitHub's own runner-queue timeout, not
        # ours. A genuine code bug fails in seconds to low minutes.
        if 13 <= duration_min <= 17:
            return True, f"most recent scan failed after {duration_min:.0f} min — matches GitHub's runner-queue timeout signature, not a code failure"
        return False, ""

    if status in ("queued", "in_progress") and age_min > 20:
        return True, f"most recent scan still {status} after {age_min:.0f} min — runner likely never assigned"

    if age_min > 90:
        return True, f"no scan run started in {age_min:.0f} min during market hours"

    return False, ""


def _stocktwits_quick_take(ticker: str) -> str:
    """
    Lightweight technical snapshot for a freshly-detected StockTwits call,
    computed at DETECTION time so the Telegram alert itself carries real
    signal ("is this worth looking at right now") instead of just "ticker
    added, check back later." Deliberately NOT a duplicate scoring engine —
    the real, full evaluation still happens via the existing immediate-
    scan-dispatch path (dman_stocktwits.yml's "Trigger immediate scan"
    step). This uses the exact same fetch_df/compute_indicators the real
    scanner uses so the numbers a human sees here match what the full
    scan sees moments later. Fails open to a short "unavailable" string —
    never blocks the alert itself from sending.
    """
    try:
        df = fetch_df(ticker, period_days=40)
        if df is None or len(df) < 20:
            return "  quick-take unavailable (insufficient data)"
        df = compute_indicators(df.copy())
        last = df.iloc[-1]
        prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else float(last["Close"])
        px = get_live_price(ticker) or float(last["Close"])
        gap_pct = (px - prev_close) / prev_close * 100 if prev_close > 0 else 0.0
        rvol = float(last.get("RVOL", 0) or 0)
        rsi  = float(last.get("RSI", 0) or 0)
        hi20 = float(df["High"].tail(20).max())
        off_high_pct = (px - hi20) / hi20 * 100 if hi20 > 0 else 0.0

        flags = [f"{gap_pct:+.1f}% today"]
        score = 1 if gap_pct >= 5 else (-1 if gap_pct <= -5 else 0)
        if rvol > 0:
            flags.append(f"RVOL {rvol:.1f}x")
            score += 1 if rvol >= 2 else 0
        if rsi:
            flags.append(f"RSI {rsi:.0f}")
            score += 1 if 50 <= rsi <= 75 else (-1 if (rsi >= 85 or rsi <= 30) else 0)
        if off_high_pct >= -2:
            flags.append("near 20d high")
            score += 1
        elif off_high_pct <= -15:
            flags.append(f"{off_high_pct:.0f}% off 20d high")
            score -= 1

        label = ("🔥 Strong setup" if score >= 3 else
                  "👀 Worth a look" if score >= 1 else
                  "⚠️ Caution — weak technicals" if score <= -2 else
                  "😐 Mixed")
        return f"  ${px:.2f}  " + " | ".join(flags) + f"\n  {label}"
    except Exception:
        return "  quick-take unavailable (data error)"


def run_stocktwits_monitor() -> None:
    """
    Fetch ProfessorDman1's recent StockTwits messages.
    Auto-adds new ticker calls to dman_smallcap_watchlist.json (data file).
    The workflow commits and pushes the file so the next scan picks them up.
    Uses a seen-ticker cache (dman_stocktwits_seen.json) — 24h dedup per ticker.
    """
    from datetime import timezone as _tz
    _SEEN_FILE = "dman_stocktwits_seen.json"
    _HOURS_BACK = 6

    # Load seen cache
    try:
        with open(_SEEN_FILE) as _f:
            _seen: dict = json.load(_f)
    except (FileNotFoundError, json.JSONDecodeError):
        _seen = {}

    # Purge entries older than 24h
    _now_utc = datetime.now(_tz.utc)
    _seen = {
        t: ts for t, ts in _seen.items()
        if (_now_utc - datetime.fromisoformat(ts)).total_seconds() < 86_400
    }

    # Fetch DMan's StockTwits stream (free public API, no key required)
    print("  📡 Fetching @ProfessorDman1 StockTwits stream...", flush=True)
    try:
        _resp = requests.get(
            "https://api.stocktwits.com/api/2/streams/user/ProfessorDman1.json",
            params={"limit": 30},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if _resp.status_code == 429:
            print("  📡 StockTwits rate-limited — skipping this run")
            return
        if _resp.status_code != 200:
            print(f"  📡 StockTwits API returned HTTP {_resp.status_code} — skipping")
            return
        _data = _resp.json()
    except Exception as _e:
        print(f"  📡 StockTwits fetch error: {_e}")
        return

    _messages = _data.get("messages", [])
    print(f"  📡 {len(_messages)} recent messages fetched")

    _known_large  = set(WATCHLIST)
    _cutoff       = _now_utc - timedelta(hours=_HOURS_BACK)
    _new_calls:   list[dict] = []
    _seen_this_run: set[str] = set()

    for _msg in _messages:
        try:
            _msg_time = datetime.fromisoformat(
                _msg.get("created_at", "").replace("Z", "+00:00")
            )
        except Exception:
            continue
        if _msg_time < _cutoff:
            continue
        for _sym in _msg.get("symbols", []):
            _ticker = _sym.get("symbol", "").upper().strip()
            if (not _ticker or _ticker in _known_large
                    or _ticker in _seen or _ticker in _seen_this_run):
                continue
            _new_calls.append({
                "ticker": _ticker,
                "body":   _msg.get("body", "")[:120].replace("\n", " "),
            })
            _seen[_ticker] = _now_utc.isoformat()
            _seen_this_run.add(_ticker)

    # Persist seen cache
    try:
        with open(_SEEN_FILE, "w") as _f:
            json.dump(_seen, _f)
    except Exception:
        pass

    if not _new_calls:
        print(f"  📡 No new DMan calls in the last {_HOURS_BACK}h")
        return

    # Auto-add to DMAN_SMALLCAP_WATCHLIST
    _to_add  = [_c["ticker"] for _c in _new_calls]
    _added   = _stocktwits_inject_tickers(_to_add)

    # Telegram — confirm what was added, with a live quick-take so the alert
    # itself answers "is this worth looking at right now" instead of just
    # "ticker added, check back later."
    _lines = []
    for _c in _new_calls:
        _tag = "✅ added to scanner" if _c["ticker"] in _added else "already tracked"
        _quick = _stocktwits_quick_take(_c["ticker"])
        _lines.append(f"  <b>{_c['ticker']}</b> [{_tag}]\n  \"{_c['body']}\"\n{_quick}\n"
                      f"  💬 <code>/options {_c['ticker']}</code> to browse strikes and buy")

    send_telegram(
        f"📡 <b>DMan StockTwits — {len(_new_calls)} call(s) detected</b>\n\n"
        + "\n\n".join(_lines)
        + (f"\n\n✅ {len(_added)} ticker(s) added to DMAN_SMALLCAP_WATCHLIST." if _added else "")
    )
    print(f"  📡 Added: {_added}  |  Already known: {[c['ticker'] for c in _new_calls if c['ticker'] not in _added]}")


_MASSIVE_NEWS_CACHE: dict[tuple, tuple[float, dict]] = {}
_MASSIVE_NEWS_CACHE_TTL_S = 600   # 10 min — matches the daemon's scan_loop cadence


def _fetch_massive_benzinga_news(tickers: list[str], hours_back: int = 20) -> dict[str, list[str]]:
    """
    Real-time per-ticker headlines via Massive's Benzinga news proxy
    (api.massive.com/benzinga/v2/news) — same reliably server-side
    ticker-filtered pipeline that fixed earnings-date lookups
    (_fetch_massive_earnings, confirmed live 2026-07-30). Requires a
    Massive plan that includes Benzinga News — a SEPARATE entitlement
    from Benzinga Earnings (confirmed live 2026-08-01: this account's
    MASSIVE_API_KEY gets 403 NOT_AUTHORIZED on /benzinga/v2/news despite
    working fine on /benzinga/v1/earnings). Returns {} on 403/no key/any
    error so callers fall through to the existing Benzinga-direct/Alpaca/
    yfinance chain unchanged — this is additive, wired in ahead of time so
    it starts working the moment the News entitlement is purchased, no
    further code changes needed.

    10-min TTL cache added 2026-08-04 — confirmed live: this had ZERO
    caching (unlike _fetch_massive_earnings, which got a 20-min cache on
    2026-07-31 for the identical reason), and gets called every scan cycle
    for the same ~80-ticker WATCHLIST batch, daemon every 10 min plus
    hourly cron on top. Result: real 429s on the SAME afternoon PLTR's
    earnings (2026-08-03) needed a news-based already-reported
    confirmation as a fallback after the primary Massive-earnings time
    field wasn't populated yet — the fallback was rate-limited into
    uselessness at exactly the moment it was needed, contributing to that
    candidate landing on UNKNOWN-TODAY and never getting an offer. Cache
    key is the exact ticker batch, stable in practice since WATCHLIST
    order doesn't change between calls.
    """
    if not MASSIVE_API_KEY:
        return {}
    from datetime import timezone as _tz
    cutoff = datetime.now(_tz.utc) - timedelta(hours=hours_back)
    result: dict[str, list[str]] = {t: [] for t in tickers}
    batch_size = 50   # matches the direct-Benzinga batching below
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        cache_key = (tuple(batch), hours_back)
        cached = _MASSIVE_NEWS_CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < _MASSIVE_NEWS_CACHE_TTL_S:
            for t, headlines in cached[1].items():
                if t in result:
                    result[t] = headlines
            continue
        try:
            resp = requests.get(
                "https://api.massive.com/benzinga/v2/news",
                params={
                    "tickers.any_of": ",".join(batch),
                    "published.gte":  cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit":          100,
                    "apiKey":         MASSIVE_API_KEY,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                if resp.status_code != 403:   # 403 = not entitled yet — expected until purchased, not worth logging every run
                    print(f"  [massive-news] batch {i}-{i+len(batch)}: HTTP {resp.status_code} "
                          f"— {resp.text[:150]}", file=sys.stderr)
                continue
            batch_result: dict[str, list[str]] = {t: [] for t in batch}
            for art in resp.json().get("results", []) or []:
                title = art.get("title", "")
                if not title:
                    continue
                for sym in art.get("tickers", []) or []:
                    sym = str(sym).upper()
                    if sym in batch_result and len(batch_result[sym]) < 5:
                        batch_result[sym].append(title)
            for t, headlines in batch_result.items():
                result[t] = headlines
            _MASSIVE_NEWS_CACHE[cache_key] = (time.time(), batch_result)
        except Exception:
            continue
    return result


_MASSIVE_SENTIMENT_CACHE: dict[tuple, tuple[float, list]] = {}
_MASSIVE_SENTIMENT_CACHE_TTL_S = 600   # matches _MASSIVE_NEWS_CACHE_TTL_S


def _fetch_massive_reference_news(ticker: str, hours_back: int = 48) -> list[dict]:
    """
    Massive's /v2/reference/news — distinct from _fetch_massive_benzinga_news's
    /benzinga/v2/news (Benzinga-only headlines, no sentiment). This one
    aggregates MULTIPLE publishers (Motley Fool, Investing.com, etc.) and
    includes per-article sentiment analysis (results[].insights[].sentiment:
    positive/neutral/negative, plus a plain-English sentiment_reasoning).

    Confirmed live 2026-08-13: accessible under the current MASSIVE_API_KEY
    (200, real multi-publisher results with sentiment) even though the
    separate analyst-ratings/bulls-bears-say/corporate-guidance tier is
    403'd on this plan (see the "no further paid API tiers" comment near
    check_insider_activity) — this endpoint closes the "market sentiment"
    gap those were meant to fill without needing a plan upgrade.

    Returns raw result dicts (title, published_utc, insights, ...); [] on
    any failure or if MASSIVE_API_KEY isn't set (fail-open to callers).
    10-min TTL cache, same reasoning as _fetch_massive_benzinga_news.
    """
    if not MASSIVE_API_KEY:
        return []
    from datetime import timezone as _tz
    cutoff = datetime.now(_tz.utc) - timedelta(hours=hours_back)
    cache_key = (ticker, hours_back)
    cached = _MASSIVE_SENTIMENT_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _MASSIVE_SENTIMENT_CACHE_TTL_S:
        return cached[1]
    try:
        resp = requests.get(
            "https://api.massive.com/v2/reference/news",
            params={
                "ticker":            ticker,
                "published_utc.gte": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit":             10,
                "apiKey":            MASSIVE_API_KEY,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            _alert_massive_api_failure(
                "reference-news", f"HTTP {resp.status_code} from /v2/reference/news")
            return []
        results = resp.json().get("results", []) or []
        _MASSIVE_SENTIMENT_CACHE[cache_key] = (time.time(), results)
        return results
    except Exception as exc:
        _alert_massive_api_failure("reference-news", str(exc))
        return []


def _news_sentiment_verdict(ticker: str, hours_back: int = 48) -> Optional[str]:
    """
    "positive", "neutral", "negative", or None (no sentiment data found) —
    majority vote across every article's insights[] entry matching this
    ticker from _fetch_massive_reference_news(). Ties resolve to whichever
    dict.max() picks first among equal counts (positive > neutral >
    negative in insertion order), which only matters for genuinely tied
    votes — not worth a tie-break rule for a 2-3 article sample size.

    Used by detect_low_float_catalyst()'s catalyst-confirmation gate to
    tell a genuine bullish catalyst apart from a volume spike driven by
    BAD news — "there is news" alone doesn't mean the news supports a
    LONG entry on a long-only system.
    """
    articles = _fetch_massive_reference_news(ticker, hours_back)
    if not articles:
        return None
    votes = {"positive": 0, "neutral": 0, "negative": 0}
    for art in articles:
        for insight in art.get("insights", []) or []:
            if str(insight.get("ticker", "")).upper() == ticker.upper():
                s = insight.get("sentiment")
                if s in votes:
                    votes[s] += 1
    if sum(votes.values()) == 0:
        return None
    return max(votes, key=votes.get)


def _news_boost_after_sentiment_veto(has_headline: bool, ticker: str) -> bool:
    """
    Refines the cheap "does a headline exist" check (has_headline) with a
    sentiment lookup — returns False if the news backing that headline is
    confirmed NEGATIVE, otherwise passes has_headline through unchanged.
    Added 2026-08-14, direct instruction to make sure news actually helps
    get into plays rather than just "a headline exists" earning the same
    +5-point score credit (and, for Gap & Hold / Bear Gap Hold, the same
    MTF override — see score_signal) regardless of what it says. Mirrors
    the exact block-on-negative philosophy detect_low_float_catalyst()'s
    catalyst gate already uses; unknown/neutral/positive all pass through
    unchanged, same as that gate. Only called when has_headline is already
    True (the batch presence check has already run) — this is the one
    network call this adds, and _news_sentiment_verdict has its own 10-min
    cache on top.
    """
    if not has_headline:
        return False
    return _news_sentiment_verdict(ticker) != "negative"


def _fetch_benzinga_ticker_news(tickers: list[str], hours_back: int = 20) -> dict[str, list[str]]:
    """
    Fetch real-time ticker-specific headlines from Benzinga Basic API.
    Returns {ticker: [headline, ...]}. Returns {} if key not set or on error.
    """
    if not BENZINGA_API_KEY:
        return {}
    from datetime import timezone as _tz
    cutoff = datetime.now(_tz.utc) - timedelta(hours=hours_back)
    result: dict[str, list[str]] = {t: [] for t in tickers}
    # Benzinga accepts comma-separated tickers but batches of ≤50 are reliable
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        try:
            resp = requests.get(
                "https://api.benzinga.com/api/v2/news",
                params={
                    "token":          BENZINGA_API_KEY,
                    "tickers":        ",".join(batch),
                    "pageSize":       100,
                    "displayOutput":  "headline",
                    "publishedAfter": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                headers={"Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                # Print WHY, not just that it failed — a 401/403 (key not yet
                # approved, or invalid) looks identical to "0 headlines" in
                # the caller's summary line otherwise, making a rejected key
                # indistinguishable from a genuinely quiet news day.
                print(f"  [benzinga] batch {i}-{i+len(batch)}: HTTP {resp.status_code} "
                      f"— {resp.text[:150]}", file=sys.stderr)
                continue
            articles = resp.json() if isinstance(resp.json(), list) else resp.json().get("result", [])
            for art in articles:
                title = art.get("title", "")
                if not title:
                    continue
                for tk_obj in art.get("stocks", []):
                    sym = (tk_obj.get("name") or "").upper()
                    if sym in result:
                        if len(result[sym]) < 5:
                            result[sym].append(title)
        except Exception:
            continue
    return result


def _fetch_benzinga_breaking_news(hours_back: int = 8) -> list[tuple[str, str, str, int]]:
    """
    Fetch real-time general market news from Benzinga Basic API.
    Returns (headline, source, time_str, impact) sorted by abs(impact) — same
    format as _fetch_breaking_news_rss so callers are drop-in compatible.
    Returns [] if key not set or on error.
    """
    if not BENZINGA_API_KEY:
        return []
    from datetime import timezone as _tz
    from email.utils import parsedate_to_datetime as _parse_date
    cutoff = datetime.now(_tz.utc) - timedelta(hours=hours_back)
    try:
        resp = requests.get(
            "https://api.benzinga.com/api/v2/news",
            params={
                "token":          BENZINGA_API_KEY,
                "pageSize":       30,
                "displayOutput":  "headline",
                "publishedAfter": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            print(f"  [benzinga] breaking news: HTTP {resp.status_code} "
                  f"— {resp.text[:150]}", file=sys.stderr)
            return []
        articles = resp.json() if isinstance(resp.json(), list) else resp.json().get("result", [])
        results: list[tuple[str, str, str, int]] = []
        for art in articles:
            title = art.get("title", "")
            if not title:
                continue
            # Parse created timestamp ("Thu, 17 Jul 2026 09:30:00 -0400" or ISO)
            created = art.get("created", "")
            ts_str = ""
            try:
                if created:
                    _dt = _parse_date(created).astimezone(MT)
                    ts_str = _dt.strftime("%I:%M %p MT")
            except Exception:
                pass
            text = title.lower()
            score = 0
            for kw, pts in _MACRO_BEARISH:
                if kw in text:
                    score += pts
            for kw, pts in _MACRO_BULLISH:
                if kw in text:
                    score += pts
            score = max(-2, min(2, score))
            results.append((title, "Benzinga", ts_str, score))
        results.sort(key=lambda x: abs(x[3]), reverse=True)
        return results[:6]
    except Exception:
        return []


def _log_news_event(symbols: list[str], headline: str, source: str = "",
                    tag: str = "watchlist", sentiment: Optional[str] = None) -> None:
    """
    Appends one headline to the rolling background news log
    (NEWS_LOG_FILE) — added 2026-08-15, direct instruction to have the
    algo "constantly internalize" news across market + extended hours
    instead of only flashing headlines past in a Telegram alert and
    losing them. Every relevant headline gets logged here regardless of
    whether it also triggers a Telegram alert (see news_data_stream_loop's
    on_news handler, now near-silent by the same instruction — held-
    position and macro news still alert, routine watchlist headlines log
    quietly here instead).

    tag is one of "held" / "macro" / "watchlist" — which relevance bucket
    matched, for filtering later without re-deriving it. Deduped by exact
    (symbols, headline) match against the current log so the SAME story
    re-appearing across repeated REST fetches (run_pro_scanner's hourly
    cron news pre-fetch has a rolling lookback window, so an already-seen
    headline can legitimately reappear in the next cycle's fetch) doesn't
    bloat the log with duplicates — the real-time stream side already has
    its own id-based dedup (_news_seen_ids) before this is ever called,
    but the REST path has no article id to dedup on, only the text.

    Sorted by ts and capped at NEWS_LOG_MAX_ENTRIES on every write, same
    pattern as _append_scan_log() — see that function's docstring for why
    a positional slice instead of a ts-sort silently drops entries the
    moment the on-disk file isn't already perfectly ordered.
    """
    if not symbols or not headline:
        return
    try:
        log: list[dict] = []
        if os.path.exists(NEWS_LOG_FILE):
            try:
                with open(NEWS_LOG_FILE) as f:
                    log = json.load(f)
            except Exception:
                log = []
        _sorted_syms = sorted(symbols)
        for _e in log:
            if _e.get("symbols") == _sorted_syms and _e.get("headline") == headline:
                return   # already logged, same story re-fetched
        log.append({
            "ts":        datetime.now(ET).isoformat(),
            "symbols":   _sorted_syms,
            "headline":  headline,
            "source":    source,
            "tag":       tag,
            "sentiment": sentiment,
        })
        log = sorted(log, key=lambda e: e.get("ts", ""), reverse=True)[:NEWS_LOG_MAX_ENTRIES][::-1]
        _write_json_atomic(NEWS_LOG_FILE, log, indent=2)
    except Exception:
        pass   # background knowledge-base write — never worth blocking a scan/alert over


def _news_sentiment_breadth(hours_back: float = 24.0) -> dict:
    """
    Rolling positive-vs-negative sentiment breadth across the curated
    universe (WATCHLIST + DMAN_SMALLCAP_WATCHLIST), computed entirely from
    the background news log (NEWS_LOG_FILE) — same "is the tape broadly
    turning" spirit as get_market_regime()'s existing IWM-breadth /
    QQQ-leadership inputs, just built from news flow instead of price.

    Added 2026-08-15 as an OBSERVATION-ONLY factor, direct instruction:
    surfaced in get_market_regime()'s details for visibility, NOT added to
    the numeric score or any gate. That score directly decides which
    trades get taken (regime_allows_signal, per-signal score
    contribution) — a sentiment signal with zero live track record on
    this account has no business influencing real entries on day one.
    Revisit wiring it into score only after watching this run for real.

    Counts LOG ENTRIES (distinct stories, after _log_news_event's own
    dedup), not ticker-mentions — a single multi-symbol headline
    shouldn't get double weight just because it's tagged with more
    tickers. "unknown" (sentiment=None — no verdict was available when
    logged) is tracked separately and excluded from breadth_pct's
    denominator; a quiet news day with mostly-unknown sentiment must
    read as low-confidence, not silently default to neutral.

    Returns a dict; every count defaults to 0 and breadth_pct to None on
    any failure or empty log — never raises, matching this file's
    fail-open convention for anything that isn't a hard trading gate.
    """
    result = {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0,
             "total": 0, "breadth_pct": None, "hours_back": hours_back}
    try:
        if not os.path.exists(NEWS_LOG_FILE):
            return result
        with open(NEWS_LOG_FILE) as f:
            log = json.load(f)
        from datetime import timezone as _tz
        cutoff = datetime.now(_tz.utc) - timedelta(hours=hours_back)
        universe = set(WATCHLIST) | set(DMAN_SMALLCAP_WATCHLIST)
        for entry in log:
            try:
                ts = datetime.fromisoformat(entry.get("ts", ""))
                if ts.tzinfo is None:
                    continue
                if ts.astimezone(_tz.utc) < cutoff:
                    continue
            except Exception:
                continue
            if not (set(entry.get("symbols", [])) & universe):
                continue
            s = entry.get("sentiment")
            if s in ("positive", "negative", "neutral"):
                result[s] += 1
            else:
                result["unknown"] += 1
            result["total"] += 1
        _scored = result["positive"] + result["negative"]
        if _scored > 0:
            result["breadth_pct"] = round((result["positive"] - result["negative"]) / _scored * 100, 1)
    except Exception:
        pass
    return result


def _fetch_alpaca_news(tickers: list[str], hours_back: int = 18) -> dict[str, list[str]]:
    """
    Fetch recent news headlines for a list of tickers.
    Priority: Massive Benzinga News → direct Benzinga API → Alpaca News API → yfinance.
    Returns {ticker: [headline, ...]}.
    """
    from datetime import timezone
    result: dict[str, list[str]] = {t: [] for t in tickers}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    # Primary: Massive's Benzinga news proxy — confirmed correctly ticker-filtered
    # server-side (same pipeline that fixed earnings lookups), unlike the direct
    # API below. No-ops to {} until the Benzinga News entitlement is purchased.
    mv = _fetch_massive_benzinga_news(tickers, hours_back=hours_back)
    filled_mv = sum(1 for v in mv.values() if v)
    if filled_mv:
        print(f"  [news] Massive Benzinga returned headlines for {filled_mv}/{len(tickers)} tickers")
        for t, headlines in mv.items():
            if headlines:
                result[t] = headlines
        if all(result[t] for t in tickers):
            return result

    # Secondary: direct Benzinga API (requires BENZINGA_API_KEY) — known
    # unreliable per-ticker filter (see _fetch_benzinga_ticker_news docstring),
    # kept as a fallback layer rather than the primary source now.
    if BENZINGA_API_KEY:
        bz = _fetch_benzinga_ticker_news(tickers, hours_back=hours_back)
        filled = sum(1 for v in bz.values() if v)
        print(f"  [news] Benzinga returned headlines for {filled}/{len(tickers)} tickers")
        # Merge — keep yfinance fallback only for tickers with no results yet
        for t, headlines in bz.items():
            if headlines and not result[t]:
                result[t] = headlines
        # If everything's covered by Massive+Benzinga combined, return early
        if all(result[t] for t in tickers):
            return result

    # Tertiary: Alpaca News API (free tier, delayed)
    try:
        from alpaca.data.historical.news import NewsClient as _NC
        from alpaca.data.requests import NewsRequest as _NR
        _nc = _NC(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
        for ticker in tickers:
            if result[ticker]:   # already have Benzinga headlines
                continue
            try:
                req  = _NR(symbols=[ticker], start=cutoff, limit=5)
                news = _nc.get_news(req)
                items = news.data.get(ticker, []) if hasattr(news, "data") else []
                if not items and hasattr(news, "news"):
                    items = news.news
                result[ticker] = [getattr(n, "headline", str(n)) for n in items][:5]
            except Exception:
                pass
        if any(result[t] for t in tickers):
            return result
    except Exception:
        pass

    # Fallback: yfinance (no API key, last resort)
    for ticker in tickers:
        if result[ticker]:
            continue
        try:
            arts = yf.Ticker(ticker).news or []
            result[ticker] = [a.get("title", "") for a in arts[:5] if a.get("title")]
        except Exception:
            pass
    return result


_CATALYST_KEYWORDS = {
    "bullish": ["fda", "approv", "clearance", "granted", "positive", "phase", "trial",
                "merger", "acqui", "contract", "awarded", "partnership", "license",
                "reverse split", "share consolidation", "squeeze", "short", "uplisting",
                "nasdaq", "nyse", "breakthrough", "milestone", "deal", "win"],
    "bearish": ["dilut", "offering", "placement", "warrant", "downgrad", "fda reject",
                "complete response", "clinical hold", "investigation", "fraud", "delisting"],
}


def _score_news_headline(headline: str) -> tuple[str, str]:
    """
    Return (sentiment, matched_keyword) for a headline.
    sentiment: 'bullish' | 'bearish' | 'neutral'
    """
    h = headline.lower()
    for kw in _CATALYST_KEYWORDS["bearish"]:
        if kw in h:
            return "bearish", kw
    for kw in _CATALYST_KEYWORDS["bullish"]:
        if kw in h:
            return "bullish", kw
    return "neutral", ""


def _check_edgar_8k(ticker: str, hours_back: int = 30) -> tuple[bool, str]:
    """
    Check SEC EDGAR full-text search for recent 8-K filings.
    Returns (found, summary_string).
    Tier A validation: real catalyst = real filing.
    """
    from urllib.parse import quote
    from datetime import timezone
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    d_from = cutoff.strftime("%Y-%m-%d")
    d_to   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    url = (f"https://efts.sec.gov/LATEST/search-index?"
           f"q=%22{quote(ticker)}%22&forms=8-K"
           f"&dateRange=custom&startdt={d_from}&enddt={d_to}")
    try:
        resp = requests.get(url, timeout=10,
                            headers={"User-Agent": "DManAlgo research@dman.algo"})
        if resp.status_code != 200:
            return False, ""
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return False, ""
        src    = hits[0].get("_source", {})
        names  = src.get("display_names", [])
        filed  = src.get("file_date", "")
        form   = src.get("form_type", "8-K")
        entity = names[0] if names else ticker
        return True, f"{form} filed {filed} — {entity}"
    except Exception:
        return False, ""


def _get_stocktwits_sentiment(ticker: str) -> tuple[float, int]:
    """
    Returns (bull_pct, message_count) from StockTwits public API (no auth).
    bull_pct: % of sentiment-tagged messages that are bullish (0–100).
    High message count + high bull_pct = retail FOMO building.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return 0.0, 0
        messages = resp.json().get("messages", [])
        bull = sum(1 for m in messages
                   if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bullish")
        bear = sum(1 for m in messages
                   if m.get("entities", {}).get("sentiment", {}).get("basic") == "Bearish")
        tagged = bull + bear
        return (bull / tagged * 100 if tagged > 0 else 0.0), len(messages)
    except Exception:
        return 0.0, 0


def _fetch_dman_stocktwits_calls(hours_back: int = 48) -> list[tuple[str, str, str]]:
    """Fetch @professorDman1's recent StockTwits posts and extract ticker mentions.
    Returns list of (ticker, message_excerpt, timestamp_str).
    StockTwits public API — no auth key required.
    """
    results: list[tuple[str, str, str]] = []
    try:
        resp = requests.get(
            "https://api.stocktwits.com/api/2/streams/user/professorDman1.json",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code != 200:
            return results
        messages = resp.json().get("messages", [])
        cutoff = datetime.now(ET) - timedelta(hours=hours_back)
        for msg in messages:
            try:
                ts = datetime.fromisoformat(
                    msg.get("created_at", "").replace("Z", "+00:00")
                ).astimezone(ET)
                if ts < cutoff:
                    continue
                symbols = [s["symbol"] for s in msg.get("entities", {}).get("symbols", [])]
                body = msg.get("body", "")[:200]
                for sym in symbols:
                    results.append((sym.upper(), body, ts.astimezone(MT).strftime("%m/%d %I:%M %p MT")))
            except Exception:
                continue
    except Exception:
        pass
    return results


def _fetch_dman_twitter_calls(hours_back: int = 48) -> list[tuple[str, str, str]]:
    """Fetch @professorDman1 tweets via Nitter RSS (no API key, no login).
    Tries multiple public Nitter instances in order; silently returns [] on full failure.
    Returns list of (ticker, tweet_excerpt, timestamp_str).
    """
    import re as _re
    import xml.etree.ElementTree as _etree
    from email.utils import parsedate_to_datetime as _parse_date

    _NITTER_INSTANCES = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.net",
        "https://nitter.lucabased.xyz",
    ]
    cutoff  = datetime.now(ET) - timedelta(hours=hours_back)
    results: list[tuple[str, str, str]] = []

    for _base in _NITTER_INSTANCES:
        try:
            _resp = requests.get(
                f"{_base}/professorDman1/rss",
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if _resp.status_code != 200:
                continue
            _root  = _etree.fromstring(_resp.content)
            _items = _root.findall(".//item")
            if not _items:
                continue
            for _item in _items:
                try:
                    _pub = _item.findtext("pubDate", "")
                    _ts  = _parse_date(_pub).astimezone(ET) if _pub else None
                    if _ts and _ts < cutoff:
                        continue
                    _raw  = (_item.findtext("title", "") + " " + _item.findtext("description", ""))
                    _text = _re.sub(r"<[^>]+>", " ", _raw).strip()
                    _syms = list(dict.fromkeys(_re.findall(r"\$([A-Za-z]{1,5})\b", _text)))
                    _ts_s = _ts.astimezone(MT).strftime("%m/%d %I:%M %p MT") if _ts else "?"
                    for _sym in _syms:
                        results.append((_sym.upper(), _text[:200], _ts_s))
                except Exception:
                    continue
            if results:
                return results   # got data from this instance — stop trying
        except Exception:
            continue

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Global Market Context — adaptive risk multiplier + breaking news
# ──────────────────────────────────────────────────────────────────────────────

# Bearish/bullish keywords for macro news headlines (lower-case matching)
_MACRO_BEARISH = [
    ("rate hike", -2), ("hawkish", -1), ("tariff", -1), ("trade war", -2),
    ("sanctions", -1), ("invasion", -2), ("bank failure", -2), ("recession", -1),
    ("inflation surge", -1), ("credit crisis", -2), ("federal reserve tight", -1),
    ("yield spike", -1), ("geopolit", -1), ("default", -1),
]
_MACRO_BULLISH = [
    ("rate cut", +2), ("dovish", +1), ("soft landing", +1), ("stimulus", +1),
    ("beat expectations", +1), ("strong jobs", +1), ("ai boom", +1),
    ("m&a", +1), ("acquisition", +1), ("deal", +1), ("fed pause", +2),
]


def _days_to_next_macro_print(today: Optional[date] = None) -> Optional[int]:
    """
    0 if NFP/CPI/PPI releases today, 1 if tomorrow, None otherwise. FOMC is
    deliberately excluded — its existing ±1 day blackout in check_macro_safe()
    already blocks entries outright, so a sizing nudge on top would be moot.
    Pulled out of _fetch_global_context() as its own function so the
    real-condition-adaptive sizing logic is testable without mocking yfinance.
    """
    today = today or date.today()
    days_away = [
        (ev - today).days
        for ev in (_nfp_dates() | _CPI_DATES | _PPI_DATES)
        if 0 <= (ev - today).days <= 1
    ]
    return min(days_away) if days_away else None


def _fetch_global_context() -> dict:
    """
    Pull overnight futures, global indices, VIX, DXY, BTC, and IWM/SPY ratio
    to compute a composite risk-tone score that adapts position sizing each session.

    Score:  -4 (extreme risk-off) → +4 (extreme risk-on)
    risk_mult: maps score → position size multiplier (0.30 → 1.30)

    Components:
      ES futures  — S&P direction
      NQ futures  — Nasdaq / tech barometer
      RTY futures — Russell 2000 (DIRECT small-cap signal)
      VIX         — fear gauge (weighted -2 if >25)
      DXY         — dollar strength (inverse for speculative names)
      BTC         — risk-on proxy (leads small-cap speculatives 12-24h)
      Asia avg    — Nikkei + Hang Seng overnight
      IWM vs SPY  — small-cap relative leadership (most direct signal)
      Gold        — safe-haven bid (risk-off flag)
      Macro cal.  — NFP/CPI/PPI today or tomorrow (risk-off flag; day-of and
                    day-before, the two windows check_macro_safe()'s hard
                    gate doesn't already cover)
    """
    components: dict[str, str] = {}
    score = 0

    try:
        _SYMS = ["ES=F", "NQ=F", "RTY=F", "^VIX", "DX-Y.NYB",
                 "BTC-USD", "^N225", "^HSI", "SPY", "IWM", "GC=F"]
        data: dict[str, dict] = {}
        for _s in _SYMS:
            try:
                _fi = yf.Ticker(_s).fast_info
                _px = float(_fi.last_price or 0)
                _pc = float(_fi.previous_close or 0)
                if _px > 0 and _pc > 0:
                    data[_s] = {"price": _px, "prev": _pc, "chg": (_px - _pc) / _pc * 100}
            except Exception:
                pass

        # ES futures
        if "ES=F" in data:
            _c = data["ES=F"]["chg"]
            _s = +1 if _c > 0.5 else (-1 if _c < -0.5 else 0)
            score += _s
            components["ES"] = f"{_c:+.1f}%"

        # NQ futures
        if "NQ=F" in data:
            _c = data["NQ=F"]["chg"]
            _s = +1 if _c > 0.7 else (-1 if _c < -0.7 else 0)
            score += _s
            components["NQ"] = f"{_c:+.1f}%"

        # RTY futures — Russell 2000 (direct small-cap predictor)
        if "RTY=F" in data:
            _c = data["RTY=F"]["chg"]
            _s = +1 if _c > 0.5 else (-1 if _c < -0.5 else 0)
            score += _s
            _tag = " ↑SC" if _s > 0 else (" ↓SC" if _s < 0 else "")
            components["RTY"] = f"{_c:+.1f}%{_tag}"

        # VIX
        if "^VIX" in data:
            _vix = data["^VIX"]["price"]
            _vc  = data["^VIX"]["chg"]
            _s = -2 if _vix > 25 else (-1 if _vix > 20 else (+1 if _vix < 15 else 0))
            score += _s
            _tag = " ⚠️HIGH" if _vix > 20 else (" ✅LOW" if _vix < 15 else "")
            components["VIX"] = f"{_vix:.1f} ({_vc:+.1f}%){_tag}"

        # DXY — dollar strength is inverse for speculative small-caps
        if "DX-Y.NYB" in data:
            _c = data["DX-Y.NYB"]["chg"]
            _s = -1 if _c > 0.5 else (+1 if _c < -0.3 else 0)
            score += _s
            components["DXY"] = f"{data['DX-Y.NYB']['price']:.1f} ({_c:+.2f}%)"

        # BTC — risk-on proxy; leads small-cap speculative 12-24h
        if "BTC-USD" in data:
            _c = data["BTC-USD"]["chg"]
            _s = +1 if _c > 3.0 else (-1 if _c < -5.0 else 0)
            score += _s
            _em = "🟢" if _s > 0 else ("🔴" if _s < 0 else "")
            components["BTC"] = f"${data['BTC-USD']['price']:,.0f} ({_c:+.1f}%) {_em}"

        # Asia overnight avg (Nikkei + Hang Seng)
        _asia = [data[k]["chg"] for k in ("^N225", "^HSI") if k in data]
        if _asia:
            _avg = sum(_asia) / len(_asia)
            _s = +1 if _avg > 0.5 else (-1 if _avg < -1.0 else 0)
            score += _s
            _parts = []
            if "^N225" in data: _parts.append(f"N225 {data['^N225']['chg']:+.1f}%")
            if "^HSI"  in data: _parts.append(f"HSI {data['^HSI']['chg']:+.1f}%")
            components["Asia"] = "  ".join(_parts)

        # IWM vs SPY relative — most direct small-cap environment signal
        if "IWM" in data and "SPY" in data:
            _rel = data["IWM"]["chg"] - data["SPY"]["chg"]
            _s = +1 if _rel > 0.3 else (-1 if _rel < -0.3 else 0)
            score += _s
            _tag = " ↑SC leading" if _s > 0 else (" ↓SC lagging" if _s < 0 else "")
            components["IWM-SPY"] = f"rel {_rel:+.1f}%{_tag}"

        # Gold — safe-haven bid = risk-off flag
        if "GC=F" in data:
            _c = data["GC=F"]["chg"]
            if _c > 1.5:
                score -= 1
                components["GOLD"] = f"${data['GC=F']['price']:,.0f} ({_c:+.1f}%) ⚠️ safe-haven"

        # Macro calendar proximity — real, current condition, not a prediction.
        # check_macro_safe() already hard-blocks entries before 10 AM ET on an
        # NFP/CPI/PPI release day, but that leaves two real gaps this same
        # scoring function should cover: the afternoon of release day (still
        # elevated-vol once the number has moved the market, no extra caution
        # today) and the day BEFORE (pre-print positioning risk, currently
        # zero adjustment at all). Added 2026-08-06 heading into NFP (Fri) +
        # CPI (next Wed). See _days_to_next_macro_print().
        _macro_days_away = _days_to_next_macro_print()
        if _macro_days_away == 0:
            score -= 2
            components["MACRO"] = "NFP/CPI/PPI today ⚠️ elevated vol"
        elif _macro_days_away == 1:
            score -= 1
            components["MACRO"] = "NFP/CPI/PPI tomorrow ⚠️ pre-print caution"

    except Exception:
        pass

    score = max(-4, min(4, score))

    if   score >=  3: risk_mult, tone = 1.30, "🟢 RISK-ON"
    elif score >=  1: risk_mult, tone = 1.00, "🟡 NEUTRAL-BULLISH"
    elif score ==  0: risk_mult, tone = 0.85, "🟡 NEUTRAL"
    elif score >= -2: risk_mult, tone = 0.60, "🟠 CAUTIOUS"
    else:             risk_mult, tone = 0.35, "🔴 RISK-OFF"

    _comp_str = "  |  ".join(f"{k}: {v}" for k, v in list(components.items())[:7])
    summary = (
        f"🌍 <b>GLOBAL CONTEXT</b>  [{tone}]  Score: {score:+d}\n"
        f"   {_comp_str}"
    )
    return {"score": score, "risk_mult": risk_mult, "tone": tone,
            "summary": summary, "components": components}


def _fetch_breaking_news_rss(hours_back: int = 8) -> list[tuple[str, str, str, int]]:
    """
    Fetch market-moving headlines.
    Primary: Benzinga real-time API (when key is set).
    Fallback: free RSS feeds (Reuters, MarketWatch, Yahoo Finance).
    Returns list of (headline, source, time_str, impact) sorted by abs(impact).
    impact: -2 (very bearish) to +2 (very bullish).
    """
    # Benzinga real-time path — skip RSS entirely when key is available
    if BENZINGA_API_KEY:
        bz_results = _fetch_benzinga_breaking_news(hours_back=hours_back)
        if bz_results:
            return bz_results

    import xml.etree.ElementTree as _etree
    from email.utils import parsedate_to_datetime as _parse_date
    import re as _re

    _RSS_FEEDS = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
        ("https://feeds.marketwatch.com/marketwatch/marketpulse/", "MarketWatch"),
        ("https://finance.yahoo.com/rss/topstories", "Yahoo Finance"),
    ]
    cutoff  = datetime.now(ET) - timedelta(hours=hours_back)
    results: list[tuple[str, str, str, int]] = []

    for _url, _src in _RSS_FEEDS:
        try:
            _resp = requests.get(_url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
            if _resp.status_code != 200:
                continue
            _root  = _etree.fromstring(_resp.content)
            _items = _root.findall(".//item")[:12]
            for _item in _items:
                try:
                    _title = _item.findtext("title", "")
                    _pub   = _item.findtext("pubDate", "")
                    if not _title:
                        continue
                    _ts = _parse_date(_pub).astimezone(ET) if _pub else None
                    if _ts and _ts < cutoff:
                        continue
                    _text  = _title.lower()
                    _score = 0
                    for _kw, _pts in _MACRO_BEARISH:
                        if _kw in _text:
                            _score += _pts
                    for _kw, _pts in _MACRO_BULLISH:
                        if _kw in _text:
                            _score += _pts
                    _score = max(-2, min(2, _score))
                    _ts_mt = _ts.astimezone(MT) if _ts else None
                    _ts_s  = _ts_mt.strftime("%I:%M %p MT") if _ts_mt else ""
                    results.append((_title, _src, _ts_s, _score))
                except Exception:
                    continue
        except Exception:
            continue

    results.sort(key=lambda x: abs(x[3]), reverse=True)
    return results[:6]


_PM_LEVEL_DEFAULTS: dict = {
    "pm_vwap": 0.0, "pm_vol": 0, "float_rotation_pct": 0.0,
    "pm_high": 0.0, "pm_low": 0.0, "price_vs_vwap_pct": 0.0,
    "entry_zone": "unknown", "pm_trend": "flat",
}


def _calc_premarket_levels(ticker: str, float_shares_m: float) -> dict:
    """
    Fetch 1-min pre+post bars (yfinance) and compute pre-market VWAP,
    float rotation %, PM high/low, price vs VWAP, trend, and entry zone.
    Only called for confirmed movers (≥5% gap) to keep scan fast.
    """
    result = dict(_PM_LEVEL_DEFAULTS)
    try:
        df = yf.download(ticker, period="1d", interval="1m",
                         prepost=True, progress=False, auto_adjust=True)
        if df is None or len(df) == 0:
            return result

        today_et = datetime.now(ET).date()
        rows = []
        for idx in df.index:
            ts = idx
            if hasattr(ts, "tz_convert"):
                ts_et = ts.tz_convert(ET)
            elif hasattr(ts, "tz_localize") and ts.tzinfo is None:
                ts_et = ts.tz_localize("UTC").tz_convert(ET)
            else:
                ts_et = ts
            if ts_et.date() != today_et:
                continue
            # Pre-market: before 9:30 AM ET
            if ts_et.hour > 9 or (ts_et.hour == 9 and ts_et.minute >= 30):
                continue
            rows.append((ts_et, df.loc[idx]))

        if not rows:
            return result

        def _fv(row, col):
            v = row[col]
            return float(v.item() if hasattr(v, "item") else v)

        total_vol = 0; total_pv = 0.0
        highs: list[float] = []; lows: list[float] = []
        for _, row in rows:
            try:
                h = _fv(row, "High"); l = _fv(row, "Low"); c = _fv(row, "Close")
                v = int(_fv(row, "Volume"))
                total_pv += (h + l + c) / 3 * v
                total_vol += v
                highs.append(h); lows.append(l)
            except Exception:
                continue

        if total_vol == 0:
            return result

        vwap         = total_pv / total_vol
        fl_shares    = float_shares_m * 1_000_000 if float_shares_m > 0 else 0
        rotation_pct = (total_vol / fl_shares * 100) if fl_shares > 0 else 0.0
        last_close   = _fv(rows[-1][1], "Close")
        price_vs_vwap = (last_close - vwap) / vwap * 100 if vwap > 0 else 0.0

        mid = len(rows) // 2
        if mid > 1:
            avg_first = sum(_fv(r, "Close") for _, r in rows[:mid]) / mid
            avg_last  = sum(_fv(r, "Close") for _, r in rows[mid:]) / (len(rows) - mid)
            pm_trend = "up" if avg_last > avg_first * 1.02 else ("down" if avg_last < avg_first * 0.98 else "flat")
        else:
            pm_trend = "flat"

        if price_vs_vwap > 15:
            entry_zone = "WAIT — overextended above VWAP; watch for dip to reclaim"
        elif price_vs_vwap >= -5:
            entry_zone = "VWAP ZONE — prime entry; VWAP is your stop"
        else:
            entry_zone = "BELOW VWAP — needs reclaim before entry"

        result.update({
            "pm_vwap":           round(vwap, 4),
            "pm_vol":            total_vol,
            "float_rotation_pct": round(rotation_pct, 1),
            "pm_high":           round(max(highs), 4),
            "pm_low":            round(min(lows), 4),
            "price_vs_vwap_pct": round(price_vs_vwap, 1),
            "entry_zone":        entry_zone,
            "pm_trend":          pm_trend,
        })
    except Exception:
        pass
    return result


_TIER_A_KW = {"fda", "approv", "clearance", "merger", "acqui", "contract",
               "awarded", "license", "phase 3", "phase iii", "breakthrough",
               "nda", "bla", "510k", "510(k)"}
_TIER_B_KW = {"phase 2", "phase ii", "trial", "partnership", "milestone", "deal",
               "uplisting", "nasdaq listing", "nyse listing", "squeeze"}
_TIER_D_KW = {"dilut", "offering", "placement", "warrant", "clinical hold",
               "fda reject", "complete response", "investigation", "fraud", "delisting"}


def _score_catalyst_tier(bull_news: list[tuple[str, str]],
                          bear_news: list[tuple[str, str]],
                          edgar_found: bool) -> str:
    """
    A: 8-K filed + strong keyword (FDA/merger/contract) — highest conviction
    B: 8-K without strong keyword, OR strong keyword without 8-K
    C: Weak catalyst / PR-only
    D: Dilution / bearish risk — avoid the long
    """
    if bear_news:
        return "D"
    combined = " ".join(h.lower() for h, _ in bull_news)
    has_a = any(kw in combined for kw in _TIER_A_KW)
    has_b = any(kw in combined for kw in _TIER_B_KW)
    if edgar_found and has_a:
        return "A"
    if edgar_found or has_a:
        return "B"
    if has_b or bull_news:
        return "C"
    return "C"


def _detect_day_n_fade(ticker: str, gap_pct: float) -> str:
    """
    Detects two momentum exhaustion patterns:

    Pattern A — Immediate collapse (IOTR Day 3):
      Day 1: huge RVOL spike → Day 2: gap+hold trade → Day 3: gaps again but
      volume drops 80%+ → sellers distributing into latecomers.

    Pattern B — Deferred collapse (SILO Day 4):
      Day 1: huge RVOL spike → Day 2-3: price drifts UP on very low RVOL
      (trapped longs holding, no new buyers) → Day 4: air pocket collapse.
      SILO Jul 10 2026: 51x RVOL Tue → 0.2x Wed/Thu → -26% Friday.

    Returns a warning string, or "" if clean.
    """
    if abs(gap_pct) < 5.0:
        return ""
    try:
        df = yf.download(ticker, period="15d", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 4:
            return ""

        def _gv(row, col):
            v = row[col]
            return float(v.item() if hasattr(v, "item") else v)

        vols   = [_gv(df.iloc[i], "Volume") for i in range(len(df))]
        closes = [_gv(df.iloc[i], "Close")  for i in range(len(df))]

        # Baseline: 5-day avg excluding the most recent 3 days
        base_vols = vols[:-3] if len(vols) > 4 else vols[:-2]
        base_avg  = sum(base_vols) / len(base_vols) if base_vols else 1.0

        # Find the peak-volume day in the last 6 sessions
        recent_vols = vols[-6:]
        peak_vol    = max(recent_vols)
        peak_rvol   = peak_vol / base_avg if base_avg > 0 else 0.0

        if peak_rvol < 3.0:
            return ""   # no hot day in recent history — not a fade setup

        peak_idx_in_recent = len(vols) - 6 + recent_vols.index(peak_vol)

        # Days since the peak-volume day
        days_since_peak = len(vols) - 1 - peak_idx_in_recent

        # Volume trend since the peak (last 2 full trading days before today)
        post_peak_vols = vols[peak_idx_in_recent + 1:]  # days after the spike
        if not post_peak_vols:
            return ""

        avg_post_peak = sum(post_peak_vols) / len(post_peak_vols)
        vol_decay_pct = avg_post_peak / peak_vol * 100  # % of peak still trading

        # Pattern A: prior day alone was hot → immediate Day N trap
        prev_vol  = vols[-1]
        prev_rvol = prev_vol / base_avg if base_avg > 0 else 0.0
        is_pattern_a = (prev_rvol >= 3.0 and days_since_peak <= 1)

        # Pattern B: peak was 2-4 days ago, price drifted up on thin volume
        # (the "deferred collapse" — price floats on no sellers, then air-pockets)
        price_above_peak_close = (closes[-1] > closes[peak_idx_in_recent]
                                   if peak_idx_in_recent < len(closes) - 1 else False)
        is_pattern_b = (days_since_peak >= 2
                        and vol_decay_pct < 30.0       # volume collapsed >70%
                        and peak_rvol >= 5.0            # original spike was real
                        and price_above_peak_close)     # price still elevated = air pocket

        if not (is_pattern_a or is_pattern_b):
            return ""

        day_label = f"Day {days_since_peak + 1}"
        if is_pattern_b:
            return (
                f"\n   ⚠️  {day_label} DEFERRED FADE RISK — {days_since_peak}d after {peak_rvol:.0f}x RVOL spike,"
                f" volume decayed to {vol_decay_pct:.0f}% of peak."
                f"\n      Price elevated on no buyers = air-pocket risk. DO NOT chase gap."
            )
        else:
            pct_str = f" ({prev_vol / peak_vol * 100:.0f}% of peak)" if peak_vol > 0 else ""
            return (
                f"\n   ⚠️  {day_label} FADE RISK — prior day {prev_rvol:.0f}x RVOL{pct_str}."
                f" Volume likely exhausted."
                f"\n      Verify live RVOL at open — gap without volume = distribution trap."
            )
    except Exception:
        return ""


def _build_premarket_entry_plan(gap_pct: float, tier: str, vwap: float, pre_px: float,
                                prev_close: float, fl_m: float, is_moon: bool) -> str:
    """
    Builds the "🎯 ENTRY PLAN" block of the pre-market mover report.

    Found in the 2026-08-16 review: is_mover (this report's own gate)
    includes gap-DOWN movers by design (the report is meant to surface
    any big move, not just gap-ups), and a catalyst-less gap-down
    defaults to tier C ("PR / weak") — not tier D — so a gap-down mover
    used to fall straight into the bullish LONG plan below: VWAP-hold/
    breakout entries and +30%/+50% T1/T2 targets computed off the
    DEPRESSED price, reading a decline as a breakout setup. Informational
    only (no auto-order), but this is the system's main human-facing
    entry surface, so a wrong-direction plan here is a real risk to a
    human acting on it manually. gap_pct < 0 now short-circuits before
    tier is even consulted.
    """
    entry_parts = ["  🎯 <b>ENTRY PLAN</b>"]
    if gap_pct < 0:
        entry_parts.append(
            "     ⛔ Gap DOWN — no long entry plan shown (this system is "
            "long-only; a bullish continuation plan would misread a decline "
            "as a breakout setup)."
        )
    elif tier != "D":
        if vwap > 0 and tier in ("A", "B"):
            entry_parts.append(
                f"     Aggressive (pre-mkt): ${vwap:.4f} VWAP hold  "
                f"← Tier {tier} catalyst, float {fl_m:.2f}M"
            )
        entry_parts.append(
            "     Moderate (9:30): first 5-min candle high breakout → trail VWAP"
        )
        entry_parts.append(
            f"     Conservative (9:45): Gap & Hold gate — hold above open ${pre_px:.4f}"
        )
        stop_px = round(vwap * 0.97, 4) if vwap > 0 else round(prev_close * 1.05, 4)
        t1 = round(pre_px * 1.30, 4)
        t2 = round(pre_px * 1.50, 4)
        moon_str = f"  T3 (2x): ${round(pre_px * 2.0, 4):.4f}" if is_moon else ""
        entry_parts.append(
            f"     Stop: ${stop_px:.4f}  |  T1: ${t1:.4f} (+30%)  T2: ${t2:.4f} (+50%){moon_str}"
        )
    else:
        entry_parts.append("     ⛔ Skip — dilution/offering risk outweighs gap.")
    return "\n".join(entry_parts)


def run_premarket_early_scan() -> None:
    """
    7 AM ET early scanner — sweeps DMAN_SMALLCAP_WATCHLIST for pre-market movers,
    runs SEC 8-K validation, pre-market VWAP/float rotation, StockTwits sentiment,
    and outputs a structured entry plan for each play on Telegram.

    Catches IOTR-style plays at 7 AM — 2.5 hours before the 9:45 gate fires.
    """
    import re as _re
    now_et = datetime.now(ET)
    print(f"\n  ⚡ EARLY PRE-MARKET SCAN — {now_et.astimezone(MT).strftime('%I:%M %p MT')}")

    # Broad pre-market gap sweep — catches movers outside the curated watchlist
    # (confirmed gap: BIYA +39% by 7 AM on 2026-07-27, never scanned because it
    # wasn't yet a "known" DMan ticker). Everything below this point — catalyst
    # tiering, EDGAR check, PDT guard, auto-submit thresholds — is unchanged and
    # applies identically to curated and dynamically-discovered tickers alike.
    _dynamic_movers = fetch_premarket_gap_universe()
    scan_universe   = list(dict.fromkeys(DMAN_SMALLCAP_WATCHLIST + _dynamic_movers))
    print(f"  Scanning {len(scan_universe)} small-cap names "
          f"({len(DMAN_SMALLCAP_WATCHLIST)} curated + {len(_dynamic_movers)} dynamic)...\n")

    # Pull global context first — drives pre-market sizing and aggression
    print("  🌍 Global context...", flush=True)
    _early_ctx  = _fetch_global_context()
    _early_news = _fetch_breaking_news_rss(hours_back=8)

    news_map = _fetch_alpaca_news(scan_universe, hours_back=20)

    mover_blocks:    list[tuple[float, str]] = []   # (abs_gap, telegram_block)
    news_alerts:     list[str] = []
    pm_auto_entries: list[dict] = []               # moon-shot auto-submit candidates

    # Time budget for the per-ticker deep dive (EDGAR/StockTwits/catalyst tier
    # are all per-ticker network calls). The Telegram alert and pre-market
    # auto-submit both happen only AFTER this loop finishes, so a timeout kill
    # mid-loop would silently drop everything found so far — worth guarding
    # now that the dynamic sweep above can add up to 60 tickers on top of the
    # curated watchlist. Budget leaves the rest of the 35-min workflow timeout
    # for setup, the universe sweep, and message composition/sending.
    _loop_start  = time.monotonic()
    _loop_budget = 22 * 60
    for _tkr_idx, ticker in enumerate(scan_universe):
        if time.monotonic() - _loop_start > _loop_budget:
            print(f"  ⏱  Time budget reached after {_tkr_idx}/{len(scan_universe)} tickers "
                  f"— sending what's found so far.", file=sys.stderr)
            break
        try:
            info       = yf.Ticker(ticker).fast_info
            pre_px     = float(info.last_price or 0)
            prev_close = float(info.previous_close or 0)
            if pre_px <= 0 or prev_close <= 0:
                continue
            gap_pct = (pre_px - prev_close) / prev_close * 100

            fl_m, si_pct, _, _ = _get_short_float_data(ticker)
            headlines = news_map.get(ticker, [])
            bull_news, bear_news = [], []
            for h in headlines:
                sent, kw = _score_news_headline(h)
                if sent == "bullish":
                    bull_news.append((h, kw))
                elif sent == "bearish":
                    bear_news.append((h, kw))

            is_mover = abs(gap_pct) >= 5.0
            is_moon  = gap_pct >= 15.0 and 0 < fl_m < 2.0

            if is_mover:
                edgar_found, edgar_summary = _check_edgar_8k(ticker, hours_back=30)
                tier      = _score_catalyst_tier(bull_news, bear_news, edgar_found)
                bull_pct, st_msgs = _get_stocktwits_sentiment(ticker)
                pm        = _calc_premarket_levels(ticker, fl_m)
                fade_warn = _detect_day_n_fade(ticker, gap_pct)

                arrow     = "🟢" if gap_pct > 0 else "🔴"
                moon_tag  = "  🚀 MOON SHOT" if (is_moon and not fade_warn) else ""
                avoid_tag = "  ⛔ AVOID" if tier == "D" else ""

                tier_labels = {"A": "TIER A — 8-K CONFIRMED", "B": "TIER B — STRONG CATALYST",
                               "C": "TIER C — PR / WEAK",     "D": "TIER D — DILUTION RISK"}

                # Header
                header = (f"{arrow} <b>{ticker}</b>  ${pre_px:.4f}  ({gap_pct:+.1f}% vs prev ${prev_close:.4f})"
                          f"  Float: {fl_m:.2f}M  SI: {si_pct:.0f}%{moon_tag}{avoid_tag}"
                          f"{fade_warn}")

                # Catalyst block
                cat_parts = [f"  📋 <b>CATALYST [{tier_labels.get(tier, tier)}]</b>"]
                if edgar_found:
                    cat_parts.append(f"     {edgar_summary}")
                if bull_news:
                    cat_parts.append(f"     \"{bull_news[0][0][:110]}\"")
                if bear_news:
                    cat_parts.append(f"     ⚠️ RISK: \"{bear_news[0][0][:110]}\"")
                cat_line = "\n".join(cat_parts)

                # Pre-market levels block
                pm_line = ""
                if pm["pm_vwap"] > 0:
                    t_arrow = {"up": "↑", "down": "↓", "flat": "→"}.get(pm["pm_trend"], "→")
                    rot_str = f"  Float rotated: {pm['float_rotation_pct']:.1f}%" if fl_m > 0 else ""
                    pm_line = (
                        f"  📊 <b>PRE-MARKET LEVELS</b>\n"
                        f"     VWAP: ${pm['pm_vwap']:.4f}  "
                        f"PM High: ${pm['pm_high']:.4f}  PM Low: ${pm['pm_low']:.4f}\n"
                        f"     Price vs VWAP: {pm['price_vs_vwap_pct']:+.1f}%  "
                        f"Trend: {t_arrow}{rot_str}\n"
                        f"     ➡️ {pm['entry_zone']}"
                    )

                # StockTwits block
                st_line = ""
                if st_msgs > 0:
                    se = "🟢" if bull_pct >= 65 else ("🔴" if bull_pct <= 35 else "🟡")
                    st_line = (f"  💬 <b>StockTwits:</b> {st_msgs} msgs  |  "
                               f"Sentiment: {bull_pct:.0f}% bullish {se}")

                # Entry plan block
                vwap = pm["pm_vwap"]
                entry_block = _build_premarket_entry_plan(
                    gap_pct, tier, vwap, pre_px, prev_close, fl_m, is_moon)

                parts = [header, cat_line]
                if pm_line:
                    parts.append(pm_line)
                if st_line:
                    parts.append(st_line)
                parts.append(entry_block)
                mover_blocks.append((abs(gap_pct), "\n".join(parts)))

                # Queue for pre-market auto-submit if criteria met
                if (ENABLE_PREMARKET_SUBMIT
                        and tier in ("A", "B")
                        and gap_pct >= 15.0
                        and 0 < fl_m < 2.0
                        and pm.get("pm_vwap", 0) > 0
                        and not fade_warn):
                    pm_auto_entries.append({
                        "ticker": ticker, "entry_px": pm["pm_vwap"],
                        "gap_pct": gap_pct, "fl_m": fl_m,
                        "tier": tier, "prev_close": prev_close,
                        "size_mult": 1.0,
                    })
                # Secondary tier — Tier A SEC catalyst, smaller gap, wider float.
                # Enters at PM VWAP to be part of the gap move from the hold level,
                # not chasing after it gaps fully at 9:45 AM.
                elif (ENABLE_PREMARKET_SUBMIT
                        and tier == "A"
                        and 5.0 <= gap_pct < 15.0
                        and 0 < fl_m < 5.0
                        and pm.get("pm_vwap", 0) > 0
                        and not fade_warn):
                    pm_auto_entries.append({
                        "ticker": ticker, "entry_px": pm["pm_vwap"],
                        "gap_pct": gap_pct, "fl_m": fl_m,
                        "tier": tier, "prev_close": prev_close,
                        "size_mult": 0.5,   # half size — less confirmed gap
                    })

            elif bull_news:
                # News only — gap hasn't triggered yet, but put on watch
                edgar_found, edgar_summary = _check_edgar_8k(ticker, hours_back=30)
                tier = _score_catalyst_tier(bull_news, [], edgar_found)
                kw_str = ", ".join(set(kw for _, kw in bull_news[:2]))
                hl_str = bull_news[0][0][:120]
                edgar_note = f"\n   SEC: {edgar_summary}" if edgar_found else ""
                news_alerts.append(
                    f"📰 <b>{ticker}</b>  [Catalyst Tier {tier}]  "
                    f"Float: {fl_m:.2f}M  SI: {si_pct:.0f}%\n"
                    f"   [{kw_str}] \"{hl_str}\"{edgar_note}\n"
                    f"   Pre-mkt: ${pre_px:.4f}  ({gap_pct:+.1f}% — gap not triggered yet)"
                )

        except Exception as _e:
            print(f"  ⚠️  [{ticker}] early scan error: {_e}")
            continue

    mover_blocks.sort(key=lambda x: x[0], reverse=True)

    lines = [f"⚡ <b>DMan 7 AM Pre-Market Scan</b>  {now_et.astimezone(MT).strftime('%a %b %d, %I:%M %p MT')}"]

    if mover_blocks:
        lines.append(f"\n🚨 <b>PRE-MARKET MOVERS</b> ({len(mover_blocks)} play(s) ≥5%)\n")
        for _, block in mover_blocks[:5]:
            lines.append(block)
            lines.append("")
    else:
        lines.append("\n📡 No small-cap movers ≥5% pre-market right now.")

    if news_alerts:
        lines.append(f"👀 <b>CATALYST WATCH</b> ({len(news_alerts)} — no gap yet)\n")
        for alert in news_alerts[:4]:
            lines.append(alert)
            lines.append("")

    if not mover_blocks and not news_alerts:
        lines.append("\n✅ Clean — no movers or catalysts. Normal open expected.")

    # Global context block
    lines.append(f"\n{_early_ctx['summary']}")
    if _early_news:
        _news_lines = []
        for _hl, _src, _ts, _imp in _early_news[:4]:
            _em = ("🔴 " if _imp <= -1 else ("🟢 " if _imp >= 1 else ""))
            _news_lines.append(f"   {_em}{_hl[:100]}  <i>[{_src} {_ts}]</i>")
        lines.append("📰 <b>BREAKING</b>\n" + "\n".join(_news_lines))

    # Dman Radar — merge X/Twitter (primary) + StockTwits (fallback)
    _dman_tw = _fetch_dman_twitter_calls(hours_back=48)
    _dman_st = _fetch_dman_stocktwits_calls(hours_back=48)
    # ticker → (body, ts_str, source_label); Twitter overwrites StockTwits on duplicates
    _radar: dict[str, tuple[str, str, str]] = {}
    for _sym, _body, _ts in _dman_st:
        if _sym not in _radar:
            _radar[_sym] = (_body, _ts, "ST")
    for _sym, _body, _ts in _dman_tw:
        _radar[_sym] = (_body, _ts, "𝕏")
    if _radar:
        _mover_set = {blk.split("<b>")[1].split("</b>")[0] for _, blk in mover_blocks if "<b>" in blk}
        _radar_lines = []
        for _sym, (_body, _ts, _src) in list(_radar.items())[:8]:
            _gap_tag = "  🔥 <b>GAPPING NOW</b>" if _sym in _mover_set else ""
            _radar_lines.append(f"  • <b>${_sym}</b>  [{_src}]{_gap_tag}  [{_ts}]  \"{_body[:100]}\"")
        _offline_note = "  <i>(𝕏 offline — StockTwits only)</i>" if not _dman_tw else ""
        lines.append(f"\n👁 <b>DMAN RADAR</b> (@professorDman1 last 48h){_offline_note}\n" + "\n".join(_radar_lines))

    lines.append("<i>Tier A=8-K filed | B=strong catalyst | C=PR | D=avoid</i>")
    lines.append("<i>Entry: Aggressive=pre-mkt VWAP | Moderate=9:30 first candle | Conservative=9:45 hold</i>")

    msg = "\n".join(lines)
    print(_re.sub(r"<[^>]+>", "", msg))
    send_telegram(msg)

    # Pre-market auto-submit: place extended-hours limit orders for Tier A/B moon shots
    if ENABLE_PREMARKET_SUBMIT and pm_auto_entries:
        _client = get_alpaca_client()
        if _client is None:
            print("  ⚠️  Pre-market submit: Alpaca unavailable — skipping orders")
        else:
            # PDT check before any pre-market orders — same guard as _submit_signals_to_alpaca
            _pm_remaining_cash: Optional[float] = None
            try:
                _pm_acct  = _client.get_account()
                _pm_eq    = float(getattr(_pm_acct, "equity", 0) or 0)
                _pm_dt    = int(getattr(_pm_acct, "daytrade_count", 0) or 0)
                _pm_remaining_cash = float(getattr(_pm_acct, "cash", 0) or 0)
                if _pm_eq < 25_000 and _pm_dt >= 3:
                    send_telegram(f"🚫 <b>PDT HALT (pre-market)</b>: {_pm_dt}/3 day trades used — no pre-market orders placed.")
                    pm_auto_entries.clear()
            except Exception as _pdt_pm:
                send_telegram(f"🚫 <b>PDT check failed (pre-market)</b>: {_pdt_pm} — skipping pre-market orders.")
                pm_auto_entries.clear()

            from alpaca.trading.requests import LimitOrderRequest as _LimReq
            from alpaca.trading.enums   import OrderSide as _Side, TimeInForce as _TIF
            _acct    = get_effective_account()
            # SMALLCAP_RISK_PCT(0.02) * MOONSHOT_RISK_MULT(5.0) * risk_mult (up to
            # 1.30 in RISK-ON regimes) can reach 13% on a SINGLE trade, with up to
            # 3 concurrent pre-market entries below and no portfolio-heat check
            # anywhere in this path — unlike the equivalent moonshot path in
            # find_smallcap_signal(), which hit the exact same conflict against
            # PORTFOLIO_HEAT_LIMIT (6%) and is fixed there for the same reason.
            # Capping per-trade risk here is a pure risk reduction, consistent
            # with that fix — it does not touch entry criteria or targets.
            _risk_pct  = min(SMALLCAP_RISK_PCT * MOONSHOT_RISK_MULT * _early_ctx["risk_mult"],
                             PORTFOLIO_HEAT_LIMIT)
            _base_risk = _acct * _risk_pct
            _pm_pt   = PositionTracker()
            for _e in pm_auto_entries[:3]:   # max 3 concurrent pre-market entries
                try:
                    _ep        = round(_e["entry_px"], 2)
                    _fl_m      = _e["fl_m"]
                    _gap_pct   = _e["gap_pct"]
                    _sm        = _e.get("size_mult", 1.0)
                    _ultra_low = _fl_m < ULTRA_LOW_FLOAT_M
                    _stop_px   = round(_ep * (1 - ULTRA_LOW_STOP_PCT), 2)
                    _rps       = _ep - _stop_px
                    _risk      = _base_risk * _sm
                    _shares    = max(1, int(_risk / _rps)) if _rps > 0 else 1
                    _cost      = _shares * _ep
                    if _cost > SMALLCAP_MAX_COST:
                        _shares = max(1, int(SMALLCAP_MAX_COST / _ep))
                        _cost   = _shares * _ep
                    # Tracked and decremented across all 3 possible entries in
                    # this loop, not just checked once — up to 3 concurrent
                    # pre-market orders submitted in immediate succession
                    # would otherwise all pass an independent per-order check
                    # against the same starting cash figure and collectively
                    # overspend it, same root cause as the 2026-08-04 margin
                    # incident (see get_available_cash docstring).
                    if _pm_remaining_cash is not None:
                        if _cost > _pm_remaining_cash:
                            print(f"  ⚠️  {_e['ticker']} pre-market: would need "
                                  f"${_cost:.0f}, only ${_pm_remaining_cash:.0f} "
                                  f"cash remaining — skipping (no margin)")
                            continue
                        _pm_remaining_cash -= _cost
                    _order = _client.submit_order(_LimReq(
                        symbol        = _e["ticker"],
                        qty           = _shares,
                        side          = _Side.BUY,
                        limit_price   = _ep,
                        time_in_force = _TIF.DAY,
                        extended_hours= True,
                    ))
                    # Targets: ultra-low float uses +50%/+150%; wider float uses gap-echo
                    # (T1 = full measured gap from entry, T2 = 1.5× echo) so targets match
                    # the actual gap move, not a fixed percentage.
                    if _ultra_low:
                        _t1 = round(_ep * (1 + ULTRA_LOW_T1_MULT), 2)
                        _t2 = round(_ep * (1 + ULTRA_LOW_T2_MULT), 2)
                        _t1_lbl, _t2_lbl = f"+{int(ULTRA_LOW_T1_MULT*100)}%", f"+{int(ULTRA_LOW_T2_MULT*100)}%"
                    else:
                        _t1 = round(_ep * (1 + _gap_pct / 100), 2)        # echo the gap
                        _t2 = round(_ep * (1 + _gap_pct / 100 * 1.5), 2)  # 1.5× echo
                        _t1_lbl = f"+{_gap_pct:.0f}% (echo gap)"
                        _t2_lbl = f"+{_gap_pct*1.5:.0f}% (1.5× echo)"
                    _t3 = round(_ep * (1 + MOONSHOT_T3_MULT), 2)
                    _setup_label = "Pre-Market Moon Shot" if _ultra_low else "Pre-Market Gap Entry"
                    # Write to PositionTracker so 9:45 AM scan skips duplicate entry
                    _pm_pt.open(OpenPosition(
                        ticker=_e["ticker"], bias="LONG", setup=_setup_label,
                        entry=_ep, stop=_stop_px, target1=_t1, target2=_t2,
                        shares=_shares, entry_date=datetime.today().strftime("%Y-%m-%d"),
                    ))
                    _size_note = "" if _sm == 1.0 else f"  <i>(half-size — {_gap_pct:.0f}% gap, Tier {_e['tier']})</i>"
                    _pm_msg = (
                        f"📤 <b>PRE-MARKET ORDER PLACED</b>{_size_note}\n"
                        f"<b>{_e['ticker']}</b>  Tier {_e['tier']}  "
                        f"Float {_fl_m:.2f}M  Gap {_gap_pct:+.0f}%\n"
                        f"Entry: ${_ep}  ({_shares} shares  ${_cost:.0f})\n"
                        f"Stop: ${_stop_px}  T1: ${_t1} ({_t1_lbl})  T2: ${_t2} ({_t2_lbl})\n"
                        f"Plan: sell 50% at T1 → move stop to ${_ep} → let rest run to T2\n"
                        f"<i>Extended-hours limit — no bracket. Momentum-watch manages exit after open.</i>"
                    )
                    send_telegram(_pm_msg)
                    print(f"  📤 Pre-market order: {_e['ticker']} {_shares}sh @ ${_ep}  [{_setup_label}]")
                except Exception as _ex:
                    print(f"  ❌ Pre-market order failed ({_e['ticker']}): {_ex}")


# ──────────────────────────────────────────────────────────────────────────────
# Intraday momentum watch — entry timing + exit management
# ──────────────────────────────────────────────────────────────────────────────

def _interval_to_resample_rule(interval: str) -> Optional[str]:
    """Maps a yfinance-style interval string ("5m", "1h", ...) to a pandas
    resample rule ("5min", "1h"). Returns None on an unrecognized format —
    caller should skip resampling rather than guess."""
    try:
        interval = interval.strip().lower()
        if interval.endswith("m"):
            return f"{int(interval[:-1])}min"
        if interval.endswith("h"):
            return f"{int(interval[:-1])}h"
        if interval.endswith("d"):
            return f"{int(interval[:-1])}D"
    except Exception:
        pass
    return None


def _fetch_intraday_bars(ticker: str, interval: str = "5m", period: str = "1d"):
    """Fetch intraday bars — Alpaca 1-min SIP primary (Algo Trader Plus), yfinance 5-min fallback."""
    # ── Alpaca 1-min bars (SIP real-time feed) ────────────────────────────────
    try:
        if ALPACA_AVAILABLE:
            dc = get_alpaca_data_client()
            if dc is not None:
                import zoneinfo as _zi
                from datetime import timezone as _tz
                _ET = _zi.ZoneInfo("America/New_York")
                _now_et = datetime.now(_ET)
                _sess_start = _now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                _alp_req = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Minute,
                    start=_sess_start.astimezone(_tz.utc),
                    feed=_resolve_stock_feed(),
                )
                _alp_resp = dc.get_stock_bars(_alp_req)
                # see _fetch_alpaca_daily — BarSet's `in` always returns False,
                # so this always silently returned [] before, meaning momentum
                # watch always fell back to the slower/coarser yfinance path.
                _alp_bars = _alp_resp.data.get(ticker, [])
                if len(_alp_bars) >= 3:
                    _records = [
                        {"Open": b.open, "High": b.high, "Low": b.low,
                         "Close": b.close, "Volume": b.volume, "Timestamp": b.timestamp}
                        for b in _alp_bars
                    ]
                    _df = pd.DataFrame(_records).set_index("Timestamp")
                    _df.index = pd.to_datetime(_df.index, utc=True)
                    # Resample Alpaca's native 1-min bars to the REQUESTED
                    # interval. Found 2026-08-16 review: this always
                    # returned raw 1-min bars regardless of `interval`,
                    # while every real caller passes "5m" and every
                    # downstream consumer (_compute_session_levels,
                    # _detect_pre_breakout, _detect_momentum_fade — the
                    # momentum-watch exit manager for live small-cap
                    # positions) is written and documented for 5-minute
                    # granularity. "No new session high in 5 bars" silently
                    # meant 5 minutes instead of 25; a breakout "tight
                    # coil" read covered 3 minutes instead of 15.
                    _rule = _interval_to_resample_rule(interval)
                    if _rule and _rule != "1min":
                        # left-closed/left-labeled: a bar's timestamp is
                        # its START time (Alpaca's and yfinance's own
                        # convention), so the 9:30 bar must anchor the
                        # [9:30, 9:35) bucket labeled 9:30, not fall into
                        # the PRIOR bucket the way right-closed would.
                        _df = _df.resample(_rule, label="left", closed="left").agg({
                            "Open": "first", "High": "max", "Low": "min",
                            "Close": "last", "Volume": "sum",
                        }).dropna(subset=["Open"])
                    return _df
    except Exception:
        pass
    # ── yfinance 5-min fallback ───────────────────────────────────────────────
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         prepost=False, progress=False, auto_adjust=True)
        if df is None or len(df) < 3:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df
    except Exception:
        return None


def _compute_session_levels(df_5m) -> dict:
    """
    From a session's 5-min bar DataFrame, extract:
    - session VWAP (volume-weighted average price)
    - first candle high (9:30-9:35 bar — key breakout reference)
    - session high (all-time-of-day high so far)
    - current price (last bar close)
    - structured bars list for further analysis

    Returns a dict with these keys (zeros if data is insufficient).
    """
    result = {
        "vwap": 0.0, "first_candle_high": 0.0, "session_high": 0.0,
        "session_low": 0.0, "cur_price": 0.0, "bars": [],
    }
    if df_5m is None or len(df_5m) < 2:
        return result

    def _fv(row, col):
        v = row[col]
        return float(v.item() if hasattr(v, "item") else v)

    total_pv = 0.0; total_vol = 0
    bars = []
    for i in range(len(df_5m)):
        row = df_5m.iloc[i]
        try:
            h = _fv(row, "High"); l = _fv(row, "Low")
            c = _fv(row, "Close"); v = int(_fv(row, "Volume"))
            total_pv += (h + l + c) / 3 * v
            total_vol += v
            bars.append({"h": h, "l": l, "o": _fv(row, "Open"), "c": c, "v": v})
        except Exception:
            continue

    if not bars or total_vol == 0:
        return result

    result["vwap"]             = round(total_pv / total_vol, 4)
    result["first_candle_high"] = bars[0]["h"]
    result["session_high"]     = max(b["h"] for b in bars)
    result["session_low"]      = min(b["l"] for b in bars)
    result["cur_price"]        = bars[-1]["c"]
    result["bars"]             = bars
    return result


def _detect_pre_breakout(levels: dict) -> dict:
    """
    Scan recent 5-min bars for pre-breakout consolidation patterns.
    Returns: {setup: bool, signals: list[str], entry_px: float, stop_px: float}

    Fires when 2+ of these are true:
      1. VWAP reclaim bounce  — price near VWAP and current candle is green
      2. Tight coil           — last 3 highs within 2.5%, lows stepping up
      3. Volume dry-up        — recent bars trading at <45% of prior volume
      4. Near breakout level  — within 2% of first candle high (the trigger)
      5. Higher lows          — last 3 bar lows are each higher than the one before
    """
    result = {"setup": False, "signals": [], "entry_px": 0.0, "stop_px": 0.0}
    bars = levels.get("bars", [])
    if len(bars) < 4:
        return result

    vwap      = levels["vwap"]
    fch       = levels["first_candle_high"]
    cur_price = levels["cur_price"]
    recent    = bars[-4:]
    last      = bars[-1]

    signals: list[str] = []

    # 1. VWAP reclaim bounce
    if vwap > 0:
        dist = (cur_price - vwap) / vwap * 100
        if 0.0 <= dist <= 6.0 and last["c"] > last["o"]:
            signals.append(f"VWAP reclaim bounce (${vwap:.4f})")

    # 2. Tight coil: last 3 highs within 2.5%, lows not making lower lows
    last3 = bars[-3:]
    if len(last3) == 3:
        high_spread = (max(b["h"] for b in last3) - min(b["h"] for b in last3)) / max(b["h"] for b in last3) * 100
        if high_spread < 2.5 and last3[-1]["l"] >= last3[0]["l"]:
            signals.append(f"tight coil ({high_spread:.1f}% high range)")

    # 3. Volume dry-up: last 2 bars avg < 45% of prior 3 bars avg
    if len(bars) >= 5:
        avg_recent = (bars[-1]["v"] + bars[-2]["v"]) / 2
        avg_prior  = sum(b["v"] for b in bars[-5:-2]) / 3
        if avg_prior > 0 and avg_recent / avg_prior < 0.45:
            signals.append(f"volume dry-up ({avg_recent/avg_prior*100:.0f}% of prior avg)")

    # 4. Near first candle high breakout level
    if fch > 0:
        dist_fch = (fch - cur_price) / cur_price * 100
        if 0.0 < dist_fch < 2.5:
            signals.append(f"within 2.5% of first candle high ${fch:.4f}")

    # 5. Higher lows: 3 consecutive bars with each low > previous
    if (len(bars) >= 3
            and bars[-1]["l"] > bars[-2]["l"]
            and bars[-2]["l"] > bars[-3]["l"]):
        signals.append("higher lows (demand stepping up)")

    if len(signals) >= 2:
        consol_low = min(b["l"] for b in bars[-3:])
        # Entry: breakout above the tightest recent high
        entry_px = round(max(b["h"] for b in bars[-3:]), 4)
        stop_px  = round(consol_low * 0.985, 4)   # 1.5% below consolidation low
        result.update({
            "setup":    True,
            "signals":  signals,
            "entry_px": entry_px,
            "stop_px":  stop_px,
        })
    return result


def _detect_momentum_fade(levels: dict, entry_px: float, float_shares_m: float) -> dict:
    """
    Scan recent 5-min bars for momentum exhaustion.
    Returns: {action: 'hold'|'trail'|'exit', reason: str, trail_stop: float}

    Exit signals (any one → EXIT):
      - Two consecutive red candles + declining volume after a session high
      - Bearish engulfing candle at or near session high

    Trail signals (any one → TRAIL STOP at 2-bar low):
      - Volume drops to <20% of session peak bar volume
      - Float rotated >2.5x intraday (distribution risk)
      - No new session high in last 5 bars (momentum stalling)
    """
    result = {"action": "hold", "reason": "", "trail_stop": 0.0}
    bars = levels.get("bars", [])
    if len(bars) < 4:
        return result

    session_high = levels["session_high"]
    cur_price    = levels["cur_price"]
    last  = bars[-1]; prev = bars[-2]

    exit_signals:  list[str] = []
    trail_signals: list[str] = []

    # EXIT 1: two consecutive red candles + declining volume, below recent high
    if len(bars) >= 3:
        red_now  = last["c"] < last["o"]
        red_prev = prev["c"] < prev["o"]
        vol_fall = last["v"] < prev["v"]
        # Only meaningful if price was elevated (above entry by >10%)
        if entry_px > 0 and cur_price > entry_px * 1.10:
            if red_now and red_prev and vol_fall and cur_price < session_high * 0.96:
                exit_signals.append("2 red candles + declining vol after high → sellers in control")

    # EXIT 2: bearish engulfing after a high (prev green, current engulfs it)
    if (prev["c"] > prev["o"]                  # prior bar was green
            and last["o"] >= prev["c"] * 0.99  # gapped up or flat into this bar
            and last["c"] < prev["o"]           # closed below prior bar open
            and cur_price < session_high * 0.97):
        exit_signals.append("bearish engulfing candle after session high")

    # TRAIL 1: volume exhaustion vs session peak bar
    all_vols = [b["v"] for b in bars]
    peak_vol = max(all_vols) if all_vols else 1
    if peak_vol > 0 and last["v"] / peak_vol < 0.20 and entry_px > 0 and cur_price > entry_px:
        trail_signals.append(f"volume at {last['v']/peak_vol*100:.0f}% of session peak (buyers thinning)")

    # TRAIL 2: float rotation >2.5x
    if float_shares_m > 0:
        session_vol = sum(b["v"] for b in bars)
        rotation    = session_vol / (float_shares_m * 1_000_000)
        if rotation > 2.5:
            trail_signals.append(f"float rotated {rotation:.1f}x — distribution risk")

    # TRAIL 3: no new session high in last 5 bars, momentum stalling
    if len(bars) >= 5:
        recent_high = max(b["h"] for b in bars[-5:])
        if recent_high < session_high and cur_price < session_high * 0.95:
            trail_signals.append("no new high in last 5 bars — momentum stalling")

    # Trailing stop = lowest low of last 2 bars - 1%
    trail_stop = round(min(last["l"], prev["l"]) * 0.99, 4)

    if exit_signals:
        result.update({"action": "exit",  "reason": " | ".join(exit_signals),  "trail_stop": trail_stop})
    elif trail_signals:
        result.update({"action": "trail", "reason": " | ".join(trail_signals), "trail_stop": trail_stop})

    return result


def _update_positions_matching(match, label: str, **fields) -> None:
    """
    Shared load/mutate/write for _update_position_field()/
    _update_option_position_field() — the two differed only in how they
    matched a position (ticker vs. OCC symbol embedded in `setup`) and
    their error-message wording; `match` is the one thing callers still
    need to specify themselves.
    """
    try:
        with open(POSITIONS_FILE) as f:
            data = json.load(f)
        changed = False
        for p in data:
            if match(p):
                p.update(fields)
                changed = True
        if changed:
            _write_json_atomic(POSITIONS_FILE, data, indent=2)
    except Exception as _e:
        print(f"  ⚠️  Could not update position {label}: {_e}")


def _update_position_field(ticker: str, **fields) -> None:
    """Update fields on a tracked position in dman_positions.json (e.g. raise stop)."""
    _update_positions_matching(lambda p: p.get("ticker") == ticker, ticker, **fields)


def _update_option_position_field(occ_symbol: str, **fields) -> None:
    """
    Like _update_position_field, but matches a SPECIFIC options leg by its
    OCC symbol embedded in `setup` (e.g. "Options Call SMCI260814C00034000
    (...)") rather than by ticker. Multiple options legs on the same
    underlying (e.g. a call AND a put, as in a manually-built earnings
    strangle) share the same `ticker` field — _update_position_field's
    ticker-only match would silently update BOTH when only one leg's
    peak_premium/milestone state should actually change. Added 2026-08-10
    alongside the options trailing-exit and P&L milestone features.
    """
    _update_positions_matching(
        lambda p: bool(occ_symbol) and occ_symbol in p.get("setup", "").split(),
        occ_symbol, **fields)


def _submit_options_close(occ_symbol: str, qty: int, reason: str) -> tuple[str, Optional[str]]:
    """
    Submit a closing SELL for an options position — the enforcement arm of the
    momentum-watch monitor (stops/targets execute instead of just alerting).

    Returns (status, order_id) where status is one of:
      "submitted"      — closing order placed
      "pending"        — a SELL is already working for this contract (no double-submit)
      "already_closed" — Alpaca no longer holds the position (sync will record P&L)
      "failed"         — submission failed; manual action needed

    Uses a marketable limit at bid−2% (fills immediately against the bid but
    caps damage on a crossed/glitched quote); falls back to a market order if
    no usable quote is available.
    """
    client = get_alpaca_client()
    if client is None:
        return "failed", None

    # Don't double-submit: if a SELL is already open for this contract, report it
    try:
        _open_orders = client.get_orders(filter=GetOrdersRequest(
            symbols=[occ_symbol], status=QueryOrderStatus.OPEN, limit=10))
        for _o in _open_orders:
            if _o.side == OrderSide.SELL:
                return "pending", str(_o.id)
    except Exception:
        pass

    # Clamp qty to what Alpaca actually holds (may differ after manual sells)
    try:
        _apos = client.get_open_position(occ_symbol)
        _held = abs(int(float(_apos.qty)))
        if _held <= 0:
            return "already_closed", None
        qty = min(qty, _held)
    except Exception as _pe:
        _pe_s = str(_pe).lower()
        if "not found" in _pe_s or "does not exist" in _pe_s or "404" in _pe_s:
            return "already_closed", None
        # Network/API error — still attempt the exit with tracker qty (fail-open
        # on protective exits; an oversized qty is rejected harmlessly by Alpaca)

    if qty <= 0:
        return "already_closed", None

    snap = _get_option_snapshot(occ_symbol)
    try:
        if snap and snap.get("bid", 0) > 0.02:
            order = client.submit_order(LimitOrderRequest(
                symbol        = occ_symbol,
                qty           = qty,
                side          = OrderSide.SELL,
                limit_price   = max(0.01, round(snap["bid"] * 0.98, 2)),
                time_in_force = TimeInForce.DAY,
            ))
        else:
            order = client.submit_order(MarketOrderRequest(
                symbol        = occ_symbol,
                qty           = qty,
                side          = OrderSide.SELL,
                time_in_force = TimeInForce.DAY,
            ))
        print(f"  📤 AUTO-CLOSE {occ_symbol} ×{qty} ({reason})  id={str(order.id)[:8]}…")
        return "submitted", str(order.id)
    except Exception as exc:
        print(f"  ❌ Auto-close failed ({occ_symbol}): {exc}")
        send_telegram(
            f"🚨 <b>AUTO-CLOSE FAILED</b> — {occ_symbol} ×{qty} ({reason})\n"
            f"{exc}\nClose manually in Alpaca NOW."
        )
        return "failed", None


OPTIONS_MILESTONE_START_PCT = 15.0   # first gain/loss %% that triggers a milestone alert —
                                      # tightened from 30.0 (2026-08-15, direct instruction
                                      # to be pinged on any meaningful UMAC move without
                                      # having to ask). Bucket-gated, not time-gated (see
                                      # _check_options_pnl_milestone), so this is safe to
                                      # tighten regardless of guard cadence — it can only
                                      # ever fire once per new price bucket, never repeat.
OPTIONS_MILESTONE_STEP_PCT  = 10.0   # every additional %% beyond the start that re-alerts

def _check_options_pnl_milestone(pos: dict, kind: str, occ: str, cur_prem: float,
                                  entry_prem: float, underlying_px: Optional[float]) -> None:
    """
    Tiered P&L milestone notifications for one options leg — added
    2026-08-10 for ongoing visibility into how a position's P&L is moving,
    independent of the stop/trail/T1 alerts (which only fire on a
    threshold CROSS, not on every step along the way). Fires at
    OPTIONS_MILESTONE_START_PCT (30%) and every further
    OPTIONS_MILESTONE_STEP_PCT (10%) beyond it, in EITHER direction.

    Tracked persistently on the position record itself
    (milestone_gain_alerted / milestone_loss_alerted), not a same-day
    dedup — an earnings-hold position can span multiple days, and a
    milestone already announced shouldn't re-fire just because a new
    calendar day started while price is still sitting at the same level.
    Only re-alerts on a NEW, FURTHER bucket (e.g. won't fire again at 32%
    once 30% was already announced, but will at 41% once past 40%).
    """
    if entry_prem <= 0 or not occ:
        return
    pnl_pct = (cur_prem - entry_prem) / entry_prem * 100
    _px_note = f"  Underlying: ${underlying_px:.2f}" if underlying_px else ""
    if pnl_pct >= OPTIONS_MILESTONE_START_PCT:
        bucket  = (int(pnl_pct) // int(OPTIONS_MILESTONE_STEP_PCT)) * OPTIONS_MILESTONE_STEP_PCT
        already = float(pos.get("milestone_gain_alerted", 0) or 0)
        if bucket <= already:
            return
        send_telegram(
            f"📈 <b>{pos.get('ticker','')} {kind} milestone</b> — +{bucket:.0f}% reached\n"
            f"Premium ${entry_prem:.2f} → ${cur_prem:.2f} ({pnl_pct:+.1f}%){_px_note}\n"
            f"{occ}"
        )
        _update_option_position_field(occ, milestone_gain_alerted=float(bucket))
    elif pnl_pct <= -OPTIONS_MILESTONE_START_PCT:
        bucket  = (int(abs(pnl_pct)) // int(OPTIONS_MILESTONE_STEP_PCT)) * OPTIONS_MILESTONE_STEP_PCT
        already = float(pos.get("milestone_loss_alerted", 0) or 0)
        if bucket <= already:
            return
        send_telegram(
            f"📉 <b>{pos.get('ticker','')} {kind} milestone</b> — -{bucket:.0f}% reached\n"
            f"Premium ${entry_prem:.2f} → ${cur_prem:.2f} ({pnl_pct:+.1f}%){_px_note}\n"
            f"{occ}"
        )
        _update_option_position_field(occ, milestone_loss_alerted=float(bucket))


def _options_trail_giveback_pct(peak_gain_pct: float) -> float:
    """
    Graduated giveback tolerance for the options trailing exit — see the
    OPTIONS_TRAIL_GIVEBACK_MIN/MAX_PCT docstring for the SMCI incident this
    replaced a flat percentage over. Linearly interpolates from MIN_PCT at
    exactly the activation threshold up to MAX_PCT once peak_gain_pct
    reaches OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT, capped at MAX_PCT beyond
    that. peak_gain_pct below the activation threshold isn't a real input
    (the trail isn't active yet) but still returns MIN_PCT rather than
    erroring, since callers may compute this before checking _trail_active.
    """
    if peak_gain_pct <= OPTIONS_TRAIL_ACTIVATE_GAIN_PCT:
        return OPTIONS_TRAIL_GIVEBACK_MIN_PCT
    _span = OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT - OPTIONS_TRAIL_ACTIVATE_GAIN_PCT
    _progress = (peak_gain_pct - OPTIONS_TRAIL_ACTIVATE_GAIN_PCT) / _span if _span > 0 else 1.0
    _progress = max(0.0, min(1.0, _progress))
    return OPTIONS_TRAIL_GIVEBACK_MIN_PCT + _progress * (OPTIONS_TRAIL_GIVEBACK_MAX_PCT - OPTIONS_TRAIL_GIVEBACK_MIN_PCT)


_option_greeks_cache: dict[str, tuple[dict, float]] = {}
_OPTION_GREEKS_CACHE_TTL_S = 300   # Greeks move slowly relative to bid/ask; a few
                                    # minutes stale is still meaningfully accurate.

def _cached_option_greeks(occ_symbol: str) -> dict:
    """
    Greeks (delta/gamma/theta/vega/iv) for occ_symbol, REST-fetched via
    _get_option_snapshot() and cached for _OPTION_GREEKS_CACHE_TTL_S.

    Found in the 2026-08-16 review: _monitor_option_position()'s theta-
    decay alert silently never fired whenever fed by the real-time
    options stream (get_snapshot_fn), since that feed only carries
    bid/ask/mid/sizes by design (OPRA/indicative quote messages don't
    include Greeks) — _snap.get("theta", 0) always read 0, and 0 can
    never clear the alert's >5.0 threshold. Unlike bid/ask, Greeks
    change slowly enough that a several-minute-stale REST-fetched value
    is still meaningfully accurate for a decay-rate alert, so this
    restores the check via a periodic REST fetch instead of reintroducing
    one on every 10s guard cycle (the exact call-volume problem the
    stream was built to avoid — see this function's own docstring above).
    """
    now = time.monotonic()
    cached = _option_greeks_cache.get(occ_symbol)
    if cached and now - cached[1] < _OPTION_GREEKS_CACHE_TTL_S:
        return cached[0]
    snap = _get_option_snapshot(occ_symbol)
    if snap:
        greeks = {"delta": snap.get("delta", 0), "gamma": snap.get("gamma", 0),
                  "theta": snap.get("theta", 0), "vega": snap.get("vega", 0),
                  "iv": snap.get("iv", 0)}
    elif cached:
        greeks = cached[0]   # REST fetch failed this cycle — keep the last known value
    else:
        greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0}
    _option_greeks_cache[occ_symbol] = (greeks, now)
    return greeks


def _monitor_option_position(pos: dict, kind: str, get_snapshot_fn=None, get_price_fn=None) -> Optional[str]:
    """
    Enforce stop / trailing-exit / T1 / DTE rules on one tracked options
    position, plus P&L milestone notifications (_check_options_pnl_milestone).
    kind: "CALL" or "PUT". Submits closing orders via _submit_options_close
    and returns a status line for the alert digest (None if record unusable).
    Shared by run_momentum_watch (hourly cron) and the always-on daemon.

    `get_snapshot_fn(occ_symbol) -> dict | None`, if given, is tried first
    (e.g. the daemon's real-time options WebSocket quote cache) before
    falling back to the REST snapshot below — same injection pattern as
    run_equity_guard's get_price_fn. The real-time feed only carries
    bid/ask/mid/sizes, not Greeks (OPRA/indicative quote messages don't
    include them) — every Greek field is read with .get(..., default)
    below, so a stream-fed snapshot just means Greeks display as 0 /
    entry-delta rather than blocking the stop/trail/T1 price checks that
    matter, which never depend on Greeks.

    `get_price_fn(ticker) -> float | None` (same shape as
    run_equity_guard's own get_price_fn) supplies the UNDERLYING stock's
    price for the milestone alert's display line — added 2026-08-15 when
    GUARD_EVERY_S dropped to 10s: get_live_price() is an uncached REST
    call, and calling it unconditionally every 10s per options position
    would have quietly 6x'd that call volume for zero benefit (the
    underlying price here is purely cosmetic — it's not used in any
    stop/trail/T1 decision). Falls back to get_live_price() on a miss,
    same as every other injection point in this file.
    """
    t     = pos.get("ticker", "")
    setup = pos.get("setup", "")
    parts = setup.split()
    _occ  = parts[2] if len(parts) >= 3 else ""
    _entry_prem = float(pos.get("entry", 0))
    if not t or not _occ or _entry_prem <= 0:
        return None
    _stop_prem  = float(pos.get("stop",    _entry_prem * 0.5))
    _t1_prem    = float(pos.get("target1", _entry_prem * 1.5))
    # target2/T2 is no longer a fixed auto-close threshold here (see
    # OPTIONS_TRAIL_ACTIVATE_GAIN_PCT) — the trailing exit replaced it
    # 2026-08-10. The field still exists on new positions for reference.
    _delta_entry= float(pos.get("atr",     0.45))   # entry delta stored in atr field
    _ctrs       = max(1, int(pos.get("shares", 100)) // 100)

    _snap = get_snapshot_fn(_occ) if get_snapshot_fn else None
    if not _snap:
        _snap = _get_option_snapshot(_occ)
    if not _snap:
        return (f"⚠️ <b>{t}</b> {kind} {_occ}\n"
                f"   Cannot fetch live quote — check position manually")

    _cur_prem  = _snap["mid"]                       # display P&L at mid
    _exit_prem = _snap.get("bid", _cur_prem)        # exits fill at bid
    _pnl_pct   = (_cur_prem - _entry_prem) / _entry_prem * 100
    # A stream-fed snapshot has no Greeks keys at all (not even a 0
    # default) — fall back to the periodically REST-refreshed cache so
    # the theta-decay alert below still has a real value to check
    # against. See _cached_option_greeks()'s docstring.
    _greeks_src = _snap if "theta" in _snap else _cached_option_greeks(_occ)
    _theta_now = _greeks_src.get("theta", 0)
    _delta_now = _greeks_src.get("delta", _delta_entry)
    _iv_now    = _greeks_src.get("iv", 0)
    _theta_pct = abs(_theta_now / _cur_prem * 100) if _cur_prem > 0 else 0
    try:
        _occ_exp = _occ[-15:-9] if len(_occ) >= 15 else ""
        _exp_dt  = datetime.strptime(_occ_exp, "%y%m%d").date() if _occ_exp else None
        _dte_now = (_exp_dt - date.today()).days if _exp_dt else 99
    except Exception:
        _dte_now = 99

    # Peak-premium tracking drives the trailing exit below — update BEFORE
    # branching so a fresh new high this exact check is still trail-active
    # against (not one check behind).
    _peak_prem = max(float(pos.get("peak_premium", 0) or 0), _cur_prem)
    if _peak_prem > float(pos.get("peak_premium", 0) or 0):
        _update_option_position_field(_occ, peak_premium=_peak_prem)
    _trail_active   = _peak_prem >= _entry_prem * (1 + OPTIONS_TRAIL_ACTIVATE_GAIN_PCT / 100)
    _peak_gain_pct  = (_peak_prem - _entry_prem) / _entry_prem * 100 if _entry_prem > 0 else 0.0
    _giveback_pct   = _options_trail_giveback_pct(_peak_gain_pct)

    # Extra P&L visibility independent of which branch below fires (or
    # doesn't) — see _check_options_pnl_milestone.
    try:
        _underlying_px = (get_price_fn(t) if get_price_fn else None) or get_live_price(t)
        _check_options_pnl_milestone(pos, kind, _occ, _cur_prem, _entry_prem, _underlying_px)
    except Exception as _me:
        print(f"  ⚠️  {t} {kind}: milestone check failed — {_me}")

    _kp    = "OPT" if kind == "CALL" else "PUT"     # legacy dedup-key prefix
    _tod   = date.today().isoformat()
    _t1k, _trailk = f"{t}_{_kp}_T1_{_tod}",   f"{t}_{_kp}_TRAIL_{_tod}"
    _stopk, _dtek = f"{t}_{_kp}_STOP_{_tod}", f"{t}_{_kp}_DTE_{_tod}"

    if not _trail_active and _exit_prem <= _stop_prem:
        # Baseline floor for a position that never became meaningfully
        # profitable — trailing can't protect a move that hasn't happened.
        _st, _coid = _submit_options_close(_occ, _ctrs, f"{t} {kind} stop")
        if _st == "submitted":
            _action = "🔴 STOP HIT — AUTO-CLOSED"
            _msg = (f"Bid ${_exit_prem:.2f} ≤ stop ${_stop_prem:.2f} "
                    f"({_pnl_pct:+.0f}%) — SELL ×{_ctrs} submitted "
                    f"(id {_coid[:8]}…). Sync will record P&L.")
        elif _st == "pending":
            _action = "🔴 STOP HIT — close order working"
            _msg = f"SELL already open (id {(_coid or '?')[:8]}…) — awaiting fill"
        elif _st == "already_closed":
            _action = "🔴 STOP — position already closed at Alpaca"
            _msg = "Nothing held — next sync records the P&L"
        else:
            _action = "🔴 STOP HIT — ⚠️ AUTO-CLOSE FAILED"
            _msg = (f"Bid ${_exit_prem:.2f} ≤ stop ${_stop_prem:.2f} "
                    f"({_pnl_pct:+.0f}%) — SELL MANUALLY NOW")
        if not _is_alerted_today(_stopk):
            send_telegram(f"🔴 <b>OPTIONS STOP</b> — {t} {kind} {_occ}\n{_msg}")
            _mark_alerted(_stopk)
    elif _trail_active and _cur_prem <= _peak_prem * (1 - _giveback_pct / 100):
        # Replaces the old fixed T2 (+150%) auto-close (2026-08-10) — reacts
        # to how the trade actually moved (peak, then a real give-back)
        # instead of one static number that could be missed on a fast
        # reversal or fire too early on a slow, healthy grind.
        _st, _coid = _submit_options_close(_occ, _ctrs, f"{t} {kind} trail")
        _giveback_desc = f"peak ${_peak_prem:.2f} → now ${_cur_prem:.2f} ({_pnl_pct:+.0f}% from entry)"
        if _st == "submitted":
            _action = "🚀 TRAIL EXIT — AUTO-CLOSED (full exit)"
            _msg = (f"Gave back {_giveback_pct:.0f}%+ off the peak — {_giveback_desc} — "
                    f"SELL ×{_ctrs} submitted (id {_coid[:8]}…). Runner banked.")
        elif _st == "pending":
            _action = "🚀 TRAIL EXIT — close order working"
            _msg = f"SELL already open (id {(_coid or '?')[:8]}…) — awaiting fill"
        elif _st == "already_closed":
            _action = "🚀 TRAIL EXIT — position already closed at Alpaca"
            _msg = "Nothing held — next sync records the P&L"
        else:
            _action = "🚀 TRAIL EXIT — ⚠️ AUTO-CLOSE FAILED"
            _msg = f"Gave back {_giveback_pct:.0f}%+ off the peak — {_giveback_desc} — SELL MANUALLY"
        if not _is_alerted_today(_trailk):
            send_telegram(f"🚀 <b>OPTIONS TRAIL EXIT</b> — {t} {kind} {_occ}\n{_msg}")
            _mark_alerted(_trailk)
    elif _cur_prem >= _t1_prem and _stop_prem < _entry_prem:
        # T1: sell half if ≥2 contracts, raise stop to breakeven either way.
        # (_stop_prem < entry guard = T1 not yet taken)
        if _ctrs >= 2:
            _half = _ctrs // 2
            _st, _coid = _submit_options_close(_occ, _half, f"{t} {kind} T1 half")
            if _st == "submitted":
                # OCC-keyed, not ticker-keyed — found 2026-08-16 review:
                # _update_option_position_field() exists specifically
                # because a ticker-keyed update silently modifies every
                # position sharing this underlying (a call+put strangle,
                # or an options leg alongside an unrelated equity position
                # on the same ticker — see that function's docstring for
                # the confirmed SMCI incident). This T1 branch was never
                # migrated to it, so a real T1 fill could overwrite an
                # unrelated position's stop/shares.
                _update_option_position_field(_occ, shares=(_ctrs - _half) * 100,
                                              stop=round(_entry_prem, 2))
                _action = "🟢 T1 HIT — ½ SOLD, stop → breakeven"
                _msg = (f"Premium ${_cur_prem:.2f} ≥ T1 ${_t1_prem:.2f} "
                        f"({_pnl_pct:+.0f}%) — sold {_half}/{_ctrs} "
                        f"(id {_coid[:8]}…), stop raised to ${_entry_prem:.2f}")
            elif _st in ("pending", "already_closed"):
                _action = "🟢 T1 — partial close in progress"
                _msg = "Half-sell order working or already done"
            else:
                _action = "🟢 T1 HIT — ⚠️ auto-sell failed"
                _msg = (f"Premium ${_cur_prem:.2f} ≥ T1 ({_pnl_pct:+.0f}%) "
                        "— sell ½ manually, raise stop to breakeven")
        else:
            _update_option_position_field(_occ, stop=round(_entry_prem, 2))
            _action = "🟢 T1 HIT — stop → breakeven (1ct runner)"
            _msg = (f"Premium ${_cur_prem:.2f} ≥ T1 ({_pnl_pct:+.0f}%) — "
                    f"single contract: riding the trailing exit, stop raised to "
                    f"breakeven ${_entry_prem:.2f} (risk-free runner)")
        if not _is_alerted_today(_t1k):
            send_telegram(f"🟢 <b>OPTIONS T1 HIT</b> — {t} {kind} {_occ}\n{_msg}")
            _mark_alerted(_t1k)
    elif _dte_now <= OPTIONS_CLOSE_DTE:
        _action = f"⏳ DTE ALERT — {_dte_now}d left, consider close"
        _msg = (f"Only {_dte_now}d to expiry — theta burning fast. "
                f"P&L: {_pnl_pct:+.0f}%. Close or roll now.")
        if not _is_alerted_today(_dtek):
            send_telegram(f"⏳ <b>DTE WARNING</b> — {t} {kind} {_occ}\n{_msg}")
            _mark_alerted(_dtek)
    elif _theta_pct > 5.0 and _pnl_pct < 10:
        _action = "⏰ THETA ALERT — consider exit"
        _msg = (f"Decaying {_theta_pct:.1f}%/day with only "
                f"{_pnl_pct:+.0f}% gain — time is working against you")
    else:
        _action = ("✅ HOLDING" if kind == "CALL" else "🐻 HOLDING")
        _msg = f"P&L: <b>{_pnl_pct:+.0f}%</b>  θ decay {_theta_pct:.1f}%/day  DTE {_dte_now}d"

    _trail_desc = (f"trailing active, peak ${_peak_prem:.2f} (exits below "
                    f"${_peak_prem * (1 - _giveback_pct / 100):.2f})"
                    if _trail_active else
                    f"not yet active (arms at +{OPTIONS_TRAIL_ACTIVATE_GAIN_PCT:.0f}%)")
    return (
        f"{_action}  <b>{t}</b> {kind} {_occ}\n"
        f"   Prem: entry ${_entry_prem:.2f} → now ${_cur_prem:.2f} (bid ${_exit_prem:.2f})  "
        f"({_pnl_pct:+.0f}%)  {_ctrs}ct × 100 = ${_cur_prem*_ctrs*100:.0f} mkt val\n"
        f"   {_msg}\n"
        f"   Δ {_delta_now:.2f}  θ {_theta_now:.3f}/d  IV {_iv_now*100:.0f}%  "
        f"T1 ${_t1_prem:.2f}  Stop ${_stop_prem:.2f}  DTE {_dte_now}d\n"
        f"   Trail: {_trail_desc}"
    )


def _close_earnings_spread(pos: dict, reason: str) -> tuple[str, Optional[str]]:
    """
    Mirrors _submit_options_close but MLEG-aware: closes every leg of an
    earnings spread atomically in ONE order, with each leg's side/intent
    inverted from how it was opened (bought-to-open -> sell-to-close,
    sold-to-open -> buy-to-close). pos['legs'] must be in the same order
    _submit_earnings_spread() submitted them: long, short[, long, short] —
    i.e. even index = was long, odd index = was short.

    Returns (status, order_id) — same vocabulary as _submit_options_close():
    "submitted"/"pending"/"already_closed"/"failed".
    """
    client = get_alpaca_client()
    if client is None:
        return "failed", None

    legs_syms = pos.get("legs", [])
    if not legs_syms or len(legs_syms) % 2 != 0:
        return "failed", None

    from alpaca.trading.requests import OptionLegRequest
    from alpaca.trading.enums import PositionIntent

    # If Alpaca shows nothing held for ANY leg, treat the whole spread as
    # already closed rather than attempting to re-close legs that don't exist.
    still_held = False
    for _sym in legs_syms:
        try:
            _apos = client.get_open_position(_sym)
            if abs(int(float(_apos.qty))) > 0:
                still_held = True
                break
        except Exception:
            continue
    if not still_held:
        return "already_closed", None

    # Don't double-submit: if a closing order for these exact legs is already
    # open, report it instead of submitting a second one.
    try:
        _open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50))
        for _o in _open_orders:
            _o_syms = {getattr(_l, "symbol", None) for _l in (getattr(_o, "legs", None) or [])}
            if _o_syms and _o_syms == set(legs_syms):
                return "pending", str(_o.id)
    except Exception:
        pass

    close_legs = []
    net_credit = 0.0
    got_all_quotes = True
    for i, sym in enumerate(legs_syms):
        was_long = (i % 2 == 0)
        side   = OrderSide.SELL if was_long else OrderSide.BUY
        intent = PositionIntent.SELL_TO_CLOSE if was_long else PositionIntent.BUY_TO_CLOSE
        close_legs.append(OptionLegRequest(symbol=sym, ratio_qty=1, side=side, position_intent=intent))
        snap = _get_option_snapshot(sym)
        if not snap:
            got_all_quotes = False
            continue
        net_credit += snap.get("bid", 0) if was_long else -snap.get("ask", 0)

    # net_credit > 0 means we net RECEIVE money closing (sell longs for more
    # than we pay to buy back shorts) -> a credit -> negative limit_price
    # per the library's own debit/credit sign convention (see _submit_earnings_spread).
    limit_price = round(-net_credit, 2) if got_all_quotes else 0.01

    try:
        qty = max(1, int(pos.get("spread_qty", 1)))
        order = client.submit_order(LimitOrderRequest(
            qty=qty, order_class=OrderClass.MLEG, time_in_force=TimeInForce.DAY,
            limit_price=limit_price, legs=close_legs,
        ))
        print(f"  📤 AUTO-CLOSE spread {pos.get('ticker', '?')} legs={legs_syms} "
              f"({reason})  id={str(order.id)[:8]}…")
        return "submitted", str(order.id)
    except Exception as exc:
        print(f"  ❌ Earnings spread auto-close failed: {exc}")
        send_telegram(f"🚨 <b>SPREAD AUTO-CLOSE FAILED</b> — {pos.get('ticker', '?')} ({reason})\n"
                     f"{exc}\nClose manually in Alpaca NOW.")
        return "failed", None


def _monitor_earnings_spread_position(pos: dict) -> Optional[str]:
    """
    Earnings debit spreads are defined-risk by construction (max loss == the
    debit already paid) — no stop-loss enforcement needed, unlike a naked
    long option. Only two actions: close at EARNINGS_SPREAD_CLOSE_DTE before
    expiry (avoid short-leg pin/assignment risk), or optionally earlier once
    current value reaches EARNINGS_SPREAD_TAKE_PROFIT_PCT of max gain. Shared
    by run_momentum_watch (hourly cron) and the always-on daemon (60s).
    """
    t = pos.get("ticker", "")
    legs_syms = pos.get("legs", [])
    max_gain  = float(pos.get("max_gain", 0))
    if not t or not legs_syms:
        return None

    try:
        _occ_exp = legs_syms[0][-15:-9] if len(legs_syms[0]) >= 15 else ""
        _exp_dt  = datetime.strptime(_occ_exp, "%y%m%d").date() if _occ_exp else None
        dte_now  = (_exp_dt - date.today()).days if _exp_dt else 99
    except Exception:
        dte_now = 99

    cur_value = 0.0
    got_quotes = True
    for i, sym in enumerate(legs_syms):
        snap = _get_option_snapshot(sym)
        if not snap:
            got_quotes = False
            break
        was_long = (i % 2 == 0)
        cur_value += snap.get("bid", 0) if was_long else -snap.get("ask", 0)
    cur_value *= 100 * max(1, int(pos.get("spread_qty", 1)))

    _tod  = date.today().isoformat()
    _dtek = f"{t}_EARNSPREAD_DTE_{_tod}"
    _tpk  = f"{t}_EARNSPREAD_TP_{_tod}"

    if dte_now <= EARNINGS_SPREAD_CLOSE_DTE:
        st, oid = _close_earnings_spread(pos, f"{t} earnings spread DTE close")
        if st == "submitted":
            action, msg = "🔴 DTE CLOSE — AUTO-CLOSED", \
                f"{dte_now}d to expiry — closing to avoid pin/assignment risk. id {oid[:8]}…"
        elif st == "pending":
            action, msg = "🔴 DTE CLOSE — close order working", "Close order already open — awaiting fill"
        elif st == "already_closed":
            action, msg = "🔴 DTE — already closed at Alpaca", "Nothing held — next sync records the outcome"
        else:
            action, msg = "🔴 DTE CLOSE — ⚠️ AUTO-CLOSE FAILED", f"{dte_now}d to expiry — CLOSE MANUALLY NOW"
        if not _is_alerted_today(_dtek):
            send_telegram(f"🔴 <b>EARNINGS SPREAD DTE</b> — {t}\n{msg}")
            _mark_alerted(_dtek)
        return f"{t}: {action}"

    if got_quotes and max_gain > 0 and cur_value >= max_gain * EARNINGS_SPREAD_TAKE_PROFIT_PCT:
        st, oid = _close_earnings_spread(pos, f"{t} earnings spread take-profit")
        if st == "submitted":
            action, msg = "🚀 TAKE-PROFIT — AUTO-CLOSED", \
                f"Value ${cur_value:.0f} >= {EARNINGS_SPREAD_TAKE_PROFIT_PCT*100:.0f}% of max gain ${max_gain:.0f}. id {oid[:8]}…"
        elif st == "pending":
            action, msg = "🚀 TAKE-PROFIT — close order working", "Close order already open — awaiting fill"
        elif st == "already_closed":
            action, msg = "🚀 TAKE-PROFIT — already closed at Alpaca", "Nothing held — next sync records the outcome"
        else:
            action, msg = "🚀 TAKE-PROFIT — ⚠️ AUTO-CLOSE FAILED", f"Value ${cur_value:.0f} — CLOSE MANUALLY to lock in gain"
        if not _is_alerted_today(_tpk):
            send_telegram(f"🚀 <b>EARNINGS SPREAD TAKE-PROFIT</b> — {t}\n{msg}")
            _mark_alerted(_tpk)
        return f"{t}: {action}"

    return None


def run_options_guard(verbose: bool = True, get_snapshot_fn=None, get_price_fn=None,
                      positions: Optional[list] = None) -> list[str]:
    """
    Enforce stops/targets on every tracked options position right now.
    The always-on daemon calls this on its guard cadence; momentum-watch
    runs the same engine hourly as backup. Returns the per-position status
    lines.

    `get_snapshot_fn(occ_symbol) -> dict | None`, if given, is passed
    through to _monitor_option_position() for naked calls/puts (e.g. the
    daemon's real-time options WebSocket quote cache) — mirrors
    run_equity_guard's get_price_fn. Earnings spreads aren't wired to it
    since _monitor_earnings_spread_position() fetches each leg's snapshot
    directly; only DTE/take-profit checks run there, not a stop/trail
    engine that benefits from tighter freshness.

    `get_price_fn(ticker) -> float | None`, if given, is also passed
    through — supplies the underlying stock's price for the milestone
    alert's display line without an uncached REST call every cycle. See
    _monitor_option_position's docstring for why this exists.

    `positions`, if given, is used instead of re-reading POSITIONS_FILE —
    found in the 2026-08-16 review: guard_loop() calls this,
    run_equity_guard(), AND _check_stop_coverage() (via PositionTracker)
    back to back every 10s guard tick, each independently re-reading and
    re-parsing the same small JSON file. Optional and defaults to the
    original self-loading behavior so every other caller (momentum-watch,
    tests) is unaffected.
    """
    alerts: list[str] = []
    if positions is None:
        try:
            with open(POSITIONS_FILE) as _f:
                positions = json.load(_f)
        except Exception:
            return alerts
    for _pos in positions:
        _setup = _pos.get("setup", "")
        if _setup.startswith("Options Call "):
            _a = _monitor_option_position(_pos, "CALL", get_snapshot_fn=get_snapshot_fn, get_price_fn=get_price_fn)
        elif _setup.startswith("Options Put "):
            _a = _monitor_option_position(_pos, "PUT", get_snapshot_fn=get_snapshot_fn, get_price_fn=get_price_fn)
        elif _setup.startswith("Earnings "):
            _a = _monitor_earnings_spread_position(_pos)
        else:
            continue
        if _a:
            alerts.append(_a)
            if verbose:
                print("  " + _a.split("\n")[0].replace("<b>", "").replace("</b>", ""))
    return alerts


def _progress_equity_stop_to_trailing(pos: "OpenPosition", cur_price: float,
                                       trigger_price: Optional[float] = None) -> Optional[str]:
    """
    Automates what the T1 alert has only ever told a HUMAN to do manually
    for equity positions — options already self-manage this via the
    daemon's own momentum-watch loop. Confirmed live 2026-08-08: CELZ sat
    at +24.58% with its original entry-time stop completely untouched —
    the T1 alert said "move stop to breakeven" but nothing ever executed
    it, leaving the entire gain exposed.

    `trigger_price` overrides the gate below pos.target1 — pass a lower
    threshold to progress the stop BEFORE a full T1 hit. Added 2026-08-10
    after CLRO ran from $9.80 to a peak of $14.08 (+43.7%), reversed hard
    within 80 minutes, and gave nearly all of it back to a real loss —
    T1 was $14.73, never reached, so this function's T1-only gate meant
    zero protection ever engaged despite a massive run-up. See
    _check_equity_position_target()'s early-profit-lock branch, the new
    caller that passes a %-gain-based trigger_price instead of waiting
    for the full target. Defaults to pos.target1 when omitted, so every
    existing T1-triggered call site is unchanged.

    Two-step transition, run once per position (tracked via stop_stage):
      1. Replace the existing stop order's stop_price to breakeven (entry)
         in place via PATCH — not a cancel+resubmit, so it can't hit the
         share-reservation race that caused this week's stuck-HELD-order
         incidents (there's only ever one order here, never two competing
         for the same shares).
      2. Cancel that (now breakeven) plain stop and submit a genuine
         Alpaca-native trailing stop, so further gains lock in
         automatically without anyone needing to keep checking in. This
         step DOES need cancel+resubmit (trail_percent isn't a replaceable
         field), which is exactly the operation that stranded W without
         protection overnight — so if the trailing submission fails for
         any reason, this immediately falls back to a plain stop at
         breakeven rather than ever leaving the position unprotected.

    Trail percent is capped so the trailing stop's INITIAL level can never
    sit below breakeven, however far past the trigger price has already
    run: min(the position's own original entry-to-stop distance%, the
    actual current-price-to-entry distance%). Preserves each setup's
    designed risk cushion (a smallcap's wider stop stays wider) while
    guaranteeing this step never introduces new risk below what's already
    locked in. Once submitted, Alpaca's native trailing stop keeps
    ratcheting up on its own as price continues rising — this doesn't need
    to be called again to keep tightening after the initial trigger.

    Returns a human-readable description of what happened, or None if
    nothing changed (already trailing, threshold not yet reached, bad
    data, or a real failure — always printed, never silently swallowed).
    """
    if pos.stop_stage == "trailing":
        return None
    _gate = trigger_price if trigger_price is not None else pos.target1
    if _gate <= 0 or cur_price < _gate or pos.entry <= 0:
        return None

    client = get_alpaca_client()
    if client is None:
        return None

    from alpaca.trading.requests import (
        GetOrdersRequest, ReplaceOrderRequest, TrailingStopOrderRequest, StopOrderRequest,
    )
    from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce, OrderType, PositionIntent

    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(
            symbols=[pos.ticker], status=QueryOrderStatus.OPEN, limit=10))
        stop_orders = [o for o in open_orders if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)]
        if not stop_orders:
            print(f"  ⚠️  {pos.ticker}: trigger reached but no live stop order found — can't progress it")
            return None
        stop_order = stop_orders[0]

        client.replace_order_by_id(stop_order.id, ReplaceOrderRequest(stop_price=round(pos.entry, 2)))
    except Exception as exc:
        print(f"  ⚠️  {pos.ticker}: failed to raise stop to breakeven — {exc}")
        return None

    original_stop_pct = (pos.entry - pos.stop) / pos.entry * 100 if pos.stop > 0 else 0
    current_gain_pct  = (cur_price - pos.entry) / cur_price * 100
    trail_pct = round(max(0.5, min(original_stop_pct, current_gain_pct)), 2)

    try:
        client.cancel_order_by_id(stop_order.id)
        trail_order = client.submit_order(TrailingStopOrderRequest(
            symbol=pos.ticker, qty=pos.shares, side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC, trail_percent=trail_pct,
            position_intent=PositionIntent.SELL_TO_CLOSE,
        ))
        _update_position_field(pos.ticker, stop_stage="trailing", stop=round(pos.entry, 2))
        return (f"stop raised to breakeven ${pos.entry:.2f}, now trailing "
                f"{trail_pct:.1f}% (id {str(trail_order.id)[:8]}…)")
    except Exception as exc:
        # The breakeven replace already succeeded above, but the trailing
        # submission failed after cancelling that order — never leave the
        # position with nothing live. Fall back to a plain stop at
        # breakeven, the same reliable order type that recovered every
        # stuck-HELD incident this week.
        print(f"  ⚠️  {pos.ticker}: trailing stop submission failed ({exc}) — falling back to a plain breakeven stop")
        try:
            fallback = client.submit_order(StopOrderRequest(
                symbol=pos.ticker, qty=pos.shares, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=round(pos.entry, 2),
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ))
            _update_position_field(pos.ticker, stop_stage="initial", stop=round(pos.entry, 2))
            return f"stop raised to breakeven ${pos.entry:.2f} (trailing failed, plain stop fallback id {str(fallback.id)[:8]}…)"
        except Exception as exc2:
            print(f"  🚨 {pos.ticker}: fallback stop ALSO failed ({exc2}) — POSITION MAY BE UNPROTECTED, check manually")
            send_telegram(
                f"🚨 <b>{pos.ticker}: stop management failed</b>\n"
                f"Breakeven raise succeeded but both the trailing stop and the "
                f"fallback plain stop failed to submit ({exc2}). "
                f"Check Alpaca directly — this position may have no live stop."
            )
            return None


EARLY_PROFIT_LOCK_GAIN_PCT = 15.0   # see _check_equity_position_target's early-lock branch

def _check_equity_position_target(pos: dict, cur_price: Optional[float] = None) -> None:
    """
    T1/T2 exit alerts + automated stop progression for one equity
    position. Extracted from run_momentum_watch() (2026-08-09) so the
    daemon's continuous equity guard loop (run_equity_guard) can reuse the
    exact same logic instead of a parallel copy that could silently drift
    out of sync with a future fix made to only one of the two call sites.
    `cur_price` lets a caller pass an already-known price (e.g. the
    daemon's real-time stream cache) instead of forcing a fresh REST
    lookup here — omit it to fetch one via get_live_price() as before.

    Fires once per target per day (_is_alerted_today dedup) no matter how
    often or from how many call sites this runs — safe to call from both
    the hourly cron's momentum-watch AND the daemon's more frequent loop
    without double-alerting, the same git-synced-dedup pattern options
    positions already rely on being checked from two separate places.
    """
    t = pos.get("ticker", "")
    e = float(pos.get("entry", 0))
    if not t or e <= 0:
        return
    _cur_eq = cur_price if cur_price is not None else get_live_price(t)
    if _cur_eq is None:
        return
    _t1e = float(pos.get("target1", 0))
    _t2e = float(pos.get("target2", 0))
    _pnle = round((_cur_eq - e) / e * 100, 1) if e > 0 else 0
    _t2k = f"{t}_T2_{date.today().isoformat()}"
    _t1k = f"{t}_T1_{date.today().isoformat()}"
    if _t2e > 0 and _cur_eq >= _t2e and not _is_alerted_today(_t2k):
        send_telegram(
            f"🎯 <b>T2 HIT</b> — {t} LONG\n"
            f"Entry ${e} → Now ${_cur_eq:.2f} (+{_pnle}%)  T2 ${_t2e}\n"
            f"<b>Sell remaining shares.</b> Best comfortable exit."
        )
        _mark_alerted(_t2k)
    elif _t1e > 0 and _cur_eq >= _t1e and not _is_alerted_today(_t1k):
        # Confirmed live 2026-08-08: this alert used to just TELL a human to
        # move the stop to breakeven — CELZ sat at +24.58% with its original
        # stop untouched because nothing ever executed it. Now actually does
        # it (see _progress_equity_stop_to_trailing).
        _stop_msg = ""
        try:
            _prog = _progress_equity_stop_to_trailing(OpenPosition(**pos), _cur_eq)
            if _prog:
                _stop_msg = f"\n🔒 {_prog}"
        except Exception as _pe:
            print(f"  ⚠️  {t}: stop progression failed — {_pe}")
        send_telegram(
            f"✅ <b>T1 HIT</b> — {t} LONG\n"
            f"Entry ${e} → Now ${_cur_eq:.2f} (+{_pnle}%)  T1 ${_t1e}\n"
            f"Consider selling 50% here manually — the rest is "
            f"riding to T2 ${_t2e} with a locked-in floor.{_stop_msg}"
        )
        _mark_alerted(_t1k)
    elif _pnle >= EARLY_PROFIT_LOCK_GAIN_PCT:
        # Confirmed live 2026-08-10: CLRO ran from $9.80 to a peak of $14.08
        # (+43.7%) in 39 minutes, reversed hard, and gave nearly all of it
        # back to a real loss within 80 minutes — T1 was $14.73, never
        # reached, so the T1-only branch above never engaged and this
        # position got ZERO automated protection despite a massive run-up.
        # "Low Float Catalyst" names are specifically known for this
        # spike-and-fade pattern. This locks in a trailing stop once gain
        # crosses EARLY_PROFIT_LOCK_GAIN_PCT, well before the full target —
        # same proven mechanism as the T1 branch, just triggered earlier.
        # Alpaca's native trailing stop then ratchets up on its own as
        # price keeps rising, so this doesn't need to re-fire to keep
        # tightening — only once per day (dedup) to avoid alert spam while
        # already-trailing positions keep clearing this same threshold.
        _lockk = f"{t}_EARLYLOCK_{date.today().isoformat()}"
        if not _is_alerted_today(_lockk):
            _trigger_px = round(e * (1 + EARLY_PROFIT_LOCK_GAIN_PCT / 100), 4)
            try:
                _prog = _progress_equity_stop_to_trailing(OpenPosition(**pos), _cur_eq, trigger_price=_trigger_px)
                if _prog:
                    send_telegram(
                        f"🔒 <b>Early profit lock</b> — {t} LONG\n"
                        f"Entry ${e} → Now ${_cur_eq:.2f} (+{_pnle}%)\n"
                        f"{_prog}\n"
                        f"Locking in gains ahead of full T1 (${_t1e}) — protects against "
                        f"a fast reversal instead of waiting for the complete target."
                    )
                    _mark_alerted(_lockk)
            except Exception as _pe:
                print(f"  ⚠️  {t}: early profit-lock progression failed — {_pe}")


def run_equity_guard(get_price_fn=None, positions: Optional[list] = None) -> None:
    """
    Continuous equity-position T1/T2/stop-progression check — the
    always-on daemon's counterpart to run_options_guard() (which only
    ever covered options/earnings-spread positions). Confirmed live
    2026-08-09: plain equity positions (e.g. CELZ, CLRO — everything
    actually held at the time) were not covered by ANY continuous daemon
    loop; only the hourly-ish cron momentum-watch checked them, meaning
    up to ~55 minutes of exposure with a stale, un-raised stop after T1.
    Reuses _check_equity_position_target(), the exact logic
    run_momentum_watch() already used, just invoked far more often from
    the daemon's guard_loop.

    `get_price_fn(ticker) -> float | None`, if given, is tried first for
    each position (e.g. the daemon's real-time WebSocket price cache).
    _check_equity_position_target() falls back to its own REST lookup
    whenever this returns None (stream not running, or the cached price
    is stale) — this never fails closed just because the stream is down,
    it just gets slower, matching the same guarantee options positions
    already have via run_options_guard()'s plain REST-based checks.

    `positions`, if given, is used instead of re-reading POSITIONS_FILE —
    see run_options_guard()'s matching docstring for why (guard_loop()
    calls both back to back every 10s guard tick). Optional and defaults
    to the original self-loading behavior.
    """
    if positions is None:
        try:
            with open(POSITIONS_FILE) as f:
                positions = json.load(f)
        except Exception:
            return
    for pos in positions:
        setup = pos.get("setup", "")
        if setup.startswith(("Options Call ", "Options Put ", "Earnings ")):
            continue   # already covered by run_options_guard()
        cur_price = get_price_fn(pos.get("ticker", "")) if get_price_fn else None
        _check_equity_position_target(pos, cur_price=cur_price)


def run_momentum_watch() -> None:
    """
    Intraday momentum watch — runs at 10:30 AM and 11:30 AM alongside the main scan.

    For each active small-cap play (open positions + watchlist movers ≥8%):
      → Fires "BREAKOUT SETUP" Telegram alert when consolidation signals align
        so you get in at the right moment before the second leg, not after it.
      → Fires "EXIT" or "TRAIL STOP" alert with exact price when momentum fades
        so you stay in for the full move but don't give back the gains.

    Entry tiers for breakout setup:
      Limit order: place at entry_px (tight above consolidation high)
      Stop:        stop_px (below consolidation low)
      Targets:     +30% / +50% / 2x from entry (Moon Shot if applicable)

    Exit / trail model:
      At T1 (+30% from entry): raise stop to break-even
      At T2 (+50%):            trail at 2-bar low
      EXIT signal:             exit market order immediately
    """
    import re as _re
    now_et = datetime.now(ET)
    print(f"\n  📡 MOMENTUM WATCH — {now_et.astimezone(MT).strftime('%I:%M %p MT')}")

    # Collect active plays
    active_plays: list[dict] = []
    options_alerts: list[str] = []

    # 1a. Open equity positions from position log
    try:
        if os.path.exists("dman_positions.json"):
            with open("dman_positions.json") as _pf:
                for pos in json.load(_pf):
                    t     = pos.get("ticker", "")
                    e     = float(pos.get("entry", 0))
                    fl    = float(pos.get("float_m", 0))
                    setup = pos.get("setup", "")
                    if not t:
                        continue
                    # Options positions — shared monitor enforces stop/T1/T2/DTE
                    # (same engine the always-on daemon runs every 60s)
                    if setup.startswith("Options Call "):
                        _oa = _monitor_option_position(pos, "CALL")
                        if _oa:
                            options_alerts.append(_oa)
                        continue
                    elif setup.startswith("Options Put "):
                        _oa = _monitor_option_position(pos, "PUT")
                        if _oa:
                            options_alerts.append(_oa)
                        continue
                    elif setup.startswith("Earnings "):
                        _oa = _monitor_earnings_spread_position(pos)
                        if _oa:
                            options_alerts.append(_oa)
                        continue

                    if e > 0:
                        # T1/T2 exit alerts + stop progression — see
                        # _check_equity_position_target() (also reused by the
                        # daemon's run_equity_guard() for continuous checking).
                        _check_equity_position_target(pos)
                        active_plays.append({"ticker": t, "entry": e, "float_m": fl, "source": "position"})
    except Exception:
        pass

    # 2. DMAN_SMALLCAP_WATCHLIST — monitor ALL tickers during market hours.
    # Use today's actual open vs prev close for gap, NOT real-time price, because by
    # 10:30 AM a gap-down recovery play (APVO -3.6% open) may already be above prev
    # close and the real-time gap would show +1%, missing the recovery signal entirely.
    # TRVI-type plays: flat/tiny gap but still run +8% intraday — always include watchlist.
    already = {p["ticker"] for p in active_plays}
    for ticker in DMAN_SMALLCAP_WATCHLIST:
        if ticker in already:
            continue
        try:
            # Use 2-day daily history to get TRUE opening gap (open vs prev close)
            _hist2 = yf.Ticker(ticker).history(period="2d", interval="1d")
            if len(_hist2) < 2:
                continue
            _today_open = float(_hist2["Open"].iloc[-1])
            _prev_close = float(_hist2["Close"].iloc[-2])
            if _today_open <= 0 or _prev_close <= 0:
                continue
            opening_gap = (_today_open - _prev_close) / _prev_close * 100
            is_gap_up       = opening_gap >= 3.0
            is_recovery_dip = -15.0 <= opening_gap < 0
            # Always include watchlist tickers during market hours regardless of gap.
            # A flat-gap ticker like TRVI (+0.2%) can still run +8% intraday.
            fl_m, _, _, _ = _get_short_float_data(ticker)
            if is_gap_up:
                _src = f"gap {opening_gap:+.1f}% at open"
            elif is_recovery_dip:
                _src = f"recovery dip (opened {opening_gap:+.1f}% → VWAP reclaim watch)"
            else:
                _src = f"watchlist (flat gap {opening_gap:+.1f}%)"
            active_plays.append({"ticker": ticker, "entry": 0.0,
                                 "float_m": fl_m, "source": _src})
        except Exception:
            continue

    if not active_plays:
        print("  No active small-cap plays — nothing to watch.")
        return

    setup_alerts: list[str] = []
    fade_alerts:  list[str] = []
    age_alerts:   list[str] = []

    # Swing position age check — alert on any position open ≥ 3 days (PDT-protected swings
    # are short-term holds; stale positions tie up PDT budget and capital)
    _SWING_AGE_LIMIT = 3
    try:
        _pt_age = PositionTracker()
        for _pos in _pt_age.positions:
            try:
                _days_in = (date.today() - date.fromisoformat(_pos.entry_date)).days
                if _days_in >= _SWING_AGE_LIMIT:
                    _age_key = f"{_pos.ticker}_SWING_AGE_{date.today().isoformat()}"
                    if not _is_alerted_today(_age_key):
                        age_alerts.append(
                            f"⏳ <b>{_pos.ticker}</b>  SWING STALE — {_days_in}d held\n"
                            f"   Entry ${_pos.entry}  Setup: {_pos.setup}\n"
                            f"   Consider closing — position is tying up capital and PDT budget"
                        )
                        _mark_alerted(_age_key)
            except Exception:
                pass
    except Exception:
        pass

    for play in active_plays:
        ticker  = play["ticker"]
        entry   = play["entry"]
        fl_m    = play["float_m"]
        source  = play["source"]

        try:
            df_5m  = _fetch_intraday_bars(ticker, interval="5m", period="1d")
            levels = _compute_session_levels(df_5m)
            if levels["cur_price"] == 0.0:
                continue

            cur = levels["cur_price"]
            vwap = levels["vwap"]

            if entry == 0.0:
                # Not in position — check VWAP reclaim first, then breakout setup
                _above_vwap = vwap > 0 and cur > vwap
                _vwap_tag   = ""
                if _above_vwap and "recovery" in source:
                    _vwap_dist = (cur - vwap) / vwap * 100
                    _vwap_tag  = f"  🔥 VWAP RECLAIMED (+{_vwap_dist:.1f}% above)"
                elif not _above_vwap and vwap > 0 and "recovery" in source:
                    _vwap_dist = (vwap - cur) / vwap * 100
                    _vwap_tag  = f"  ⏳ below VWAP ({_vwap_dist:.1f}% away — watching)"

                bp = _detect_pre_breakout(levels)
                # Fire alert on breakout setup OR on VWAP reclaim from recovery dip
                _fire = bp["setup"] or (_above_vwap and "recovery" in source)
                if _fire:
                    if bp["setup"]:
                        entry_px = bp["entry_px"]
                        stop_px  = bp["stop_px"]
                        sig_str  = " + ".join(bp["signals"][:3])
                    else:
                        # Pure VWAP reclaim: entry at current price, stop at session low
                        entry_px = round(cur * 1.002, 4)   # slight limit above current
                        _sl_base = levels.get("session_low") or 0
                        stop_px  = round((_sl_base if _sl_base > 0 else cur * 0.92) * 0.99, 4)
                        sig_str  = f"VWAP reclaim ({source})"
                    risk_px = round(max(entry_px - stop_px, 0.001), 4)
                    t1 = round(entry_px * 1.30, 4)
                    t2 = round(entry_px * 1.50, 4)
                    t3_str = f"  T3 2x: ${round(entry_px * 2.0, 4):.4f}" if fl_m > 0 and fl_m < 2.0 else ""
                    _label = "🔥 VWAP RECLAIM" if (not bp["setup"] and _above_vwap) else "BREAKOUT SETUP"
                    setup_alerts.append(
                        f"🟡 <b>{ticker}</b>  {_label}  [{source}]{_vwap_tag}\n"
                        f"   {sig_str}\n"
                        f"   Entry: <b>${entry_px:.4f}</b>  Stop: ${stop_px:.4f}  "
                        f"(risk ${risk_px:.4f}/sh)\n"
                        f"   T1: ${t1:.4f} (+30%)  T2: ${t2:.4f} (+50%){t3_str}\n"
                        f"   Curr: ${cur:.4f}  VWAP: ${vwap:.4f}"
                    )
            else:
                # In position — check fade + trailing stop levels
                gain_pct = (cur - entry) / entry * 100 if entry > 0 else 0.0
                fd = _detect_momentum_fade(levels, entry, fl_m)

                # Always compute dynamic trailing levels regardless of fade signal
                # T1 hit (+30%): stop moves to break-even
                # T2 hit (+50%): stop trails to 2-bar low
                be_stop   = round(entry * 1.002, 4)   # break-even + 0.2% buffer
                trail_now = fd["trail_stop"]
                if gain_pct >= 50.0:
                    stop_rec = f"${trail_now:.4f} (2-bar low trail — T2 reached)"
                elif gain_pct >= 30.0:
                    stop_rec = f"${be_stop:.4f} (move to break-even — T1 reached)"
                else:
                    stop_rec = f"${be_stop:.4f} (original stop — below entry)"

                if fd["action"] in ("exit", "trail"):
                    emoji = "🔴" if fd["action"] == "exit" else "🟡"
                    action_label = "EXIT NOW" if fd["action"] == "exit" else "TRAIL STOP"
                    fade_alerts.append(
                        f"{emoji} <b>{ticker}</b>  {action_label}  [{source}]\n"
                        f"   {fd['reason']}\n"
                        f"   Entry: ${entry:.4f}  Curr: ${cur:.4f}  "
                        f"P&L: <b>{gain_pct:+.1f}%</b>\n"
                        f"   Recommended stop → {stop_rec}\n"
                        f"   Session high: ${levels['session_high']:.4f}  VWAP: ${vwap:.4f}"
                    )
                else:
                    # Momentum intact — still report trailing levels so you know where your stop is
                    if gain_pct >= 10.0:   # Only report if we have meaningful gains
                        fade_alerts.append(
                            f"✅ <b>{ticker}</b>  MOMENTUM INTACT  [{source}]\n"
                            f"   Entry: ${entry:.4f}  Curr: ${cur:.4f}  "
                            f"P&L: <b>{gain_pct:+.1f}%</b>\n"
                            f"   Active stop → {stop_rec}\n"
                            f"   Session high: ${levels['session_high']:.4f}  VWAP: ${vwap:.4f}"
                        )

        except Exception:
            continue

    if not setup_alerts and not fade_alerts and not options_alerts and not age_alerts:
        print(f"  All clear at {now_et.strftime('%H:%M')} — no setups or fade signals.")
        return

    lines = [f"📡 <b>DMan Momentum Watch</b>  {now_et.astimezone(MT).strftime('%I:%M %p MT')}"]

    if age_alerts:
        lines.append(f"\n⏳ <b>STALE POSITIONS ({len(age_alerts)})</b>\n")
        for a in age_alerts:
            lines.append(a)
            lines.append("")

    if options_alerts:
        lines.append(f"\n🎯 <b>OPTIONS POSITIONS ({len(options_alerts)})</b>\n")
        for a in options_alerts:
            lines.append(a)
            lines.append("")

    if setup_alerts:
        lines.append(f"\n🔔 <b>BREAKOUT SETUP ({len(setup_alerts)})</b> — entry NOW before next leg\n")
        for a in setup_alerts:
            lines.append(a)
            lines.append("")

    if fade_alerts:
        lines.append(f"\n📊 <b>POSITION STATUS ({len(fade_alerts)})</b>\n")
        for a in fade_alerts:
            lines.append(a)
            lines.append("")

    lines.append("<i>Trailing model: T1(+30%)→move stop to break-even | T2(+50%)→trail 2-bar low | EXIT signal→market out</i>")

    msg = "\n".join(lines)
    print(_re.sub(r"<[^>]+>", "", msg))
    send_telegram(msg)


def run_premarket_briefing() -> None:
    """
    Daily 9:10 AM ET pre-market briefing.
    Sends a Telegram summary covering regime, macro env, seasonal, live WR,
    monthly P&L, and filter suggestions. Never modifies code autonomously.
    """
    now_et = datetime.now(ET)
    date_str = now_et.strftime("%A %b %d, %Y")

    # ── 0a. GTC swing fill reconciliation ─────────────────────────────
    # If a GTC entry filled overnight, update PositionTracker entry price to the
    # actual avg fill so stop/target math is anchored to the real fill, not the limit.
    try:
        _rc = get_alpaca_client()
        _pt_r = PositionTracker()
        if _rc and _pt_r.positions:
            _alp_positions = {p.symbol: p for p in _rc.get_all_positions()}
            _updated = []
            for _rp in _pt_r.positions:
                if _rp.setup.startswith("SWING") and _rp.ticker in _alp_positions:
                    _ap = _alp_positions[_rp.ticker]
                    _actual_entry = float(getattr(_ap, "avg_entry_price", 0) or 0)
                    if _actual_entry > 0 and abs(_actual_entry - _rp.entry) / max(_rp.entry, 0.01) > 0.005:
                        # Fill price differs from limit by > 0.5% — re-anchor stop and target
                        _risk = abs(_rp.entry - _rp.stop)
                        _rp.entry   = _actual_entry
                        _rp.stop    = round(_actual_entry - _risk, 2)
                        _rp.target1 = round(_actual_entry + 2.5 * _risk, 2)
                        _rp.target2 = round(_actual_entry + 4.0 * _risk, 2)
                        _updated.append(_rp.ticker)
            if _updated:
                _pt_r._save()
                print(f"  🔄 GTC reconciliation: re-anchored {', '.join(_updated)} to actual fill prices")
    except Exception as _rc_exc:
        print(f"  [swing reconcile] {_rc_exc}")

    # ── 0. Scanner health watchdog ────────────────────────────────────
    scanner_health_line = ""
    try:
        if os.path.exists(SCAN_LOG_FILE):
            with open(SCAN_LOG_FILE) as _swf:
                _sw_log = json.load(_swf)
            if _sw_log:
                _sw_ts  = _sw_log[-1].get("ts", "")
                _sw_dt  = datetime.fromisoformat(_sw_ts).astimezone(ET)
                _sw_hrs = (now_et - _sw_dt).total_seconds() / 3600
                if _sw_hrs < 2:
                    scanner_health_line = f"✅ Scanner healthy — last run {int(_sw_hrs * 60)}min ago"
                elif _sw_hrs < 27:   # within a trading day + overnight gap
                    scanner_health_line = f"✅ Scanner ran {int(_sw_hrs)}h ago"
                else:
                    _sw_days = int(_sw_hrs / 24)
                    scanner_health_line = (
                        f"⚠️ <b>SCANNER DOWN</b> — last run {_sw_days}d ago "
                        f"({_sw_dt.strftime('%b %d')}). Check GitHub Actions → Actions tab."
                    )
            else:
                scanner_health_line = "⚠️ Scan log empty — scanner may not have run yet"
        else:
            scanner_health_line = "⚠️ No scan log found — scanner may not be running"
    except Exception:
        pass

    # ── 0.5. Global market context + breaking news ───────────────────
    print("  [0.5/7] Fetching global market context + breaking news...")
    _global_ctx_section = ""
    try:
        _briefing_ctx  = _fetch_global_context()
        _briefing_news = _fetch_breaking_news_rss(hours_back=12)
        _tone_icons = {
            "strong bull": "🟢🟢", "bull": "🟢", "neutral": "🟡",
            "caution": "🟠", "risk-off": "🔴",
        }
        _g_icon = _tone_icons.get(_briefing_ctx.get("tone", "neutral"), "⚪")
        _ctx_msg_lines = [f"{_g_icon} {_briefing_ctx.get('summary', '')}"]
        _ctx_comps = _briefing_ctx.get("components", {})
        if _ctx_comps:
            _comp_parts = []
            for _ck, _cv in list(_ctx_comps.items())[:6]:
                _comp_parts.append(f"{_ck}: {_cv:+d}" if isinstance(_cv, int) else f"{_ck}: {_cv}")
            _ctx_msg_lines.append("  " + " | ".join(_comp_parts))
        if _briefing_news:
            _ctx_msg_lines.append("📰 <b>Breaking</b>")
            for _hl, _src, _ts, _imp in _briefing_news[:4]:
                _em = "🔴 " if _imp <= -1 else ("🟢 " if _imp >= 1 else "")
                _ctx_msg_lines.append(f"   {_em}{_hl[:100]}  <i>[{_src} {_ts}]</i>")
        _global_ctx_section = "\n\n🌍 <b>GLOBAL CONTEXT</b>  (risk mult: {:.2f}x)\n".format(
            _briefing_ctx.get("risk_mult", 1.0)
        ) + "\n".join(_ctx_msg_lines)
    except Exception as _gce:
        print(f"  ⚠️  Global context fetch failed: {_gce}")

    # ── 1. Market regime + macro context ─────────────────────────────
    print("  [1/7] Checking market regime + macro context...")
    regime_line  = "Unable to fetch regime"
    regime_line2 = ""
    macro_env_section = ""
    warnings_section  = ""
    vix = "?"
    try:
        regime   = get_market_regime()
        top_secs = get_top_sectors()
        r_type   = regime.get("regime", "UNKNOWN")
        r_score  = regime.get("score", 0)
        det      = regime["details"]
        vix      = det.get("VIX", "?")

        # Fixed key names — keys are "SPY vs EMA20" / "QQQ vs EMA20"
        spy_above_ema20 = det.get("SPY vs EMA20", False)
        spy_above_ema50 = det.get("SPY vs EMA50", False)
        qqq_above_ema20 = det.get("QQQ vs EMA20", False)
        qqq_above_ema50 = det.get("QQQ vs EMA50", False)
        spy_ema20_dist  = det.get("SPY EMA20 dist", 0.0)
        spy_ema50_dist  = det.get("SPY EMA50 dist", 0.0)
        qqq_ema20_dist  = det.get("QQQ EMA20 dist", None)
        spy_ok   = "✅" if spy_above_ema20 else "⚠️"
        qqq_ok   = "✅" if qqq_above_ema20 else "⚠️"

        # Build key-level context lines
        spy_ema20_str = f"{spy_ema20_dist:+.1f}% {'above' if spy_above_ema20 else 'below'} EMA20"
        spy_ema50_str = f"{abs(spy_ema50_dist):.1f}% {'above' if spy_above_ema50 else 'below'} EMA50"
        qqq_note      = det.get("QQQ Note", "N/A")

        regime_line  = f"<b>{r_type}</b> ({r_score}/19)  VIX: {vix}"
        regime_line2 = (
            f"SPY {spy_ok} {spy_ema20_str}  |  {spy_ema50_str}\n"
            f"QQQ {qqq_ok} {qqq_note}\n"
            f"Top sectors: {', '.join(top_secs[:3])}"
        )

        # VIX shock + term structure + defensive rotation warnings at the top
        regime_warnings = []
        vix_shock_str = det.get("VIX Shock", "none")
        vix_term_str  = det.get("VIX Term Structure", "N/A")
        def_rot_str   = det.get("Def Rotation", "none")
        if vix_shock_str and vix_shock_str != "none":
            regime_warnings.append(f"⚡ <b>VIX SHOCK</b>: {vix_shock_str} — min score +5 today")
        if vix_term_str and "INVERTED" in vix_term_str:
            regime_warnings.append(f"📉 <b>VIX term inverted</b>: {vix_term_str} — "
                                   f"acute fear; size reduction active")
        if def_rot_str and def_rot_str != "none":
            regime_warnings.append(f"🔄 <b>DEFENSIVE ROTATION</b>: {def_rot_str} — tech longs -5pts")
        vix_comp_str = det.get("VIX Complacency", "none")
        if vix_comp_str and vix_comp_str != "none":
            regime_warnings.append(f"😴 {vix_comp_str}")
        if qqq_ema20_dist is not None and qqq_ema20_dist < -0.5:
            regime_warnings.append(
                f"🔴 <b>QQQ below EMA20</b> ({qqq_ema20_dist:+.2f}%) — tech index in declining trend; "
                f"Gap & Hold setups have lower follow-through probability; tighten score threshold or stand aside"
            )
        elif qqq_ema20_dist is not None and abs(qqq_ema20_dist) < 0.5:
            regime_warnings.append(
                f"⚠️ <b>QQQ at EMA20</b> ({qqq_ema20_dist:+.2f}%) — tech at decision level; "
                f"gap failure here weakens regime score"
            )
        warnings_section = ("\n" + "\n".join(regime_warnings)) if regime_warnings else ""

        # Extended macro environment (rates, dollar, breadth)
        tlt_note  = det.get("TLT Note", "N/A")
        dxy_note  = det.get("DXY Note", "N/A")
        tlt_trend = det.get("TLT Trend", "flat")
        dxy_trend = det.get("DXY Trend", "flat")
        tlt_emoji = "📈" if tlt_trend == "rising" else ("📉" if tlt_trend == "falling" else "➡️")
        dxy_emoji = "💪" if dxy_trend == "strong" else ("🔻" if dxy_trend == "weak" else "➡️")
        breadth   = det.get("IWM Breadth", "N/A")
        macro_env_section = (
            f"\n\n🌐 <b>MACRO ENVIRONMENT</b>\n"
            f"TLT {tlt_emoji}: {tlt_note}  (rates {'falling ✅' if tlt_trend == 'rising' else ('rising ⚠️' if tlt_trend == 'falling' else 'flat')})\n"
            f"DXY {dxy_emoji}: {dxy_note}  ({'headwind ⚠️' if dxy_trend == 'strong' else ('tailwind ✅' if dxy_trend == 'weak' else 'neutral')})\n"
            f"Breadth: {breadth}"
        )
    except Exception as e:
        regime_line2 = f"⚠️ regime fetch error: {str(e)[:60]}"
        send_telegram(f"⚠️ <b>DMan pre-market</b>: regime fetch failed\n<code>{str(e)[:120]}</code>")

    # ── 2. Macro calendar ─────────────────────────────────────────────
    print("  [2/6] Checking macro calendar...")
    try:
        today_d = now_et.date()
        nfp_set = _nfp_dates()
        events_lines = []
        for offset in range(8):  # today through 7 calendar days ahead
            d = today_d + timedelta(days=offset)
            if offset == 0:
                day_lbl = "today"
            elif offset == 1:
                day_lbl = f"tomorrow ({d.strftime('%a %b %d')})"
            else:
                day_lbl = f"in {offset}d ({d.strftime('%a %b %d')})"
            if d in _FOMC_DATES:
                if offset == 0:
                    events_lines.append(f"⛔ FOMC today — full-day blackout")
                elif offset == 1:
                    events_lines.append(f"⛔ FOMC tomorrow — full-day blackout; avoid new entries today")
                elif offset == 2:
                    # Today is the last clear session before the ±1 day blackout begins tomorrow
                    events_lines.append(
                        f"⛔ FOMC {day_lbl} — <b>TODAY is the LAST CLEAR SESSION</b> "
                        f"(blackout begins tomorrow through Thu). Maximize quality setups today."
                    )
                else:
                    events_lines.append(f"⛔ FOMC {day_lbl} — approaching; consider strangle / reduce size")
            if d in nfp_set:
                if offset == 0:
                    events_lines.append(f"📊 NFP today (6:30 AM MT) — blackout until 8:00 AM MT, then open")
                else:
                    events_lines.append(f"📊 NFP {day_lbl} — prepare for gap at open")
            if d in _CPI_DATES:
                if offset == 0:
                    events_lines.append(f"📊 CPI today (6:30 AM MT) — blackout until 8:00 AM MT, then open")
                else:
                    events_lines.append(f"📊 CPI {day_lbl} — prepare for gap at open")
            if d in _PPI_DATES:
                if offset == 0:
                    events_lines.append(f"📊 PPI today (6:30 AM MT) — blackout until 8:00 AM MT, then open")
                else:
                    events_lines.append(f"📊 PPI {day_lbl} — inflation data; gap risk at open")
            # Major unscheduled macro events (tariff deadlines, etc.) — same
            # ±1 day full blackout as FOMC, and the same multi-day advance
            # warning treatment here. This is a manually-maintained set
            # (see _MAJOR_MACRO_EVENT_DATES) so it stays empty most of the
            # time; when populated, the briefing should flag it exactly
            # like a scheduled event, not leave it as a same-day surprise.
            if d in _MAJOR_MACRO_EVENT_DATES:
                if offset == 0:
                    events_lines.append(f"⛔ Major macro event today — full-day blackout")
                elif offset == 1:
                    events_lines.append(f"⛔ Major macro event tomorrow — full-day blackout; avoid new entries today")
                else:
                    events_lines.append(f"⛔ Major macro event {day_lbl} — approaching; see code comment for details")
            # OPEX markers
            if d.weekday() == 4 and d == _get_third_friday(d.year, d.month):
                if offset == 0:
                    events_lines.append(f"⚡ OPEX today — gamma pinning; size down, expect wider spreads")
                elif offset == 1:
                    events_lines.append(f"⚡ OPEX tomorrow — consider SPY/QQQ strangle before close")
                else:
                    events_lines.append(f"⚡ OPEX {day_lbl} — monthly options expiration approaching")
        # NFP printed on a market holiday: the first trading session after that closed
        # NFP day IS the reaction session — note it so the user watches for gap at open.
        for _nfp_past in sorted(nfp_set, reverse=True):
            if _nfp_past >= today_d:
                continue
            if (today_d - _nfp_past).days > 5:
                break
            if _nfp_past in _MARKET_HOLIDAYS:
                _nfp_react = _nfp_past + timedelta(days=1)
                while _nfp_react.weekday() >= 5 or _nfp_react in _MARKET_HOLIDAYS:
                    _nfp_react += timedelta(days=1)
                if today_d == _nfp_react:
                    events_lines.insert(0,
                        f"📊 NFP printed {_nfp_past.strftime('%a %b %d')} (market closed) — "
                        f"first reaction session today; watch for gap at open"
                    )
            break

        # Quarter-end proximity: within 3 calendar days of Mar 31, Jun 30, Sep 30, Dec 31.
        # Hardcoded day thresholds (no calendar import needed): Mar/Dec=31-day months,
        # Jun/Sep=30-day months.
        _qe_thresholds = {3: 29, 6: 28, 9: 28, 12: 29}
        _qe_reported = False
        for _qoff in range(8):
            _qd = today_d + timedelta(days=_qoff)
            if _qe_reported:
                break
            if _qd.month not in _qe_thresholds or _qd.day < _qe_thresholds[_qd.month]:
                continue
            if _qd.weekday() >= 5 or _qd in _MARKET_HOLIDAYS:
                continue
            _qnum = _qd.month // 3
            if _qoff == 0:
                events_lines.append(
                    f"📅 Q{_qnum} quarter-end today — window dressing &amp; rebalancing flows; expect elevated vol"
                )
            elif _qoff == 1:
                events_lines.append(
                    f"📅 Q{_qnum} quarter-end tomorrow ({_qd.strftime('%a %b %d')})"
                    f" — position for window dressing / rebalancing"
                )
            else:
                events_lines.append(
                    f"📅 Q{_qnum} quarter-end in {_qoff}d ({_qd.strftime('%a %b %d')})"
                    f" — watch for rebalancing flows"
                )
            _qe_reported = True
        macro_line = ("\n".join(events_lines) if events_lines
                      else "✅ No macro events in next 7 days — clean tape")
    except Exception:
        macro_line = "✅ Macro check OK"

    # ── 3. Seasonal filter ────────────────────────────────────────────
    print("  [3/6] Checking seasonal status...")
    curr_month  = now_et.month
    month_name  = now_et.strftime("%B")
    if curr_month in SEASONAL_WEAK_MONTHS:
        seasonal_line = f"⚠️ {month_name} — weak month (min score raised to {SEASONAL_MIN_SCORE})"
    else:
        seasonal_line = f"✅ {month_name} — normal conditions (min score: {MIN_CONFLUENCE})"
    try:
        vix_f = float(vix)
        if vix_f > 25:
            seasonal_line += f"\n⚡ VIX {vix_f:.1f} > 25 — score also raised to 90"
    except Exception:
        pass

    # ── 3.5. Sector ETF health ────────────────────────────────────────
    print("  [3.5/7] Checking sector ETF health...")
    sector_health_section = ""
    try:
        _etf_rows = []
        _blocked_sectors = []
        for _sec_name, _etf_sym in SECTOR_ETFS.items():
            try:
                _etf_df = fetch_df(_etf_sym)
                if _etf_df is None or len(_etf_df) < 50:
                    continue
                _etf_df = compute_indicators(_etf_df.copy())
                if "EMA50" not in _etf_df.columns:
                    continue
                _etf_last = _etf_df.iloc[-1]
                _above = float(_etf_last["Close"]) > float(_etf_last["EMA50"])
                _pct   = (float(_etf_last["Close"]) - float(_etf_last["EMA50"])) / float(_etf_last["EMA50"]) * 100
                _icon  = "✅" if _above else "⚠️"
                _etf_rows.append(f"{_icon} {_etf_sym} ({_sec_name[:4]})")
                if not _above:
                    _blocked_sectors.append(_etf_sym)
            except Exception:
                continue
        if _etf_rows:
            # 3 ETFs per line for compact layout
            _lines = [" | ".join(_etf_rows[i:i+3]) for i in range(0, len(_etf_rows), 3)]
            if _blocked_sectors:
                _block_note = f"\n⚠️ {', '.join(_blocked_sectors)} below EMA50 — sector gate will block these plays"
            else:
                _block_note = "\n✅ All sectors above EMA50 — sector gate open for all plays"
            sector_health_section = (
                f"\n\n📡 <b>SECTOR ETF HEALTH</b> (above EMA50 = gate open)\n"
                + "\n".join(_lines)
                + _block_note
            )
    except Exception:
        pass

    # ── 4. Live outcomes ──────────────────────────────────────────────
    print("  [4/7] Reading live outcomes...")
    live_line = "No live outcome data yet — logger active, accumulating."
    suggestion_line = ""
    try:
        if os.path.exists(LIVE_OUTCOMES_FILE):
            df_live = pd.read_csv(LIVE_OUTCOMES_FILE)
            if not df_live.empty:
                total  = len(df_live)
                wr_all = (df_live["outcome"] == "WIN").mean() * 100
                last20 = df_live.tail(20)
                wr_20  = (last20["outcome"] == "WIN").mean() * 100
                wins   = (df_live["outcome"] == "WIN").sum()
                losses = (df_live["outcome"] == "LOSS").sum()
                avg_w  = df_live.loc[df_live["pnl_pct"] > 0,  "pnl_pct"].mean()
                avg_l  = df_live.loc[df_live["pnl_pct"] < 0,  "pnl_pct"].mean()
                pf     = (wins * avg_w) / (losses * abs(avg_l)) if losses > 0 and avg_l != 0 else 0
                live_line = (f"Total: {total} trades  |  WR: {wr_all:.1f}% (last 20: {wr_20:.1f}%)\n"
                             f"PF: {pf:.2f}x  |  Avg W: {avg_w:+.1f}%  Avg L: {avg_l:+.1f}%\n"
                             f"Backtest baseline: 76.0% WR | 6.10x PF | Sharpe 12.51")

                setup_lines = []
                for setup, grp in df_live.groupby("setup"):
                    g_wr = (grp["outcome"] == "WIN").mean() * 100
                    setup_lines.append(f"{setup}: {g_wr:.0f}% WR ({len(grp)} trades)")
                if setup_lines:
                    live_line += "\n" + "  ".join(setup_lines)

                # Alert if any setup's live WR drops >8 pts below current backtest baseline
                bt_baseline = {
                    "Gap & Hold":    76.0,
                    "Morning Runner": 75.0,
                    "Vol Breakdown":  50.0,
                }
                suggestions = []
                for setup, grp in df_live.groupby("setup"):
                    if len(grp) >= 8:
                        g_wr = (grp["outcome"] == "WIN").mean() * 100
                        bt   = bt_baseline.get(setup, 65.0)
                        if g_wr < bt - 8:
                            suggestions.append(
                                f"⚠️ {setup} live WR {g_wr:.0f}% vs backtest {bt:.0f}% — "
                                f"consider raising SETUP_MIN_CONFLUENCE to "
                                f"{SETUP_MIN_CONFLUENCE.get(setup, MIN_CONFLUENCE) + 3}"
                            )
                suggestion_line = ("\n\n💡 <b>CODE SUGGESTIONS</b>\n" + "\n".join(suggestions)
                                   if suggestions else "\n\n💡 <b>CODE SUGGESTIONS</b>: None — filters on track.")
    except Exception as e:
        live_line = f"Error reading live outcomes: {e}"

    # ── 5. Monthly P&L ────────────────────────────────────────────────
    print("  [5/7] Checking monthly P&L...")
    try:
        month_loss = get_this_month_loss()
        limit_pct  = MONTHLY_LOSS_LIMIT * 100
        if month_loss <= -limit_pct:
            monthly_line = f"🛑 MONTHLY LIMIT HIT: {month_loss:.1f}% — trading halted"
        elif month_loss < -(limit_pct * 0.6):
            monthly_line = f"⚠️ Down {abs(month_loss):.1f}% this month (limit: {limit_pct:.0f}%)"
        elif month_loss < 0:
            monthly_line = f"📉 Down {abs(month_loss):.1f}% this month (limit: {limit_pct:.0f}%)"
        else:
            monthly_line = f"📈 Up {month_loss:.1f}% this month"
    except Exception:
        monthly_line = "Monthly P&L: unavailable"

    # ── 5.5. Weekend open-position risk (Fridays only) ───────────────
    weekend_section = ""
    if now_et.weekday() == 4:  # Friday
        try:
            _pending_wk = []
            if os.path.exists(LIVE_SIGNALS_FILE):
                with open(LIVE_SIGNALS_FILE) as _fw:
                    _pending_wk = json.load(_fw).get("pending", [])
            # Find nearest upcoming FOMC for context
            _td_wk = now_et.date()
            _fomc_wk_str = ""
            for _ev_wk in sorted(_FOMC_DATES):
                _days_wk = (_ev_wk - _td_wk).days
                if 1 <= _days_wk <= 7:
                    _fomc_wk_str = f" FOMC {_ev_wk.strftime('%a %b %d')} in {_days_wk}d."
                    break
                if _days_wk > 7:
                    break
            if _pending_wk:
                _open_tickers_wk = ", ".join(p.get("ticker", "?") for p in _pending_wk)
                weekend_section = (
                    f"\n\n🏖 <b>WEEKEND RISK — {len(_pending_wk)} open position(s)</b>\n"
                    f"Pending: <b>{_open_tickers_wk}</b>\n"
                    f"These carry over the weekend.{_fomc_wk_str} "
                    f"Plan your exit before close — hit T1 or tighten stop."
                )
            else:
                weekend_section = (
                    f"\n\n🏖 <b>Friday</b> — no open positions.{_fomc_wk_str} "
                    f"Clean slate heading into the weekend."
                )
        except Exception:
            pass

    # ── 6. Pre-market gap scanner ─────────────────────────────────────
    print("  [6/7] Scanning pre-market gaps...")
    gap_lines      = []
    near_gap_lines: list[tuple[float, str]] = []  # 1.0–1.5% READY — near-threshold watch
    try:
        for ticker in WATCHLIST:
            try:
                info       = yf.Ticker(ticker).fast_info
                pre_px     = float(info.last_price or 0)
                prev_close = float(info.previous_close or 0)
                if pre_px <= 0 or prev_close <= 0:
                    continue
                gap_pct = (pre_px - prev_close) / prev_close * 100
                if gap_pct < 1.0:   # capture near-threshold (1.0-1.5%) too
                    continue
                est_stop = round(pre_px * 0.985, 2)
                vol_tag  = " ⚡" if ticker in VOLATILE_TICKERS else ""

                # Technical readiness: check the two primary Gap & Hold filters
                # (MACD > 0 and prior-day green) so the user knows at 8 AM which
                # gaps are real candidates vs. likely to fade at the open.
                tech_label = ""
                entry_note = f"→ Watch entry near open  |  Est. stop ~${est_stop}"
                _macd_ok   = False  # initialise so near-threshold routing can see them
                _prior_grn = False
                try:
                    _gdf = fetch_df(ticker, period_days=50)
                    if _gdf is not None and len(_gdf) >= 30:
                        _gdf       = compute_indicators(_gdf)
                        _macd_ok   = bool(_gdf["MACD"].iloc[-1] > 0)
                        _prior_grn = bool(_gdf["Close"].iloc[-2] > _gdf["Open"].iloc[-2])
                        if _macd_ok and _prior_grn:
                            tech_label = " ✅ <b>READY</b> (MACD+ ✓ prior green ✓)"
                            entry_note = f"→ <b>Gap &amp; Hold candidate</b> — confirm hold at 9:45 AM  |  Stop ~${est_stop}"
                            # MACD just crossed positive but still near zero — recently crossed,
                            # still fragile; one bad session flips it back negative.
                            _macd_raw = float(_gdf["MACD"].iloc[-1])
                            _cls_px   = float(_gdf["Close"].iloc[-1])
                            if _cls_px > 0 and (_macd_raw / _cls_px * 100) < 0.5:
                                _macd_pct = _macd_raw / _cls_px * 100
                                tech_label += f" ⚡ <b>MACD near-zero</b> ({_macd_pct:+.2f}%)"
                                entry_note = (
                                    f"→ <b>Gap &amp; Hold candidate</b> — MACD just crossed, still fragile; "
                                    f"9:45 confirm is critical  |  Stop ~${est_stop}"
                                )
                        elif _macd_ok:
                            tech_label = " ⚠️ <b>PARTIAL</b> (MACD+ ✓ prior RED ✗)"
                            entry_note = f"→ Prior-green filter may block — monitor  |  Stop ~${est_stop}"
                        elif _prior_grn:
                            tech_label = " ⚠️ <b>PARTIAL</b> (MACD- ✗ prior green ✓)"
                            entry_note = f"→ MACD filter may block — monitor  |  Stop ~${est_stop}"
                            # MACD within 0.5% of price = essentially at zero, crossover imminent
                            _macd_raw = float(_gdf["MACD"].iloc[-1])
                            _cls_px   = float(_gdf["Close"].iloc[-1])
                            if _cls_px > 0 and (_macd_raw / _cls_px * 100) > -0.5:
                                _macd_pct = _macd_raw / _cls_px * 100
                                tech_label += f" 🔄 <b>MACD near-zero</b> ({_macd_pct:.2f}%)"
                                entry_note = (
                                    f"→ MACD {_macd_raw:.1f} ({_macd_pct:.2f}%) — crossover imminent; "
                                    f"may turn READY by open  |  Stop ~${est_stop}"
                                )
                        else:
                            tech_label = " ❌ <b>NOT READY</b> (MACD- ✗ prior RED ✗)"
                            entry_note = f"→ Likely to fade — do not chase  |  Stop ~${est_stop}"
                        # Fade-risk override: large pre-mkt gap after a heavy prior session.
                        # Stocks down >4% the prior day often bounce pre-market and then
                        # reverse hard at the 9:30 regular-session open (ghost-gap pattern).
                        if len(_gdf) >= 3 and gap_pct >= 3.0:
                            _prior_net = (
                                (float(_gdf["Close"].iloc[-2]) - float(_gdf["Close"].iloc[-3]))
                                / float(_gdf["Close"].iloc[-3]) * 100
                            )
                            if _prior_net <= -4.0:
                                tech_label += " 🔴 <b>FADE RISK</b>"
                                entry_note = (
                                    f"→ ⚠️ Prior session {_prior_net:.1f}% — pre-mkt bounce"
                                    f" may reverse at 9:30 open"
                                    f"  |  Wait for 9:45 confirm before acting"
                                    f"  |  Stop ~${est_stop}"
                                )
                except Exception:
                    pass

                if gap_pct >= 1.5:
                    gap_lines.append(
                        (gap_pct, f"🔥 <b>{ticker}</b>{vol_tag}  +{gap_pct:.1f}%  "
                         f"pre-mkt ${pre_px:.2f}{tech_label}\n"
                         f"   {entry_note}")
                    )
                elif _macd_ok and _prior_grn:
                    # READY but below 1.5% gate — show as near-threshold watch only
                    near_gap_lines.append(
                        (gap_pct, f"<b>{ticker}</b>{vol_tag} +{gap_pct:.1f}% ${pre_px:.2f}")
                    )
            except Exception:
                continue
        gap_lines.sort(key=lambda x: x[0], reverse=True)
        near_gap_lines.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        pass

    # ── 6b. Small-cap pre-market movers ──────────────────────────────────
    # Scan DMAN_SMALLCAP_WATCHLIST for pre-market moves ≥5%.
    # Lower threshold than large-caps — penny stocks move 20-100% before open.
    sc_gap_lines: list[tuple[float, str]] = []
    try:
        _sc_scan_list = list(dict.fromkeys(DMAN_SMALLCAP_WATCHLIST))
        for _sc_t in _sc_scan_list:
            try:
                _sc_info   = yf.Ticker(_sc_t).fast_info
                _sc_pm     = float(_sc_info.last_price or 0)
                _sc_prev   = float(_sc_info.previous_close or 0)
                if _sc_pm <= 0 or _sc_prev <= 0:
                    continue
                _sc_gap = (_sc_pm - _sc_prev) / _sc_prev * 100
                if abs(_sc_gap) < 5.0:   # only surface meaningful moves
                    continue
                _sc_fl_m, _sc_si, _, _ = _get_short_float_data(_sc_t)
                _float_str = f"  Float {_sc_fl_m:.2f}M" if _sc_fl_m > 0 else ""
                _si_str    = f" | SI {_sc_si:.0f}%" if _sc_si >= 15 else ""
                _dir = "🚀" if _sc_gap > 0 else "📉"
                _moon = " 🌙 MOON SHOT WATCH" if (_sc_gap >= 15 and _sc_fl_m > 0 and _sc_fl_m < 2.0) else ""
                sc_gap_lines.append(
                    (_sc_gap, f"{_dir} <b>{_sc_t}</b>  {_sc_gap:+.1f}%  "
                     f"pre-mkt ${_sc_pm:.4f}{_float_str}{_si_str}{_moon}")
                )
            except Exception:
                continue
        sc_gap_lines.sort(key=lambda x: abs(x[0]), reverse=True)
    except Exception:
        pass

    if gap_lines:
        gap_section = (f"\n\n🚨 <b>PRE-MARKET GAPS ≥1.5%</b> — entry setups forming\n"
                       + "\n".join(line for _, line in gap_lines[:5])
                       + "\n<i>✅ = passes MACD + prior-green filters. Confirm hold at 9:45 AM.</i>")
    else:
        gap_section = "\n\n📡 <b>PRE-MARKET</b>: No significant gaps right now."
    if near_gap_lines:
        gap_section += (
            "\n\n👀 <b>Near-threshold READY</b> (1.0–1.5% gap, MACD+ &amp; prior green): "
            + " | ".join(line for _, line in near_gap_lines[:4])
            + "\n<i>Below 1.5% scanner gate — watch if gap expands to the open.</i>"
        )
    if sc_gap_lines:
        gap_section += (
            "\n\n⚡ <b>SMALL-CAP PRE-MARKET MOVERS</b> (≥5% pre-mkt):\n"
            + "\n".join(line for _, line in sc_gap_lines[:6])
            + "\n<i>Low-float catalyst plays — confirm RVOL at open before entry.</i>"
        )

    # ── 6.5. Upcoming earnings — large-cap watchlist AND DMan's small-cap
    # watchlist. Small-caps were previously excluded from this check
    # entirely, despite being where DMan's actual edge lives (ultra-low-
    # float, catalyst-driven names move far more violently on an earnings
    # surprise than a mega-cap does) — labeled separately below since the
    # two carry very different risk/opportunity character, not merged
    # indiscriminately into one undifferentiated list.
    print("  [6.5/7] Scanning earnings calendar...")  # noqa: label kept for operator readability
    earnings_section = ""
    try:
        upcoming    = get_upcoming_earnings(WATCHLIST, days_ahead=5)
        upcoming_sc = get_upcoming_earnings(DMAN_SMALLCAP_WATCHLIST, days_ahead=5)

        def _fmt_earnings_line(item: dict) -> str:
            da = item["days_away"]
            tag = "today" if da == 0 else ("tomorrow" if da == 1 else f"in {da}d")
            in_bl = "" if da > EARNINGS_BLACKOUT else " 🚫 BLACKOUT"
            return f"  {item['ticker']:<6} — {item['earn_date'].strftime('%a %b %d')} ({tag}){in_bl}"

        if upcoming:
            earnings_section += (
                "\n\n📆 <b>EARNINGS THIS WEEK (watchlist)</b>\n"
                + "\n".join(_fmt_earnings_line(i) for i in upcoming)
                + "\n<i>Tickers marked BLACKOUT are skipped until 5d after report</i>"
            )
        else:
            earnings_section += "\n\n📆 <b>EARNINGS</b>: No watchlist tickers report in next 5 days."

        if upcoming_sc:
            earnings_section += (
                "\n\n⚡ <b>EARNINGS THIS WEEK (DMan small-cap watch)</b>\n"
                + "\n".join(_fmt_earnings_line(i) for i in upcoming_sc)
                + "\n<i>Low-float names move far harder on a surprise — size accordingly</i>"
            )
    except Exception as _e:
        earnings_section = f"\n\n📆 <b>EARNINGS</b>: scan error ({str(_e)[:60]})"

    # ── PDT budget + account milestone tracker ────────────────────────
    _pdt_section     = ""
    _milestone_section = ""
    try:
        _pdt_live = _get_pdt_status()
        _live_eq  = _pdt_live["equity"]
        _dt_used  = _pdt_live["used"]
        _dt_rem   = _pdt_live["remaining"]
        _sw_on    = _pdt_live["swing_mode"]

        if _live_eq > 0:
            # PDT budget line
            if _live_eq >= 25_000:
                _pdt_section = "\n\n🔓 <b>PDT</b>: Unlimited day trades (equity ≥ $25k)"
            elif _dt_rem == 0:
                _pdt_section = (f"\n\n🚫 <b>PDT HALT</b>: {_dt_used}/3 day trades used — "
                                "window resets Monday. No new day trades until reset.")
            elif _sw_on:
                _pdt_section = (f"\n\n🔄 <b>PDT — SWING MODE</b>: {_dt_used}/3 used · "
                                f"1 remaining · New entries will be GTC swings (overnight)")
            else:
                _pdt_section = (f"\n\n🎯 <b>PDT</b>: {_dt_used}/3 day trades used · "
                                f"{_dt_rem} remaining this window")

            # Growth milestone tracker
            _MILESTONES = [2_000, 5_000, 10_000, 25_000]
            _next_ms = next((m for m in _MILESTONES if m > _live_eq), None)
            if _next_ms:
                _ms_pct   = _live_eq / _next_ms * 100
                _ms_gap   = _next_ms - _live_eq
                _bar_fill = int(_ms_pct / 10)   # 0-10 blocks
                _bar      = "█" * _bar_fill + "░" * (10 - _bar_fill)
                _pdt_flag = "  ← PDT UNLOCK 🔓" if _next_ms == 25_000 else ""
                _milestone_section = (
                    f"\n\n📈 <b>ACCOUNT GROWTH</b>\n"
                    f"${_live_eq:,.2f}  →  ${_next_ms:,.0f}{_pdt_flag}\n"
                    f"[{_bar}] {_ms_pct:.1f}%  (${_ms_gap:,.0f} to go)"
                )
            else:
                _milestone_section = (
                    f"\n\n📈 <b>ACCOUNT</b>: ${_live_eq:,.2f} — all milestones cleared 🏆"
                )
    except Exception:
        pass

    # ── Format & send ─────────────────────────────────────────────────
    msg = (
        f"🌅 <b>DMan PRO Pre-Market Briefing</b>\n"
        f"{date_str} — 7:10 AM MT\n"
        + (f"{scanner_health_line}\n\n" if scanner_health_line else "\n")
        + f"📊 <b>MARKET REGIME</b>\n{regime_line}\n{regime_line2}"
        f"{warnings_section}"
        f"{macro_env_section}"
        f"{_global_ctx_section}\n\n"
        f"📅 <b>MACRO CALENDAR</b>\n{macro_line}\n\n"
        f"🌡 <b>SEASONAL FILTER</b>\n{seasonal_line}"
        f"{sector_health_section}\n\n"
        f"💰 <b>MONTHLY P&amp;L</b>\n{monthly_line}"
        f"{_pdt_section}"
        f"{_milestone_section}"
        f"{weekend_section}"
        f"{gap_section}"
        f"{earnings_section}"
        f"{suggestion_line}"
    )

    print(f"\n{'═'*60}")
    print(f"  PRE-MARKET BRIEFING  —  {date_str}")
    print(f"{'─'*60}")
    print(f"  Regime   : {regime_line}")
    print(f"  Macro    : {macro_line}")
    print(f"  Seasonal : {seasonal_line}")
    print(f"  Monthly  : {monthly_line}")
    print(f"{'═'*60}\n")

    sent = send_telegram(msg)
    if sent:
        print("  ✅ Telegram briefing sent.")
    else:
        print("  ⚠️  Telegram not configured — briefing printed above only.")

    # ── Pre-event strangle advisory ───────────────────────────────────
    # Triggers (buy before IV crush, catch the move):
    #   FOMC same-day (2 PM announcement — premarket still pre-event)
    #   FOMC tomorrow or within 3 days — strangle while premium is still building
    #   CPI / NFP / PPI tomorrow (8:30 AM releases)
    #   OPEX tomorrow (3rd Friday) — gamma explosion event
    print("  [7/7] Checking for pre-event strangle opportunities...")
    try:
        _today    = now_et.date()
        _tomorrow = _today + timedelta(days=1)
        _nfp      = _nfp_dates()
        strangle_events = []

        # FOMC: same-day or tomorrow = immediate; 2-3 days out = early premium entry
        if _today in _FOMC_DATES:
            strangle_events.append("FOMC today 12 PM MT")
        if _tomorrow in _FOMC_DATES:
            strangle_events.append("FOMC tomorrow 12 PM MT")
        else:
            for _off in range(2, 4):  # 2 or 3 days out
                _fd = _today + timedelta(days=_off)
                if _fd in _FOMC_DATES:
                    strangle_events.append(
                        f"FOMC in {_off}d ({_fd.strftime('%a %b %d')}) — enter strangle early")
                    break

        # Data releases the next morning
        if _tomorrow in _CPI_DATES:
            strangle_events.append("CPI tomorrow 6:30 AM MT")
        if _tomorrow in _PPI_DATES:
            strangle_events.append("PPI tomorrow 6:30 AM MT")
        if _tomorrow in _nfp:
            strangle_events.append("NFP tomorrow 6:30 AM MT")

        # OPEX eve: tomorrow is the 3rd Friday — gamma explosion
        if _tomorrow.weekday() == 4 and _tomorrow == _get_third_friday(_tomorrow.year, _tomorrow.month):
            strangle_events.append("OPEX tomorrow (3rd Friday) — gamma event; SPY/QQQ strangle")

        if strangle_events:
            generate_strangle_advisory(" | ".join(strangle_events))
        else:
            print("  No catalyst tomorrow — skipping strangle advisory.")
    except Exception as _e:
        print(f"  [strangle] advisory error: {_e}", file=sys.stderr)

    # ── Pre-build dynamic universe for 9:45 AM Gap & Hold scan ────────────
    # The 9:45 scan runs --universe curated (fast), which normally limits it
    # to the ~15-name WATCHLIST and misses any gapper outside that list.
    # Building and caching the full 500-ticker RVOL universe here (at 9 AM)
    # lets the 9:45 scan load it for free — no rebuild time at the gate.
    print("\n  [8/8] Pre-building dynamic scan universe for 9:45 AM gate...")
    try:
        _prebuilt = build_scan_universe()
        _cache_payload = {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "tickers": _prebuilt,
        }
        with open("dman_universe_cache.json", "w") as _ucf:
            json.dump(_cache_payload, _ucf)
        print(f"  ✅ Universe cached: {len(_prebuilt)} tickers → dman_universe_cache.json "
              f"(9:45 AM scan will use this instead of {len(WATCHLIST)}-name list)")
    except Exception as _ue:
        print(f"  ⚠️  Universe pre-build failed ({_ue}) — 9:45 scan will fall back to curated list")


def print_live_performance() -> None:
    """Print live-trade WR stats from the outcomes CSV."""
    if not os.path.exists(LIVE_OUTCOMES_FILE):
        print("\n  No live outcome data yet. Signals log to dman_live_signals.json")
        print("  and resolve automatically at the next scan after market close.\n")
        return

    try:
        df = pd.read_csv(LIVE_OUTCOMES_FILE)
    except Exception as e:
        print(f"  Error reading {LIVE_OUTCOMES_FILE}: {e}")
        return

    if df.empty:
        print("  Live outcomes file is empty.")
        return

    total  = len(df)
    wins   = (df["outcome"] == "WIN").sum()
    losses = (df["outcome"] == "LOSS").sum()
    wr     = wins / total * 100 if total else 0
    avg_w  = df.loc[df["pnl_pct"] > 0, "pnl_pct"].mean() if wins else 0
    avg_l  = df.loc[df["pnl_pct"] < 0, "pnl_pct"].mean() if losses else 0
    pf     = (wins * avg_w) / (losses * abs(avg_l)) if losses > 0 and avg_l != 0 else float("inf")

    print(f"\n{'═'*60}")
    print(f"  LIVE TRADE OUTCOMES — Ground-Truth WR")
    print(f"{'─'*60}")
    print(f"  Total Trades   : {total}")
    print(f"  Win Rate       : {wr:.1f}%  ({wins}W / {losses}L)")
    print(f"  Profit Factor  : {pf:.2f}x")
    print(f"  Avg Win %      : {avg_w:+.2f}%")
    print(f"  Avg Loss %     : {avg_l:+.2f}%")
    print(f"{'─'*60}")

    if "setup" in df.columns:
        print(f"  BY SETUP:")
        for setup, grp in df.groupby("setup"):
            g_wr = (grp["outcome"] == "WIN").mean() * 100
            g_avg = grp["pnl_pct"].mean()
            print(f"  {setup:<20} {len(grp):3d} trades | {g_wr:5.1f}% WR | avg {g_avg:+.2f}%")

    if "exit_reason" in df.columns:
        print(f"\n  EXIT BREAKDOWN:")
        for reason, grp in df.groupby("exit_reason"):
            g_wr = (grp["outcome"] == "WIN").mean() * 100
            print(f"  {reason:<10} {len(grp):3d} ({len(grp)/total*100:.0f}%)  {g_wr:.0f}% WR")

    # Backtest comparison line
    print(f"\n  Backtest baseline : 60.0% WR | 2.43x PF (115 trades, 2yr walk-forward)")
    print(f"  Live delta        : WR {wr - 60.0:+.1f}%  |  PF {pf - 2.43:+.2f}x")
    print(f"{'═'*60}\n")


def _sector_etf_above_ema50(ticker: str) -> bool:
    """Return True if the ticker's sector ETF is above its 50-day EMA (bullish sector trend)."""
    sector = TICKER_SECTOR.get(ticker, "")
    etf    = SECTOR_ETFS.get(sector, "")
    if not etf:
        return True  # no sector mapping — don't block the signal
    try:
        etf_df = fetch_df(etf)
        if etf_df is None or len(etf_df) < 50:
            return True
        etf_df = compute_indicators(etf_df.copy())
        if "EMA50" not in etf_df.columns:
            return True
        last = etf_df.iloc[-1]
        return float(last["Close"]) > float(last["EMA50"])
    except Exception:
        return True  # on any error, don't block


_float_cache: dict[str, tuple[float, float, float, float]] = {}  # → (float_M, short_pct, insider_pct, cash_to_mc)

def _get_short_float_data(ticker: str) -> tuple[float, float, float, float]:
    """
    Return (float_M, short_pct, insider_pct, cash_to_mc_ratio) for a ticker.
    All four pulled from one yf.Ticker.info call (cached per session).

    cash_to_mc_ratio: totalCash / marketCap.
        >= 1.0 = trading below cash value (Dman: "undervalue, trading under cash")
        >= 0.5 = substantial cash cushion vs market cap
    Returns (0, 0, 0, 0) on failure.
    """
    if ticker in _float_cache:
        return _float_cache[ticker]
    try:
        info         = yf.Ticker(ticker).info
        float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 0
        short_pct    = info.get("shortPercentOfFloat") or 0.0
        insider_pct  = info.get("heldPercentInsiders") or 0.0
        total_cash   = info.get("totalCash") or 0
        market_cap   = info.get("marketCap") or 0
        cash_to_mc   = float(total_cash) / float(market_cap) if market_cap > 0 else 0.0
        result = (float_shares / 1_000_000, float(short_pct) * 100,
                  float(insider_pct) * 100, cash_to_mc)
    except Exception:
        result = (0.0, 0.0, 0.0, 0.0)
    _float_cache[ticker] = result
    return result


def _supertrend_bull(h_arr, l_arr, c_arr, period: int = 10, mult: float = 3.0) -> np.ndarray:
    """Returns bool array: True where price is above Supertrend (bullish)."""
    n = len(c_arr)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(h_arr[i]-l_arr[i], abs(h_arr[i]-c_arr[i-1]), abs(l_arr[i]-c_arr[i-1]))
    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = float(np.nanmean(tr[1:period+1]))
        for i in range(period+1, n):
            atr[i] = (atr[i-1]*(period-1) + tr[i]) / period
    hl2 = (h_arr + l_arr) / 2.0
    bu = hl2 + mult*atr;  bl = hl2 - mult*atr
    fu = np.full(n, np.nan);  fl = np.full(n, np.nan)
    bull = np.ones(n, dtype=bool)
    for i in range(period, n):
        fu[i] = bu[i] if i==period or bu[i] < fu[i-1] or c_arr[i-1] > fu[i-1] else fu[i-1]
        fl[i] = bl[i] if i==period or bl[i] > fl[i-1] or c_arr[i-1] < fl[i-1] else fl[i-1]
        bull[i] = (c_arr[i] > fl[i]) if (i > period and bull[i-1]) else (c_arr[i] > fu[i])
    return bull


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Full indicator suite on daily OHLCV data."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # Moving averages
    for span, name in [(9,"EMA9"),(20,"EMA20"),(50,"EMA50")]:
        df[name] = c.ewm(span=span, adjust=False).mean()
    df["SMA200"] = c.rolling(200).mean()

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # MACD (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["MACD_sig"]  = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]

    # Bollinger Bands
    bb_mid        = c.rolling(20).mean()
    bb_std        = c.rolling(20).std()
    df["BB_upper"] = bb_mid + 2*bb_std
    df["BB_lower"] = bb_mid - 2*bb_std
    df["BB_pct"]   = (c - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # ATR (14)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ADX (14) — trend strength
    plus_dm  = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    # Wilder: zero the smaller directional move on each bar (both evaluated on originals)
    plus_dm, minus_dm = (plus_dm.where(plus_dm >= minus_dm, 0.0),
                         minus_dm.where(minus_dm > plus_dm, 0.0))
    atr14    = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean()  / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"]      = dx.rolling(14).mean()
    df["PLUS_DI"]  = plus_di
    df["MINUS_DI"] = minus_di

    # Stochastic (14,3)
    lo14 = l.rolling(14).min(); hi14 = h.rolling(14).max()
    df["STOCH_K"] = 100*(c-lo14)/(hi14-lo14+1e-9)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    # Volume
    df["AvgVol20"] = v.rolling(20).mean()
    df["RVOL"]     = v / df["AvgVol20"].replace(0, np.nan)
    df["OBV"]      = (np.sign(c.diff()) * v).cumsum()

    # Price change
    df["Chg1d"]  = c.pct_change(1)*100
    df["Chg5d"]  = c.pct_change(5)*100
    df["Chg20d"] = c.pct_change(20)*100

    # Supertrend (period=10, multiplier=3.0)
    df["ST_bull"] = _supertrend_bull(h.values, l.values, c.values)

    return df


def _compute_indicators_cached(ticker: str, raw: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
    """
    compute_indicators(), cached per ticker for the same lifetime as
    fetch_df()'s own _cache (identical key convention, cleared at the
    same point). Found in the 2026-08-16 review: the intended cache-reuse
    guard at several call sites ("MACD" not in df.columns) never actually
    tripped, because fetch_df() only ever caches the RAW (pre-indicator)
    dataframe — compute_indicators() was always called fresh on a
    .copy() of it. A ticker appearing in both the main scan universe and
    a later pass (small-cap catalyst, momentum-watch radar) got its
    entire indicator suite — including a pure-Python triple loop
    (_supertrend_bull) — recomputed a second time, every scan and every
    fast-trigger, for ~200 tickers.

    Always returns a fresh .copy() of the cached result, cache hit or
    miss: several callers do an in-place dropna() on what this returns
    (e.g. df.dropna(..., inplace=True)), which would otherwise silently
    corrupt the shared cached dataframe for every later caller in the
    same scan pass. The copy costs far less than recomputing the whole
    indicator suite, so the cache still nets a large win.
    """
    key = f"{ticker}_{interval}"
    if key not in _indicator_cache:
        _indicator_cache[key] = compute_indicators(raw.copy())
    return _indicator_cache[key].copy()


def compute_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute key indicators on weekly bars for MTF analysis."""
    c = df["Close"]
    df["W_EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["W_EMA50"] = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["W_RSI"]   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["W_MACD"]     = ema12 - ema26
    df["W_MACD_sig"] = df["W_MACD"].ewm(span=9, adjust=False).mean()
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3.5 — PATTERN HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def detect_candle_pattern(r, p, bias: str) -> tuple[str, int]:
    """Returns (pattern_name, score 0-8). r/p are the last two DataFrame rows."""
    try:
        o, h, l, c = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
        full_range = h - l
        if full_range < 1e-6:
            return "", 0
        body        = abs(c - o)
        upper_wick  = h - max(c, o)
        lower_wick  = min(c, o) - l
        body_pct    = body / full_range
        p_body      = abs(float(p["Close"]) - float(p["Open"]))

        if bias == "LONG":
            if body > 0 and lower_wick >= 2*body and upper_wick <= body*0.5 and c > o:
                return "Hammer", 8
            if (c > o and float(p["Close"]) < float(p["Open"]) and body > p_body
                    and o <= float(p["Close"]) and c >= float(p["Open"])):
                return "Bull Engulf", 8
            if body_pct < 0.1 and lower_wick > upper_wick * 1.5:
                return "Doji Support", 4
            if (c - l) / full_range > 0.75 and body_pct > 0.4:
                return "Strong Close", 5
        else:
            if body > 0 and upper_wick >= 2*body and lower_wick <= body*0.5 and c < o:
                return "Shoot Star", 8
            if (c < o and float(p["Close"]) > float(p["Open"]) and body > p_body
                    and o >= float(p["Close"]) and c <= float(p["Open"])):
                return "Bear Engulf", 8
            if (h - c) / full_range > 0.75 and body_pct > 0.4:
                return "Weak Close", 5
    except Exception:
        pass
    return "", 0


def detect_vcp(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Volatility Contraction Pattern (Minervini): stock within 15% of 52-week high
    with three progressively tighter 20-bar ranges and drying volume.
    """
    try:
        if len(df) < 60:
            return False, ""
        hi52  = float(df["High"].iloc[-252:].max()) if len(df) >= 252 else float(df["High"].max())
        close = float(df["Close"].iloc[-1])
        if (hi52 - close) / hi52 > 0.15:
            return False, ""
        w = df.iloc[-60:]
        def _rng(sl: pd.DataFrame) -> float:
            mn = float(sl["Close"].mean())
            return (float(sl["High"].max()) - float(sl["Low"].min())) / mn if mn > 0 else 1.0
        r1, r2, r3 = _rng(w.iloc[:20]), _rng(w.iloc[20:40]), _rng(w.iloc[40:])
        vol_mid  = float(w.iloc[20:40]["Volume"].mean())
        vol_late = float(w.iloc[40:]["Volume"].mean())
        if r1 > r2 > r3 and vol_late < vol_mid * 0.85 and r3 < 0.08:
            off_hi = (hi52 - close) / hi52 * 100
            vol_drop = (1 - vol_late / vol_mid) * 100 if vol_mid > 0 else 0
            return True, (f"VCP -{off_hi:.1f}% from 52wk hi | "
                          f"ranges {r1*100:.1f}%→{r2*100:.1f}%→{r3*100:.1f}% | "
                          f"vol -{vol_drop:.0f}%")
    except Exception:
        pass
    return False, ""


def get_regime_from_window(spy_window: pd.DataFrame) -> dict:
    """Compute market regime from a historical SPY slice — no live API call."""
    _default = {"regime": "CHOP", "score": 7, "vix_ok": True, "spy_trend": True}
    try:
        if spy_window is None or len(spy_window) < 50:
            return _default
        spy = compute_indicators(spy_window.copy())
        sr  = spy.iloc[-1]
        above20  = float(sr["Close"]) > float(sr["EMA20"])
        above50  = float(sr["Close"]) > float(sr["EMA50"])
        above200 = float(sr["Close"]) > float(sr["SMA200"])
        adx_val  = float(sr["ADX"]) if not pd.isna(sr["ADX"]) else 0
        bull_di  = float(sr["PLUS_DI"]) > float(sr["MINUS_DI"])
        score = ((4 if above20 else 0) + (4 if above50 else 0)
                 + (3 if above200 else 0) + (2 if adx_val >= 20 else 0) + 1)
        # Mirror the 4 bonus points from live get_market_regime() so backtest/live scores align
        try:
            _iwm_col = "IWM" if "IWM" in spy_window.columns else None
            if _iwm_col and len(spy_window) >= 22:
                _iwm_ret = (float(spy_window[_iwm_col].iloc[-1]) / float(spy_window[_iwm_col].iloc[-21]) - 1) * 100
                _spy_ret = (float(spy_window["Close"].iloc[-1])  / float(spy_window["Close"].iloc[-21])  - 1) * 100
                if _iwm_ret > _spy_ret - 5:
                    score += 1  # IWM breadth
        except Exception:
            pass
        if score >= 10 and bull_di:
            regime = "BULL"
        elif score <= 4 or not bull_di:
            regime = "BEAR"
        else:
            regime = "CHOP"
        return {"regime": regime, "score": score, "vix_ok": True, "spy_trend": above50}
    except Exception:
        return _default


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SIGNAL DATACLASS (enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ProSignal:
    # Core fields
    ticker:     str
    setup:      str
    bias:       str          # "LONG" | "SHORT"
    entry:      float
    stop:       float
    target1:    float        # 2R
    target2:    float        # 3R
    rr:         float
    rsi:        float
    rvol:       float
    reason:     str
    date:       str = field(default_factory=lambda: datetime.today().strftime("%Y-%m-%d"))

    # Position sizing
    shares:         int   = 0
    cost:           float = 0.0
    risk_usd:       float = 0.0
    kelly_frac:     float = 0.0
    vix_adj:        float = 1.0    # VIX size multiplier applied (1.0 = no reduction)
    float_rotation: float = 0.0   # small-cap only: today_vol / float_shares
    target3:        float = 0.0   # Moon Shot T3 at +100% (2x entry) — ultra-low float only
    is_moonshot:    bool  = False  # True when Moon Shot tier conditions are met
    swing_mode:     bool  = False  # True when PDT budget ≤ 1 → GTC entry + stop only, no T1 TP
    news_boost:     bool  = False  # True when ticker has a news headline in the last 4 hours

    # Scoring
    confluence_score:  int  = 0   # 0-100
    ai_score:          int  = 0   # 0-10 from Claude
    final_score:       float = 0.0  # weighted composite

    # Filter verdicts
    regime_ok:   bool = False
    mtf_ok:      bool = False
    rs_ok:       bool = False
    sector_ok:   bool = False
    earnings_ok: bool = False
    macro_ok:    bool = True
    fib_ok:      bool = False
    divergence_free: bool = True
    candle_pattern:  str  = ""

    # Breakdown
    score_breakdown: dict = field(default_factory=dict)

    # Market context (populated in score_signal)
    atr:  float = 0.0
    beta: float = 1.0

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    def passed_all_gates(self) -> bool:
        return (self.regime_ok and self.mtf_ok and self.rs_ok
                and self.sector_ok and self.earnings_ok and self.macro_ok
                and self.divergence_free
                and self.confluence_score >= MIN_CONFLUENCE)

    def summary(self) -> str:
        gates = {
            "Regime": self.regime_ok, "MTF": self.mtf_ok,
            "RS": self.rs_ok, "Sector": self.sector_ok,
            "Earnings": self.earnings_ok, "Macro": self.macro_ok,
            "NoDiverg": self.divergence_free,
        }
        passed = [k for k,v in gates.items() if v]
        failed = [k for k,v in gates.items() if not v]
        return (f"Score:{self.confluence_score}/100  "
                f"AI:{self.ai_score}/10  "
                f"Pass:[{','.join(passed)}]  "
                f"Fail:[{','.join(failed)}]")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FILTER 01: MARKET REGIME
# ═══════════════════════════════════════════════════════════════════════════

def get_market_regime() -> dict:
    """
    Classify the overall market as BULL, BEAR, or CHOP using:
      • SPY price vs EMA20 / EMA50 / SMA200
      • QQQ relative to SPY (tech leadership)
      • ADX (trend strength) — CHOP if ADX < 20
      • VIX level (fear gauge)

    Returns dict with regime, score (0-19), and details.
    """
    result = {
        "regime":    "UNKNOWN",
        "score":     0,
        "spy_trend": False,
        "adx_strong":False,
        "vix_ok":    False,
        "details":   {},
    }

    try:
        # SPY analysis
        spy_df = fetch_df("SPY")
        if spy_df is None:
            return result
        spy = compute_indicators(spy_df.copy())
        spy = spy.dropna(subset=["Close"])   # drop incomplete after-hours row
        if len(spy) == 0:
            return result
        sr  = spy.iloc[-1]

        spy_above_20  = float(sr["Close"]) > float(sr["EMA20"])
        spy_above_50  = float(sr["Close"]) > float(sr["EMA50"])
        spy_above_200 = float(sr["Close"]) > float(sr["SMA200"])
        adx_val       = float(sr["ADX"]) if not pd.isna(sr["ADX"]) else 0
        adx_strong    = adx_val >= 20
        bull_di       = float(sr["PLUS_DI"]) > float(sr["MINUS_DI"])

        # VIX
        vix_df  = fetch_df("^VIX")
        vix_val = float(vix_df["Close"].iloc[-1]) if vix_df is not None else 25
        vix_low = vix_val < 20          # calm market
        vix_mid = vix_val < 30          # manageable

        # Scoring
        score = 0
        if spy_above_20:   score += 4
        if spy_above_50:   score += 4
        if spy_above_200:  score += 3
        if adx_strong:     score += 2
        if vix_low:        score += 2
        elif vix_mid:      score += 1

        # IWM breadth: small-caps must not be severely lagging SPY (narrowing rally warning)
        iwm_df = fetch_df("IWM")
        breadth_note = "N/A"
        if iwm_df is not None and len(iwm_df) >= 22:
            _iwm_clean = iwm_df.dropna(subset=["Close"])
            _spy_clean = spy_df.dropna(subset=["Close"])
            iwm_ret = (float(_iwm_clean["Close"].iloc[-1]) / float(_iwm_clean["Close"].iloc[-21]) - 1) * 100
            spy_ret = (float(_spy_clean["Close"].iloc[-1]) / float(_spy_clean["Close"].iloc[-21]) - 1) * 100
            if iwm_ret > spy_ret - 5:   # IWM not lagging SPY by more than 5 pts
                score += 1
            breadth_note = f"IWM {iwm_ret:+.1f}% vs SPY {spy_ret:+.1f}%"

        # QQQ (tech leadership) — tech must confirm the move
        qqq_above_ema20 = False
        qqq_above_ema50 = False
        qqq_ema20_dist  = 0.0
        qqq_note = "N/A"
        try:
            qqq_df = fetch_df("QQQ")
            if qqq_df is not None and len(qqq_df) >= 55:
                qqq_ind = compute_indicators(qqq_df.copy())
                qqq_ind = qqq_ind.dropna(subset=["Close"])
                qr = qqq_ind.iloc[-1]
                qqq_above_ema20 = float(qr["Close"]) > float(qr["EMA20"])
                qqq_above_ema50 = float(qr["Close"]) > float(qr["EMA50"])
                qqq_ema20_dist  = (float(qr["Close"]) - float(qr["EMA20"])) / float(qr["EMA20"]) * 100
                _qqq_clean = qqq_df.dropna(subset=["Close"])
                qqq_chg5 = (float(_qqq_clean["Close"].iloc[-1]) / float(_qqq_clean["Close"].iloc[-6]) - 1) * 100
                if qqq_above_ema20:
                    score += 1   # tech leading = bull confirmation
                qqq_note = f"{'✓' if qqq_above_ema20 else '✗'} EMA20  {'✓' if qqq_above_ema50 else '✗'} EMA50  5d {qqq_chg5:+.1f}%"
        except Exception:
            pass

        # TLT (20Y bond ETF) — proxy for rate environment
        # Rising TLT = falling yields = tailwind for growth stocks
        tlt_trend = "flat"
        tlt_note = "N/A"
        try:
            tlt_df = fetch_df("TLT")
            if tlt_df is not None and len(tlt_df) >= 22:
                _tlt_clean = tlt_df.dropna(subset=["Close"])
                tlt_now  = float(_tlt_clean["Close"].iloc[-1])
                tlt_20d  = float(_tlt_clean["Close"].iloc[-21]) if len(_tlt_clean) >= 22 else tlt_now
                tlt_chg  = (tlt_now - tlt_20d) / tlt_20d * 100
                tlt_trend = "rising" if tlt_chg > 1.5 else ("falling" if tlt_chg < -1.5 else "flat")
                if tlt_trend == "rising":
                    score += 1   # falling rates = growth tailwind
                tlt_note = f"${tlt_now:.1f}  20d {tlt_chg:+.1f}%  ({tlt_trend})"
        except Exception:
            pass

        # DXY proxy via UUP (DB USD Bull ETF) — strong dollar = headwind for risk assets
        dxy_trend = "flat"
        dxy_note = "N/A"
        try:
            uup_df = fetch_df("UUP")
            if uup_df is not None and len(uup_df) >= 22:
                _uup_clean = uup_df.dropna(subset=["Close"])
                uup_now = float(_uup_clean["Close"].iloc[-1])
                uup_20d = float(_uup_clean["Close"].iloc[-21]) if len(_uup_clean) >= 22 else uup_now
                uup_chg = (uup_now - uup_20d) / uup_20d * 100
                dxy_trend = "strong" if uup_chg > 1 else ("weak" if uup_chg < -1 else "flat")
                if dxy_trend == "weak":
                    score += 1   # weak dollar = risk-on tailwind
                dxy_note = f"${uup_now:.2f}  20d {uup_chg:+.1f}%  ({dxy_trend})"
        except Exception:
            pass

        # VIX shock detector — fires when EITHER:
        #   • 1-day VIX change ≥ 20% (sudden fear spike, e.g. +39.7% on a single session)
        #   • OR VIX ≥ 1.30× its 5-day average (sustained elevated fear, ~30% above norm)
        # The session AFTER a shock is historically a "digestion" period: vol stays elevated,
        # momentum longs swim against the current. Raises the min-score floor in the scanner.
        vix_shock = False
        vix_shock_note = ""
        try:
            if vix_df is not None and len(vix_df) >= 6:
                vix_prev   = float(vix_df["Close"].iloc[-2])
                vix_5d_avg = float(vix_df["Close"].iloc[-6:-1].mean())
                vix_1d_chg = (vix_val - vix_prev) / vix_prev * 100
                vix_vs_avg = vix_val / vix_5d_avg if vix_5d_avg > 0 else 1.0
                if vix_1d_chg >= 20 or vix_vs_avg >= 1.30:
                    vix_shock = True
                    vix_shock_note = (
                        f"SHOCK — 1d +{vix_1d_chg:.0f}%  "
                        f"(vs 5d avg {vix_5d_avg:.1f}, ratio {vix_vs_avg:.2f}x)"
                    )
        except Exception:
            pass

        # VIX term structure — VIX/VIX3M ratio reveals whether fear is acute or structural.
        # Normal (contango): VIX < VIX3M — near-term calm relative to future uncertainty.
        # Inverted (backwardation): VIX > VIX3M — acute fear dominating short-term, often
        # marks a volatility spike event. VIX sizing already handles this via raw VIX level;
        # term structure shows HOW the market is pricing that fear (spike vs regime shift).
        vix_term_note = "N/A"
        vix_complacency_warn = ""
        try:
            vix3m_df = fetch_df("^VIX3M")
            if vix3m_df is not None and len(vix3m_df) >= 1:
                vix3m_val = float(vix3m_df["Close"].iloc[-1])
                ts_ratio  = vix_val / vix3m_val if vix3m_val > 0 else 1.0
                if ts_ratio >= 1.10:
                    vix_term_note = (f"⚠️ INVERTED {ts_ratio:.2f}x "
                                     f"(VIX {vix_val:.1f} > VIX3M {vix3m_val:.1f}) "
                                     f"— acute fear spike, reduce size further")
                elif ts_ratio >= 1.0:
                    vix_term_note = (f"flat {ts_ratio:.2f}x "
                                     f"(VIX {vix_val:.1f} ≈ VIX3M {vix3m_val:.1f})")
                else:
                    vix_term_note = (f"normal {ts_ratio:.2f}x "
                                     f"(VIX {vix_val:.1f} < VIX3M {vix3m_val:.1f})")
        except Exception:
            pass

        # VIX complacency warning — when VIX is 10%+ below its own 20-day EMA,
        # the market is pricing in near-zero risk. This often precedes sharp
        # reversals because any surprise triggers outsized moves.
        # VIX Fri Jul 10 2026: 15.0 vs EMA20=17.1 = -12% → complacency alert.
        try:
            _vix_df_ema = fetch_df("^VIX")
            if _vix_df_ema is not None and len(_vix_df_ema) >= 20:
                _vix_closes = [float(_vix_df_ema["Close"].iloc[i])
                               for i in range(len(_vix_df_ema))]
                _vix_ema20  = sum(_vix_closes[-20:]) / 20   # simple avg as proxy
                _vix_vs_ema = (vix_val - _vix_ema20) / _vix_ema20 * 100
                if _vix_vs_ema <= -10.0:
                    vix_complacency_warn = (
                        f"⚠️ VIX COMPLACENCY: {vix_val:.1f} is {abs(_vix_vs_ema):.0f}% "
                        f"below its 20d avg ({_vix_ema20:.1f}) — market pricing near-zero risk. "
                        f"Surprises hit harder in this environment; size conservatively."
                    )
        except Exception:
            pass

        # Defensive rotation detector — when XLP/XLU/XLV outperform XLK by >5%
        # on a single day, institutional money is rotating out of growth into safety.
        # A "defensive rotation" day invalidates most Gap & Hold tech long setups.
        defensive_rotation = False
        def_rotation_note  = ""
        try:
            def_tickers = ["XLK","XLP","XLU","XLV"]
            def_data    = yf.download(def_tickers, period="3d", progress=False,
                                      auto_adjust=True)["Close"]
            if isinstance(def_data.columns, pd.MultiIndex):
                def_data.columns = def_data.columns.droplevel(1)
            if len(def_data) >= 2:
                xlk_chg  = (float(def_data["XLK"].iloc[-1])  / float(def_data["XLK"].iloc[-2])  - 1) * 100
                xlp_chg  = (float(def_data["XLP"].iloc[-1])  / float(def_data["XLP"].iloc[-2])  - 1) * 100
                xlu_chg  = (float(def_data["XLU"].iloc[-1])  / float(def_data["XLU"].iloc[-2])  - 1) * 100
                xlv_chg  = (float(def_data["XLV"].iloc[-1])  / float(def_data["XLV"].iloc[-2])  - 1) * 100
                def_avg  = (xlp_chg + xlu_chg + xlv_chg) / 3
                spread   = def_avg - xlk_chg  # positive = defensives winning
                if spread > 5.0:
                    defensive_rotation = True
                    def_rotation_note = (
                        f"ACTIVE — XLK {xlk_chg:+.1f}%  "
                        f"DEF avg {def_avg:+.1f}%  spread {spread:+.1f}%"
                    )
        except Exception:
            pass

        # Regime classification
        # BULL_TECH: SPY in CHOP but QQQ clearly above EMA20 + XLK leading + VIX calm.
        # Captures post-selloff tech recoveries where broad market lags but semis/tech
        # are already bouncing (e.g., June 8 2026 — QQQ +1.56%, XLK +2.15%, SPY flat).
        # Treated as BULL for LONG scoring (min score cap lifted for tech signals).
        qqq_leading = qqq_above_ema20 and (not spy_above_20) and vix_low
        if score >= 10 and bull_di:
            regime = "BULL"
        elif qqq_leading and score >= 7 and bull_di:
            regime = "BULL_TECH"   # QQQ leads while SPY lags — tech BULL sub-regime
        elif score <= 4 or (not bull_di and vix_val > 30):
            regime = "BEAR"
        else:
            regime = "CHOP"

        # SPY key level distances (for briefing context)
        try:
            spy_ema20_dist = (float(sr["Close"]) - float(sr["EMA20"])) / float(sr["EMA20"]) * 100
            spy_ema50_dist = (float(sr["Close"]) - float(sr["EMA50"])) / float(sr["EMA50"]) * 100
        except Exception:
            spy_ema20_dist = spy_ema50_dist = 0.0

        # News sentiment breadth — OBSERVATION ONLY (2026-08-15, direct
        # instruction). Deliberately not added to `score` or referenced by
        # regime_allows_signal()/any gate — see _news_sentiment_breadth()'s
        # docstring for why a signal with zero live track record has no
        # business influencing real entries yet. Purely a display note for
        # now, same visibility tier as the VIX complacency/term-structure
        # notes above before those were ever considered for scoring either.
        _nb = None
        try:
            _nb = _news_sentiment_breadth(hours_back=24.0)
            if _nb["breadth_pct"] is None:
                news_breadth_note = f"no scored sentiment in last 24h ({_nb['total']} logged, {_nb['unknown']} unscored)"
            else:
                news_breadth_note = (f"{_nb['breadth_pct']:+.0f}% "
                                     f"({_nb['positive']}pos/{_nb['negative']}neg/{_nb['neutral']}neu, "
                                     f"{_nb['unknown']} unscored, 24h)")
        except Exception:
            news_breadth_note = "N/A"

        result.update({
            "regime":     regime,
            "score":      score,
            "spy_trend":  spy_above_50,
            "adx_strong": adx_strong,
            "vix_ok":     vix_mid,
            "vix_shock":  vix_shock,
            "defensive_rotation": defensive_rotation,
            # Raw numbers behind the display note above — added 2026-08-15
            # so callers (run_pro_scanner's scan-log entry, chiefly) can
            # persist a per-scan snapshot for trend review over the coming
            # days, without re-parsing the formatted string or re-reading
            # the news log a second time. None if the lookup itself failed.
            "news_breadth": _nb,
            "details": {
                "SPY vs EMA20":      spy_above_20,
                "SPY vs EMA50":      spy_above_50,
                "SPY vs SMA200":     spy_above_200,
                "SPY EMA20 dist":    round(spy_ema20_dist, 2),
                "SPY EMA50 dist":    round(spy_ema50_dist, 2),
                "ADX":               round(adx_val, 1),
                "VIX":               round(vix_val, 1),
                "+DI > -DI":         bull_di,
                "IWM Breadth":       breadth_note,
                "QQQ vs EMA20":      qqq_above_ema20,
                "QQQ vs EMA50":      qqq_above_ema50,
                "QQQ EMA20 dist":    round(qqq_ema20_dist, 2),
                "QQQ Note":          qqq_note,
                "TLT Trend":         tlt_trend,
                "TLT Note":          tlt_note,
                "DXY Trend":         dxy_trend,
                "DXY Note":          dxy_note,
                "VIX Shock":         vix_shock_note or ("none" if not vix_shock else "detected"),
                "VIX Term Structure": vix_term_note,
                "VIX Complacency":   vix_complacency_warn or "none",
                "Def Rotation":      def_rotation_note or ("none" if not defensive_rotation else "detected"),
                "News Sentiment Breadth (obs. only, not scored)": news_breadth_note,
            }
        })
    except Exception as e:
        result["details"]["error"] = str(e)

    return result


def regime_allows_signal(regime: dict, bias: str) -> tuple[bool, int]:
    """
    Check if regime allows a LONG or SHORT signal.
    Returns (allowed, score_contribution 0-15).
    """
    r = regime.get("regime", "UNKNOWN")
    s = regime.get("score",  0)
    v = regime.get("vix_ok", False)

    if bias == "LONG":
        if r == "BULL":      return True,  min(15, s)
        if r == "BULL_TECH": return True,  min(12, s)  # tech leading — near-BULL credit
        if r == "CHOP":      return True,  min(8,  s)  # allowed but reduced score
        return False, 0    # BEAR — no longs

    else:  # SHORT
        if r == "BEAR":      return True,  min(15, 15 - s)
        if r == "CHOP":      return True,  min(8,  8  - s//2)
        return False, 0    # BULL / BULL_TECH — no shorts


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FILTER 02: MULTI-TIMEFRAME (WEEKLY)
# ═══════════════════════════════════════════════════════════════════════════

def check_mtf(ticker: str, bias: str) -> tuple[bool, int]:
    """
    Weekly chart must agree with the daily signal.
    LONG : weekly EMA20 > EMA50 AND weekly RSI > 45 AND weekly MACD > signal
    SHORT: weekly EMA20 < EMA50 AND weekly RSI < 55 AND weekly MACD < signal

    Returns (passes, score 0-20).
    """
    try:
        wdf = fetch_weekly(ticker)
        if wdf is None or len(wdf) < 30:
            return True, 10   # data unavailable — partial credit

        wdf = compute_weekly_indicators(wdf.copy())
        wr  = wdf.iloc[-1]
        wp  = wdf.iloc[-2]

        ema_bull = float(wr["W_EMA20"]) > float(wr["W_EMA50"])
        rsi_val  = float(wr["W_RSI"])
        macd_bull= float(wr["W_MACD"]) > float(wr["W_MACD_sig"])
        macd_rising = float(wr["W_MACD"]) > float(wp["W_MACD"])

        if bias == "LONG":
            score = 0
            if ema_bull:    score += 8
            if rsi_val > 45: score += 6
            if macd_bull:   score += 6
            passes = ema_bull and rsi_val > 45
        else:  # SHORT
            score = 0
            if not ema_bull: score += 8
            if rsi_val < 55: score += 6
            if not macd_bull: score += 6
            passes = (not ema_bull) and rsi_val < 55

        return passes, min(20, score)
    except Exception:
        return True, 10


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 7 — FILTER 03: RELATIVE STRENGTH vs SPY
# ═══════════════════════════════════════════════════════════════════════════

def check_relative_strength(ticker: str, bias: str) -> tuple[bool, int]:
    """
    Stock must be outperforming (LONG) or underperforming (SHORT) SPY
    over multiple lookback periods: 5-day, 20-day, 60-day.
    At least 2 of 3 periods must align.

    Returns (passes, score 0-15).
    """
    if ticker in ("SPY", "QQQ", "IWM"):
        return True, 10   # index ETFs — skip

    try:
        stock_df = fetch_df(ticker)
        spy_df   = fetch_df("SPY")
        if stock_df is None or spy_df is None:
            return True, 8

        stock_c = stock_df["Close"]
        spy_c   = spy_df["Close"]

        # Align on common dates
        common = stock_c.index.intersection(spy_c.index)
        if len(common) < 65:
            return True, 8
        sc = stock_c.loc[common]
        sp = spy_c.loc[common]

        rs5  = (sc.pct_change(5).iloc[-1]  - sp.pct_change(5).iloc[-1])  * 100
        rs20 = (sc.pct_change(20).iloc[-1] - sp.pct_change(20).iloc[-1]) * 100
        rs60 = (sc.pct_change(60).iloc[-1] - sp.pct_change(60).iloc[-1]) * 100

        if bias == "LONG":
            beats = [rs5 > 0, rs20 > 0, rs60 > 0]
        else:
            beats = [rs5 < 0, rs20 < 0, rs60 < 0]

        n_pass = sum(beats)
        passes = n_pass >= 2
        score  = n_pass * 5   # 0, 5, 10, or 15
        return passes, score
    except Exception:
        return True, 8


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 8 — FILTER 04: SECTOR ROTATION
# ═══════════════════════════════════════════════════════════════════════════

_sector_cache: Optional[list[str]] = None
_sector_cache_ts: Optional[datetime] = None

def get_top_sectors(n: int = 4) -> list[str]:
    """
    Rank all 11 sectors by 20-day momentum and return the top n.
    Cached for 4 hours to avoid spamming yfinance.

    The cache stores the FULL ranking (all scored sectors), not a
    truncated top-n — found 2026-08-16 review: it used to cache
    `ranked[:n]`, so whichever call happened to populate the cache first
    fixed its size for the next 4 hours regardless of what a LATER caller
    asked for. The premarket briefing calls this with the default n=4
    every morning; for hours afterward, check_sector()'s SHORT path (which
    wants the bottom 4 of all 11 via `get_top_sectors(11)`) would silently
    receive only 4 elements back — reversed, that's the same top-4
    *strongest* sectors, inverting the SHORT gate into requiring a
    candidate's sector be among the market's strongest to pass. A LONG
    call for `get_top_sectors(6)` after an n=11 caller had populated the
    cache had the opposite problem — all 11 sectors, effectively
    disabling the gate. Slicing to `n` on every return (hit or miss) means
    one shared 4-hour cache correctly serves every caller regardless of
    which one happens to run first.
    """
    global _sector_cache, _sector_cache_ts

    if (_sector_cache is not None and _sector_cache_ts is not None
            and (datetime.now() - _sector_cache_ts).total_seconds() < 14400):
        return _sector_cache[:n]

    scores = {}
    for sector, etf in SECTOR_ETFS.items():
        try:
            df = fetch_df(etf)
            if df is None or len(df) < 25:
                continue
            perf_20d = (float(df["Close"].iloc[-1]) /
                        float(df["Close"].iloc[-21]) - 1) * 100
            scores[sector] = perf_20d
        except Exception:
            continue

    ranked = sorted(scores, key=scores.get, reverse=True)
    _sector_cache    = ranked
    _sector_cache_ts = datetime.now()
    return _sector_cache[:n]


def check_sector(ticker: str, bias: str) -> tuple[bool, int]:
    """
    For LONG : ticker's sector must be in top-4 performing sectors (score 10).
    For SHORT: ticker's sector should be in bottom-4 (weakest).
    Returns (passes, score 0-10).
    """
    if ticker in ("SPY","QQQ","IWM"):
        return True, 8

    sector = TICKER_SECTOR.get(ticker, "")
    if not sector:
        return True, 6   # unknown sector — partial credit

    try:
        top4 = get_top_sectors(4)
        if bias == "LONG":
            passes = sector in top4
            score  = 10 if sector in top4 else (5 if sector in get_top_sectors(6) else 0)
        else:
            top4_weak = list(reversed(get_top_sectors(11)))[:4]
            passes = sector in top4_weak
            score  = 10 if passes else 0
        return passes, score
    except Exception:
        return True, 5


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9 — FILTER 05: EARNINGS BLACKOUT
# ═══════════════════════════════════════════════════════════════════════════

_MASSIVE_EARNINGS_CACHE: dict[tuple, tuple[float, list[dict]]] = {}
_MASSIVE_EARNINGS_CACHE_TTL_S = 1200   # 20 min

def _fetch_massive_earnings(ticker: str, date_from: date, date_to: date) -> list[dict]:
    """
    Query Massive's Benzinga-earnings proxy for a single ticker within a date
    window. Confirmed live 2026-07-30: the `ticker` filter is applied
    server-side correctly (unlike BENZINGA_API_KEY's direct calendar
    endpoint, see _fetch_benzinga_earnings_time docstring) — a ticker=AAPL
    request returns only AAPL records. Returns raw result dicts (date, time,
    date_status, actual_eps, ...); [] on any failure or if MASSIVE_API_KEY
    isn't set (fail-open to callers' existing fallbacks).

    20-min TTL cache added 2026-07-31 — no published rate limit for this
    endpoint, and this function sits behind check_earnings_safe(), which
    _extract_earnings_dates() (via this function) gets called for on every
    candidate ticker in the confluence pipeline, every scan cycle (daemon
    every 10 min + hourly cron), with zero caching before this. An earnings
    date doesn't change within a session; caching removes that redundant
    volume rather than betting on an undocumented limit never getting hit
    on the first day this integration sees real daily traffic.

    One retry on a non-200/exception added 2026-08-04 — confirmed live:
    three identical requests for the same ticker+date in immediate
    succession got a mix of successes and failures (no consistent pattern
    by query shape, genuinely intermittent). _resolve_earnings_timing()
    calls this with an exact single-day window and has NO retry of its
    own — one transient hit here used to fall straight through to the
    secondary fallbacks (also unreliable — see _fetch_benzinga_earnings_time
    and _fetch_massive_benzinga_news docstrings) and land on UNKNOWN-TODAY
    for the rest of the day, since the earnings-scan only gets a narrow
    pre-close retry window. A single short retry costs under a second and
    meaningfully raises the odds today's classification actually succeeds.
    """
    if not MASSIVE_API_KEY:
        return []
    key = (ticker, date_from.isoformat(), date_to.isoformat())
    cached = _MASSIVE_EARNINGS_CACHE.get(key)
    if cached and (time.time() - cached[0]) < _MASSIVE_EARNINGS_CACHE_TTL_S:
        return cached[1]
    for _attempt in range(2):
        try:
            resp = requests.get(
                "https://api.massive.com/benzinga/v1/earnings",
                params={
                    "ticker":      ticker,
                    "date.gte":    date_from.isoformat(),
                    "date.lte":    date_to.isoformat(),
                    "limit":       10,
                    "apiKey":      MASSIVE_API_KEY,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", []) or []
                _MASSIVE_EARNINGS_CACHE[key] = (time.time(), results)
                return results
            if resp.status_code in (401, 403):
                break   # auth/entitlement error — retrying won't help
        except Exception:
            pass
        if _attempt == 0:
            time.sleep(1)
    return cached[1] if cached else []   # serve stale-but-valid data over nothing after both attempts fail


def _recent_earnings_surprise(ticker: str, days_back: int = 2) -> Optional[dict]:
    """
    Real beat/miss data for a ticker that reported within the last
    `days_back` days — grounds a Gap & Hold/Morning Runner alert's "why"
    in an actual verified number instead of just "gap detected." Confirmed
    live 2026-08-05: Massive's /benzinga/v1/earnings response already
    carries actual_eps/eps_surprise_percent/actual_revenue/
    revenue_surprise_percent for reported quarters — this data was being
    fetched (via _fetch_massive_earnings, used elsewhere for the
    estimate-only fields) but never surfaced anywhere. Analyst ratings/
    price-target/bull-bear endpoints exist on Massive but return 403 (not
    entitled on the current plan) — this sticks to data actually
    available rather than silently no-op'ing on a paywalled endpoint.
    Returns None if nothing reported in the window, or actual_eps is
    still null (estimate-only / not yet confirmed).
    """
    today = date.today()
    items = _fetch_massive_earnings(ticker, today - timedelta(days=days_back), today)
    best = None
    for item in items:
        if str(item.get("ticker", "")).upper() != ticker.upper():
            continue
        if item.get("actual_eps") is None:
            continue   # estimate only — hasn't actually reported yet
        if best is None or item.get("date", "") > best.get("date", ""):
            best = item
    return best


def _format_earnings_surprise_note(surprise: dict) -> str:
    """Plain-text one-liner for a Telegram alert — real numbers only."""
    eps_pct = surprise.get("eps_surprise_percent")
    rev_pct = surprise.get("revenue_surprise_percent")
    parts = []
    if eps_pct is not None:
        parts.append(f"EPS {'beat' if eps_pct >= 0 else 'missed'} by {abs(eps_pct)*100:.1f}%")
    if rev_pct is not None:
        parts.append(f"Revenue {'beat' if rev_pct >= 0 else 'missed'} by {abs(rev_pct)*100:.1f}%")
    if not parts:
        return ""
    return f"\n📊 <b>Earnings {surprise.get('date','')}</b>: " + ", ".join(parts)


EARNINGS_REACTION_LOOKBACK_DAYS = 2   # matches _recent_earnings_surprise's default window

def check_earnings_reaction(ticker: str, bias: str) -> tuple[bool, int]:
    """
    Confluence bonus (up to 7 pts) if a REAL, recently-reported earnings
    surprise fundamentally confirms this signal's direction — added
    2026-08-11 so a signal riding a genuine earnings beat scores higher
    than an identical-looking one with no fundamental backing (previously
    _recent_earnings_surprise's data only ever reached alert TEXT, never
    the score itself).

    Deliberately does NOT average EPS and revenue surprise together.
    Confirmed live 2026-08-10: RIOT missed EPS by -106% (a GAAP loss
    heavily distorted by non-cash crypto-asset mark-to-market accounting)
    but beat revenue by +14.6%, and the market rallied +22.9% overnight —
    an averaged score would have read that as net negative and missed a
    genuinely strong, fundamentally-backed setup. Revenue is scored as
    the primary signal (harder to flatter than EPS via cost-cutting, more
    directly reflects real business momentum); a same-direction EPS
    result adds a smaller bonus on top. An EPS miss is never itself a
    penalty here — the opposite-direction price action that would result
    from a truly bad quarter is already what keeps a contradicting ticker
    from generating a same-direction signal in the first place (e.g. a
    real down-move on bad numbers won't produce a Gap & Hold LONG
    candidate to apply this bonus to).

    Never a hard gate (always returns ok=True) — most signals simply have
    no earnings in the window, which isn't itself bearish or bullish.
    """
    try:
        surprise = _recent_earnings_surprise(ticker, days_back=EARNINGS_REACTION_LOOKBACK_DAYS)
        if not surprise:
            return True, 0
        rev_pct = surprise.get("revenue_surprise_percent")
        eps_pct = surprise.get("eps_surprise_percent")
        want_long = (bias == "LONG")
        score = 0
        if rev_pct is not None:
            signed_rev = rev_pct if want_long else -rev_pct
            if signed_rev >= 0.10:
                score += 5
            elif signed_rev >= 0.05:
                score += 3
        if eps_pct is not None:
            signed_eps = eps_pct if want_long else -eps_pct
            if signed_eps >= 0:
                score += 2
        return True, min(score, 7)
    except Exception:
        return True, 0


def _extract_earnings_dates(ticker: str) -> list[date]:
    """
    Shared calendar parser for check_earnings_safe()/get_upcoming_earnings().

    Primary source: Massive's Benzinga earnings proxy (_fetch_massive_earnings,
    -3d to +30d window) — ticker-filtered correctly server-side, confirmed
    live 2026-07-30. Falls back to yfinance's .calendar if MASSIVE_API_KEY is
    unset or the call returns nothing.

    yfinance's .calendar is a plain dict — {'Earnings Date': [date(...), ...], ...} —
    not a DataFrame. The old code here checked cal.empty/cal.columns, which raised
    AttributeError on every real call (dict has neither), silently caught by the
    bare except below each call site — meaning check_earnings_safe() always
    returned "safe" and get_upcoming_earnings() always returned [], in production,
    confirmed empirically. EARNINGS_BLACKOUT never actually blocked anything.
    """
    today = date.today()
    massive = _fetch_massive_earnings(ticker, today - timedelta(days=3), today + timedelta(days=30))
    if massive:
        out = []
        for item in massive:
            try:
                out.append(date.fromisoformat(str(item.get("date", ""))[:10]))
            except Exception:
                continue
        if out:
            return out
    try:
        cal = yf.Ticker(ticker).calendar
    except Exception:
        return []
    if not cal:
        return []
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date", [])
        out = []
        for d in raw if isinstance(raw, (list, tuple)) else [raw]:
            try:
                out.append(d if isinstance(d, date) else pd.Timestamp(d).date())
            except Exception:
                continue
        return out
    # Legacy DataFrame shape — kept in case yfinance reverts this upstream.
    try:
        if hasattr(cal, "empty") and not cal.empty and "Earnings Date" in cal.columns:
            return [d.date() for d in pd.to_datetime(cal["Earnings Date"]).dropna()]
    except Exception:
        pass
    return []


def check_earnings_safe(ticker: str) -> tuple[bool, int]:
    """
    Blocks signals if earnings are UPCOMING and unreported, within
    EARNINGS_BLACKOUT days — protects against entering right before an
    unknown reaction. Does NOT block a real post-earnings gap: earnings
    from yesterday, or a same-day report already confirmed released this
    morning, are both KNOWN reactions by the time a signal can fire on
    them.

    Found 2026-08-16 review: the old `-1 <= days_away <= EARNINGS_BLACKOUT`
    range blocked days_away == -1 (reported yesterday) and days_away == 0
    (today, regardless of whether it already happened) unconditionally —
    meaning a 96/100-scoring post-earnings gap-and-hold got rejected with
    "EARNINGS BLACKOUT" every single time, on principle, and
    fetch_earnings_mover_tickers()'s discoveries (every one of which
    reported within the last day, by construction) could never produce a
    tradeable signal through this gate at all. days_away == -1 is
    unambiguous (any earnings time yesterday is now in the past) and never
    blocks. days_away == 0 is ambiguous — could be a BMO report already
    out, or an AMC report still pending later today — so it's the one case
    that still needs a real check: _check_earnings_already_reported()
    fails closed (still blocks) when it can't confirm, so an unconfirmed
    same-day report is treated exactly as conservatively as before.

    Returns (safe, score 0-5).
    """
    try:
        today = date.today()
        for ed in _extract_earnings_dates(ticker):
            days_away = (ed - today).days
            if 1 <= days_away <= EARNINGS_BLACKOUT:
                return False, 0   # upcoming, unreported — unknown reaction, stay out
            if days_away == 0 and not _check_earnings_already_reported(ticker):
                return False, 0   # today, not yet confirmed reported — could still be AMC-pending
        return True, 5
    except Exception:
        return True, 5


def get_upcoming_earnings(tickers: list, days_ahead: int = 5) -> list[dict]:
    """
    Return list of {ticker, earn_date, days_away} for tickers with earnings
    in the next `days_ahead` calendar days. Used by premarket briefing.
    Skips silently on API failure so it never blocks the briefing.
    """
    results = []
    today = date.today()
    for ticker in tickers:
        try:
            for ed in _extract_earnings_dates(ticker):
                days_away = (ed - today).days
                if 0 <= days_away <= days_ahead:
                    results.append({
                        "ticker":     ticker,
                        "earn_date":  ed,
                        "days_away":  days_away,
                    })
                    break  # one entry per ticker
        except Exception:
            continue
    results.sort(key=lambda x: x["days_away"])
    return results


def _classify_bmo_amc(time_str: str, earn_date: date) -> Optional[str]:
    """
    Precise BMO/AMC classification from a Benzinga "HH:MM:SS" time string.
    Massive's docs are explicit this field is fixed EST (UTC-5), NOT
    DST-aware ET — during EDT (roughly March-November, which covers most
    of the trading year, including right now), a naive hour<12 check reads
    the wrong wall-clock hour by up to 1 hour. This converts the
    fixed-UTC-5 timestamp to real America/New_York local time (DST-aware,
    via the module's existing ET zoneinfo) before comparing against actual
    market boundaries (9:30 open, 16:00 close) instead of a coarse noon
    split.

    Returns "BMO" (strictly before 9:30 ET), "AMC" (16:00 ET or later), or
    None for a report that timestamps to during market hours — genuinely
    ambiguous for this feature's purposes, not a parse failure, so the
    caller falls through to its other checks rather than guessing. In
    practice this coarse-vs-precise distinction almost never changes the
    BMO/AMC call (real earnings releases cluster well before 9:30 or well
    after 16:00, not near the old noon threshold), but "almost never"
    isn't "never," and the precise version costs nothing extra to get right.
    """
    try:
        hh, mm, ss = (int(x) for x in time_str.split(":")[:3])
    except Exception:
        return None
    try:
        from datetime import timezone as _tz, time as _dt_time
        est = _tz(timedelta(hours=-5))   # fixed UTC-5, no DST — matches Benzinga's documented "EST"
        naive = datetime(earn_date.year, earn_date.month, earn_date.day, hh, mm, ss, tzinfo=est)
        local_t = naive.astimezone(ET).time()
    except Exception:
        return None
    if local_t < _dt_time(9, 30):
        return "BMO"
    if local_t >= _dt_time(16, 0):
        return "AMC"
    return None   # reported during market hours — ambiguous, let caller fall through


def _fetch_benzinga_earnings_time(ticker: str, earn_date: date) -> Optional[str]:
    """
    Best-effort BMO/AMC lookup via Benzinga's calendar endpoint. Returns "BMO",
    "AMC", or None (unknown/unavailable).

    NOTE — empirically tested 2026-07-29 against the live BENZINGA_API_KEY on
    this account: the endpoint's `tickers` filter is NOT applied server-side
    (a `tickers=META` request still returns an unrelated generic page of
    future estimated dates), and the bracketed `parameters[tickers]=META`
    form returns zero results. This function therefore fetches whatever page
    Benzinga returns and filters for `ticker`+`earn_date` client-side, which
    only helps if the requested name happens to already be on that page. In
    practice this currently returns None almost always on this account/plan
    tier — get_earnings_spread_candidates() is written to treat that as the
    normal case and fall through to the IV-backwardation heuristic below, not
    as an error. Revisit if the Benzinga plan/endpoint behavior changes.
    """
    if not BENZINGA_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://api.benzinga.com/api/v2.1/calendar/earnings",
            params={"token": BENZINGA_API_KEY, "tickers": ticker},
            headers={"Accept": "application/json"},
            timeout=8,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        items = data if isinstance(data, list) else data.get("earnings", [])
        for item in items:
            if str(item.get("ticker", "")).upper() != ticker.upper():
                continue
            try:
                if date.fromisoformat(str(item.get("date", ""))[:10]) != earn_date:
                    continue
            except Exception:
                continue
            time_str = str(item.get("time", "")).strip()
            if not time_str:
                continue
            return _classify_bmo_amc(time_str, earn_date)
    except Exception:
        return None
    return None


def _get_atm_iv(client, ticker: str, current_price: float, target_dte: int) -> Optional[float]:
    """
    ATM implied vol for the expiry nearest target_dte, via the same
    GetOptionContractsRequest + _get_option_snapshot pattern _find_best_call_contract
    uses. Returns None on any lookup/data failure (fail-closed for the caller).
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    today = date.today()
    target_expiry = None
    best_diff = float("inf")
    for offset in range(1, target_dte + 21):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 4:   # Friday
            diff = abs(offset - target_dte)
            if diff < best_diff:
                best_diff = diff
                target_expiry = candidate
    if not target_expiry:
        return None

    incr = 1.0 if current_price < 25 else (2.5 if current_price < 200 else 5.0)
    atm = round(round(current_price / incr) * incr, 2)
    try:
        raw = client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker],
            expiration_date=target_expiry,
            type=ContractType.CALL,
            strike_price_gte=str(round(atm - 0.01, 2)),
            strike_price_lte=str(round(atm + 0.01, 2)),
            limit=1,
        ))
        items = getattr(raw, "option_contracts", None) or (raw if isinstance(raw, list) else [])
        if not items:
            return None
        snap = _get_option_snapshot(items[0].symbol)
        if not snap or snap.get("iv", 0) <= 0:
            return None
        return float(snap["iv"])
    except Exception:
        return None


_EARNINGS_REPORTED_KEYWORDS = (
    "reports q", "reports fiscal", "reports third quarter", "reports fourth quarter",
    "reports first quarter", "reports second quarter", "earnings per share",
    "quarterly results", "beats estimates", "misses estimates", "eps of",
    "revenue of $", "earnings call transcript",
)


def _check_earnings_already_reported(ticker: str, hours_back: int = 8) -> bool:
    """
    Direct confirmation via Benzinga headlines that a ticker's earnings have
    already been released, rather than inferring it indirectly from options
    IV term structure. Checks for a recent headline matching common
    earnings-release phrasing. Fails closed on any error/no-match (returns
    False = "can't confirm reported" — caller still has the IV fallback as a
    second check, not a bare guess).

    Tries Massive's Benzinga news proxy first (_fetch_massive_benzinga_news,
    correctly ticker-filtered server-side once the News entitlement is
    purchased — no-ops to {} until then). Falls back to the direct Benzinga
    API below, which has a KNOWN LIMITATION, confirmed live 2026-07-29:
    Benzinga's single-ticker `tickers=X` filter is unreliable on this
    account's plan tier — querying META or TSLA alone returned ZERO
    articles even with no time filter at all, while AAPL returned 5 in the
    same call. This isn't a bug in this function; it inherits the same
    per-ticker unreliability already documented in
    _fetch_benzinga_earnings_time(). This check has real value when either
    source's filter happens to work for a given ticker, but will silently
    return False (not "confirmed clear," just "couldn't confirm") if both
    miss — it's an additive signal, not a replacement for the IV fallback
    that already existed (or the Massive-earnings actual_eps check that now
    runs before this in _resolve_earnings_timing).
    """
    try:
        news = _fetch_massive_benzinga_news([ticker], hours_back=hours_back)
        headlines = news.get(ticker, [])
        if not headlines:
            news = _fetch_benzinga_ticker_news([ticker], hours_back=hours_back)
            headlines = news.get(ticker, [])
        return any(kw in h.lower() for h in headlines for kw in _EARNINGS_REPORTED_KEYWORDS)
    except Exception:
        return False


def _resolve_earnings_timing(client, ticker: str, earn_date: date, current_price: float,
                             days_away: int) -> str:
    """
    Returns "BMO", "AMC", "ALREADY-REPORTED", "PENDING-TOMORROW", or
    "UNKNOWN-TODAY". days_away == 1 doesn't need real BMO/AMC resolution —
    entering today before close is safe either way (a day early for a true
    BMO print costs a little extra theta, nothing more). days_away == 0 is
    the only case that actually needs disambiguating, since a same-day BMO
    print (or, confirmed live 2026-07-29, an AMC print that's already
    happened when this runs late in the evening) may have already occurred.

    Checks Massive's Benzinga-earnings proxy FIRST (_fetch_massive_earnings) —
    its `actual_eps` field is only populated once the report has actually
    been released (a harder confirmation than headline keyword matching) and
    its `time` field gives an exact BMO/AMC read for this ticker+date,
    confirmed server-side accurate live 2026-07-30. Falls back to the direct
    Benzinga news confirmation, then the direct Benzinga calendar endpoint,
    then IV term structure, only if Massive is unavailable or inconclusive
    for this ticker/date (e.g. MASSIVE_API_KEY unset).
    """
    if days_away >= 1:
        return "PENDING-TOMORROW"
    for item in _fetch_massive_earnings(ticker, earn_date, earn_date):
        if str(item.get("ticker", "")).upper() != ticker.upper():
            continue
        try:
            if date.fromisoformat(str(item.get("date", ""))[:10]) != earn_date:
                continue
        except Exception:
            continue
        if item.get("actual_eps") is not None:
            return "ALREADY-REPORTED"
        time_str = str(item.get("time", "")).strip()
        if time_str:
            classified = _classify_bmo_amc(time_str, earn_date)
            if classified:
                return classified
        break   # matched record but no usable actual_eps/time classification — fall through below
    if _check_earnings_already_reported(ticker):
        return "ALREADY-REPORTED"
    bz = _fetch_benzinga_earnings_time(ticker, earn_date)
    if bz:
        return bz
    front_iv = _get_atm_iv(client, ticker, current_price, EARNINGS_SPREAD_TARGET_DTE)
    back_iv  = _get_atm_iv(client, ticker, current_price, EARNINGS_SPREAD_TARGET_DTE + 21)
    if front_iv is not None and back_iv is not None and front_iv - back_iv >= EARNINGS_IV_BACKWARDATION_MIN:
        return "AMC"   # front IV still elevated relative to back-month = event still pending
    return "UNKNOWN-TODAY"   # can't confirm still-pending — caller should skip, not guess


def get_earnings_spread_candidates(client) -> list[dict]:
    """
    WATCHLIST tickers (excluding index ETFs — TICKER_SECTOR is empty for
    SPY/QQQ/IWM) with earnings today or tomorrow. Each item:
    {ticker, earn_date, days_away, timing, current_price}.
    Skips tickers whose current price can't be fetched (fail-closed).
    """
    out = []
    today = date.today()
    for ticker in WATCHLIST:
        if not TICKER_SECTOR.get(ticker):
            continue
        for ed in _extract_earnings_dates(ticker):
            days_away = (ed - today).days
            if days_away not in (0, 1):
                continue
            try:
                price = float(yf.Ticker(ticker).fast_info.last_price or 0)
            except Exception:
                price = 0.0
            if price <= 0:
                continue
            timing = _resolve_earnings_timing(client, ticker, ed, price, days_away)
            out.append({
                "ticker": ticker, "earn_date": ed, "days_away": days_away,
                "timing": timing, "current_price": price,
            })
            break
    return out


def run_earnings_spread_scan() -> None:
    """
    Detect WATCHLIST tickers reporting earnings today/tomorrow, build a plan
    for each new one, and send a Telegram approval request. Shared by the
    always-on daemon's earnings_loop (10s dispatch, once per day at a fixed
    trigger time) AND the hourly cron scanner's "--mode earnings-scan"
    dispatch — the cron path exists specifically so a same-day code deploy
    doesn't have to wait for the daemon's next scheduled restart to pick up
    new code: the daemon does ONE git checkout per session and doesn't
    hot-reload, but the cron scanner re-checks-out fresh code every single
    run. Confirmed gap: this feature was pushed live at 2:47 PM ET on
    2026-07-29 (the day it shipped) while the daemon's already-running
    session had checked out at 1:59 PM ET — 48 minutes earlier — so despite
    being "live" in the repo, it never actually executed that day; META
    reported earnings after that close with no offer ever sent. Both
    dispatch paths call this exact function so they can't drift apart, and
    it's idempotent (dedup via _is_alerted_today) — running it more than
    once in a day is safe, it just does nothing after the first offer.

    Gated on is_market_open(): confirmed live 2026-07-29 that calling this
    after-hours (an ad-hoc manual/CLI invocation, not the real cron/daemon
    schedule, which are both already time-windowed) sends real approval
    offers whose EARNINGS_APPROVAL_TIMEOUT_MIN countdown expires overnight,
    well before the next real trading session even opens — an offer for
    tomorrow's earnings that's dead before tomorrow starts is worse than no
    offer at all. The real cron/daemon triggers only ever fire during market
    hours anyway, so this is a no-op gate for them and a real safety net for
    anything else that calls this function.
    """
    print("  📅 Running earnings-spread scan...")
    try:
        if not is_market_open():
            print("  ⏸️  Market closed — earnings scan skipped (avoids sending an "
                  "approval offer whose timer would expire before the next session opens)")
            return
        client = get_alpaca_client()
        if client is None:
            print("  ⚠️  earnings scan skipped — Alpaca unavailable")
            return
        candidates = get_earnings_spread_candidates(client)
        pending = _load_earnings_pending()
        already_offered = {(e["ticker"], e.get("earn_date")) for e in pending}

        # Confirmed live 2026-07-30: when build_earnings_spread_plan() returns
        # None (no liquid leg pair), this loop used to just `continue` —
        # total silence, no Telegram message at all. That day it happened for
        # every single candidate (AAPL included) because a runaway websocket
        # reconnect storm elsewhere in the daemon had rate-limited the
        # account's options data (see dman_daemon.py:stream_loop), so "no
        # liquid leg pair" was actually "couldn't get real data" wearing a
        # liquidity-sounding message — and the user had zero visibility that
        # the scan even ran, let alone why nothing came through. Now tracked
        # and reported once per ticker/day via the same dedup mechanism as
        # successful offers, so a real rerun (hourly cron) doesn't re-spam it.
        skipped_no_legs = []
        for c in candidates:
            key = (c["ticker"], c["earn_date"].isoformat())
            dedup_key = f"{c['ticker']}_EARNSPREAD_OFFER_{c['earn_date'].isoformat()}"
            skip_dedup_key = f"{c['ticker']}_EARNSPREAD_SKIP_{c['earn_date'].isoformat()}"
            if key in already_offered or _is_alerted_today(dedup_key):
                continue
            # _resolve_earnings_timing()'s own docstring says "caller should
            # skip, not guess" on UNKNOWN-TODAY — this loop never actually did
            # that until now; it just passed timing through to the message as
            # a display string, still building and offering a plan regardless.
            # ALREADY-REPORTED (new Benzinga-confirmed state) must skip for
            # the same reason META correctly got filtered out by liquidity
            # alone tonight, but that's a coincidence, not a real check.
            if c["timing"] in ("UNKNOWN-TODAY", "ALREADY-REPORTED"):
                print(f"  ⚠️  {c['ticker']} earnings spread: timing={c['timing']} — skipping, not guessing")
                continue
            plan = build_earnings_spread_plan(
                client, c["ticker"], c["current_price"], c["earn_date"], c["timing"])
            if not plan:
                if not _is_alerted_today(skip_dedup_key):
                    _mark_alerted(skip_dedup_key)
                    skipped_no_legs.append(c["ticker"])
                continue
            _mark_alerted(dedup_key)
            pending.append({
                "ticker": c["ticker"], "earn_date": c["earn_date"].isoformat(),
                "created_at": datetime.now(ET).isoformat(),
                "expires_at": (datetime.now(ET)
                              + timedelta(minutes=EARNINGS_APPROVAL_TIMEOUT_MIN)).isoformat(),
                "status": "awaiting_approval", "plan": plan,
            })
            send_telegram(format_earnings_spread_telegram(plan))
            print(f"  📤 Earnings spread offer sent for {c['ticker']}")
        if skipped_no_legs:
            send_telegram(
                "📅 Earnings spread scan ran, no order sent for: <b>"
                + ", ".join(skipped_no_legs)
                + "</b> — no tradeable long/short leg pair found at any width. Could be "
                  "genuinely illiquid strikes, or Alpaca options data was degraded/"
                  "rate-limited. Worth a manual look if this repeats.")
            print(f"  📤 Skip summary sent for: {', '.join(skipped_no_legs)}")
        _save_earnings_pending(pending)
    except Exception as exc:
        print(f"  ⚠️  earnings scan error: {exc}", file=sys.stderr)


def expire_earnings_spread_offers() -> None:
    """
    Sweeps pending offers for expiry independent of any reply arriving —
    _handle_earnings_approval_reply() only checks expiry when a NEW reply
    comes in, so an offer nobody ever replies to needs this separate sweep
    to actually notify "expired, no order placed" instead of just going
    quiet forever. Shared by the daemon loop and the cron scanner dispatch.
    """
    try:
        pending = _load_earnings_pending()
        if not pending:
            return
        now = datetime.now(ET)
        changed = False
        still_pending = []
        for entry in pending:
            if entry.get("status") == "awaiting_approval":
                try:
                    if now >= datetime.fromisoformat(entry["expires_at"]):
                        send_telegram(
                            f"⏰ {entry['ticker']} earnings spread offer expired — no reply "
                            f"within {EARNINGS_APPROVAL_TIMEOUT_MIN} min, no order placed.")
                        changed = True
                        continue
                except Exception:
                    pass
            still_pending.append(entry)
        if changed:
            _save_earnings_pending(still_pending)
    except Exception as exc:
        print(f"  ⚠️  earnings expiry sweep error: {exc}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9.5 — FILTER: MACRO CALENDAR BLACKOUT (FOMC / NFP)
# ═══════════════════════════════════════════════════════════════════════════

# FOMC decision dates — update annually from federalreserve.gov/monetarypolicy/fomccalendars.htm
# 2027/2028 are estimated; verify and replace each December.
_FOMC_DATES: set[date] = {
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5,  7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025,10, 29), date(2025,12, 17),
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026,10, 28), date(2026,12, 16),
    # 2027 estimated
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28),
    date(2027, 6, 16), date(2027, 7, 28), date(2027, 9, 15),
    date(2027,10, 27), date(2027,12, 15),
    # 2028 estimated
    date(2028, 1, 26), date(2028, 3, 15), date(2028, 4, 26),
    date(2028, 6, 14), date(2028, 7, 26), date(2028, 9, 20),
    date(2028,10, 25), date(2028,12, 13),
}
_FOMC_LAST_CONFIRMED_YEAR = 2026  # raise this after verifying new dates

# NYSE market holidays — algo skips scans and omits these days when computing
# "blackout lifts" dates. Update each December from nyse.com/markets/hours-calendars.
_MARKET_HOLIDAYS: set[date] = {
    # 2025
    date(2025,  1,  1),  # New Year's Day
    date(2025,  1, 20),  # MLK Day
    date(2025,  2, 17),  # Presidents Day
    date(2025,  4, 18),  # Good Friday
    date(2025,  5, 26),  # Memorial Day
    date(2025,  6, 19),  # Juneteenth
    date(2025,  7,  4),  # Independence Day
    date(2025,  9,  1),  # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026,  1,  1),  # New Year's Day
    date(2026,  1, 19),  # MLK Day
    date(2026,  2, 16),  # Presidents Day
    date(2026,  4,  3),  # Good Friday
    date(2026,  5, 25),  # Memorial Day
    date(2026,  6, 19),  # Juneteenth  ← caught by this week's scan
    date(2026,  7,  3),  # Independence Day observed (Jul 4 = Sat)
    date(2026,  9,  7),  # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027 — verify Good Friday from nyse.com each December
    date(2027,  1,  1),  # New Year's Day
    date(2027,  1, 18),  # MLK Day
    date(2027,  2, 15),  # Presidents Day
    date(2027,  3, 26),  # Good Friday (Easter Mar 28)
    date(2027,  5, 31),  # Memorial Day
    date(2027,  6, 18),  # Juneteenth observed (Jun 19 = Sat)
    date(2027,  7,  5),  # Independence Day observed (Jul 4 = Sun)
    date(2027,  9,  6),  # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas observed (Dec 25 = Sat)
}

# CPI release dates (8:30 AM ET) — the single biggest monthly market mover after NFP.
# Drops pre-open so the gap reaction IS the signal, but intraday whipsaws make stops
# unreliable. Blackout day-of and ±1 day. Update each December from bls.gov/schedule.
_CPI_DATES: set[date] = {
    # 2026 — verified from BLS schedule (eskisignal.com/cpi-release-dates-2026)
    date(2026,  1, 14), date(2026,  2, 11), date(2026,  3, 11),
    date(2026,  4, 10), date(2026,  5, 13), date(2026,  6, 10),
    date(2026,  7, 14), date(2026,  8, 12), date(2026,  9,  9),
    date(2026, 10, 14), date(2026, 11, 12), date(2026, 12, 10),
    # 2027 — estimated mid-month; verify from bls.gov each December
    date(2027,  1, 13), date(2027,  2, 10), date(2027,  3, 10),
    date(2027,  4,  9), date(2027,  5, 12), date(2027,  6,  9),
    date(2027,  7, 14), date(2027,  8, 11), date(2027,  9,  8),
    date(2027, 10, 13), date(2027, 11, 10), date(2027, 12,  9),
}

# PPI release dates (8:30 AM ET) — released ~1 business day after CPI; similar gap/whipsaw risk.
# Blackout: same as CPI — pre-10 AM on release day only. Update each December from bls.gov.
#
# Found in the 2026-08-16 review: the estimate below was calendar-day
# (CPI+1 day) arithmetic, not next-BUSINESS-day as the comment claimed.
# April CPI lands on a Friday both 2026 and 2027, so CPI+1 CALENDAR day
# landed on a Saturday both years -- a date BLS never actually releases
# on -- meaning the real April PPI print (the following Monday) had
# zero blackout coverage either year. Corrected those two entries to the
# next business day; every other month's CPI falls on a weekday that
# doesn't hit this collision.
_PPI_DATES: set[date] = {
    # 2026 — estimated as CPI+1 business day (verify from bls.gov/schedule)
    date(2026,  1, 15), date(2026,  2, 12), date(2026,  3, 12),
    date(2026,  4, 13), date(2026,  5, 14), date(2026,  6, 11),
    date(2026,  7, 15), date(2026,  8, 13), date(2026,  9, 10),
    date(2026, 10, 15), date(2026, 11, 13), date(2026, 12, 11),
    # 2027 — estimated; verify from bls.gov each December
    date(2027,  1, 14), date(2027,  2, 11), date(2027,  3, 11),
    date(2027,  4, 12), date(2027,  5, 13), date(2027,  6, 10),
    date(2027,  7, 15), date(2027,  8, 12), date(2027,  9,  9),
    date(2027, 10, 14), date(2027, 11, 11), date(2027, 12, 10),
}

# Manually-maintained: major macro catalysts that aren't on a fixed schedule
# (tariff deadlines, major geopolitical shocks) — unlike FOMC/NFP/CPI there's
# no calendar formula for these, so add entries by hand as they're announced.
# Each needs its own comment: what it is, why it's here, and the source date
# the assessment was made (so a stale/resolved entry is easy to spot and prune).
_MAJOR_MACRO_EVENT_DATES: set[date] = {
    date(2026, 7, 24),  # New global tariffs (10-12.5% on ~60 trading partners,
                        # 99.4% of imports) take effect 12:01 AM ET — replaces
                        # an expiring temporary 10% stopgap. Reporting as of
                        # 2026-07-23 described genuine uncertainty over exact
                        # implementation details hours before effect, and
                        # markets already moved hard on the announcement
                        # (S&P -1.2%, Nasdaq -2.2% same day). Full-day
                        # blackout, same reasoning as FOMC: single-session
                        # whipsaw risk from a high-uncertainty event makes
                        # stops unreliable regardless of setup quality.
    date(2026, 8, 28),  # Jackson Hole Economic Policy Symposium keynote
                        # (symposium runs Aug 27-29; the chair's address has
                        # historically landed Friday morning). Confirmed as
                        # of 2026-08-15: Kevin Warsh was sworn in as Fed
                        # chair 2026-05-13 (54-45 Senate vote — the
                        # narrowest confirmation in Fed history) and this is
                        # his FIRST Jackson Hole keynote as chair, with
                        # inflation readings recently described as
                        # complicating the rate path. A brand-new, narrowly-
                        # confirmed chair's first major policy address is at
                        # least as market-moving as a routine FOMC decision
                        # — same single-session whipsaw/stop-reliability
                        # reasoning applies. ±1 day blackout covers Aug 27-28
                        # (Aug 29 is a Saturday, moot). Update/remove after
                        # the fact if the keynote's actual date/impact
                        # turned out different from this estimate.
}


def _nfp_dates(years: int = 2) -> set[date]:
    """Generate NFP dates (first Friday of each month) for the next `years` years."""
    today = date.today()
    result: set[date] = set()
    for yr in range(today.year, today.year + years + 1):
        for mo in range(1, 13):
            first = date(yr, mo, 1)
            days_to_fri = (4 - first.weekday()) % 7
            result.add(first + timedelta(days=days_to_fri))
    return result


def check_macro_safe() -> tuple[bool, int]:
    """
    Block signals near macro catalysts that create stop-blowing whipsaws.

    Blackout rules (by event type):
    • FOMC decision (2 PM ET): full-day blackout ±1 day — afternoon announcement plus
      prior-day positioning makes the whole session unreliable.
    • NFP / CPI (8:30 AM ET pre-market): block only until 10:00 AM ET on release day.
      By 10 AM the market has had 90 min to price in the number; Gap & Hold setups at
      9:45+ AM are valid reads on the post-number trend.  Day-before / day-after: open.
    • Major unscheduled macro events (tariff deadlines, etc., see
      _MAJOR_MACRO_EVENT_DATES): full-day blackout ±1 day, same reasoning as
      FOMC — applies uniformly to every setup since the concern is stop
      reliability during a high-uncertainty session, not pattern validity.

    Returns (safe, score 0-5).
    """
    try:
        today  = date.today()
        now_et = datetime.now(ET)

        if today.year > _FOMC_LAST_CONFIRMED_YEAR:
            import sys as _sys
            print(f"  ⚠️  FOMC dates beyond {_FOMC_LAST_CONFIRMED_YEAR} are estimated — "
                  f"update _FOMC_DATES from federalreserve.gov", file=_sys.stderr)

        # FOMC: full-day blackout on release day ±1 calendar day
        for ev in _FOMC_DATES:
            days_away = (ev - today).days
            if -1 <= days_away <= MACRO_BLACKOUT:
                return False, 0

        # Major unscheduled macro events: same full-day ±1 blackout as FOMC
        for ev in _MAJOR_MACRO_EVENT_DATES:
            days_away = (ev - today).days
            if -1 <= days_away <= MACRO_BLACKOUT:
                return False, 0

        # NFP / CPI / PPI: pre-market 8:30 AM release — block only before 10:00 AM ET
        for ev in (_nfp_dates() | _CPI_DATES | _PPI_DATES):
            days_away = (ev - today).days
            if days_away == 0 and now_et.hour < 10:
                return False, 0

        return True, 5
    except Exception:
        return True, 5


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 9B — FILTER 05b: INSIDER BUYING (SEC FORM 4, FREE)
# ═══════════════════════════════════════════════════════════════════════════
#
# Free alternative surfaced 2026-08-08 to fill the "more edge" gap left by
# Massive's paid analyst-ratings / bulls-bears-say / corporate-guidance
# tiers (all confirmed 403 not-entitled, and explicitly off-limits per
# direct instruction — no further paid API tiers for now). SEC EDGAR's
# Form 4 insider-transaction data is fully public and free.
#
# Signal: an insider (officer/director/10% owner) trading on the OPEN
# MARKET with their own money is a genuine conviction signal.
# transactionCode == "P" (open-market purchase) is the bullish read;
# "S" (open-market sale) is the bearish mirror, used for SHORT-side
# confirmation. Codes "M" (option exercise) and "F" (tax-withholding share
# disposal) are routine comp mechanics, NOT predictive — confirmed by
# pulling a real AAPL Form 4 containing both and neither meaning anything
# directionally. Only P/S are ever counted.

_sec_cik_map: Optional[dict] = None

def _load_sec_cik_map() -> dict:
    """
    Ticker -> zero-padded 10-digit CIK, for the SEC submissions API.
    SEC publishes one ~800KB JSON for the whole market (no per-ticker
    lookup endpoint) — cached to disk with a 1-week TTL since listings
    change rarely, plus an in-memory cache so one process only reads the
    file once. Falls back to a stale on-disk copy if SEC is unreachable
    (fail-open — this is a bonus signal, never a hard gate).
    """
    global _sec_cik_map
    if _sec_cik_map is not None:
        return _sec_cik_map
    try:
        if os.path.exists(_SEC_CIK_MAP_FILE):
            age = time.time() - os.path.getmtime(_SEC_CIK_MAP_FILE)
            if age < _SEC_CIK_MAP_TTL_S:
                with open(_SEC_CIK_MAP_FILE, "r") as f:
                    _sec_cik_map = json.load(f)
                    return _sec_cik_map
    except Exception:
        pass
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": SEC_EDGAR_USER_AGENT},
            timeout=15,
        )
        if resp.status_code == 200:
            cik_map = {
                str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
                for v in resp.json().values()
            }
            try:
                _write_json_atomic(_SEC_CIK_MAP_FILE, cik_map)
            except Exception:
                pass
            _sec_cik_map = cik_map
            return _sec_cik_map
    except Exception:
        pass
    try:
        if os.path.exists(_SEC_CIK_MAP_FILE):
            with open(_SEC_CIK_MAP_FILE, "r") as f:
                _sec_cik_map = json.load(f)
                return _sec_cik_map
    except Exception:
        pass
    _sec_cik_map = {}
    return _sec_cik_map


_INSIDER_TXN_CACHE: dict[str, tuple[float, list]] = {}
_INSIDER_TXN_CACHE_TTL_S = 6 * 3600   # 6 hr — only hit for already-scored candidates, filings don't move intraday

# SEC EDGAR is slow — live-tested 2026-08-08, a single ticker with a few
# Form 4s to check took 72s (submissions call + per-filing XML fetches,
# each close to the 10s timeout below). score_signal() runs for every
# scanned candidate every cycle, so calling this unconditionally would
# blow scan latency and risk the GitHub Actions workflow timeout. A hard
# per-process budget bounds the worst case regardless of how many
# candidates qualify for a check in a given run; combined with the 6-hr
# cache TTL, coverage rotates across cycles over the course of a day
# rather than stalling any single scan.
_INSIDER_NETWORK_BUDGET = 6
_insider_network_calls_used = 0

def _fetch_recent_insider_transactions(ticker: str, days_back: int = 14, max_filings: int = 2) -> list:
    """
    Recent Form 4 open-market transactions for `ticker`, filtered to
    transactionCode in {P, S} — the only two codes reflecting genuine
    insider conviction. Returns [] on any failure (no CIK match, SEC
    unreachable, no recent Form 4s, or per-process network budget
    exhausted — see _INSIDER_NETWORK_BUDGET) — fail-open.

    Checks only the most recent `max_filings` Form 4s within `days_back`,
    not the full filing history, to keep per-ticker latency bounded —
    this runs for already-scored candidates late in the pipeline, not
    the whole scan universe.
    """
    global _insider_network_calls_used
    cached = _INSIDER_TXN_CACHE.get(ticker)
    if cached and (time.time() - cached[0]) < _INSIDER_TXN_CACHE_TTL_S:
        return cached[1]

    if _insider_network_calls_used >= _INSIDER_NETWORK_BUDGET:
        return []
    _insider_network_calls_used += 1

    import xml.etree.ElementTree as _etree
    result: list = []
    try:
        cik = _load_sec_cik_map().get(ticker.upper())
        if not cik:
            _INSIDER_TXN_CACHE[ticker] = (time.time(), result)
            return result

        headers = {"User-Agent": SEC_EDGAR_USER_AGENT}
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=headers, timeout=10,
        )
        if resp.status_code != 200:
            _INSIDER_TXN_CACHE[ticker] = (time.time(), result)
            return result

        recent       = resp.json().get("filings", {}).get("recent", {})
        forms        = recent.get("form", [])
        accessions   = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        cutoff       = date.today() - timedelta(days=days_back)
        cik_int      = int(cik)

        checked = 0
        for i, form in enumerate(forms):
            if form != "4" or checked >= max_filings:
                continue
            try:
                f_date = date.fromisoformat(filing_dates[i])
            except Exception:
                continue
            if f_date < cutoff:
                continue
            checked += 1
            acc = accessions[i].replace("-", "")
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/form4.xml"
            try:
                rx = requests.get(xml_url, headers=headers, timeout=10)
                if rx.status_code != 200:
                    continue
                root = _etree.fromstring(rx.content)
                owner_el   = root.find(".//reportingOwner/reportingOwnerId/rptOwnerName")
                owner_name = owner_el.text.strip() if owner_el is not None and owner_el.text else ""
                for txn in root.findall(".//nonDerivativeTransaction"):
                    code_el = txn.find("./transactionCoding/transactionCode")
                    code    = code_el.text.strip() if code_el is not None and code_el.text else ""
                    if code not in ("P", "S"):
                        continue
                    shares_el = txn.find("./transactionAmounts/transactionShares/value")
                    price_el  = txn.find("./transactionAmounts/transactionPricePerShare/value")
                    date_el   = txn.find("./transactionDate/value")
                    shares = float(shares_el.text) if shares_el is not None and shares_el.text else 0.0
                    price  = float(price_el.text)  if price_el  is not None and price_el.text  else 0.0
                    result.append({
                        "code":   code,
                        "owner":  owner_name,
                        "shares": shares,
                        "price":  price,
                        "value":  round(shares * price, 2),
                        "date":   date_el.text.strip() if date_el is not None and date_el.text else filing_dates[i],
                    })
            except Exception:
                continue
    except Exception:
        pass

    _INSIDER_TXN_CACHE[ticker] = (time.time(), result)
    return result


def check_insider_activity(ticker: str, bias: str) -> tuple:
    """
    +4 pts (or +2 for a small/token-sized transaction) if a genuine
    open-market insider transaction in the signal's direction landed in
    the last 14 days — code "P" (buy) confirms LONG, code "S" (sale)
    confirms SHORT. Never blocks a signal (always returns ok=True): most
    tickers simply have no recent Form 4 at all, and absence of a filing
    isn't itself bearish or bullish, just silence.
    """
    try:
        txns    = _fetch_recent_insider_transactions(ticker)
        want    = "P" if bias == "LONG" else "S"
        matches = [t for t in txns if t["code"] == want]
        if not matches:
            return True, 0
        total_value = sum(t["value"] for t in matches)
        return True, (4 if total_value >= 25_000 else 2)
    except Exception:
        return True, 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 10 — FILTER 06: FIBONACCI RETRACEMENT
# ═══════════════════════════════════════════════════════════════════════════

FIB_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

def get_fib_levels(df: pd.DataFrame, lookback: int = 60) -> dict[float, float]:
    """
    Calculate Fibonacci retracement levels from the 60-bar swing high/low.
    Returns {fib_ratio: price_level}.
    """
    window = df.iloc[-lookback:]
    hi     = float(window["High"].max())
    lo     = float(window["Low"].min())
    rng    = hi - lo
    return {level: round(hi - level * rng, 2) for level in FIB_LEVELS}


def check_fibonacci(df: pd.DataFrame, entry: float, tolerance: float = 0.015) -> tuple[bool, int]:
    """
    Check if the entry price is within tolerance% of a key Fibonacci level.
    If yes: stronger support/resistance zone → higher confidence.
    Returns (near_fib, score 0-10).
    """
    try:
        fibs = get_fib_levels(df)
        for ratio, price in fibs.items():
            if abs(entry - price) / price <= tolerance:
                # Higher score for stronger fib levels (50% and 61.8%)
                strength = 10 if ratio in (0.500, 0.618) else 7
                return True, strength
        return False, 0
    except Exception:
        return False, 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 11 — FILTER 07: VWAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Compute a rolling VWAP proxy on daily data (true intraday VWAP needs 1m data).
    Uses (H+L+C)/3 * Volume / rolling Volume sum — good approximation for swing trades.
    """
    typical  = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tpv  = (typical * df["Volume"]).rolling(window).sum()
    cum_vol  = df["Volume"].rolling(window).sum()
    return cum_tpv / cum_vol.replace(0, np.nan)


def check_vwap(df: pd.DataFrame, bias: str) -> tuple[bool, int]:
    """
    LONG  : price closing above 20-day VWAP proxy = bullish institutional flow
    SHORT : price closing below 20-day VWAP proxy = bearish institutional flow
    Returns (passes, score 0-8).
    """
    try:
        vwap = compute_vwap(df)
        price = float(df["Close"].iloc[-1])
        vwap_val = float(vwap.iloc[-1])
        if pd.isna(vwap_val):
            return True, 4

        if bias == "LONG":
            passes = price > vwap_val
        else:
            passes = price < vwap_val

        # Score higher if price is meaningfully above/below
        pct_diff = abs(price - vwap_val) / vwap_val * 100
        score = 8 if (passes and pct_diff > 0.5) else (4 if passes else 0)
        return passes, score
    except Exception:
        return True, 4


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 12 — FILTER 08: VOLUME PROFILE (POINT OF CONTROL)
# ═══════════════════════════════════════════════════════════════════════════

def compute_poc(df: pd.DataFrame, lookback: int = 30) -> float:
    """
    Approximate the Point of Control (POC) — the price level with the highest
    traded volume over the lookback period. Calculated by bucketing prices
    into 20 bins and finding the heaviest bucket's midpoint.
    """
    try:
        window  = df.iloc[-lookback:].copy()
        typical = (window["High"] + window["Low"] + window["Close"]) / 3
        bins    = pd.cut(typical, bins=20)
        vol_by_bin = window.groupby(bins, observed=True)["Volume"].sum()
        poc_bin    = vol_by_bin.idxmax()
        poc_price  = float(poc_bin.mid)
        return poc_price
    except Exception:
        return float(df["Close"].iloc[-1])


def check_poc_alignment(df: pd.DataFrame, entry: float, bias: str) -> tuple[bool, int]:
    """
    For LONG  : entry above POC = buyers in control of high-volume zone
    For SHORT : entry below POC = sellers dominating
    Also checks if stop is below POC (long) or above POC (short) for extra protection.
    Returns (aligned, score 0-7).
    """
    try:
        poc = compute_poc(df)
        if bias == "LONG":
            passes = entry > poc
        else:
            passes = entry < poc
        score = 7 if passes else 0
        return passes, score
    except Exception:
        return True, 3


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13 — FILTER 09: MOMENTUM DIVERGENCE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def check_divergence_free(df: pd.DataFrame, bias: str) -> tuple[bool, int]:
    """
    Detect bearish RSI/MACD divergence for longs (or bullish for shorts).
    Divergence = price making new high but RSI/MACD making lower high → WARNING.

    For LONG  : fail if bearish divergence detected (price high, RSI/MACD lower high)
    For SHORT : fail if bullish divergence detected (price low, RSI/MACD higher low)

    Returns (divergence_free, score 0-5). Score=5 means no divergence.
    """
    try:
        window = df.iloc[-20:]
        price  = window["Close"]
        rsi    = window["RSI"] if "RSI" in window.columns else pd.Series(dtype=float)
        macd_h = window["MACD_hist"] if "MACD_hist" in window.columns else pd.Series(dtype=float)

        if len(rsi) < 10 or rsi.isna().all():
            return True, 3

        if bias == "LONG":
            # Bearish divergence: price high > prior high but RSI peak < prior RSI peak
            price_new_high = float(price.iloc[-1]) >= float(price.iloc[-10:-1].max())
            rsi_lower_high = float(rsi.iloc[-1])   <  float(rsi.iloc[-10:-1].max()) - 3
            divergence = price_new_high and rsi_lower_high
        else:
            # Bullish divergence: price low < prior low but RSI trough > prior trough
            price_new_low  = float(price.iloc[-1]) <= float(price.iloc[-10:-1].min())
            rsi_higher_low = float(rsi.iloc[-1])   >  float(rsi.iloc[-10:-1].min()) + 3
            divergence = price_new_low and rsi_higher_low

        return (not divergence), (0 if divergence else 5)
    except Exception:
        return True, 3


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13.4 — FILTER: ATR PERCENTILE / COMPRESSION SCORE
# ═══════════════════════════════════════════════════════════════════════════

def check_atr_percentile(df: pd.DataFrame, setup: str) -> int:
    """
    Rank today's ATR vs the last 126 bars (≈6 months).
    Low percentile = volatility compression = ideal for breakout setups (VCP, Gap, Vol Breakout).
    High percentile = extended volatility = better for mean-reversion setups (OS Bounce, OB Rev).
    Returns score 0-8.
    """
    try:
        atr = df["ATR"].dropna()
        if len(atr) < 20:
            return 4
        window  = atr.iloc[-126:] if len(atr) >= 126 else atr
        current = float(atr.iloc[-1])
        pctile  = float((window.iloc[:-1] < current).sum()) / max(len(window) - 1, 1)

        breakout_setups  = {"Vol Breakout", "Gap & Hold", "VCP", "EMA Pullback", "Morning Runner"}
        reversal_setups  = {"OS Bounce", "OB Reversal", "MACD Cross", "MACD Bear",
                            "Gap & Short", "EMA Breakdown", "Vol Breakdown"}

        if setup in breakout_setups:
            # Compressed ATR = ideal entry before expansion
            return (8 if pctile < 0.25 else
                    6 if pctile < 0.40 else
                    3 if pctile < 0.60 else 0)
        elif setup in reversal_setups:
            # Elevated ATR = momentum available for mean-reversion snap
            return (8 if pctile > 0.70 else
                    5 if pctile > 0.50 else 2)
        return 4   # neutral for unlisted setups
    except Exception:
        return 4


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13.5 — FILTER: ICHIMOKU CLOUD
# ═══════════════════════════════════════════════════════════════════════════

def check_ichimoku(df: pd.DataFrame, bias: str) -> tuple[bool, int]:
    """
    Ichimoku Cloud confluence check (no forward-shift version — uses current values).
    LONG : price above cloud, Tenkan > Kijun, bullish kumo twist = 10 pts.
    SHORT: price below cloud, Tenkan < Kijun, bearish kumo twist = 10 pts.
    Returns (passes, score 0-10).
    """
    try:
        if len(df) < 80:
            return True, 5
        h, l, c = df["High"], df["Low"], df["Close"]
        tenkan = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kijun  = (h.rolling(26).max() + l.rolling(26).min()) / 2
        raw_span_a = (tenkan + kijun) / 2
        raw_span_b = (h.rolling(52).max() + l.rolling(52).min()) / 2
        # Standard Ichimoku: Senkou spans are displaced 26 bars forward;
        # so the cloud visible at the current bar was calculated 26 bars ago.
        span_a = raw_span_a.shift(26)
        span_b = raw_span_b.shift(26)

        price      = float(c.iloc[-1])
        sa         = float(span_a.iloc[-1])
        sb         = float(span_b.iloc[-1])
        cloud_top  = max(sa, sb);  cloud_bot = min(sa, sb)
        tk_bull    = float(tenkan.iloc[-1]) > float(kijun.iloc[-1])
        # Kumo twist: is the FUTURE cloud (26 bars ahead) bullish?
        twist_bull = float(raw_span_a.iloc[-1]) > float(raw_span_b.iloc[-1])

        if bias == "LONG":
            above  = price > cloud_top
            in_cld = cloud_bot <= price <= cloud_top
            passes = above or (in_cld and tk_bull)
            score  = (10 if above and tk_bull and twist_bull else
                      7  if above and (tk_bull or twist_bull) else
                      4  if above else
                      2  if in_cld and tk_bull else 0)
        else:
            below  = price < cloud_bot
            in_cld = cloud_bot <= price <= cloud_top
            passes = below or (in_cld and not tk_bull)
            score  = (10 if below and not tk_bull and not twist_bull else
                      7  if below and (not tk_bull or not twist_bull) else
                      4  if below else
                      2  if in_cld and not tk_bull else 0)

        return passes, score
    except Exception:
        return True, 5


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 13.6 — FILTER: ANCHORED VWAP
# ═══════════════════════════════════════════════════════════════════════════

def compute_anchored_vwap(df: pd.DataFrame, bias: str) -> float:
    """
    VWAP anchored to the most significant recent swing point.
    LONG : anchor to 52-week (or available) low bar — institutional cost basis from capitulation.
    SHORT: anchor to 52-week high bar.
    """
    try:
        lookback = df.iloc[-252:] if len(df) >= 252 else df
        anchor_idx = lookback["Low"].idxmin() if bias == "LONG" else lookback["High"].idxmax()
        anchored   = df.loc[anchor_idx:]
        if len(anchored) < 2:
            return float(df["Close"].iloc[-1])
        typical    = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3
        cum_tpv    = (typical * anchored["Volume"]).cumsum()
        cum_vol    = anchored["Volume"].cumsum()
        avwap      = cum_tpv / cum_vol.replace(0, np.nan)
        return float(avwap.iloc[-1])
    except Exception:
        return float(df["Close"].iloc[-1])


def check_anchored_vwap(df: pd.DataFrame, entry: float, bias: str) -> tuple[bool, int]:
    """
    LONG : entry above anchored VWAP = price trading above institutional avg cost from swing low.
    SHORT: entry below anchored VWAP from swing high.
    Returns (passes, score 0-8).
    """
    try:
        avwap = compute_anchored_vwap(df, bias)
        passes = entry > avwap if bias == "LONG" else entry < avwap
        pct_diff = abs(entry - avwap) / avwap * 100
        score = 8 if (passes and pct_diff > 1.0) else (5 if passes else 0)
        return passes, score
    except Exception:
        return True, 4


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 14 — FILTER 10: ATR-OPTIMIZED STOP PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

def optimize_stop(df: pd.DataFrame, raw_stop: float,
                  entry: float, bias: str) -> float:
    """
    Improve the raw stop price by snapping it to a nearby support/resistance
    level — either the recent swing low/high or an ATR-derived level.

    Principle: stops at REAL price structure hold better than arbitrary ones.
    - Long  : stop = max(raw_stop, recent_swing_low - 0.5*ATR)
    - Short : stop = min(raw_stop, recent_swing_high + 0.5*ATR)

    Tightening is capped at OPTIMIZE_STOP_MIN_RISK_FRACTION of the raw
    risk-per-share. Found 2026-08-16 review: the previous "never tighten so
    much it invalidates 2R" clamp was mathematically dead code — both the
    swing-low/ATR candidate selection AND the old clamp line included
    raw_stop/raw risk as one term of a max()/min(), which guarantees the
    clamp could never actually change the result (it can only ever widen
    toward raw risk, and the selected stop was already at least that wide
    by construction). There was no real cap on tightening at all. On a
    tight-consolidation setup (VCP, Vol Breakout — exactly what this
    system targets) a shallow recent swing low close to entry could
    collapse risk-per-share by 80%+, inflating the resulting share count
    and notional exposure far beyond what Kelly sizing intended for the
    same modeled dollar risk.
    """
    try:
        atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0
        if atr == 0:
            return raw_stop

        window   = df.iloc[-15:]
        raw_risk = abs(entry - raw_stop)
        if raw_risk <= 0:
            return raw_stop
        min_risk = raw_risk * OPTIMIZE_STOP_MIN_RISK_FRACTION

        if bias == "LONG":
            swing_low  = float(window["Low"].min())
            atr_stop   = entry - 1.5 * atr
            # Use the HIGHER of raw and swing-low stop (tighter but real)
            best_stop  = max(swing_low - 0.3*atr, raw_stop, atr_stop)
            # Cap tightening: risk-per-share may never shrink below min_risk.
            best_stop  = min(best_stop, entry - min_risk)
            return round(best_stop, 2)

        else:  # SHORT
            swing_high = float(window["High"].max())
            atr_stop   = entry + 1.5 * atr
            best_stop  = min(swing_high + 0.3*atr, raw_stop, atr_stop)
            best_stop  = max(best_stop, entry + min_risk)
            return round(best_stop, 2)

    except Exception:
        return raw_stop


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 15 — FILTER 11: AI SETUP SCORER (Claude API)
# ═══════════════════════════════════════════════════════════════════════════

def ai_score_signal(signal: ProSignal, regime: dict) -> Optional[int]:
    """
    Send signal details to Claude and get a 1-10 confidence score.
    Returns None if the API call/response couldn't produce a real score
    (no key, timeout, rate limit, malformed response) — the caller must
    treat that as "couldn't get an AI opinion this time," not as a low
    score. Found 2026-08-16 review: this used to return 0 on every one of
    those failure paths, and the caller's `sig.ai_score < 6` check couldn't
    tell "the AI said this is bad" apart from "the AI call itself failed" —
    a rate-limited or timed-out request silently rejected the signal with
    "AI score too low" instead of falling back to the confluence-only
    judgment. Also bumped max_tokens 10->20: a single-digit-or-two answer
    plus any small amount of response overhead was tight enough to
    plausibly truncate the actual answer out of the response entirely,
    which fed the same silent-failure path.

    Prompt is concise to minimise tokens; macro env added for context-aware scoring.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "":
        return None   # AI scoring skipped — no key

    det = regime.get("details", {})
    tlt_trend = det.get("TLT Trend", "flat")
    dxy_trend = det.get("DXY Trend", "flat")
    qqq_ok    = "above" if det.get("QQQ vs EMA20") else "below"
    rate_env  = ("falling (tailwind)" if tlt_trend == "rising"
                 else "rising (headwind)" if tlt_trend == "falling" else "flat")
    dollar    = ("strong (headwind)" if dxy_trend == "strong"
                 else "weak (tailwind)" if dxy_trend == "weak" else "neutral")
    vix_val   = det.get("VIX", "?")

    prompt = f"""You are an expert technical analyst evaluating a US equity swing trade.
Score this setup from 1-10 based on quality and probability of success.
Return ONLY a single integer (1-10), nothing else.

Setup Details:
- Ticker      : {signal.ticker}
- Bias        : {signal.bias}
- Pattern     : {signal.setup}
- Entry       : ${signal.entry}
- Stop Loss   : ${signal.stop}  (risk: ${signal.risk_per_share:.2f}/share)
- Target 1    : ${signal.target1}  (2R)
- Target 2    : ${signal.target2}  (3R)
- R/R Ratio   : {signal.rr}:1
- RSI (14)    : {signal.rsi}
- Rel Volume  : {signal.rvol}x
- Confluence  : {signal.confluence_score}/100
- Setup Reason: {signal.reason}

Macro Environment:
- Market Regime: {regime.get('regime','?')} (SPY score {regime.get('score',0)}/19)
- VIX           : {vix_val}
- QQQ vs EMA20  : {qqq_ok} (tech {'leading ✅' if qqq_ok == 'above' else 'lagging ⚠️'})
- Rates (TLT)   : {rate_env}
- Dollar (DXY)  : {dollar}

Score 8-10 only if: textbook setup, trending market, strong volume, clean R/R, macro tailwinds.
Score 5-7 for borderline or mixed signals.
Score 1-4 for weak setups, macro headwinds, or conflicting signals."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-5",
                "max_tokens": 20,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        content = resp.json()["content"]
        text = next((b["text"] for b in content if b.get("type") == "text"), "").strip()
        if not text:
            return None
        m = re.search(r'\b(10|[1-9])\b', text)
        score = int(m.group(1)) if m else 5
        return max(1, min(10, score))
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 16 — FILTER 12: KELLY CRITERION POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

_PNL_ENTRY_MAX_AGE_DAYS = 45   # comfortably more than one month, bounds file growth

def _normalize_pnl_data(data) -> list[dict]:
    """
    Extract the entries list from already-parsed P&L JSON, in whatever
    shape it happens to be in. Shared by _load_pnl_entries() (reads from
    disk) and sync_daily_pnl_with_remote()/sync_monthly_pnl_with_remote()
    (operate on JSON already parsed off local disk / `git show`) so both
    paths upgrade a pre-migration legacy file the same way — a sync
    running against a not-yet-migrated local file must not treat its one
    running total as "no entries" and silently drop it during the merge.
    """
    if isinstance(data, dict) and "entries" in data:
        return data["entries"]
    if isinstance(data, list):   # backward-compat: never actually shipped this shape
        return data
    # Legacy single-accumulator shape ({"date"/"month": ..., "pnl_pct": ...})
    # from before this file became an entry log — treat its one running
    # total as a single historical entry so an in-place upgrade doesn't
    # silently discard whatever was already accumulated today/this month.
    if isinstance(data, dict) and "pnl_pct" in data:
        _ts = data.get("date") or data.get("month") or datetime.now(ET).date().isoformat()
        return [{"ts": f"{_ts}T00:00:00", "pnl_pct": data["pnl_pct"]}]
    return []


def _load_pnl_entries(filepath: str) -> list[dict]:
    try:
        with open(filepath) as f:
            data = json.load(f)
        return _normalize_pnl_data(data)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _period_pnl_total(filepath: str, period_key: str) -> float:
    """Shared sum logic for get_todays_loss()/get_this_month_loss() —
    period_key is a date or "%Y-%m" string; entries are matched by ts
    prefix (a day's entries all start with its date, a month's with its
    "%Y-%m")."""
    entries = _load_pnl_entries(filepath)
    total = sum(float(e.get("pnl_pct", 0.0)) for e in entries
                if str(e.get("ts", "")).startswith(period_key))
    return round(total, 4)


def _record_period_pnl(filepath: str, pnl_pct: float, max_age_days: int) -> None:
    """
    Shared append logic for record_daily_pnl()/record_monthly_pnl().

    An append-only entry log, not a single mutated running total — found
    2026-08-16 review: dman_daily_pnl.json/dman_monthly_pnl.json are
    written by BOTH the daemon and the hourly cron scanner (both call
    sync_alpaca_fills(), which calls this), from separate checkouts, with
    no semantic merge — unlike every other multi-writer file in this
    project (positions, scan log, win rate, etc.), which all got one
    specifically because a naive git merge can silently keep only one
    side's update. A single mutated total has no way to recover a lost
    update after the fact; an append-only log of individual contributions
    can be merged the exact same way sync_scan_log_with_remote()/
    sync_news_log_with_remote() already merge their own append-only logs
    (union by content, sorted by ts) — see sync_daily_pnl_with_remote().
    ET, not date.today() (system/UTC): the evening cloud daemon session
    runs until 20:05 ET, past midnight UTC, so date.today() on that
    runner reads "tomorrow" for the last ~5 min of every trading day.
    """
    entries = _load_pnl_entries(filepath)
    entries.append({"ts": datetime.now(ET).isoformat(), "pnl_pct": round(pnl_pct, 4)})
    _cutoff = (datetime.now(ET) - timedelta(days=max_age_days)).isoformat()
    entries = sorted([e for e in entries if str(e.get("ts", "")) >= _cutoff],
                      key=lambda e: e.get("ts", ""))
    _write_json_atomic(filepath, {"entries": entries}, indent=2)


def get_todays_loss() -> float:
    """Return today's realized P&L as a signed percentage of account
    size. Negative = loss. Returns 0.0 if no trades recorded today."""
    return _period_pnl_total(DAILY_PNL_FILE, datetime.now(ET).date().isoformat())


def record_daily_pnl(pnl_pct: float) -> None:
    """Append pnl_pct (signed %) as a new entry to today's P&L log.
    See _record_period_pnl()'s docstring for the merge-safety reasoning."""
    _record_period_pnl(DAILY_PNL_FILE, pnl_pct, _PNL_ENTRY_MAX_AGE_DAYS)


def get_this_month_loss() -> float:
    """Return this calendar month's realized P&L as a signed % of account. 0.0 if none."""
    return _period_pnl_total(MONTHLY_PNL_FILE, datetime.now(ET).strftime("%Y-%m"))


def record_monthly_pnl(pnl_pct: float) -> None:
    """Append pnl_pct (signed %) as a new entry to this month's P&L log.
    See _record_period_pnl()'s docstring — same append-only-log reasoning,
    same multi-writer merge safety, same ET-vs-UTC fix."""
    _record_period_pnl(MONTHLY_PNL_FILE, pnl_pct, 400)   # ~13 months of history


_live_equity_cache: dict = {"equity": 0.0, "ts": 0.0}


def get_effective_account() -> float:
    """
    Live Alpaca equity (5-min cache) — sizing compounds automatically as the
    account grows/shrinks, no manual ACCOUNT_SIZE secret updates needed.
    Falls back to ACCOUNT_SIZE (adjusted by today's realized losses) when
    Alpaca is unreachable. Live equity already reflects realized P&L, so no
    double-adjustment is applied on that path.
    """
    try:
        if time.time() - _live_equity_cache["ts"] > 300:
            _client = get_alpaca_client()
            if _client is not None:
                _eq = float(getattr(_client.get_account(), "equity", 0) or 0)
                if _eq > 0:
                    _live_equity_cache["equity"] = _eq
                    _live_equity_cache["ts"]     = time.time()
    except Exception:
        pass
    # Checked outside the try/except above on purpose: a transient exception
    # during the re-fetch attempt (e.g. a network blip) must not also skip
    # this stale-but-still-good cached value — before this fix, any fetch
    # exception fell straight through to the ACCOUNT_SIZE fallback below
    # even when a perfectly usable (just slightly stale) cached equity
    # figure was sitting right here. Found 2026-08-15 while testing the
    # fallback-alert path added below.
    if _live_equity_cache["equity"] > 0:
        return _live_equity_cache["equity"]
    # Reached only when the cache has NEVER been populated (cold start with
    # no successful fetch yet) — a stale-but-previously-good cached value is
    # still returned above and never falls through here, so this path means
    # position sizing is running on the static ACCOUNT_SIZE secret instead
    # of real account state, silently, until the next successful fetch.
    # Added 2026-08-15: no prior alerting existed on this fallback at all.
    if not _is_duplicate_alert("__EQUITY_FALLBACK__"):
        try:
            send_telegram(
                "⚠️ <b>Live equity unavailable</b> — position sizing is using the "
                f"static ACCOUNT_SIZE (${ACCOUNT_SIZE:,.0f}, loss-adjusted) instead "
                "of real Alpaca account equity. Check Alpaca API connectivity."
            )
            _save_last_alert("__EQUITY_FALLBACK__")
        except Exception:
            pass
    pnl_pct = get_todays_loss()   # signed %; 0 or negative on loss days
    adjusted = ACCOUNT_SIZE * (1 + pnl_pct / 100)
    return max(adjusted, ACCOUNT_SIZE * 0.5)   # floor at 50% of configured size


_live_cash_cache: dict = {"cash": None, "ts": 0.0}
_reserved_cash_cache: dict = {"reserved": 0.0, "ts": 0.0}


def get_available_cash() -> Optional[float]:
    """
    Live Alpaca cash balance — 30s cache (much shorter than
    get_effective_account()'s 5-min equity cache, since cash moves as
    orders fill within a single scan while equity barely does). Returns
    None if unreachable; callers must treat that as "can't verify, fail
    safe," never as "assume unlimited cash."

    Added 2026-08-04: equity-based sizing doesn't shrink as positions
    open — only cash does — so nothing was checking whether the account
    actually had real money for a NEW trade on top of what was already
    deployed. Confirmed live: three ordinary signals (FERG, AMZN, W),
    each individually well-sized against equity, collectively cost
    $5,060 against a $3,000 account, pushing cash to -$2,062 — real
    margin usage on an account explicitly meant to stay cash-only.
    """
    try:
        if time.time() - _live_cash_cache["ts"] > 30:
            _client = get_alpaca_client()
            if _client is not None:
                _cash = float(getattr(_client.get_account(), "cash", 0) or 0)
                _live_cash_cache["cash"] = _cash
                _live_cash_cache["ts"]   = time.time()
        return _live_cash_cache["cash"]
    except Exception:
        return _live_cash_cache["cash"]   # last known value, possibly still None


def _reserved_cash_for_open_orders() -> float:
    """
    Dollar notional still reserved by open BUY-side orders that haven't
    filled yet (queued GTC swing entries, earnings-spread MLEG debit
    legs, manual options buys). Alpaca's account.cash field is NOT
    reduced until an order actually FILLS, so _cash_available_for()
    checking raw cash alone can green-light a new order that, stacked on
    top of what's already queued, would overspend the instant those
    queued orders also fill (real risk: GTC entries can sit open
    overnight, well past this function's own 30s cache window). Same 30s
    cache cadence as get_available_cash() so the two figures stay in
    sync with each other. Returns 0.0 (fails OPEN, not closed) on any
    error — deliberately the opposite of get_available_cash()'s fail-
    closed None, since this is a subtracted correction, not the primary
    balance: losing sight of reserved orders only makes the check as
    permissive as it was before this fix existed, never more so.
    """
    try:
        if time.time() - _reserved_cash_cache["ts"] > 30:
            client = get_alpaca_client()
            if client is None:
                return _reserved_cash_cache["reserved"]
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus, OrderSide, OrderClass, AssetClass
            orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))
            reserved = 0.0
            for o in orders:
                if o.order_class == OrderClass.MLEG:
                    # Multi-leg spreads omit side/asset_class at the top level —
                    # qty is sets, limit_price is net debit per set (see
                    # _submit_earnings_spread()'s identical *100 convention).
                    px = float(o.limit_price or 0)
                    if px > 0:   # a net credit (px<0) reserves no cash
                        reserved += float(o.qty or 0) * px * 100
                    continue
                if o.side != OrderSide.BUY:
                    continue   # a protective SELL stop/TP doesn't reserve cash for a NEW buy
                qty = float(o.qty or 0)
                px  = float(o.limit_price or o.stop_price or 0)
                mult = 100 if o.asset_class == AssetClass.US_OPTION else 1
                reserved += qty * px * mult
            _reserved_cash_cache["reserved"] = reserved
            _reserved_cash_cache["ts"] = time.time()
        return _reserved_cash_cache["reserved"]
    except Exception:
        return _reserved_cash_cache["reserved"]


def _cash_available_for(cost: float) -> tuple[bool, str]:
    """
    True if `cost` can be covered by real cash on hand — net of what's
    already reserved by open BUY orders that haven't filled yet — without
    touching margin. Fails CLOSED if cash can't be verified — an unknown
    balance is not a green light to spend real money. See
    get_available_cash() / _reserved_cash_for_open_orders().
    """
    cash = get_available_cash()
    if cash is None:
        return False, "cash balance unavailable — skipping rather than risk margin"
    reserved = _reserved_cash_for_open_orders()
    free = cash - reserved
    if cost > free:
        return False, (f"would need ${cost:.0f}, only ${free:.0f} cash free "
                       f"(${cash:.0f} cash − ${reserved:.0f} reserved by open orders, no margin)")
    return True, ""


def _last_n_earnings_moves(ticker: str, n: int = EARNINGS_DIRECTIONAL_LOOKBACK) -> list[float]:
    """
    Signed overnight-gap % for the n most recent ~quarterly gap days (proxy
    for "past earnings moves"). yf.Ticker().earnings_dates requires the
    'lxml' dependency (HTML-scraping based, not installed, not in
    requirements.txt) — rather than add a fragile new dependency, this
    reuses the same overnight-gap approach used to manually analyze META's
    real earnings history earlier tonight.

    Buckets the trailing ~2 years into ~91-day (quarterly) windows and takes
    the single BIGGEST-magnitude gap within each window, most recent first —
    NOT simply the global top-N gaps by size. Confirmed live on META: the
    global-top-N approach skipped the actual most recent quarter's real move
    (2026-04-30, -7.4%) in favor of an older, bigger one (2024-08-01, +9.7%)
    from over a year prior, which would have given a materially wrong read
    on "recent" directional bias. Quarterly bucketing usually recovers the
    real sequence instead, but has a known remaining failure mode (also
    confirmed live): if some OTHER large, unrelated gap (macro news, sector
    move) falls in the same ~91-day window as a real but smaller earnings
    reaction, the unrelated gap can win that window and mask the real one —
    on META, 2026-07-01 (+7.9%, unrelated) briefly outranked the real
    2026-04-30 (-7.4%) earnings reaction within one window. Accepted as-is
    rather than over-engineered further: this only feeds an OPTIONAL
    directional-bias decision, the safe fallback on ambiguous data is
    always the non-directional double spread (defined risk either way),
    and every trade this influences still goes through the Telegram
    human-approval gate before an order is submitted — that gate is the
    real safety net here, not this function's precision.
    """
    df = _fetch_alpaca_daily(ticker, period_days=730)
    if df is None or len(df) < 10:
        return []
    gaps = []
    for i in range(1, len(df)):
        prev_close = float(df["Close"].iloc[i - 1])
        o = float(df["Open"].iloc[i])
        if prev_close <= 0:
            continue
        gaps.append((df.index[i], (o - prev_close) / prev_close * 100))

    window_days = 91
    today_ts = pd.Timestamp(date.today())
    buckets: dict[int, tuple] = {}   # window index (0 = most recent) -> (abs_gap, signed_gap)
    for ts, g in gaps:
        age_days = (today_ts - ts).days
        if age_days < 0:
            continue
        window = age_days // window_days
        if window not in buckets or abs(g) > buckets[window][0]:
            buckets[window] = (abs(g), g)

    ordered_windows = sorted(buckets.keys())[:n]
    return [buckets[w][1] for w in ordered_windows]


def _earnings_spread_ai_analysis(ticker: str, plan: dict, earn_date: date) -> str:
    """
    Short technicals + fundamentals + historical-move + real recent
    headlines to sit alongside the Telegram YES/NO approval message, so
    approving/rejecting doesn't require opening a separate chart or
    research tab first — the trader has ~30 min before the offer expires
    and isn't always at a desk. Grounded ONLY in real fetched data (price
    history via compute_indicators, Massive's actual consensus estimates,
    the plan's own historical earnings-move pattern, real headlines via
    _fetch_massive_benzinga_news) — the prompt explicitly forbids
    inventing unverifiable "outside research" (analyst names, price
    targets) since this feeds a real trading decision on a live account;
    Claude is asked to reason only from what's supplied and say so
    plainly when data is thin. Massive's analyst-ratings/bulls-bears-say
    endpoints would be the more direct fit for "market sentiment" but
    return 403 on this account's current plan (confirmed live
    2026-08-05) — this sticks to entitled data rather than silently
    no-op'ing. Returns "" on any failure (no key, no price data, API
    error) — this is a nice-to-have annotation, never a reason to
    withhold the offer itself.
    """
    if not ANTHROPIC_API_KEY:
        return ""
    try:
        tech_block = "Technical data unavailable."
        df = fetch_df(ticker, period_days=120)
        if df is not None and len(df) >= 30:
            df = compute_indicators(df)
            last = df.iloc[-1]
            price = float(last["Close"])
            rsi   = float(last["RSI"])   if pd.notna(last.get("RSI"))   else None
            macd  = float(last["MACD"])  if pd.notna(last.get("MACD"))  else None
            ema20 = float(last["EMA20"]) if pd.notna(last.get("EMA20")) else None
            ema50 = float(last["EMA50"]) if pd.notna(last.get("EMA50")) else None
            trend5 = ((price / float(df["Close"].iloc[-6]) - 1) * 100
                      if len(df) >= 6 else None)
            parts = [f"price ${price:.2f}"]
            if rsi is not None:
                parts.append(f"RSI(14) {rsi:.1f}")
            if macd is not None:
                parts.append(f"MACD {'positive' if macd > 0 else 'negative'}")
            if ema20 is not None:
                parts.append(f"{'above' if price > ema20 else 'below'} EMA20")
            if ema50 is not None:
                parts.append(f"{'above' if price > ema50 else 'below'} EMA50")
            if trend5 is not None:
                parts.append(f"5-day trend {trend5:+.1f}%")
            tech_block = ", ".join(parts)

        fund_block = "Consensus estimates unavailable."
        for item in _fetch_massive_earnings(ticker, earn_date, earn_date):
            if str(item.get("ticker", "")).upper() != ticker.upper():
                continue
            est_eps, prev_eps = item.get("estimated_eps"), item.get("previous_eps")
            est_rev, prev_rev = item.get("estimated_revenue"), item.get("previous_revenue")
            if est_eps is not None:
                fund_block = f"consensus EPS estimate ${est_eps}"
                if prev_eps is not None:
                    fund_block += f" (prior qtr actual ${prev_eps})"
                if est_rev is not None:
                    fund_block += f", revenue estimate ${est_rev/1e9:.2f}B"
                    if prev_rev is not None:
                        fund_block += f" (prior ${prev_rev/1e9:.2f}B)"
            break

        moves = plan.get("last_moves_pct", [])
        moves_block = (f"last {len(moves)} post-earnings moves: "
                       + "/".join(f"{m:+.1f}%" for m in moves)) if moves \
            else "no reliable earnings-move history found"

        # Real headlines, not analyst opinions — Massive's analyst-ratings/
        # bull-bear-case endpoints exist but return 403 on this account's
        # plan (confirmed live 2026-08-05, not entitled). News IS entitled
        # and already integrated elsewhere; surfacing actual headlines here
        # gives real "other research" context without inventing anything.
        news_block = "No recent headlines found."
        try:
            _news = _fetch_massive_benzinga_news([ticker], hours_back=48).get(ticker, [])
            if _news:
                news_block = "; ".join(_news[:5])
        except Exception:
            pass

        prompt = f"""You are briefing a retail trader who must reply YES or NO in Telegram within {EARNINGS_APPROVAL_TIMEOUT_MIN} minutes to approve or reject a pre-built earnings options spread. You are not deciding for them, only briefing them. Use ONLY the data given below — do not invent analyst names, price targets, or any fact not listed here; if data is thin, say so plainly instead of guessing.

Ticker: {ticker}
Earnings date: {earn_date.isoformat()} ({plan.get('timing', '?')})
Spread: {plan.get('directional') or 'double-sided'} debit spread, net debit ${plan.get('total_cost', 0):.0f}, max loss ${plan.get('max_loss', 0):.0f}
Technicals: {tech_block}
Fundamentals: {fund_block}
History: {moves_block}
Recent headlines (last 48h): {news_block}

Write exactly 4-5 short plain-text sentences (no markdown, no bullets — this goes straight into a Telegram message): (1) what the technicals suggest about current trend/momentum, (2) what the consensus estimates suggest relative to prior performance, (3) how the historical earnings-move pattern relates to this spread's structure, (4) whether the recent headlines support, contradict, or say nothing useful about the setup, (5) one honest risk or data-gap caveat."""

        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-5",
                "max_tokens": 300,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        return resp.json()["content"][0]["text"].strip()
    except Exception:
        return ""


def build_earnings_spread_plan(client, ticker: str, current_price: float,
                               earn_date: date, timing: str) -> Optional[dict]:
    """
    Orchestrates the whole earnings-spread decision for one candidate: risk
    budget -> directional check -> _find_spread_legs() (one or both sides) ->
    full pricing (net debit, max loss/gain, breakevens). Returns a plan dict
    ready for the Telegram approval message/order submission, or None if
    nothing viable fits (never a degenerate 0-contract/0-cost plan — a skip
    is explicit, with a reason, not silent).
    """
    budget = get_effective_account() * EARNINGS_SPREAD_RISK_PCT

    moves = _last_n_earnings_moves(ticker)
    go_single_sided = None   # None = double spread; "CALL" or "PUT" = single-sided
    if len(moves) >= EARNINGS_DIRECTIONAL_MIN_MOVES:
        same_sign_up   = sum(1 for m in moves if m > 0)
        same_sign_down = sum(1 for m in moves if m < 0)
        avg_mag = sum(abs(m) for m in moves) / len(moves)
        if avg_mag >= EARNINGS_DIRECTIONAL_MIN_AVG_PCT:
            if same_sign_up >= EARNINGS_DIRECTIONAL_MIN_MOVES:
                go_single_sided = "CALL"
            elif same_sign_down >= EARNINGS_DIRECTIONAL_MIN_MOVES:
                go_single_sided = "PUT"

    sides = [go_single_sided] if go_single_sided else ["CALL", "PUT"]
    legs: dict[str, dict] = {}
    for side in sides:
        per_side_budget = budget if len(sides) == 1 else budget / 2
        found = _find_spread_legs(client, ticker, current_price, side, per_side_budget)
        if found:
            legs[side.lower()] = found

    if not legs:
        print(f"  ⚠️  {ticker} earnings spread: no liquid leg pair found on either side — skipping")
        return None

    total_debit_per_share = sum(l["net_debit"] for l in legs.values())
    total_cost = total_debit_per_share * 100   # 1 set
    if total_cost > budget * EARNINGS_SPREAD_BUDGET_SLACK:
        print(f"  ⚠️  {ticker} earnings spread: even minimum-width cost ${total_cost:.0f} "
              f"exceeds budget ${budget:.0f} x {EARNINGS_SPREAD_BUDGET_SLACK} — skipping "
              f"rather than force a degenerate structure")
        return None

    sets = 1   # budget already targets ~1 set at minimum viable width; see open risk in plan doc

    plan = {
        "ticker": ticker, "earn_date": earn_date.isoformat(), "timing": timing,
        "current_price": current_price, "sets": sets,
        "net_debit": round(total_debit_per_share, 2),
        "total_cost": round(total_cost * sets, 2),
        "directional": go_single_sided,
        "last_moves_pct": [round(m, 1) for m in moves],
    }
    for side_key, leg in legs.items():
        max_gain_per_set = (leg["short_strike"] - leg["long_strike"] - leg["net_debit"]) * 100 \
            if side_key == "call" else (leg["long_strike"] - leg["short_strike"] - leg["net_debit"]) * 100
        breakeven = (leg["long_strike"] + leg["net_debit"]) if side_key == "call" \
            else (leg["long_strike"] - leg["net_debit"])
        plan[side_key] = {
            **leg,
            "max_gain": round(max_gain_per_set * sets, 2),
            "breakeven": round(breakeven, 2),
        }
    plan["max_loss"] = plan["total_cost"]
    plan["ai_analysis"] = _earnings_spread_ai_analysis(ticker, plan, earn_date)
    return plan


def kelly_fraction(win_rate: float, avg_win_r: float,
                   avg_loss_r: float = 1.0) -> float:
    """
    Kelly Criterion: f* = (W*B - L) / B
    Where W = win rate, L = loss rate, B = avg win / avg loss

    Fractional Kelly (25% of full Kelly) is used to reduce volatility
    while still sizing larger on higher-edge trades.
    Returns fraction of account to risk (capped at 3%).
    """
    if win_rate <= 0 or avg_win_r <= 0:
        return 0.02   # default to 2%
    w = win_rate
    l = 1 - win_rate
    b = avg_win_r / max(avg_loss_r, 0.01)
    full_kelly = (w * b - l) / b
    fractional = full_kelly * 0.25   # 25% Kelly
    return round(max(0.005, min(0.03, fractional)), 4)


def size_position_kelly(signal: ProSignal, account: float,
                         win_rate: float, avg_win_r: float,
                         avg_loss_r: float = 1.0,
                         vix: float = VIX_SIZE_BASE) -> ProSignal:
    """Apply Kelly-optimal, beta-adjusted, VIX-scaled position sizing to a signal.

    avg_loss_r defaults to 1.0 for any OTHER caller (there are none today,
    but the parameter is new) — the one real caller, score_signal(), always
    passes the setup's actual average loss now. Found 2026-08-16 review:
    this was previously hardcoded inside kelly_fraction() itself with no
    way to pass a real value in, so the payoff ratio b = avg_win/avg_loss
    always used a loss of exactly 1.0 regardless of the setup's real
    average loss — for setup_stats()'s percentage-based avg_win_r values
    (typically several points, e.g. 6-10 for a good setup), that made b
    artificially huge, which made full_kelly (and therefore the fractional
    Kelly size) come out strongly positive for nearly every setup
    regardless of actual win rate, saturating the 3% cap on every trade
    instead of sizing down setups with a real edge below what the
    documented RISK_PER_TRADE (2%) target assumes.
    """
    kf  = kelly_fraction(win_rate, avg_win_r, avg_loss_r)

    # Beta adjustment: scale down proportionally for high-volatility stocks.
    # beta=1.0 → no change; beta=2.0 → half size; beta=0.5 → no increase (capped at 1x).
    beta = signal.beta if signal.beta > 0 else 1.0
    if beta > 1.0:
        kf = kf / beta
    kf = max(0.005, kf)   # floor at 0.5%

    rps = signal.risk_per_share
    if rps <= 0:
        return signal

    # VIX-adjusted sizing: reduce shares proportionally when fear is elevated.
    # At VIX=20 (baseline): 1.0x.  VIX=30: 0.67x.  VIX=40: 0.50x.
    # Never sizes UP below baseline — this is a risk-reduction tool only.
    vix_mult = min(1.0, VIX_SIZE_BASE / max(vix, 10.0))

    risk_budget = account * kf
    raw_shares  = int(risk_budget / rps)
    adj_shares  = int(raw_shares * vix_mult)

    # Found in the 2026-08-16 review: forcing at least 1 share regardless
    # of budget (the old unconditional max(1, ...)) could push real
    # dollar risk several times past the sized Kelly fraction on a wide-
    # stop or higher-priced name -- when risk_budget is smaller than one
    # share's worth of stop distance, actual risk becomes the FULL stop
    # distance (rps), not the sized budget, defeating the entire point of
    # Kelly sizing. Only floor to 1 share when doing so stays within a
    # reasonable rounding margin of the intended budget (1.5x); a name
    # where even 1 share risks meaningfully more than that is skipped
    # (shares=0) instead of force-bought — the caller must treat 0 shares
    # as "sizing failed, don't trade this."
    if adj_shares < 1:
        adj_shares = 1 if rps <= risk_budget * 1.5 else 0

    if vix_mult < 0.98 and raw_shares != adj_shares:
        print(f"     📉 VIX {vix:.1f} size adj: {raw_shares}→{adj_shares} shares "
              f"({vix_mult:.0%} of Kelly)")

    signal.shares     = adj_shares
    signal.cost       = round(adj_shares * signal.entry, 2)
    signal.risk_usd   = round(adj_shares * rps, 2)
    signal.kelly_frac = kf
    signal.vix_adj    = round(vix_mult, 2)
    return signal


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 17 — FILTER 13: WIN RATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    ticker:    str
    date:      str
    bias:      str
    setup:     str
    entry:     float
    exit:      float
    outcome:   str   # "WIN" | "LOSS" | "BE"
    pnl_pct:   float
    score:     int
    is_live:   bool = False   # True = real Alpaca fill, False = backtest simulation.
                               # Added 2026-08-07: dman_win_rate.json's 500 records span
                               # back to 2024 — almost entirely backtest data, with real
                               # trades so sparse they barely register in the rolling-50
                               # window sync_alpaca_fills()/adaptive_min_score() read from.
                               # Confirmed live: a str(status)!="filled" bug meant
                               # sync_alpaca_fills() had NEVER once auto-recorded a real
                               # close (fixed separately) — every real trade before today
                               # only exists here because of manual --mode record calls.
                               # Defaults False so old records without this field (all
                               # backtest-era) classify correctly on load.


class WinRateTracker:
    """
    Persists trade outcomes to disk and computes rolling win rate,
    average R won, and consecutive loss count.
    Auto-tightens MIN_CONFLUENCE when win rate drops below target.
    """

    def __init__(self, filepath: str = WIN_RATE_FILE):
        self.filepath = filepath
        self.records:  list[TradeRecord] = []
        self._load()

    def _load(self):
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            self.records = [TradeRecord(**r) for r in data]
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            self.records = []

    def _save(self):
        # Cap at 500 records — rolling_stats() only needs the last 50.
        # Prevents dman_win_rate.json growing unboundedly and slowing GitHub Actions.
        if len(self.records) > 500:
            self.records = self.records[-500:]
        _write_json_atomic(self.filepath, [asdict(r) for r in self.records], indent=2)

    def record(self, trade: TradeRecord):
        self.records.append(trade)
        self._save()

    def rolling_stats(self, n: int = 50, live_only: bool = False) -> dict:
        """
        Stats over the last n closed trades. live_only=True restricts to
        real Alpaca fills (TradeRecord.is_live) — use this whenever the
        number needs to honestly represent live performance rather than
        the backtest-dominated full pool (see TradeRecord.is_live).
        """
        pool   = [r for r in self.records if r.is_live] if live_only else self.records
        recent = pool[-n:] if len(pool) >= n else pool
        if not recent:
            return {"win_rate": 0.60, "avg_win_r": 2.2,
                    "avg_loss_r": 1.0, "consec_losses": 0,
                    "total": 0, "wins": 0, "losses": 0}

        wins   = [r for r in recent if r.outcome == "WIN"]
        losses = [r for r in recent if r.outcome == "LOSS"]

        win_rate  = len(wins) / len(recent)
        avg_win_r = (sum(r.pnl_pct for r in wins) / len(wins)
                     if wins else 2.2)
        avg_loss_r= (abs(sum(r.pnl_pct for r in losses)) / len(losses)
                     if losses else 1.0)

        # Consecutive losses (from end)
        consec = 0
        for r in reversed(recent):
            if r.outcome == "LOSS":
                consec += 1
            else:
                break

        # Consecutive wins (from end)
        consec_wins = 0
        for r in reversed(recent):
            if r.outcome == "WIN":
                consec_wins += 1
            else:
                break

        return {
            "win_rate":     round(win_rate, 3),
            "avg_win_r":    round(avg_win_r, 2),
            "avg_loss_r":   round(avg_loss_r, 2),
            "consec_losses":consec,
            "consec_wins":  consec_wins,
            "total":        len(recent),
            "wins":         len(wins),
            "losses":       len(losses),
        }

    def setup_stats(self, setup: str, n: int = 30) -> dict:
        """
        Rolling win-rate stats for a specific setup pattern (e.g. "VCP").
        Falls back to aggregate LIVE stats if fewer than 5 real trades exist
        for this setup.

        Live-only (TradeRecord.is_live), unlike rolling_stats()'s own
        default — this is the one function real position sizing
        (size_position_kelly(), via score_signal()) reads from, and mixing
        in backtest-era records here isn't a cosmetic display choice the
        way it is elsewhere (rolling_stats()'s unfiltered default still
        backs adaptive_min_score() deliberately). Found 2026-08-16 review:
        before this filter, "Gap & Hold" showed 475 backtest records and
        exactly 1 real live trade, but setup_stats() reported the blended
        83% win rate as if it were a proven live track record — sizing a
        real position on an account's actual single live data point
        dressed up as 475 trades of confidence.
        """
        recent = [r for r in self.records[-200:] if r.setup == setup and r.is_live][-n:]
        if len(recent) < 5:
            return self.rolling_stats(live_only=True)   # not enough data — use live-only aggregate
        wins   = [r for r in recent if r.outcome == "WIN"]
        losses = [r for r in recent if r.outcome == "LOSS"]
        wr     = len(wins) / len(recent)
        avg_win_r  = sum(r.pnl_pct for r in wins)   / len(wins)   if wins   else 2.2
        avg_loss_r = abs(sum(r.pnl_pct for r in losses)) / len(losses) if losses else 1.0
        consec = 0
        for r in reversed(recent):
            if r.outcome == "LOSS":
                consec += 1
            else:
                break
        return {
            "win_rate":     round(wr, 3),
            "avg_win_r":    round(avg_win_r, 2),
            "avg_loss_r":   round(avg_loss_r, 2),
            "total":        len(recent),
            "wins":         len(wins),
            "losses":       len(losses),
            "consec_losses": consec,
        }

    def adaptive_min_score(self, target_wr: float = 0.80) -> int:
        """
        If rolling win rate < target, raise the min confluence score.
        If rolling win rate > target + 5%, relax it slightly.
        """
        stats = self.rolling_stats()
        wr    = stats["win_rate"]
        base  = MIN_CONFLUENCE

        if stats["total"] < 10:
            return base   # not enough data yet

        if wr < target_wr - 0.10:
            return min(90, base + 10)   # struggling — much stricter
        elif wr < target_wr - 0.05:
            return min(85, base + 5)    # below target — tighten
        elif wr > target_wr + 0.05:
            return max(70, base - 3)    # above target — can relax a bit
        return base

    def setup_performance_drift(self, min_trades: int = SETUP_PERFORMANCE_ALERT_MIN_TRADES,
                                 wr_floor: float = SETUP_PERFORMANCE_ALERT_WR_FLOOR) -> list[dict]:
        """
        Scan every distinct setup with enough LIVE (real Alpaca fill, not
        backtest) trades to mean something, and flag any whose rolling win
        rate has dropped below wr_floor. This is the generalized, automatic
        version of the manual investigation that found Low Float Catalyst's
        0% WR / -24.9% avg loss (which led to the SETUP_MIN_CONFLUENCE
        override above) — that discovery only happened because it was
        chased down by hand after a string of visible losses. Restricted to
        is_live trades deliberately: live fills are the ones with real
        slippage/execution risk baked in, and mixing in backtest-era
        records (which dominate by volume, see TradeRecord.is_live) would
        dilute or hide a real live-only problem.
        """
        live = [r for r in self.records if r.is_live]
        setups = sorted({r.setup for r in live if r.setup})
        drifting = []
        for setup in setups:
            recent = [r for r in live if r.setup == setup][-30:]
            if len(recent) < min_trades:
                continue
            wins   = [r for r in recent if r.outcome == "WIN"]
            losses = [r for r in recent if r.outcome == "LOSS"]
            wr = len(wins) / len(recent)
            if wr < wr_floor:
                avg_loss_r = (abs(sum(r.pnl_pct for r in losses)) / len(losses)
                              if losses else 0.0)
                drifting.append({
                    "setup": setup, "win_rate": round(wr, 3), "total": len(recent),
                    "wins": len(wins), "losses": len(losses),
                    "avg_loss_pct": round(avg_loss_r, 1),
                })
        return drifting

    def print_report(self):
        stats = self.rolling_stats()
        W = 56
        print(f"\n{'═'*W}")
        print(f"  D🔥man Win Rate Tracker  —  Last {stats['total']} trades")
        print(f"{'─'*W}")
        print(f"  Win Rate     : {stats['win_rate']*100:.1f}%"
              f"  ({stats['wins']}W / {stats['losses']}L)")
        print(f"  Avg Win      : +{stats['avg_win_r']:.2f}R")
        print(f"  Avg Loss     : -{stats['avg_loss_r']:.2f}R")
        print(f"  Consec Losses: {stats['consec_losses']}")
        print(f"  Adaptive Min Score: {self.adaptive_min_score()}/100")
        print(f"{'─'*W}")

        # Real-money-only figures — the block above is dominated by backtest
        # data (see TradeRecord.is_live); this is what the account has
        # actually done. Shown separately, never blended into the number
        # above, so a small live sample can't quietly masquerade as the
        # backtest-scale win rate.
        live_stats = self.rolling_stats(live_only=True)
        if live_stats["total"] > 0:
            print(f"  📈 REAL TRADES ONLY ({live_stats['total']} live fill(s)):")
            print(f"     Win Rate: {live_stats['win_rate']*100:.1f}%  "
                  f"({live_stats['wins']}W / {live_stats['losses']}L)")
            if live_stats["total"] < 20:
                print(f"     ⚠️  Sample too small to mean much yet — "
                      f"not used for adaptive scoring.")
            print(f"{'─'*W}")

        by_setup: dict[str,list] = {}
        for r in self.records[-100:]:
            by_setup.setdefault(r.setup, []).append(r)
        print("  BY SETUP:")
        for setup, rs in sorted(by_setup.items(), key=lambda x: -len(x[1])):
            w  = sum(1 for r in rs if r.outcome=="WIN")
            wr = w/len(rs)*100
            print(f"  {setup:<22} {len(rs):>3} trades | {wr:>5.1f}% WR")
        print(f"{'═'*W}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 17.5 — OPEN POSITION TRACKER
# ═══════════════════════════════════════════════════════════════════════════

_ALERT_DEDUP_FILE = "dman_alerts_dedup.json"

def _is_alerted_today(key: str) -> bool:
    """Return True if this alert key was already sent today."""
    try:
        with open(_ALERT_DEDUP_FILE) as _f:
            _d = json.load(_f)
        return _d.get(key, "")[:10] == date.today().isoformat()
    except Exception:
        return False

def _mark_alerted(key: str) -> None:
    """Record that this alert key was sent today."""
    try:
        try:
            with open(_ALERT_DEDUP_FILE) as _f:
                _d = json.load(_f)
        except Exception:
            _d = {}
        _d[key] = datetime.now().isoformat()
        with open(_ALERT_DEDUP_FILE, "w") as _f:
            json.dump(_d, _f)
    except Exception:
        pass

@dataclass
class OpenPosition:
    ticker:     str
    bias:       str
    setup:      str
    entry:      float
    stop:       float
    target1:    float
    target2:    float
    shares:     int
    entry_date: str
    atr:        float = 0.0
    score:      int   = 0
    # ── Earnings vertical/double debit-spread fields — 0/empty for every
    # other position type. legs holds the OCC symbols in submission order
    # (long, short[, long, short] for a double spread) since a spread is one
    # OpenPosition but 2-4 real option legs, unlike every other position type
    # here which maps 1:1 to a single tradable symbol.
    legs:       list[str] = field(default_factory=list)
    spread_qty: int   = 0
    max_loss:   float = 0.0
    max_gain:   float = 0.0
    earn_date:  str   = ""
    stop_stage: str   = "initial"   # "initial" -> "trailing" (equity only —
                                     # options already self-manage via the
                                     # daemon's own momentum-watch loop, see
                                     # _progress_equity_stop_to_trailing()).
                                     # Confirmed live 2026-08-08: CELZ sat at
                                     # +24.58% with its original entry-time
                                     # stop untouched — the T1 alert told the
                                     # human to "move stop to breakeven" but
                                     # nothing ever executed it.
    milestone_gain_alerted: float = 0.0   # furthest P&L% gain milestone already
    milestone_loss_alerted: float = 0.0   # alerted for this position (options only,
                                            # see _check_options_pnl_milestone) — 0.0
                                            # means none yet. Added 2026-08-10 for
                                            # extra visibility into P&L swings on top
                                            # of the existing stop/T1/T2 alerts.
    peak_premium: float = 0.0   # options only — highest premium seen since entry
                                  # (or since T1, once taken). Drives the trailing-
                                  # exit in _monitor_option_position: added 2026-08-10
                                  # so exits react to how the trade is actually
                                  # moving instead of only a fixed stop/T2 number —
                                  # see OPTIONS_TRAIL_ACTIVATE_GAIN_PCT.


def _position_identity(ticker: str, setup: str) -> str:
    """
    Unique key for a tracked position. Multiple options legs on the same
    underlying (e.g. an earnings call+put strangle) share `ticker` but are
    genuinely different positions — use the OCC symbol embedded in `setup`
    instead wherever it identifies one, so ticker-keyed operations (dedup
    on open, merge, close) can't collide two legs into one.

    Confirmed live 2026-08-11: merge_positions_snapshots() keyed purely by
    ticker and silently dropped the SMCI call leg during a routine
    concurrent-write merge ("4 local + 4 remote -> 3 merged") because it
    shares ticker "SMCI" with the SMCI put — the call went completely
    untracked (no stop, no milestone alerts) for hours before this was
    caught by manual account inspection, not by the system itself.
    """
    if setup.startswith("Options Call ") or setup.startswith("Options Put "):
        parts = setup.split()
        if len(parts) >= 3:
            return parts[2]
    return ticker.upper()


class PositionTracker:
    """Persists open positions to disk and provides a live P&L dashboard."""

    def __init__(self, filepath: str = POSITIONS_FILE):
        self.filepath  = filepath
        self.positions: list[OpenPosition] = []
        self._load()

    def _load(self):
        try:
            with open(self.filepath) as f:
                data = json.load(f)
            self.positions = [OpenPosition(**p) for p in data]
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            self.positions = []

    def _save(self):
        _write_json_atomic(self.filepath, [asdict(p) for p in self.positions], indent=2)

    def open(self, pos: OpenPosition) -> bool:
        _ident = _position_identity(pos.ticker, pos.setup)
        self.positions = [p for p in self.positions if _position_identity(p.ticker, p.setup) != _ident]
        if len(self.positions) >= MAX_POSITIONS:
            print(f"  ⚠️  MAX_POSITIONS ({MAX_POSITIONS}) reached — cannot open {pos.ticker}.")
            return False
        self.positions.append(pos)
        self._save()
        return True

    def close(self, ticker: str, occ_symbol: Optional[str] = None) -> Optional[OpenPosition]:
        # occ_symbol disambiguates which leg to close when the ticker alone
        # is shared by multiple open options legs (see _position_identity).
        ident = occ_symbol if occ_symbol else ticker.upper()
        found = next((p for p in self.positions if _position_identity(p.ticker, p.setup) == ident), None)
        self.positions = [p for p in self.positions if _position_identity(p.ticker, p.setup) != ident]
        self._save()
        return found

    def show(self):
        if not self.positions:
            print("\n  No open positions tracked.\n")
            return

        W = 72
        print(f"\n{'═'*W}")
        print(f"  D🔥man Open Positions  —  {datetime.today().strftime('%A %b %d, %Y')}")
        print(f"{'═'*W}")

        total_unreal = 0.0
        for p in self.positions:
            if p.setup.startswith("Earnings "):
                # Comparing STOCK price against entry/stop/target below is
                # meaningless for a spread (already imprecise for existing
                # single-leg options positions too, but a spread's P&L isn't
                # even directionally related to the stock price the same way).
                # Cost/max-loss/max-gain are the real, defined-risk numbers.
                days_in = (date.today() - date.fromisoformat(p.entry_date)).days
                print(f"\n  ◆ {p.ticker}  {p.setup}")
                print(f"    Cost ${p.entry:.0f}  |  Max loss ${p.max_loss:.0f}  |  "
                      f"Max gain ${p.max_gain:.0f}  |  {days_in}d held  |  "
                      f"legs: {', '.join(p.legs)}")
                continue

            if p.setup.startswith("Options Call ") or p.setup.startswith("Options Put "):
                # Found in the 2026-08-16 review: only earnings spreads
                # were excluded from the equity path above -- a single-
                # leg options position fell through to it and had the
                # UNDERLYING STOCK price compared against premium-
                # denominated entry/stop/target fields (e.g. entry=$9.22
                # premium vs. a stock price in the tens/hundreds), the
                # same class of bug this branch fixes for spreads, one
                # tier down. That's not just a cosmetic display bug: a
                # stock price that happens to numerically exceed a
                # premium target field (unrelated scales) fires a FALSE
                # "T2 HIT — take remaining profits" Telegram alert.
                _occ_parts = p.setup.split()
                _occ = _occ_parts[2] if len(_occ_parts) >= 3 else ""
                _snap = _get_option_snapshot(_occ) if _occ else None
                if not _snap:
                    print(f"\n  ◆ {p.ticker}  {p.setup}\n    Unable to fetch live option quote")
                    continue
                _cur_prem = _snap["mid"]
                _ctrs     = max(1, int(p.shares) // 100)
                _unreal   = (_cur_prem - p.entry) * _ctrs * 100
                _pnl_pct  = (_cur_prem - p.entry) / p.entry * 100 if p.entry > 0 else 0
                total_unreal += _unreal
                _days_in = (date.today() - date.fromisoformat(p.entry_date)).days
                print(f"\n  ◆ {p.ticker}  {p.setup}")
                print(f"    Premium ${p.entry:.2f}  →  ${_cur_prem:.2f}  "
                      f"({'+' if _pnl_pct >= 0 else ''}{_pnl_pct:.1f}%)  "
                      f"{'+' if _unreal >= 0 else ''}${_unreal:,.0f}  |  "
                      f"{_days_in}d held  |  {_ctrs}ct")
                print(f"    Stop ${p.stop:.2f}  T1 ${p.target1:.2f}  T2 ${p.target2:.2f}")
                continue

            cur = get_live_price(p.ticker)
            if cur is None:
                print(f"\n  {p.ticker}: unable to fetch price")
                continue

            is_lo   = p.bias == "LONG"
            unreal  = (cur - p.entry) * p.shares if is_lo else (p.entry - cur) * p.shares
            pnl_pct = (cur - p.entry) / p.entry * 100 if is_lo else (p.entry - cur) / p.entry * 100
            total_unreal += unreal

            t1_hit  = cur >= p.target1 if is_lo else cur <= p.target1
            t2_hit  = cur >= p.target2 if is_lo else cur <= p.target2
            stopped = cur <= p.stop    if is_lo else cur >= p.stop

            if t2_hit:
                status = "🎯 T2 HIT — take remaining profits"
                _k = f"{p.ticker}_T2_{date.today().isoformat()}"
                if not _is_alerted_today(_k):
                    send_telegram(
                        f"🎯 <b>T2 HIT</b> — {p.ticker} {p.bias}\n"
                        f"Entry ${p.entry} → Now ${cur:.2f} (+{pnl_pct:.1f}%)  T2 ${p.target2}\n"
                        f"<b>Sell remaining shares.</b> This is the best comfortable exit."
                    )
                    _mark_alerted(_k)
            elif t1_hit:
                status = f"✅ T1 HIT — sell 50% now, move stop to ${p.entry}"
                _k = f"{p.ticker}_T1_{date.today().isoformat()}"
                if not _is_alerted_today(_k):
                    send_telegram(
                        f"✅ <b>T1 HIT</b> — {p.ticker} {p.bias}\n"
                        f"Entry ${p.entry} → Now ${cur:.2f} (+{pnl_pct:.1f}%)  T1 ${p.target1}\n"
                        f"<b>Sell 50% HERE.</b>  Move stop to ${p.entry} (breakeven).  "
                        f"Let rest ride to T2 ${p.target2}."
                    )
                    _mark_alerted(_k)
            elif stopped:
                status = "🛑 AT STOP — exit immediately"
                send_telegram(
                    f"🛑 <b>STOP HIT</b> — {p.ticker} {p.bias}\n"
                    f"Entry ${p.entry} → Now ${cur:.2f} | Stop ${p.stop}\n"
                    f"P&L: {'+' if pnl_pct>=0 else ''}{pnl_pct:.1f}%  Exit NOW."
                )
            else:
                status = "⏳ Active"

            days_in = (date.today() - date.fromisoformat(p.entry_date)).days
            arrow   = "▲" if is_lo else "▼"
            sign    = "+" if unreal >= 0 else ""

            print(f"\n  {arrow} {p.ticker}  {p.bias}  ─  {p.setup}   [{status}]")
            print(f"    Entry ${p.entry}  →  Now ${cur:.2f}  "
                  f"({'+' if pnl_pct>=0 else ''}{pnl_pct:.1f}%)  "
                  f"{sign}${unreal:,.0f}  |  {days_in}d held  |  {p.shares} shares")
            print(f"    Stop  ${p.stop:<10}  T1 ${p.target1:<10}  T2 ${p.target2}")
            if p.atr > 0 and t1_hit and not t2_hit:
                trail = round(p.target1 - p.atr if is_lo else p.target1 + p.atr, 2)
                print(f"    → Trail stop to ${trail} (1 ATR from T1)")

        sign = "+" if total_unreal >= 0 else ""
        print(f"\n{'─'*W}")
        print(f"  Total unrealized P&L : {sign}${total_unreal:,.0f}")
        print(f"{'═'*W}\n")


def merge_positions_snapshots(local_list: list[dict], remote_list: list[dict],
                              closed_identities: Optional[set] = None) -> list[dict]:
    """
    Semantic per-ticker merge for dman_positions.json — replaces git's blind
    "whoever wins the conflict" resolution with a rule that can't regress
    protective state.

    The always-on daemon (60s cadence) and the hourly cron scanner both
    independently monitor and protectively update the SAME open positions
    (raise stop to breakeven after T1, reduce shares after a partial sell),
    on separate concurrency groups that can be active at the same time. A
    naive file-level merge can silently discard whichever side did MORE
    protective work if both push within the same short window.

    Rule per ticker present in both copies: keep whichever record has
    progressed further — fewer shares (a partial sale already executed)
    wins first; a higher stop breaks ties. Every live position here is
    long-equivalent (long equity, long calls, long puts valued by rising
    premium), so "higher stop" is strictly more protective in all cases —
    this assumption breaks only if short equity setups are ever re-enabled
    (ALLOW_SHORTS is False today).

    Tickers present in only one copy are normally kept (union), not
    dropped — EXCEPT `closed_identities`: identities this same process
    already confirmed closed at Alpaca (via sync_alpaca_fills(), ground
    truth) but that are present ONLY in `remote_list` (i.e. absent from
    `local_list`, meaning THIS run's own pt.close() already removed it
    locally) are never resurrected, full stop. Confirmed live 2026-08-13:
    without this, a genuinely-closed position (FGL, stopped out -37.89%)
    resurrected on EVERY git_sync() cycle for 35+ minutes straight, not
    "a few minutes" as originally assumed — sync_positions_with_remote()
    runs BEFORE this cycle's own commit+push, so origin/main never catches
    up to the local close in time for the NEXT cycle's `git show
    origin/main:...` read, and the union rule kept re-adding it back from
    that stale remote copy before every single commit, forever. Guarded on
    local ABSENCE (not a blanket identity filter) so a same-day legitimate
    re-entry on the same ticker — which shows up present in local_list via
    its own pt.open() — is never blocked by a stale tombstone.

    Keyed by _position_identity(), not bare ticker: two options legs on
    the same underlying (e.g. an earnings call+put strangle) share ticker
    but are different positions. Confirmed live 2026-08-11 — a ticker-keyed
    version of this merge silently collapsed a real SMCI call+put pair into
    one record, dropping the call from tracking entirely for hours.

    Options-only progress fields (peak_premium, milestone_gain_alerted,
    milestone_loss_alerted, stop_stage) are NOT covered by the shares/stop
    tie-break above — for an options position those two fields are set once
    at entry and don't change again until T1, so they're tied on nearly
    every merge that happens pre-T1 (the normal, common case), and the
    tie-break's "keep prev" default then discarded whichever side had
    actually progressed further in real time. Confirmed live 2026-08-15:
    a real open UMAC call's peak_premium was found pinned below its own
    entry price, permanently, because of exactly this gap — disarming its
    trailing-exit protection with no error anywhere. Fixed by always
    carrying forward the higher (more-progressed) value of each field from
    whichever side loses the shares/stop comparison, and never letting
    stop_stage regress from "trailing" back to "initial".
    """
    closed_identities = closed_identities or set()
    local_idents = {_position_identity(rec.get("ticker", ""), rec.get("setup", ""))
                    for rec in local_list if rec.get("ticker")}
    by_ident: dict[str, dict] = {}
    for rec in remote_list + local_list:
        t = rec.get("ticker")
        if not t:
            continue
        ident = _position_identity(t, rec.get("setup", ""))
        if ident not in local_idents and ident in closed_identities:
            continue   # confirmed closed by this process; not local's job to re-add it
        prev = by_ident.get(ident)
        if prev is None:
            by_ident[ident] = rec
            continue
        rec_shares, prev_shares = float(rec.get("shares", 0)), float(prev.get("shares", 0))
        rec_stop,   prev_stop   = float(rec.get("stop", 0)),   float(prev.get("stop", 0))
        if rec_shares < prev_shares or (rec_shares == prev_shares and rec_stop > prev_stop):
            winner, loser = rec, prev
        else:
            winner, loser = prev, rec
        merged = dict(winner)
        merged["peak_premium"] = max(float(winner.get("peak_premium", 0) or 0),
                                      float(loser.get("peak_premium", 0) or 0))
        merged["milestone_gain_alerted"] = max(float(winner.get("milestone_gain_alerted", 0) or 0),
                                                float(loser.get("milestone_gain_alerted", 0) or 0))
        merged["milestone_loss_alerted"] = max(float(winner.get("milestone_loss_alerted", 0) or 0),
                                                float(loser.get("milestone_loss_alerted", 0) or 0))
        if loser.get("stop_stage") == "trailing":
            merged["stop_stage"] = "trailing"
        by_ident[ident] = merged
    return list(by_ident.values())


def sync_positions_with_remote() -> None:
    """
    Pre-merge dman_positions.json against origin/main's copy BEFORE staging
    a commit, using merge_positions_snapshots() instead of relying on git's
    line-based conflict resolution. Call this right after `git fetch` and
    before `git add` in the persist step — safe no-op if the file is absent
    on either side, if git/network is unavailable, or if there's nothing to
    merge (single-writer case).
    """
    import subprocess
    try:
        remote_raw = subprocess.run(
            ["git", "show", "origin/main:" + POSITIONS_FILE],
            capture_output=True, text=True, timeout=15,
        )
        if remote_raw.returncode != 0 or not remote_raw.stdout.strip():
            return   # file absent upstream (or git unavailable) — nothing to merge
        remote_list = json.loads(remote_raw.stdout)
    except Exception:
        return

    try:
        with open(POSITIONS_FILE) as f:
            local_list = json.load(f)
    except Exception:
        local_list = []

    if not remote_list and not local_list:
        return
    merged = merge_positions_snapshots(local_list, remote_list,
                                       closed_identities=_recent_closed_identities())
    if merged != local_list:
        _write_json_atomic(POSITIONS_FILE, merged, indent=2)
        print(f"  🔀 Merged dman_positions.json with origin/main "
              f"({len(local_list)} local + {len(remote_list)} remote → {len(merged)} merged)")


def merge_json_lists(local_list: list, remote_list: list, key_fn=None,
                     max_entries: int = None) -> list:
    """
    Generic union-merge for append-only JSON list files that can be
    concurrently written by separate processes (the cron scanner and the
    daemon both append to dman_scan_log.json and dman_win_rate.json on
    independent schedules). Reproduced directly: a rebase conflict on one
    of these files, resolved via `git checkout --theirs -- file` (correct
    for keeping THIS run's own new entry — see the rebase ours/theirs fix
    from 2026-07-23), still replaces the file WHOLESALE — silently
    discarding whichever entries were unique to the losing side, even
    though that side had already been successfully pushed to origin.
    Confirmed as the actual cause of dman_scan_log.json going a full
    trading day (2026-07-27) with zero new entries despite 8+ genuinely
    successful scans: the file is rewritten so frequently (every cron
    scan AND every 10-min daemon scan) that it's the single most
    contested file in the repo.

    Two earlier designs of this function were both wrong in ways only
    caught by testing against the REAL dman_win_rate.json before shipping:
    - v1 deduplicated across the entire combined list. The real file
      already contained 4,852 pre-existing exact-duplicate records
      (unrelated to this fix — almost certainly from re-running backtests
      over overlapping periods without ever deduplicating). A full-history
      dedup silently collapsed 6,282 real records to 1,430 on first run —
      a completely different, undiscussed, much bigger change than "stop
      losing new entries to a race," bundled in as an unreviewed side
      effect that had to be caught and reverted before it reached git.
    - v2 tried a "only dedupe the most recent N" window, but still
      unconditionally concatenated local + remote first. When local and
      remote are IDENTICAL (the common, everything-already-synced case),
      that doubles the file's size on every single invocation, since nothing
      recognizes the two lists share the same already-synced history.

    Correct approach: local is the base, taken exactly as-is — including
    any pre-existing duplicates or quirks it may already contain, none of
    which this function's job is to judge or clean up. Only entries from
    remote whose key_fn identity ISN'T already present anywhere in local
    get appended. If local and remote are identical, remote contributes
    nothing and the result is byte-for-byte local, unchanged. If remote
    has a genuinely new entry local is missing (the actual race being
    fixed), it gets added once.

    max_entries truncation is purely positional (keep the last N, no
    re-sorting) to match WinRateTracker._save()'s own `records[-500:]`.
    A "ts"/"date" field looks like an obvious sort key, but real
    dman_win_rate.json data proved it isn't reliable for this: entries
    come from backtests run over arbitrary historical windows, so the
    date field is NOT chronological-by-insertion (e.g. three consecutive
    real records dated 2024-10-03, 2025-07-29, 2024-11-12). Sorting by
    it before capping would silently keep whichever entries happen to
    have the highest date value instead of the 500 most recently
    appended ones — corrupting exactly what rolling_stats()'s "last N
    trades" is supposed to mean for Kelly sizing / adaptive scoring.
    """
    if key_fn is None:
        # No identity available to tell what's actually new in remote —
        # assume whichever list is longer is at least as complete, rather
        # than blindly concatenating (which could double-count in the
        # common case where both sides already agree).
        return local_list if len(local_list) >= len(remote_list) else remote_list

    local_keys = {key_fn(item) for item in local_list}
    new_from_remote = [item for item in remote_list if key_fn(item) not in local_keys]
    combined = list(local_list) + new_from_remote

    if max_entries is not None and len(combined) > max_entries:
        combined = combined[-max_entries:]
    return combined


def _sync_json_file_via_merge(filepath: str, extract, rebuild, label: str) -> None:
    """
    Shared plumbing for sync_scan_log_with_remote() / sync_win_rate_with_remote()
    / etc: fetch origin's copy, merge with the local copy via the caller-
    supplied extract/rebuild functions (which know each file's specific
    shape — flat list, or a list nested under a dict key), write back only
    if something actually changed. Same fail-safe pattern as
    sync_positions_with_remote(): any git/parse error is a silent no-op,
    never a crash.
    """
    import subprocess
    try:
        remote_raw = subprocess.run(
            ["git", "show", f"origin/main:{filepath}"],
            capture_output=True, text=True, timeout=15,
        )
        if remote_raw.returncode != 0 or not remote_raw.stdout.strip():
            return
        remote_data = json.loads(remote_raw.stdout)
    except Exception:
        return

    try:
        with open(filepath) as f:
            local_data = json.load(f)
    except Exception:
        return

    try:
        local_list, local_extra   = extract(local_data)
        remote_list, remote_extra = extract(remote_data)
    except Exception:
        return

    merged_list = merge_json_lists(local_list, remote_list,
                                   key_fn=lambda x: json.dumps(x, sort_keys=True)
                                   if isinstance(x, (dict, list)) else x)
    if len(merged_list) == len(local_list) and local_extra == remote_extra:
        return   # nothing new from remote and no extra-field change — avoid a needless rewrite
    rebuilt = rebuild(merged_list, local_extra, remote_extra)
    _write_json_atomic(filepath, rebuilt, indent=2)
    print(f"  🔀 Merged {label} with origin/main "
          f"({len(local_list)} local + {len(remote_list)} remote → {len(merged_list)} merged)")


def sync_scan_log_with_remote() -> None:
    """
    dman_scan_log.json — flat list, capped at 20 most recent.

    Confirmed live 2026-08-11: positional `merged[-20:]` (matching
    merge_json_lists()'s own max_entries convention) silently froze this
    file at a full day-old snapshot. merge_json_lists() concatenates as
    local + new-from-remote, so whenever local and remote have each been
    independently appended-and-capped-at-20 by separate runs (the normal
    case — hourly scanner + 60s daemon, each writing its own checkout) and
    their content has diverged, EVERY remote entry can look "new" to
    local's key_fn (byte-for-byte match only) — putting all 20 remote
    entries after local's in the concatenation. `merged[-20:]` then keeps
    only that remote tail, discarding 100% of local's fresh entries
    including the newest one just appended. The rewritten local file then
    exactly matches origin, so the next git diff finds nothing to commit —
    silently repeating forever with no error, no print, nothing to catch.
    Unlike dman_win_rate.json (whose "date" field is backtest data, NOT
    reliably chronological-by-insertion — see merge_json_lists()'s
    docstring), every entry here comes from a single call site using
    datetime.now(ET).isoformat() — always a real wall-clock append time —
    so sorting by it before capping is safe here specifically and fixes
    the eviction: genuinely newest entries always survive regardless of
    which side of the local/remote concatenation they landed on.
    """
    _sync_json_file_via_merge(
        SCAN_LOG_FILE,
        extract=lambda d: (d, None),
        rebuild=lambda merged, _le, _re: sorted(
            merged, key=lambda e: e.get("ts", ""), reverse=True)[:20][::-1],
        label="dman_scan_log.json",
    )


def sync_news_log_with_remote() -> None:
    """
    dman_news_log.json — flat list, capped at NEWS_LOG_MAX_ENTRIES most
    recent. Added 2026-08-15 alongside _log_news_event() itself, proactively
    using the same ts-sort-before-cap merge sync_scan_log_with_remote()
    needed a real production incident to discover was necessary — this file
    has the identical high-frequency multi-writer shape (the cron scanner's
    hourly + premarket-early REST pre-fetch AND the daemon's continuous
    real-time stream both append to it), so the same positional-cap data
    loss is a real risk here from day one, not a hypothetical. Every entry
    uses datetime.now(ET).isoformat() from a single call site, so sorting
    by ts before capping is safe here for the identical reason it is for
    the scan log.
    """
    _sync_json_file_via_merge(
        NEWS_LOG_FILE,
        extract=lambda d: (d, None),
        rebuild=lambda merged, _le, _re: sorted(
            merged, key=lambda e: e.get("ts", ""), reverse=True)[:NEWS_LOG_MAX_ENTRIES][::-1],
        label="dman_news_log.json",
    )


def sync_win_rate_with_remote() -> None:
    """dman_win_rate.json — flat list, capped at 500 most recent (see
    WinRateTracker._save())."""
    _sync_json_file_via_merge(
        WIN_RATE_FILE,
        extract=lambda d: (d, None),
        rebuild=lambda merged, _le, _re: merged[-500:],
        label="dman_win_rate.json",
    )


def sync_live_signals_with_remote() -> None:
    """
    dman_live_signals.json — "pending" list nested in a dict. Unlike
    scan_log/win_rate this list also has entries REMOVED (once resolved),
    not just appended, so a blind union could resurrect an already-
    resolved signal. Accepted deliberately, same reasoning as
    merge_positions_snapshots(): resolve_live_outcomes() re-evaluates
    every pending entry against real price data on the next run regardless
    of how it got there, so a resurrected-then-immediately-re-resolved
    entry self-heals within one cycle — silently losing a signal that
    should still be tracked is the worse failure mode to guard against.
    """
    _sync_json_file_via_merge(
        LIVE_SIGNALS_FILE,
        extract=lambda d: (d.get("pending", []), None),
        rebuild=lambda merged, _le, _re: {"pending": merged},
        label="dman_live_signals.json",
    )


def sync_alpaca_sync_state_with_remote() -> None:
    """
    dman_alpaca_sync.json — recorded_ids list nested in a dict, plus a
    last_sync scalar, plus a closed_identities tombstone dict (see
    _mark_identity_closed()). Losing an entry from recorded_ids risks
    RE-processing an already-recorded Alpaca fill on the next sync,
    double-counting it in the win-rate tracker — this file's whole purpose
    is preventing exactly that, so it's worth protecting the same way.
    last_sync takes whichever of the two ISO timestamps is chronologically
    later — a plain string comparison works here since both are always
    produced by the same isoformat() call site.

    closed_identities is unioned (per-identity, keeping the later
    timestamp) and re-pruned to the same TTL _mark_identity_closed() uses
    — added 2026-08-16 after finding the previous rebuild() silently
    dropped this field on every rewrite (it only ever reconstructed
    last_sync/recorded_ids). That is the actual, still-live root cause of
    the documented FGL resurrection incident: this function runs on nearly
    every cycle (last_sync changes on every sync_alpaca_fills() call, so
    it almost never matches remote's last_sync and a rewrite happens), and
    it runs BEFORE this cycle's own commit — so the tombstone
    merge_positions_snapshots() depends on to keep a genuinely-closed
    position from reappearing was being wiped almost as fast as it was
    written, re-arming the exact bug the tombstone exists to prevent.
    """
    def _extract(d):
        return d.get("recorded_ids", []), (d.get("last_sync", ""), d.get("closed_identities", {}))

    def _rebuild(merged, local_extra, remote_extra):
        local_ts,  local_closed  = local_extra
        remote_ts, remote_closed = remote_extra
        now = time.time()
        merged_closed = dict(local_closed)
        for ident, ts in remote_closed.items():
            try:
                merged_closed[ident] = max(float(ts), float(merged_closed.get(ident, 0)))
            except (TypeError, ValueError):
                continue   # a corrupted timestamp on the remote side must not poison a good local one
        merged_closed = {ident: ts for ident, ts in merged_closed.items()
                         if now - float(ts) < _CLOSED_IDENTITY_TOMBSTONE_S}
        return {
            "last_sync": max(local_ts or "", remote_ts or ""),
            "recorded_ids": merged[-500:],
            "closed_identities": merged_closed,
        }

    _sync_json_file_via_merge(
        ALPACA_SYNC_FILE,
        extract=_extract,
        rebuild=_rebuild,
        label="dman_alpaca_sync.json",
    )


def sync_daily_pnl_with_remote() -> None:
    """
    dman_daily_pnl.json — append-only list of {ts, pnl_pct} entries (see
    record_daily_pnl()). Same multi-writer shape as scan_log/news_log
    (daemon + hourly cron scanner both append via sync_alpaca_fills()), so
    the same union-then-sort merge applies: every entry's ts comes from a
    single datetime.now(ET).isoformat() call site, so sorting by it is
    safe, and a union (not a positional concat-then-cap) means neither
    side's contribution to today's realized P&L can be silently dropped —
    which get_todays_loss() feeds directly into DAILY_LOSS_LIMIT, so a
    lost entry here isn't just a display bug, it's a live risk-guard that
    can silently under-count today's real loss.
    """
    _sync_json_file_via_merge(
        DAILY_PNL_FILE,
        extract=lambda d: (_normalize_pnl_data(d), None),
        rebuild=lambda merged, _le, _re: {
            "entries": sorted(merged, key=lambda e: e.get("ts", ""))[-2000:]
        },
        label="dman_daily_pnl.json",
    )


def sync_monthly_pnl_with_remote() -> None:
    """dman_monthly_pnl.json — same reasoning and shape as
    sync_daily_pnl_with_remote(), feeding MONTHLY_LOSS_LIMIT instead."""
    _sync_json_file_via_merge(
        MONTHLY_PNL_FILE,
        extract=lambda d: (_normalize_pnl_data(d), None),
        rebuild=lambda merged, _le, _re: {
            "entries": sorted(merged, key=lambda e: e.get("ts", ""))[-2000:]
        },
        label="dman_monthly_pnl.json",
    )


def sync_earnings_pending_with_remote() -> None:
    """
    dman_earnings_pending.json — pending list nested in a dict, plus a
    consumed-identity tombstone dict (see _consume_earnings_offer_save()).

    Found in the 2026-08-16 review: unlike every sibling multi-writer
    state file, this one had no dedicated semantic merge at all — just
    git's default whole-file last-writer-wins. That's a live-money risk
    specifically here: this daemon (continuous) and the hourly cron
    scanner both call _handle_earnings_approval_reply() from separate
    checkouts, and these offers gate real earnings-spread orders. If the
    process that just consumed (approved/rejected) an offer loses a git
    merge to a stale remote copy still showing it "awaiting_approval", a
    later YES — meant for a genuinely new offer, or a stray duplicate
    reply — could re-submit a real spread order that was already placed
    (or explicitly rejected) once. Guarded the same way
    merge_positions_snapshots() guards its FGL tombstone: an identity
    this process just consumed is never re-added from remote, full stop.
    """
    def _extract(d):
        if isinstance(d, list):   # pre-migration legacy shape: bare list, no tombstones yet
            return d, {}
        return d.get("pending", []), d.get("consumed", {})

    def _rebuild(merged, local_consumed, remote_consumed):
        now = time.time()
        merged_consumed = dict(local_consumed)
        for ident, ts in remote_consumed.items():
            try:
                merged_consumed[ident] = max(float(ts), float(merged_consumed.get(ident, 0)))
            except (TypeError, ValueError):
                continue   # a corrupted timestamp on the remote side must not poison a good local one
        merged_consumed = {ident: ts for ident, ts in merged_consumed.items()
                           if now - float(ts) < _EARNINGS_OFFER_TOMBSTONE_S}
        merged_pending = [e for e in merged
                          if _earnings_offer_identity(e) not in merged_consumed]
        return {"pending": merged_pending, "consumed": merged_consumed}

    _sync_json_file_via_merge(
        EARNINGS_SPREAD_PENDING_FILE,
        extract=_extract,
        rebuild=_rebuild,
        label="dman_earnings_pending.json",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 18 — CORE SIGNAL DETECTORS (from v2, inline)
# ═══════════════════════════════════════════════════════════════════════════

RVOL_MIN       = 1.3
RVOL_MIN_SHORT = 1.2

def _day2_continuation_not_overextended(entry_price: float, pre_gap_close: float) -> bool:
    """
    Day 2 Continuation's other gates (d1_gap, d1_held, RVOL, RSI) all check
    the SHAPE of day 1's gap — none of them check how far price has already
    run in total by the time day 2's entry fires. Confirmed live 2026-08-06:
    AMZN passed every one of those gates by a wide margin (12.5% d1_gap vs
    4% min, 80% d1_held vs 60% min — not a marginal pass) and still bought
    the exact top of its whole holding period, already +20.8% from the
    pre-gap baseline. A single day's gap can be any size (d1_gap only has a
    floor, no ceiling), so this caps the TOTAL cumulative move instead of
    guessing at a tighter single-day gap limit that would also reject
    perfectly healthy, more modest setups.
    """
    if pre_gap_close <= 0:
        return True   # can't compute — don't block on bad data
    total_move_pct = (entry_price - pre_gap_close) / pre_gap_close * 100
    return total_move_pct <= DAY2_MAX_CUMULATIVE_MOVE_PCT


def _raw_signals(df: pd.DataFrame, ticker: str) -> Optional[ProSignal]:
    """
    Evaluate all long and short patterns; return the highest-RR qualifying signal.
    Quality gates (ATR%, dollar volume) applied once at entry — confluence scorer
    handles fine-grained filtering so individual patterns can be relaxed.
    """
    r, p, p2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    c = float(r["Close"])

    # ── Hard quality gates ────────────────────────────────────────────────
    atr_val  = float(r["ATR"])      if ("ATR"      in r.index and not pd.isna(r["ATR"]))      else 0
    avg_vol  = float(r["AvgVol20"]) if ("AvgVol20" in r.index and not pd.isna(r["AvgVol20"])) else 0
    atr_pct  = atr_val / c * 100 if c > 0 else 0
    avg_dv   = c * avg_vol
    if atr_pct < ATR_PCT_MIN or avg_dv < AVG_DOLLAR_VOL_MIN:
        return None

    def _long(setup, stop, t1_mult=2.0, t2_mult=3.0, reason=""):
        stop = float(stop)
        t1 = round(c + t1_mult * (c - stop), 2)
        t2 = round(c + t2_mult * (c - stop), 2)
        rr = round((t1 - c) / (c - stop), 2) if (c - stop) > 0 else 0
        return ProSignal(ticker, setup, "LONG", round(c, 2), round(stop, 2), t1, t2,
                         rr, round(float(r["RSI"]), 1), round(float(r["RVOL"]), 2), reason)

    def _short(setup, stop, t1_mult=2.0, t2_mult=3.0, reason=""):
        stop = float(stop)
        t1 = round(c - t1_mult * (stop - c), 2)
        t2 = round(c - t2_mult * (stop - c), 2)
        rr = round((c - t1) / (stop - c), 2) if (stop - c) > 0 else 0
        return ProSignal(ticker, setup, "SHORT", round(c, 2), round(stop, 2), t1, t2,
                         rr, round(float(r["RSI"]), 1), round(float(r["RVOL"]), 2), reason)

    rec  = df.iloc[-20:-1]
    res  = float(rec["High"].quantile(0.90))
    candidates: list[ProSignal] = []

    # ── LONG patterns ─────────────────────────────────────────────────────

    # L1: EMA Pullback — disabled (33.3% WR trips consec-loss halt; avg +0.77% not worth it)
    if ENABLE_EMA_PULLBACK:
        if (float(r["EMA20"]) > float(r["EMA50"])
                and min(float(p["Low"]), float(p2["Low"])) <= float(p["EMA20"]) * 1.005
                and c > float(r["EMA20"]) and 35 < float(r["RSI"]) < 65
                and float(r["RVOL"]) >= RVOL_MIN
                and float(r["MACD_hist"]) > float(p["MACD_hist"])):
            sig = _long("EMA Pullback", float(r["EMA20"]) * 0.985,
                        reason=f"EMA20 pullback reclaimed, RVOL {float(r['RVOL']):.1f}x")
            if sig.rr >= MIN_RR:
                candidates.append(sig)

    # L2: Volume Breakout — disabled (39.1% WR / avg -1.53%, losing after tightening)
    if ENABLE_VOL_BREAKOUT:
        if (c > res * 1.005 and float(r["RVOL"]) >= 2.5 and float(r["RSI"]) > 52
                and float(r["MACD"]) > float(r["MACD_sig"])
                and float(r["EMA20"]) > float(r["EMA50"])):
            sig = _long("Vol Breakout", res * 0.985, 2.5, 4.0,
                        reason=f"Broke ${res:.2f} on {float(r['RVOL']):.1f}x vol, EMA trend aligned")
            if sig.rr >= MIN_RR:
                candidates.append(sig)

    # L3: Oversold Bounce — disabled (0% WR / 1 trade, avg -0.47%; reversal setups underperform in BULL regime)
    if ENABLE_OS_BOUNCE:
        if (float(p2["RSI"]) < 33 and float(r["RSI"]) > float(p["RSI"]) > float(p2["RSI"])
                and float(r["RSI"]) > 35
                and c > float(r["EMA9"]) and c > float(r["Open"])
                and float(r["RVOL"]) >= 1.8
                and float(r["EMA20"]) > float(r["EMA50"])
                and float(r["MACD_hist"]) > float(p["MACD_hist"])
                and float(r["STOCH_K"]) > float(r["STOCH_D"])):
            sig = _long("OS Bounce",
                        min(float(p["Low"]), float(p2["Low"])) * 0.99,
                        reason=f"RSI bounced from {float(p2['RSI']):.0f}, EMA9 reclaimed")
            if sig.rr >= MIN_RR:
                candidates.append(sig)

    # L4: MACD Cross Bull — disabled (47.1% WR / 17 trades, below breakeven in BULL regime)
    if ENABLE_MACD_CROSS:
        if (float(p["MACD"]) < float(p["MACD_sig"]) and float(r["MACD"]) > float(r["MACD_sig"])
                and float(r["EMA20"]) > float(r["EMA50"])   # uptrend structure
                and 48 < float(r["RSI"]) < 68               # trend zone, not overbought
                and float(r["RVOL"]) >= 1.5                 # real volume on the cross day
                and float(r["MACD_hist"]) > float(p["MACD_hist"])  # histogram accelerating
                and c > float(r["EMA20"])):                 # price holding above short-term MA
            sig = _long("MACD Cross", float(r["EMA50"]) * 0.98, 2.0, 3.5,
                        reason=f"Fresh MACD bull cross, EMA20>EMA50, RVOL {float(r['RVOL']):.1f}x")
            if sig.rr >= MIN_RR:
                candidates.append(sig)

    # L5: VCP — disabled (backtest: avg -0.94%, no edge)
    if ENABLE_VCP:
        vcp_ok, vcp_reason = detect_vcp(df)
        if vcp_ok:
            vcp_stop = float(df["Low"].iloc[-10:].min()) * 0.99
            sig = _long("VCP", vcp_stop, 2.5, 4.0, reason=vcp_reason)
            if sig.rr >= MIN_RR:
                candidates.append(sig)

    # L6: Gap & Hold — gap up ≥1.5% from prior close, holding above the open
    try:
        gap_pct = (float(r["Open"]) - float(p["Close"])) / float(p["Close"]) * 100
        _gh_dollar_vol = c * float(r.get("AvgVol20", 0))
        if (gap_pct >= 1.5 and c >= float(r["Open"]) * 0.995
                and float(r["RVOL"]) >= 2.0               # raised from 1.5 — real institutional volume
                and float(r["RSI"]) > 50
                and float(r["MACD"]) > float(r["MACD_sig"])
                and float(r["MACD"]) > 0                   # confirmed uptrend, not just recovering
                and float(p["Close"]) > float(p["Open"])   # prior day green — continuation not reversal
                and _gh_dollar_vol >= 500_000              # min $500K avg daily dollar volume
                and _sector_etf_above_ema50(ticker)):
            gap_stop = min(float(r["Low"]) * 0.99, float(r["Open"]) * 0.985)
            sig = _long("Gap & Hold", gap_stop, 2.5, 4.0,
                        reason=f"Gap up +{gap_pct:.1f}% from prior close, holding, RVOL {float(r['RVOL']):.1f}x")
            # Targets: take the larger of R-multiple or gap-echo. For small gaps
            # (1.5-4%) the R-multiple is bigger; for large gaps (8%+) the echo wins.
            _echo_t1 = round(c * (1 + gap_pct / 100), 2)
            _echo_t2 = round(c * (1 + gap_pct / 100 * 1.5), 2)
            sig.target1 = max(sig.target1, _echo_t1)
            sig.target2 = max(sig.target2, _echo_t2)
            _gap_risk   = c - gap_stop
            sig.rr = round((sig.target1 - c) / _gap_risk, 2) if _gap_risk > 0 else 0
            if sig.rr >= MIN_RR:
                candidates.append(sig)
    except Exception:
        pass

    # L7: Morning Runner — news-catalyst gap ≥5%, holding above open.
    # RVOL lowered from 5x to 3x: mega-cap names (NVDA, META) legitimately move
    # with 3-4x RVOL on catalyst days; the 5x bar excluded them with no edge benefit.
    try:
        gap_up = (float(r["Open"]) - float(p["Close"])) / float(p["Close"]) * 100
        if (gap_up >= 5.0 and c >= float(r["Open"]) * 0.97
                and float(r["RVOL"]) >= 3.0
                and 50 <= float(r["RSI"]) <= 72
                and float(r["MACD"]) > float(r["MACD_sig"])):
            mr_stop = min(float(r["Low"]) * 0.99, float(r["Open"]) * 0.96)
            fl_m, sh_pct, _, _cash = _get_short_float_data(ticker)
            float_tag = ""
            if fl_m > 0 and fl_m < 10:
                float_tag = f" | ULTRA-LOW FLOAT {fl_m:.1f}M"
            elif fl_m > 0 and fl_m < 50 and sh_pct >= 10:
                float_tag = f" | Float {fl_m:.0f}M, Short {sh_pct:.0f}%"
            elif fl_m > 0 and fl_m < 50:
                float_tag = f" | Float {fl_m:.0f}M"
            sig = _long("Morning Runner", mr_stop, 2.5, 4.0,
                        reason=f"News gap +{gap_up:.1f}% on {float(r['RVOL']):.1f}x vol, holding open{float_tag}")
            if sig.rr >= MIN_RR:
                candidates.append(sig)
    except Exception:
        pass

    # L8: Day 2 Continuation — yesterday's gap-and-hold follows through today.
    # Pattern: Day 1 gapped ≥4% and closed strong (held ≥ 80% of gap range).
    #          Day 2 opens near or above Day 1 close, RVOL still elevated ≥ 1.5x.
    # Institutional flow is continuous — they don't finish buying in one day.
    try:
        if len(df) >= 3:
            p3 = df.iloc[-3]   # two days ago (the day BEFORE the original gap)
            d1_gap     = (float(p["Open"]) - float(p3["Close"])) / float(p3["Close"]) * 100
            d1_range   = float(p["High"]) - float(p["Open"])
            d1_held    = (float(p["Close"]) - float(p["Open"])) / d1_range if d1_range > 0 else 0
            d2_above   = c >= float(p["Close"]) * 0.98   # today still above Day 1 close
            d2_rvol    = float(r["RVOL"]) >= 1.5
            d2_rsi     = 45 < float(r["RSI"]) < 75
            _d2_dollar = c * float(r.get("AvgVol20", 0))
            d2_not_overextended = _day2_continuation_not_overextended(c, float(p3["Close"]))
            if (d1_gap >= 4.0 and d1_held >= 0.6 and d2_above
                    and d2_rvol and d2_rsi and d2_not_overextended
                    and float(r["MACD"]) > float(r["MACD_sig"])
                    and _d2_dollar >= 500_000):
                d2_stop = round(float(p["Close"]) * 0.97, 2)   # stop below Day 1 close
                sig = _long("Day 2 Continuation", d2_stop, 2.0, 3.5,
                            reason=f"Day 1 gapped +{d1_gap:.1f}%, held {d1_held*100:.0f}% of range; Day 2 holding")
                # T1 = Day 1 gap echoed from entry
                sig.target1 = max(sig.target1, round(c * (1 + d1_gap / 100), 2))
                sig.target2 = max(sig.target2, round(c * (1 + d1_gap / 100 * 1.5), 2))
                _d2_risk = c - d2_stop
                sig.rr = round((sig.target1 - c) / _d2_risk, 2) if _d2_risk > 0 else 0
                if sig.rr >= MIN_RR:
                    candidates.append(sig)
    except Exception:
        pass

    # ── SHORT patterns ────────────────────────────────────────────────────
    if ALLOW_SHORTS:
        sup2 = float(rec["Low"].quantile(0.10))

        # S1: EMA Breakdown — disabled (backtest: 39.5% WR, avg -0.14%, no edge)
        if ENABLE_EMA_BREAKDOWN:
            if (float(r["EMA20"]) < float(r["EMA50"])
                    and max(float(p["High"]), float(p2["High"])) >= float(p["EMA20"]) * 0.995
                    and c < float(r["EMA20"]) and 35 < float(r["RSI"]) < 65
                    and float(r["RVOL"]) >= RVOL_MIN_SHORT
                    and float(r["MACD_hist"]) < float(p["MACD_hist"])):
                sig = _short("EMA Breakdown", float(r["EMA20"]) * 1.015,
                             reason=f"EMA20 rejection, {float(r['RVOL']):.1f}x volume")
                if sig.rr >= MIN_RR:
                    candidates.append(sig)

        # S2: Volume Breakdown — disabled alongside all other short setups.
        # Backtest: insufficient sample size in current bull-dominant setup mix.
        # Also: ALLOW_SHORTS = False prevents live short order submission.
        if ALLOW_SHORTS and False:   # explicit double-guard until shorts are re-evaluated
            try:
                _range = float(r["High"]) - float(r["Low"])
                _atr   = float(r["ATR"]) if not pd.isna(r["ATR"]) else _range
                _news_candle = _range >= _atr * 1.5
            except Exception:
                _news_candle = True
            if (c < sup2 * 0.988 and float(r["RVOL"]) >= 3.5 and float(r["RSI"]) < 32
                    and _news_candle
                    and float(r["MACD"]) < float(r["MACD_sig"])
                    and float(r["EMA20"]) < float(r["EMA50"])
                    and float(r["EMA50"]) < float(p["EMA50"])
                    and float(r["MACD_hist"]) < float(p["MACD_hist"])):
                sig = _short("Vol Breakdown", sup2 * 1.015, 2.5, 4.0,
                             reason=f"Broke ${sup2:.2f} on {float(r['RVOL']):.1f}x vol, range {_range/(_atr or 1):.1f}x ATR")
                if sig.rr >= MIN_RR:
                    candidates.append(sig)

        # S3: Overbought Reversal — disabled (backtest: 41% WR, avg -4.77%, consistent loser)
        if ENABLE_OB_REVERSAL:
            if (float(p2["RSI"]) > 65 and float(r["RSI"]) < float(p["RSI"]) < float(p2["RSI"])
                    and c < float(r["EMA9"]) and c < float(r["Open"])
                    and float(r["RVOL"]) >= 1.0):
                sig = _short("OB Reversal", max(float(p["High"]), float(p2["High"])) * 1.01,
                             reason=f"RSI curling from {float(p2['RSI']):.0f}, EMA9 broken")
                if sig.rr >= MIN_RR:
                    candidates.append(sig)

        # S4: MACD Bear Cross — disabled (0% WR / 1 trade; short in BULL-dominant algo)
        if ENABLE_MACD_BEAR and (float(p["MACD"]) > float(p["MACD_sig"]) and float(r["MACD"]) < float(r["MACD_sig"])
                and float(r["MACD"]) < 0
                and float(r["EMA20"]) < float(r["EMA50"])
                and float(r["EMA50"]) < float(p["EMA50"])
                and 42 <= float(r["RSI"]) <= 58
                and float(r["RVOL"]) >= 1.8
                and float(r["MACD_hist"]) < float(p["MACD_hist"])):
            sig = _short("MACD Bear", float(r["EMA50"]) * 1.02, 2.0, 3.5,
                         reason="Fresh MACD bear cross below zero, EMA20<EMA50 declining")
            if sig.rr >= MIN_RR:
                candidates.append(sig)

        # S5: Gap & Short — disabled (40% WR / avg +1.51% in backtest, consistent drag)
        if ENABLE_GAP_SHORT:
            try:
                gap_dn = (float(p["Close"]) - float(r["Open"])) / float(p["Close"]) * 100
                gap_unfilled = float(r["High"]) < float(p["Close"]) * 0.998
                if (gap_dn >= 3.0 and gap_unfilled and c <= float(r["Open"]) * 1.005
                        and float(r["RVOL"]) >= 3.0 and float(r["RSI"]) < 45
                        and float(r["MACD"]) < float(r["MACD_sig"])
                        and float(r["EMA20"]) < float(r["EMA50"])
                        and float(r["MACD_hist"]) < float(p["MACD_hist"])):
                    gap_stop = max(float(r["High"]) * 1.01, float(r["Open"]) * 1.015)
                    sig = _short("Gap & Short", gap_stop, 2.5, 4.0,
                                 reason=f"Gap down -{gap_dn:.1f}% unfilled, RVOL {float(r['RVOL']):.1f}x")
                    if sig.rr >= MIN_RR:
                        candidates.append(sig)
            except Exception:
                pass

    # L9: Bear Gap Hold — bearish mirror of L6.
    # Gap DOWN ≥1.5%, holding BELOW open (failed recovery), RVOL ≥2x,
    # prior day red, MACD bearish, sector ETF weak.
    # Signal bias = SHORT but ALLOW_SHORTS is False for shares —
    # execution layer routes to _submit_options_put() instead.
    if OPTIONS_ENABLE_PUTS:
        try:
            # Gap % from today's OPEN vs prior close — not today's current/
            # close price. Found 2026-08-16 review: this used `c` (current
            # price) as the gap endpoint, so a stock that opened FLAT and
            # simply drifted down 2% intraday read as a "gap down 2%" and
            # could trigger a real ITM put purchase on ordinary noise, not
            # an actual gap. Matches L6 Gap & Hold's (correct) convention.
            _bg_gap_pct  = (float(p["Close"]) - float(r["Open"])) / float(p["Close"]) * 100
            _bg_dv       = c * float(r.get("AvgVol20", 0))
            if (_bg_gap_pct >= 1.5
                    and c <= float(r["Open"]) * 1.005          # holding at/below open
                    and float(r["RVOL"]) >= 2.0
                    and float(r["RSI"]) < 50
                    and float(r["MACD"]) < float(r["MACD_sig"])
                    and float(r["MACD"]) < 0
                    and float(p["Close"]) < float(p["Open"])    # prior day red
                    and _bg_dv >= 500_000):
                _bg_stop   = max(float(r["High"]) * 1.01, float(r["Open"]) * 1.015)
                sig = _short("Bear Gap Hold", _bg_stop, 2.5, 4.0,
                             reason=f"Gap down -{_bg_gap_pct:.1f}%  holding below open  "
                                    f"RVOL {float(r['RVOL']):.1f}x  bearish MACD")
                # Echo targets: T1 = entry × (1 - gap_pct/100), T2 = 1.5× echo
                _bg_echo_t1 = round(c * (1 - _bg_gap_pct / 100), 2)
                _bg_echo_t2 = round(c * (1 - _bg_gap_pct / 100 * 1.5), 2)
                sig.target1 = min(sig.target1, _bg_echo_t1)   # more aggressive of the two
                sig.target2 = min(sig.target2, _bg_echo_t2)
                # Recompute rr against the (possibly echo-overridden) target1 —
                # matches L6 Gap & Hold's pattern. Found 2026-08-16 review:
                # this was missing here, so both the MIN_RR gate just below
                # and the single-pattern-per-ticker selector at the bottom
                # of this function (max(candidates, key=lambda s: s.rr))
                # were comparing a stale rr that no longer matched the
                # signal's real target whenever the echo target won.
                _bg_risk = _bg_stop - c
                sig.rr = round((c - sig.target1) / _bg_risk, 2) if _bg_risk > 0 else 0
                if sig.rr >= MIN_RR:
                    candidates.append(sig)
        except Exception:
            pass

    if not candidates:
        return None
    return max(candidates, key=lambda s: s.rr)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19 — MASTER CONFLUENCE SCORER
# ═══════════════════════════════════════════════════════════════════════════

def score_signal(signal: ProSignal, df: pd.DataFrame,
                 regime: dict, tracker: WinRateTracker) -> ProSignal:
    """
    Run all 18 filters and compute a 0-100 confluence score.
    Updates the signal in-place with all filter verdicts and scores.
    """
    breakdown: dict[str, int] = {}

    # 1. Market Regime (15 pts)
    regime_ok, regime_score = regime_allows_signal(regime, signal.bias)
    signal.regime_ok  = regime_ok
    breakdown["Regime"] = regime_score

    # 2. Multi-Timeframe (20 pts)
    mtf_ok, mtf_score = check_mtf(signal.ticker, signal.bias)
    # Catalyst override for Gap & Hold / Bear Gap Hold: these are SUDDEN
    # reversal plays (earnings beat/miss, guidance change) — by definition
    # the weekly trend hasn't caught up yet on day one, and waiting for it
    # to confirm defeats the entire purpose of the setup (the move is mostly
    # over by the time a multi-week EMA structure flips). A REAL same-day
    # news catalyst is independent, more current evidence than a lagging
    # weekly indicator, so it overrides MTF for these two setups only.
    # Confirmed against GOOGL/TSLA's July 23 earnings gaps: both scored
    # 96-100/100 and passed every other gate, but mtf_ok=False blocked them
    # solely because their weekly EMA20 was still above EMA50 from the prior
    # uptrend — exactly the scenario this override exists for.
    if (signal.setup in ("Gap & Hold", "Bear Gap Hold")
            and getattr(signal, "news_boost", False) and not mtf_ok):
        mtf_ok, mtf_score = True, max(mtf_score, 10)
    signal.mtf_ok     = mtf_ok
    breakdown["MTF"] = mtf_score

    # 3. Relative Strength (15 pts)
    rs_ok, rs_score   = check_relative_strength(signal.ticker, signal.bias)
    signal.rs_ok      = rs_ok
    breakdown["Rel Strength"] = rs_score

    # 4. Sector (10 pts)
    sec_ok, sec_score = check_sector(signal.ticker, signal.bias)
    signal.sector_ok  = sec_ok
    breakdown["Sector"] = sec_score

    # 4.5 Sector ETF momentum confirmation (8 pts)
    # If the stock's sector ETF is green on the day, money flows are aligned —
    # adds conviction that this isn't an idiosyncratic pop against a falling sector.
    _sector_name = TICKER_SECTOR.get(signal.ticker, "")
    _sector_etf  = SECTOR_ETFS.get(_sector_name, "")
    _etf_score   = 0
    if _sector_etf:
        try:
            _etf_df = fetch_df(_sector_etf)
            if _etf_df is not None and len(_etf_df) >= 2:
                _etf_chg = (float(_etf_df["Close"].iloc[-1]) /
                            float(_etf_df["Close"].iloc[-2]) - 1) * 100
                if signal.bias == "LONG":
                    _etf_score = 8 if _etf_chg >= 1.0 else (4 if _etf_chg > 0 else 0)
                else:
                    _etf_score = 8 if _etf_chg <= -1.0 else (4 if _etf_chg < 0 else 0)
        except Exception:
            pass
    breakdown["Sector ETF"] = _etf_score

    # 4.6 AI theme momentum bonus (+3 pts) — additive on top of the sector
    # score above, not a replacement. AI isn't a GICS/SPDR sector, so an
    # AI-specific move (e.g. NVDA rallying on a chip headline) only ever
    # showed up as generic Technology/XLK momentum otherwise, which can
    # mute a real AI-specific signal against a flat broader tech tape.
    _ai_score = 0
    if signal.ticker in AI_THEME_TICKERS:
        try:
            _ai_df = fetch_df(AI_THEME_ETF)
            if _ai_df is not None and len(_ai_df) >= 2:
                _ai_chg = (float(_ai_df["Close"].iloc[-1]) /
                           float(_ai_df["Close"].iloc[-2]) - 1) * 100
                if signal.bias == "LONG":
                    _ai_score = 3 if _ai_chg >= 1.0 else 0
                else:
                    _ai_score = 3 if _ai_chg <= -1.0 else 0
        except Exception:
            pass
    breakdown["AI Theme"] = _ai_score

    # 4.7 Insider buying confirmation (+4 pts, free SEC Form 4 data) —
    # additive, never a gate. Only fires for tickers with a genuine
    # open-market insider transaction (P=buy for LONG, S=sale for SHORT)
    # in the last 14 days. Gated on partial score-so-far (MTF+RS+Sector+
    # SectorETF+AI, max 56) so the slow SEC lookup (see
    # _INSIDER_NETWORK_BUDGET docstring) only runs for candidates already
    # showing real strength, not the whole scan universe.
    _partial_score = mtf_score + rs_score + sec_score + _etf_score + _ai_score
    insider_score = 0
    if _partial_score >= 25:
        _, insider_score = check_insider_activity(signal.ticker, signal.bias)
    breakdown["Insider"] = insider_score

    # 5. Earnings safety (5 pts)
    earn_ok, earn_score = check_earnings_safe(signal.ticker)
    signal.earnings_ok  = earn_ok
    breakdown["Earnings"] = earn_score

    # 5b. Macro calendar safety (5 pts) — FOMC / NFP blackout
    macro_ok, macro_score = check_macro_safe()
    signal.macro_ok = macro_ok
    breakdown["Macro"] = macro_score

    # 5c. Earnings reaction confirmation (up to 7 pts, free — already-fetched
    # Massive/Benzinga data). Added 2026-08-11: _recent_earnings_surprise()
    # already existed and was surfaced in alert TEXT, but never fed the
    # actual score — a signal could look identical whether or not real
    # numbers backed it up. See check_earnings_reaction() for why this
    # deliberately doesn't average EPS and revenue together (RIOT missed
    # EPS by -106% but beat revenue +14.6% and still rallied +22.9%
    # overnight, confirmed live 2026-08-10 — an averaged score would have
    # scored that setup negative).
    _, earn_react_score = check_earnings_reaction(signal.ticker, signal.bias)
    breakdown["Earn React"] = earn_react_score

    # 6. Fibonacci (10 pts)
    fib_ok, fib_score = check_fibonacci(df, signal.entry)
    signal.fib_ok     = fib_ok
    breakdown["Fibonacci"] = fib_score

    # 7. VWAP (8 pts)
    _, vwap_score = check_vwap(df, signal.bias)
    breakdown["VWAP"] = vwap_score

    # 8. Volume Profile / POC (7 pts)
    _, poc_score  = check_poc_alignment(df, signal.entry, signal.bias)
    breakdown["POC"] = poc_score

    # 9. Divergence-free (5 pts)
    div_free, div_score = check_divergence_free(df, signal.bias)
    signal.divergence_free = div_free
    breakdown["No Divergence"] = div_score

    # 10. RSI sweet-spot bonus (5 pts)
    rsi = signal.rsi
    if signal.bias == "LONG":
        rsi_bonus = 5 if 45 <= rsi <= 62 else (2 if 38 <= rsi < 45 else 0)
    else:
        rsi_bonus = 5 if 38 <= rsi <= 55 else (2 if 55 < rsi <= 62 else 0)
    breakdown["RSI Zone"] = rsi_bonus

    # 11. 52-week high proximity (10 pts for longs; 52wk low proximity for shorts)
    try:
        if signal.bias == "LONG":
            hi52    = float(df["High"].iloc[-252:].max()) if len(df) >= 252 else float(df["High"].max())
            off_hi  = (hi52 - signal.entry) / hi52 * 100
            prox_score = 10 if off_hi <= 5 else (7 if off_hi <= 15 else 0)
        else:
            lo52    = float(df["Low"].iloc[-252:].min()) if len(df) >= 252 else float(df["Low"].min())
            off_lo  = (signal.entry - lo52) / lo52 * 100
            prox_score = 10 if off_lo <= 5 else (7 if off_lo <= 15 else 0)
    except Exception:
        prox_score = 0
    breakdown["52wk Prox"] = prox_score

    # 12. Candlestick pattern (8 pts)
    candle_name, candle_score = detect_candle_pattern(df.iloc[-1], df.iloc[-2], signal.bias)
    signal.candle_pattern = candle_name
    breakdown["Candle"] = candle_score

    # 13. Supertrend direction (8 pts) — ATR-based dynamic trend alignment
    if "ST_bull" in df.columns and not pd.isna(df.iloc[-1]["ST_bull"]):
        st_bull = bool(df.iloc[-1]["ST_bull"])
        st_score = 8 if (signal.bias == "LONG" and st_bull) or (signal.bias == "SHORT" and not st_bull) else 0
    else:
        st_score = 4
    breakdown["Supertrend"] = st_score

    # 14. Per-stock ADX trend strength (5 pts) — uses already-computed ADX/DI columns
    r_last = df.iloc[-1]
    adx_v = float(r_last["ADX"])      if ("ADX"      in r_last.index and not pd.isna(r_last["ADX"]))      else 0
    pdi_v = float(r_last["PLUS_DI"])  if ("PLUS_DI"  in r_last.index and not pd.isna(r_last["PLUS_DI"]))  else 0
    mdi_v = float(r_last["MINUS_DI"]) if ("MINUS_DI" in r_last.index and not pd.isna(r_last["MINUS_DI"])) else 0
    if signal.bias == "LONG":
        adx_score = 5 if adx_v >= 25 and pdi_v > mdi_v else (2 if adx_v >= 20 else 0)
    else:
        adx_score = 5 if adx_v >= 25 and mdi_v > pdi_v else (2 if adx_v >= 20 else 0)
    breakdown["ADX Trend"] = adx_score

    # 15. Ichimoku Cloud (10 pts)
    _, ichi_score = check_ichimoku(df, signal.bias)
    breakdown["Ichimoku"] = ichi_score

    # 16. Anchored VWAP (8 pts)
    _, avwap_score = check_anchored_vwap(df, signal.entry, signal.bias)
    breakdown["Anchored VWAP"] = avwap_score

    # 17. ATR percentile / compression score (0-8 pts)
    breakdown["ATR Pctile"] = check_atr_percentile(df, signal.setup)

    # 18. Regime-adaptive setup bonus (0-8 pts)
    _rtype = regime.get("regime", "CHOP")
    _mom_long  = {"Vol Breakout", "Gap & Hold", "VCP", "EMA Pullback", "Morning Runner"}
    _mom_short = {"Vol Breakdown", "Gap & Short", "EMA Breakdown"}
    _rev_long  = {"OS Bounce", "MACD Cross"}
    _rev_short = {"OB Reversal", "MACD Bear"}
    if _rtype == "BULL" and signal.bias == "LONG":
        _rs_bonus = 8 if signal.setup in _mom_long  else 4
    elif _rtype == "BEAR" and signal.bias == "SHORT":
        _rs_bonus = 8 if signal.setup in _mom_short else 4
    elif _rtype == "CHOP":
        _pref = _rev_long if signal.bias == "LONG" else _rev_short
        _rs_bonus = 8 if signal.setup in _pref else 3
    else:
        _rs_bonus = 4
    breakdown["RegimeSetup"] = _rs_bonus

    # 19. Short float / squeeze potential (0-10 pts) — Gap & Hold and Morning Runner
    if signal.setup in {"Morning Runner", "Gap & Hold"}:
        fl_m, sh_pct, _, _cash = _get_short_float_data(signal.ticker)
        if fl_m > 0 and fl_m < 10:            # ultra-low float (<10M) — wildfire move potential
            float_score = 10
        elif fl_m > 0 and fl_m < 50 and sh_pct >= 15:   # low float + high short → squeeze
            float_score = 10
        elif fl_m > 0 and fl_m < 50 and sh_pct >= 10:   # low float + moderate short
            float_score = 7
        elif fl_m > 0 and fl_m < 50:                     # low float alone
            float_score = 4
        elif sh_pct >= 20:                               # high short interest regardless of float
            float_score = 5
        else:
            float_score = 0
        breakdown["Float/Short"] = float_score
    else:
        breakdown["Float/Short"] = 0

    # 20. RVOL tier bonus (0-6 pts) — live scorer was missing this; backtest already has it
    rvol_bonus = 6 if signal.rvol >= 3.0 else (3 if signal.rvol >= 2.0 else 0)
    breakdown["RVOL Tier"] = rvol_bonus

    # 21. Gap size bonus (0-5 pts) — Gap & Hold only; larger gaps = stronger institutional conviction
    if signal.setup == "Gap & Hold" and len(df) >= 2:
        try:
            _gap_pct = (float(df["Open"].iloc[-1]) - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2]) * 100
            gap_bonus = 5 if _gap_pct >= 5.0 else (3 if _gap_pct >= 3.0 else 0)
        except Exception:
            gap_bonus = 0
        breakdown["Gap Size"] = gap_bonus
    else:
        breakdown["Gap Size"] = 0

    # 22. News catalyst recency (0-5 pts) — confirmed headline in last 4 hours
    breakdown["News Catalyst"] = 5 if getattr(signal, "news_boost", False) else 0

    # Populate context fields on the signal
    signal.atr  = float(r_last["ATR"]) if ("ATR" in r_last.index and not pd.isna(r_last["ATR"])) else 0.0
    signal.beta = get_beta(signal.ticker)

    total = sum(breakdown.values())
    signal.confluence_score = min(100, total)
    signal.score_breakdown  = breakdown

    # Apply ATR stop optimizer
    signal.stop = optimize_stop(df, signal.stop, signal.entry, signal.bias)

    # RR against the (possibly ATR-tightened) stop — target1/target2 are
    # deliberately left untouched here. Found 2026-08-16 review: this block
    # used to unconditionally overwrite both targets with a flat 2.0x/3.0x
    # multiplier of the NEW stop distance, discarding whatever
    # _raw_signals() had actually set — Gap & Hold's and Bear Gap Hold's
    # gap-echo targets (a specific technical price level the setup is
    # betting on, not a multiple of risk — rescaling it against a
    # different stop would be wrong in the OTHER direction too) and every
    # setup's real multiplier (2.5x/4.0x for several setups, not a
    # universal 2.0x/3.0x). A second, independently hardcoded multiplier
    # (also 2.5x/4.0x, not this function's 2.0x/3.0x) then got reapplied
    # again at live order-submission time on top of THIS overwrite — so
    # what got logged/alerted here never matched what the broker bracket
    # order actually used, and the WinRateTracker's "ground truth" live
    # outcome resolution was being scored against a target the trade never
    # had. optimize_stop() only refines WHERE the stop sits based on
    # ATR/swing structure; it was never meant to imply the profit target
    # should move too.
    rps = abs(signal.entry - signal.stop)
    if rps > 0:
        signal.rr = round(abs(signal.target1 - signal.entry) / rps, 2)

    # Kelly sizing — prefer setup-specific win rate when enough data exists
    setup_s  = tracker.setup_stats(signal.setup)
    _vix_now = float(regime.get("details", {}).get("VIX", VIX_SIZE_BASE))
    signal = size_position_kelly(
        signal,
        account    = get_effective_account(),
        win_rate   = max(0.5, setup_s["win_rate"]),
        avg_win_r  = max(1.5, setup_s["avg_win_r"]),
        avg_loss_r = max(1.0, setup_s["avg_loss_r"]),
        vix        = _vix_now,
    )

    # Final weighted score: confluence (70%) + AI score (30%)
    signal.final_score = round(signal.confluence_score * 0.70 +
                                signal.ai_score * 10 * 0.30, 1)

    return signal


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.5 — SMALL-CAP / LOW FLOAT CATALYST MODULE (Dman style)
# ═══════════════════════════════════════════════════════════════════════════

_reverse_split_cache: dict[tuple[str, int], bool] = {}

def _is_recent_reverse_split(ticker: str, days: int = 45) -> bool:
    """
    Return True if ticker had a reverse split within `days` days.
    Post-RS plays are Professor Dman's #1 category: float collapses, shorts get trapped.

    Cached per session (like its sibling _get_short_float_data()) — found
    in the 2026-08-16 review: this had no cache at all, unlike every
    other yfinance-backed lookup in this file, so the same live
    yf.Ticker.splits call could fire up to 3x for one candidate within a
    single scan pass. Split history for a given `days` window doesn't
    change within a session, so this is safe to cache unconditionally.
    """
    _key = (ticker, days)
    if _key in _reverse_split_cache:
        return _reverse_split_cache[_key]
    try:
        splits = yf.Ticker(ticker).splits
        if splits is None or splits.empty:
            result = False
        else:
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
            recent = splits[splits.index >= cutoff]
            result = bool((recent < 1.0).any())   # ratio < 1.0 = reverse split
    except Exception:
        result = False
    _reverse_split_cache[_key] = result
    return result


def _smallcap_pullback_tolerance_pct(gap_pct: float) -> float:
    """
    How far off today's high a low-float candidate may sit before the
    intraday-high pullback guard blocks it — scaled down for extreme gaps.
    Confirmed live 2026-08-06: CLRO gapped +217% intraday, CELZ +70%, both
    far beyond the ~15-50% range SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT was
    originally calibrated against. A stock that's already run 200%+ gives
    back ground faster and harder than one that's gapped a more modest
    amount, so the same flat cushion isn't equally safe at every gap size.
    """
    if gap_pct >= 150.0:
        return 6.0
    if gap_pct >= 75.0:
        return 9.0
    return SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT   # 12.0, the original baseline


def detect_low_float_catalyst(df: pd.DataFrame, ticker: str) -> Optional[ProSignal]:
    """
    Detects Professor Dman's core small-cap setup (from 2yr Discord data):
    - Float < 5M (ultra-nano < 1M is the sweet spot)
    - MACD "curling upside" at 20D/50D MA support — buys OVERSOLD, not above EMA
    - RVOL spike as catalyst proxy
    - Bonus: post-RS, high short interest, high insider ownership
    Bypasses AVG_DOLLAR_VOL_MIN — micro-caps are illiquid on normal days by definition.
    """
    if not ENABLE_SMALLCAP:
        return None
    try:
        r  = df.iloc[-1]
        p  = df.iloc[-2]
        c  = float(r["Close"])

        # Price range gate — not sub-penny, not large-cap
        if not (SMALLCAP_MIN_PRICE <= c <= SMALLCAP_MAX_PRICE):
            return None

        # Float gate — core of Dman's filter
        fl_m, sh_pct, insider_pct, cash_to_mc = _get_short_float_data(ticker)
        if fl_m <= 0 or fl_m > SMALLCAP_MAX_FLOAT_M:
            return None

        # Volume gate — RVOL spike = catalyst proxy for dynamic discovery.
        # For Dman's personally curated watchlist, his public call IS the catalyst;
        # lower bar applies so quiet accumulation days (CAST/LABT/YHC Monday pattern) pass.
        rvol = float(r["RVOL"]) if not pd.isna(r.get("RVOL", float("nan"))) else 0
        _rvol_gate = DMAN_WATCHLIST_MIN_RVOL if ticker in DMAN_SMALLCAP_WATCHLIST else SMALLCAP_MIN_RVOL
        if rvol < _rvol_gate:
            return None

        macd      = float(r["MACD"])
        macd_sig  = float(r["MACD_sig"])
        macd_hist = float(r["MACD_hist"])
        p_hist    = float(p["MACD_hist"])
        rsi       = float(r["RSI"])

        # Dman's exact entry: MACD "curling upside" at or near the bottom chart
        # He buys BELOW the 50D MA — at 52-week lows — when MACD is turning bullish
        # RSI 15-45 is his typical entry zone (deeply oversold or just recovering)
        macd_bull    = macd > macd_sig      # MACD crossed above signal — "curling upside"
        macd_curling = macd_hist > p_hist   # histogram accelerating (momentum turning)
        rsi_ok       = 15 <= rsi <= 65      # oversold to early recovery — his buy zone

        if not (macd_bull and macd_curling and rsi_ok):
            return None

        # 52-week low proximity — "bottom chart" is his #1 phrase
        # He buys within 30% of the 52-week low (SMALLCAP_52WK_LOW_PCT)
        try:
            lookback = min(len(df), 252)
            low_52wk = float(df["Low"].iloc[-lookback:].min())
            near_bottom = c <= low_52wk * (1 + SMALLCAP_52WK_LOW_PCT)
        except Exception:
            # Found in the 2026-08-16 review: this branch intended "data
            # unavailable — don't block," but left low_52wk unassigned.
            # The bot_note f-string further down unconditionally
            # references low_52wk whenever near_bottom is True, so a
            # missing 52wk-low computation raised a silent NameError
            # there, caught by this function's own outer catch-all —
            # dropping the ENTIRE signal instead of letting it through as
            # intended. low_52wk=c makes the fallback genuinely fail
            # open: the bot_note then reads "0% off 52wk low" (unknown,
            # treated as at it) instead of crashing the whole detector.
            near_bottom = True
            low_52wk = c

        # Must be either near 52wk low OR have very high RVOL (catalyst-driven spike)
        if not (near_bottom or rvol >= 5.0):
            return None

        # Gap size computed here (moved up from the Moon Shot section below)
        # so the pullback guard right below can scale its tolerance by it.
        try:
            today_gap_pct = (float(df.iloc[-1]["Open"]) - float(df.iloc[-2]["Close"])) / \
                             float(df.iloc[-2]["Close"]) * 100
        except Exception:
            today_gap_pct = 0.0

        # Intraday-high pullback guard — confirmed live 2026-08-06: CLRO scored
        # a qualifying setup on RVOL/float/MACD/RSI (all computed from the DAILY
        # bar, which says nothing about the shape of TODAY specifically) while
        # actually 14.4% off its own intraday high of 22 minutes earlier and
        # still falling — bought a knife-catch mid-drop, not a breakout. Every
        # other gate above is blind to this because none of them compare price
        # to today's own high; r["High"] on an in-progress daily bar already IS
        # the running intraday high, so this needs no new data. A stock that
        # legitimately bases and breaks out again later in the same session
        # will have a fresh, lower "high so far" by then and pass normally —
        # this only blocks buying while a spike is actively unwinding.
        #
        # Tolerance scales down for extreme gaps — confirmed live 2026-08-06:
        # CLRO gapped +217% intraday, CELZ +70%, both far beyond the ~15-50%
        # range this tolerance was originally calibrated against. A stock
        # that's already run 200%+ gives back ground faster and harder than
        # one that's gapped a more modest amount — the same flat 12% cushion
        # isn't equally safe at every gap size.
        try:
            today_high = float(r["High"])
            pullback_from_high_pct = (today_high - c) / today_high * 100 if today_high > 0 else 0.0
        except Exception:
            pullback_from_high_pct = 0.0
        _pullback_tolerance = _smallcap_pullback_tolerance_pct(today_gap_pct)
        if pullback_from_high_pct > _pullback_tolerance:
            return None

        # Day N fade pre-check: compute _day_n_fade here so it's available for
        # Moon Shot suppression below (re-computed after gap calc; defined early
        # so the gate below can skip signals that are pure distribution traps)
        try:
            _vols_early = [float(df["Volume"].iloc[i]) for i in range(len(df))]
            _base_early = sum(_vols_early[:-2]) / len(_vols_early[:-2]) if len(_vols_early) > 2 else 1.0
            _prev_rvol_early = _vols_early[-2] / _base_early if _base_early > 0 else 0.0
            # If prior day was very hot (≥10x) and today's RVOL is collapsing (<20%),
            # skip the signal entirely — high probability distribution trap
            if _prev_rvol_early >= 10.0 and rvol < _prev_rvol_early * 0.20:
                return None
        except Exception:
            pass

        # Ultra-low float tier: < 2M float = "thin walls" — use wider targets
        ultra_low = 0 < fl_m < ULTRA_LOW_FLOAT_M
        t1_mult   = ULTRA_LOW_T1_MULT  if ultra_low else SMALLCAP_T1_MULT
        t2_mult   = ULTRA_LOW_T2_MULT  if ultra_low else SMALLCAP_T2_MULT
        stop_pct  = ULTRA_LOW_STOP_PCT if ultra_low else SMALLCAP_STOP_PCT

        # Moon Shot tier — ultra-low float + massive gap + extreme RVOL
        # Example: IOTR Jul 8 2026 — 0.64M float, +40.87% gap, 17x RVOL
        # (today_gap_pct now computed earlier, before the pullback guard above)

        # Day N fade guard: if the prior trading day had very high RVOL and
        # today is still gapping, this is likely a distribution trap (e.g. IOTR Day 3).
        # Suppress Moon Shot tier and reduce RVOL gate to prevent entry.
        try:
            vols = [float(df["Volume"].iloc[i]) for i in range(len(df))]
            base_avg  = sum(vols[:-2]) / len(vols[:-2]) if len(vols) > 2 else 1.0
            prev_rvol = vols[-2] / base_avg if base_avg > 0 else 0.0
            _day_n_fade = (prev_rvol >= 5.0 and rvol < prev_rvol * 0.30
                           and today_gap_pct >= 5.0)
        except Exception:
            _day_n_fade = False

        moonshot = (ultra_low
                    and today_gap_pct >= MOONSHOT_MIN_GAP_PCT
                    and rvol >= MOONSHOT_MIN_RVOL
                    and not _day_n_fade)   # never Moon Shot a Day-N fade trap

        # Entry / stop / targets — wider stops than large-cap (penny stocks whipsaw)
        entry = round(c * 1.002, 4)
        stop  = round(entry * (1 - stop_pct), 4)
        t1    = round(entry * (1 + t1_mult), 2)
        t2    = round(entry * (1 + t2_mult), 2)
        t3    = round(entry * (1 + MOONSHOT_T3_MULT), 2) if moonshot else 0.0
        rr    = round((t1 - entry) / (entry - stop), 2) if entry > stop else 0

        if rr < 1.5:
            return None

        # Build reason string with all Dman-relevant context
        squeeze_note = ""
        if sh_pct >= 30:
            squeeze_note = f" | 🔥 SQUEEZE {sh_pct:.0f}% SI"
        elif sh_pct >= 15:
            squeeze_note = f" | SI {sh_pct:.0f}%"
        insider_note = f" | Insiders {insider_pct:.0f}%" if insider_pct >= 30 else ""
        post_rs  = _is_recent_reverse_split(ticker)
        rs_note  = " | ✅ POST-RS" if post_rs else ""
        bot_note  = f" | Bottom chart ({((c/low_52wk-1)*100):.0f}% off 52wk low)" if near_bottom else ""
        cash_note = f" | 💰 Below cash value ({cash_to_mc:.1f}x)" if cash_to_mc >= 1.0 else (
                    f" | Cash {cash_to_mc:.1f}x MC" if cash_to_mc >= 0.5 else "")
        if rsi < 35:
            pattern_note = f"MACD curling from oversold RSI {rsi:.0f}"
        elif near_bottom:
            pattern_note = "MACD bullish at 52wk low support"
        else:
            pattern_note = "MACD curling bullish, RVOL spike"

        # Float rotation — how many times the entire float has traded hands today.
        # 1x = every share changed hands once; 3x+ = violent conviction/squeeze in progress.
        # Dman watches this explicitly: "this thing has rotated 3x already" = big signal.
        try:
            _today_vol    = float(df["Volume"].iloc[-1])
            _float_shares = fl_m * 1_000_000
            float_rotation = _today_vol / _float_shares if _float_shares > 0 else 0.0
        except Exception:
            float_rotation = 0.0

        rotation_note = ""
        if float_rotation >= 3.0:
            rotation_note = f" | 🔥 Float {float_rotation:.1f}x rotated"
        elif float_rotation >= 1.5:
            rotation_note = f" | Float {float_rotation:.1f}x rotated"
        elif float_rotation >= 0.5:
            rotation_note = f" | Float {float_rotation:.1f}x"

        ultra_note  = " | ⚡ ULTRA-LOW FLOAT" if ultra_low else ""
        moon_note   = f" | 🚀 MOON SHOT (gap {today_gap_pct:+.0f}% / {rvol:.0f}x RVOL / T3 +100%)" if moonshot else ""

        # Catalyst confirmation — everything above (float, RVOL, MACD, RSI,
        # 52wk-low proximity) treats a volume spike as a PROXY for a real
        # catalyst, never actually confirming one exists. Confirmed live
        # 2026-08-12: FGL cleared every one of those gates (28x float
        # rotation on the day) with no findable news or filing behind it —
        # direct instruction after that trade: require a real, confirmed
        # catalyst before committing capital to a low-float spike, not
        # just trust volume alone. DMan's own curated watchlist is exempt
        # — his public call already IS the catalyst (same reasoning as
        # DMAN_WATCHLIST_MIN_RVOL's lower bar above), so a separate
        # news-API hit would be redundant, not additional confirmation.
        # Fails CLOSED (blocks entry) on no headlines OR an API failure —
        # the safer default for a speculative micro-cap is to skip a trade
        # we can't verify, not force one through.
        if ticker not in DMAN_SMALLCAP_WATCHLIST:
            _news_map = _fetch_massive_benzinga_news([ticker], hours_back=CATALYST_NEWS_LOOKBACK_HOURS)
            if not _news_map.get(ticker):
                return None
            # "There is news" alone doesn't mean the news supports a LONG
            # entry -- confirmed live 2026-08-13 that Massive's separate
            # /v2/reference/news endpoint provides real per-article
            # sentiment (positive/neutral/negative) even though the
            # analyst-ratings/bulls-bears-say tier is 403'd. A volume spike
            # whose only real news is negative is a sell-off, not a
            # catalyst worth buying on a long-only system -- block it the
            # same way a missing catalyst is blocked. Unknown/no-sentiment-
            # data (None) does NOT block here -- the primary news-existence
            # check above already confirmed a real catalyst exists; this is
            # an additional refinement on top; not finding a sentiment
            # verdict isn't grounds to override that.
            _sentiment = _news_sentiment_verdict(ticker, hours_back=CATALYST_NEWS_LOOKBACK_HOURS)
            if _sentiment == "negative":
                return None

        reason = (f"Float {fl_m:.1f}M | RVOL {rvol:.1f}x"
                  f"{ultra_note}{moon_note}{rotation_note}{squeeze_note}{insider_note}"
                  f"{rs_note}{bot_note}{cash_note} | {pattern_note}")

        sig = ProSignal(
            ticker=ticker, setup="Low Float Catalyst", bias="LONG",
            entry=entry, stop=stop, target1=t1, target2=t2,
            rr=rr, rsi=round(rsi, 1), rvol=round(rvol, 2),
            reason=reason,
            float_rotation=round(float_rotation, 2),
            target3=t3,
            is_moonshot=moonshot,
        )

        # Position sizing — Moon Shot tier gets MOONSHOT_RISK_MULT allocation; normal uses SMALLCAP_RISK_PCT.
        acct      = get_effective_account()
        risk_pct  = SMALLCAP_RISK_PCT * (MOONSHOT_RISK_MULT if moonshot else 1.0)
        # SMALLCAP_RISK_PCT(0.02) * MOONSHOT_RISK_MULT(5.0) = 10% — bigger than
        # PORTFOLIO_HEAT_LIMIT (6% across ALL open positions combined), which
        # means a moonshot signal was being sized to exceed the account's
        # entire risk ceiling by itself, with zero other positions open.
        # Confirmed live 2026-07-27: LGHL scored 100 and re-qualified as a
        # moonshot on every single scan that day, and was excluded every
        # time with "Heat cap 6% reached (0.0% used)" — the cap wasn't
        # protecting against real exposure, the trade's own intended size
        # already broke it. Capping here preserves the "moonshots get
        # outsized risk" intent while keeping it inside what the account
        # can actually ever approve — it degrades to the max the heat cap
        # allows instead of silently dying every time.
        risk_pct  = min(risk_pct, PORTFOLIO_HEAT_LIMIT)
        risk_amt  = acct * risk_pct
        rps = entry - stop
        shares = max(1, int(risk_amt / rps)) if rps > 0 else 1
        cost = shares * entry
        if cost > SMALLCAP_MAX_COST:
            shares = max(1, int(SMALLCAP_MAX_COST / entry))
            cost   = shares * entry
        sig.shares   = shares
        sig.cost     = round(cost, 2)
        sig.risk_usd = round(shares * rps, 2)

        return sig
    except Exception:
        return None


def score_smallcap_signal(sig: ProSignal) -> int:
    """
    Simplified 100-pt scorer calibrated for micro-cap/penny stocks.
    MTF, Sector rotation, and RS vs SPY are meaningless for these names.
    Focuses on what actually matters: float tightness, short interest,
    RVOL conviction, MACD alignment, and price vs 20D MA.
    """
    fl_m, sh_pct, insider_pct, cash_to_mc = _get_short_float_data(sig.ticker)
    score = 0

    # Float size (40 pts) — ultra-nano is Dman's core setup; sub-1M is the sweet spot
    if   fl_m > 0 and fl_m < 0.5:   score += 40   # ultra-nano (<500K) — maximum explosive
    elif fl_m > 0 and fl_m < 1.0:   score += 35   # nano (500K-1M) — Dman's sweet spot
    elif fl_m > 0 and fl_m < 2.0:   score += 28   # micro (1-2M) — solid
    elif fl_m > 0 and fl_m < 3.5:   score += 20   # small-micro (2-3.5M)
    elif fl_m > 0 and fl_m <= 5.0:  score += 12   # small (3.5-5M) — marginal

    # Short interest / squeeze potential (20 pts)
    if   sh_pct >= 40: score += 20
    elif sh_pct >= 30: score += 15
    elif sh_pct >= 20: score += 10
    elif sh_pct >= 15: score +=  5

    # RVOL conviction (20 pts) — catalyst proxy; Dman looks for massive RVOL spikes
    if   sig.rvol >= 10: score += 20
    elif sig.rvol >=  5: score += 15
    elif sig.rvol >=  3: score += 10
    elif sig.rvol >=  2: score +=  5

    # Float rotation bonus (15 pts) — how many times the entire float has traded today.
    # Dman: "this thing has rotated 3x" = conviction that shorts are trapped and longs
    # are stepping in. A high float rotation on a small float = extreme scarcity of shares.
    if   sig.float_rotation >= 5.0: score += 15   # 5x+ = short squeeze ignition
    elif sig.float_rotation >= 3.0: score += 12   # 3x+ = Dman's verbal cue for urgency
    elif sig.float_rotation >= 1.5: score +=  8   # 1.5x = strong conviction
    elif sig.float_rotation >= 0.5: score +=  4   # 0.5x = meaningful activity

    # MACD curling at support (10 pts) — hard gate already, guaranteed
    score += 10

    # Post-RS bonus (10 pts) — Dman's #1 category: fresh reverse split collapses float
    if _is_recent_reverse_split(sig.ticker):
        score += 10

    # Insider ownership (10 pts) — Dman explicitly checks 13G/13D filings
    if   insider_pct >= 60: score += 10
    elif insider_pct >= 40: score +=  7
    elif insider_pct >= 25: score +=  4

    # Price range bonus — Dman's sweet spot is $0.50-$5 (5 pts)
    if 0.50 <= sig.entry <= 5.0:
        score += 5

    # Oversold bonus (5 pts) — he buys when RSI is deeply oversold (< 35)
    if sig.rsi < 35:
        score += 5

    # 52-week low proximity proxy (5 pts) — oversold RSI correlates with near-52wk-low
    if sig.rsi < 40:
        score += 5

    # Cash vs market cap (10 pts) — "trading below cash value" = Dman's specific phrase
    # Means company has more cash on hand than its entire market cap = extreme undervalue
    if   cash_to_mc >= 1.0: score += 10   # trading below cash value (his direct signal)
    elif cash_to_mc >= 0.5: score +=  5   # substantial cash cushion

    # Dman watchlist bonus (10 pts) — personally curated picks have informational edge
    if sig.ticker in DMAN_SMALLCAP_WATCHLIST:
        score += 10

    return min(100, score)


def format_smallcap_telegram(sig: ProSignal, fl_m: float, sh_pct: float,
                              insider_pct: float = 0.0, post_rs: bool = False) -> str:
    """Telegram alert format for Low Float Catalyst signals — distinct from large-cap alerts."""
    squeeze = f"  🔥 SQUEEZE ({sh_pct:.0f}% SI)" if sh_pct >= 20 else (
              f"  SI: {sh_pct:.0f}%" if sh_pct > 0 else "")
    insider = f"  Insiders: {insider_pct:.0f}%" if insider_pct >= 25 else ""
    rs_tag  = "  ✅ POST-RS" if post_rs else ""
    rotation_line = ""
    if sig.float_rotation >= 1.5:
        rotation_line = f"  🔄 Float rotated <b>{sig.float_rotation:.1f}x</b> today\n"
    elif sig.float_rotation >= 0.5:
        rotation_line = f"  Float rotated {sig.float_rotation:.1f}x today\n"
    t3_line = (f"🚀 <b>T3 (+100% / 2x): ${sig.target3}</b>  ← Moon Shot target\n"
               if sig.is_moonshot and sig.target3 > 0 else "")
    moon_header = "🚀 <b>Dman MOON SHOT Alert</b>" if sig.is_moonshot else "🔥 <b>Dman Small-Cap Alert</b>"
    stop_pct_used = ULTRA_LOW_STOP_PCT if fl_m < ULTRA_LOW_FLOAT_M else SMALLCAP_STOP_PCT
    return (
        f"{moon_header}\n"
        f"🟢 LONG <b>{sig.ticker}</b> — {sig.setup}\n"
        f"Entry: <b>${sig.entry}</b>  Stop: ${sig.stop} (-{stop_pct_used*100:.0f}%)\n"
        f"T1 (+{int(ULTRA_LOW_T1_MULT*100 if fl_m < ULTRA_LOW_FLOAT_M else SMALLCAP_T1_MULT*100)}%): "
        f"${sig.target1}   T2 (+{int(ULTRA_LOW_T2_MULT*100 if fl_m < ULTRA_LOW_FLOAT_M else SMALLCAP_T2_MULT*100)}%): "
        f"${sig.target2}\n"
        f"{t3_line}"
        f"Float: <b>{fl_m:.1f}M</b>{squeeze}{insider}{rs_tag}\n"
        f"RVOL: {sig.rvol}x  RSI: {sig.rsi}  Score: {sig.confluence_score}/100\n"
        f"{rotation_line}"
        f"Size: {sig.shares} shares  Cost: ${sig.cost:,.0f}  Risk: ${sig.risk_usd:.0f}\n"
        f"⚠️ Micro-cap: smaller position, wider stop, news-driven\n"
        f"{sig.reason}\n"
        f"💬 <code>/options {sig.ticker}</code> to browse strikes and buy (if a liquid chain exists)"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.8 — OPTIONS LAYER (Dman style: ITM calls on large-cap signals)
# ═══════════════════════════════════════════════════════════════════════════

def _build_occ_symbol(ticker: str, strike: float, expiration: str,
                      call_put: str = "C") -> str:
    """Build OCC option symbol: TICKER + YYMMDD + C/P + STRIKE*1000 (8 digits zero-padded)."""
    try:
        exp = date.fromisoformat(expiration)
        strike_int = int(round(strike * 1000))
        return f"{ticker}{exp.strftime('%y%m%d')}{call_put}{strike_int:08d}"
    except Exception:
        return ""


# ── Strangles (pre-event, direction-neutral) ──────────────────────────────

def select_strangle_legs(ticker: str, current_price: float) -> Optional[dict]:
    """
    Select OTM call + OTM put for a pre-event strangle on weekly expiration.
    Both legs target STRANGLE_OTM_PCT away from current price.
    Returns a combined result dict or None.
    """
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        today = date.today()
        def _dte(e): return (date.fromisoformat(e) - today).days
        candidates = [(e, _dte(e)) for e in expirations
                      if STRANGLE_MIN_DTE <= _dte(e) <= STRANGLE_MAX_DTE]
        if not candidates:
            return None
        best_exp, best_dte = min(candidates, key=lambda x: abs(x[1] - STRANGLE_TARGET_DTE))
        chain = tk.option_chain(best_exp)

        call_target = current_price * (1 + STRANGLE_OTM_PCT)
        put_target  = current_price * (1 - STRANGLE_OTM_PCT)

        otm_calls = chain.calls[
            (chain.calls["strike"] >= current_price) &
            (chain.calls["strike"] <= current_price * 1.12)
        ].copy()
        otm_puts = chain.puts[
            (chain.puts["strike"] <= current_price) &
            (chain.puts["strike"] >= current_price * 0.88)
        ].copy()

        if otm_calls.empty or otm_puts.empty:
            return None

        def _pick_leg(df, target, option_type):
            df = df.copy()
            df["_dist"] = (df["strike"] - target).abs()
            df = df.sort_values("_dist")
            liq = df[(df["volume"].fillna(0) > 0) | (df["openInterest"].fillna(0) > 10)]
            row = liq.iloc[0] if not liq.empty else df.iloc[0]
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            if bid <= 0 or ask <= 0:
                return None
            mid = (bid + ask) / 2
            if (ask - bid) / mid > 0.20:  # allow wider spread for strangles
                return None
            iv  = float(row.get("impliedVolatility", 0) or 0)
            occ = _build_occ_symbol(ticker, float(row["strike"]), best_exp,
                                    "C" if option_type == "call" else "P")
            return {
                "strike":  float(row["strike"]),
                "premium": round(mid, 2),
                "bid":     round(bid, 2),
                "ask":     round(ask, 2),
                "iv_pct":  round(iv * 100, 1),
                "volume":  int(row.get("volume", 0) or 0),
                "oi":      int(row.get("openInterest", 0) or 0),
                "occ":     occ,
            }

        call_leg = _pick_leg(otm_calls, call_target, "call")
        put_leg  = _pick_leg(otm_puts,  put_target,  "put")
        if not call_leg or not put_leg:
            return None

        total_prem     = round(call_leg["premium"] + put_leg["premium"], 2)
        call_breakeven = round(call_leg["strike"] + total_prem, 2)
        put_breakeven  = round(put_leg["strike"]  - total_prem, 2)
        move_needed    = round(total_prem / current_price * 100, 1)

        return {
            "ticker":         ticker,
            "price":          current_price,
            "expiration":     best_exp,
            "dte":            best_dte,
            "call":           call_leg,
            "put":            put_leg,
            "total_premium":  total_prem,
            "call_breakeven": call_breakeven,
            "put_breakeven":  put_breakeven,
            "move_needed_pct": move_needed,
        }
    except Exception:
        return None


def format_strangle_telegram(result: dict, event: str) -> str:
    """Telegram message for a pre-event strangle advisory."""
    exp_fmt    = date.fromisoformat(result["expiration"]).strftime("%b %d")
    ticker     = result["ticker"]
    c          = result["call"]
    p          = result["put"]
    total      = result["total_premium"]
    cost_each  = round(total * 100, 2)
    strangles  = size_strangle_trade(total)
    total_cost = round(cost_each * strangles, 0) if strangles > 0 else cost_each
    size_line  = (f"Suggested: <b>{strangles} strangle(s)</b> = ${total_cost:,.0f} total "
                  f"(1% acct risk)\n" if strangles > 0
                  else f"Cost: ${cost_each:.0f}/strangle (budget check: may need larger acct)\n")
    return (
        f"⚡ <b>DMan STRANGLE Advisory</b> — {ticker}  [{event}]\n\n"
        f"<b>📈 CALL leg</b>  ${c['strike']:.0f}C  exp {exp_fmt}  ({result['dte']}d)\n"
        f"  Premium: <b>${c['premium']}</b>  (bid ${c['bid']} / ask ${c['ask']})\n"
        f"  IV: {c['iv_pct']:.0f}%  |  Vol: {c['volume']:,}  OI: {c['oi']:,}\n"
        f"  <code>{c['occ']}</code>\n\n"
        f"<b>📉 PUT leg</b>   ${p['strike']:.0f}P  exp {exp_fmt}  ({result['dte']}d)\n"
        f"  Premium: <b>${p['premium']}</b>  (bid ${p['bid']} / ask ${p['ask']})\n"
        f"  IV: {p['iv_pct']:.0f}%  |  Vol: {p['volume']:,}  OI: {p['oi']:,}\n"
        f"  <code>{p['occ']}</code>\n\n"
        f"Cost: <b>${cost_each:.0f}</b>/strangle  |  "
        f"Need <b>{result['move_needed_pct']}%+</b> move to profit\n"
        f"Break-even ↑ ${result['call_breakeven']}  |  "
        f"Break-even ↓ ${result['put_breakeven']}\n"
        f"{size_line}"
        f"📋 Advisory — buy both legs simultaneously\n"
        f"🛑 Stop: exit either leg if premium drops 50%\n"
        f"🎯 Target: exit at +80-100% on the winning leg"
    )


def generate_strangle_advisory(event: str) -> None:
    """
    Fire pre-event strangle advisories on STRANGLE_TICKERS (SPY + QQQ by default).
    Called from premarket briefing on catalyst days (CPI tomorrow, FOMC today/tomorrow).
    """
    if not ENABLE_OPTIONS:
        return
    print(f"  [options] Generating strangle advisories ({event})...")
    for ticker in STRANGLE_TICKERS:
        price = get_live_price(ticker)
        if not price:
            continue
        result = select_strangle_legs(ticker, price)
        if not result:
            print(f"  [options] {ticker}: no liquid strangle found", file=sys.stderr)
            continue
        msg = format_strangle_telegram(result, event)
        send_telegram(msg)
        print(f"  ⚡ Strangle: {ticker}  "
              f"${result['call']['strike']:.0f}C / ${result['put']['strike']:.0f}P  "
              f"exp {result['expiration']} ({result['dte']}d)  "
              f"total ${result['total_premium']*100:.0f}/strangle  "
              f"need {result['move_needed_pct']}%+ move")


_STOP_RESTORE_COOLDOWN_KEY_FMT = "{ticker}_STOP_RESTORE_ATTEMPT"

def _auto_restore_missing_stop(client, ticker: str, qty: float) -> tuple[bool, str]:
    """
    Attempts to restore live stop protection for `ticker`, which
    _check_open_position_risk() has already confirmed has zero live
    STOP-type order right now. Returns (success, detail) — the caller
    folds `detail` into the Telegram alert either way, so a failure is
    never silent, just no longer purely manual either.

    Added 2026-08-15 after this exact gap (a position's stop stuck HELD
    or CANCELED with nothing re-establishing it) hit twice in 48 hours —
    LITX (stop stuck HELD since entry, its take-profit sibling had
    claimed all shares via Alpaca's held_for_orders accounting, breaking
    the OCO link — the same "W" incident shape from 2026-08-04) and ARTL
    (a duplicate bracket-order race left the real position's stop
    CANCELED). The alert alone caught both, correctly — but still needed
    a human to notice and manually cancel the conflicting order(s) before
    a fresh stop could get share allocation. This automates that sequence:
      1. Look up the intended stop price (and target1, for TP restoration)
         from PositionTracker. If the ticker isn't tracked there (a
         genuine orphan, not just a broken bracket), there's no known-safe
         price to use, so this refuses to guess and returns False rather
         than inventing a stop level.
      2. Cancel only STOP-type orders first — those are the broken ones,
         and cancelling them can never affect a healthy take-profit leg.
         Try submitting the fresh stop with just that. Only if THAT still
         fails (the take-profit leg is the one holding the shares via
         Alpaca's held_for_orders accounting — the actual LITX/W failure
         shape) does this fall back to also cancelling the take-profit
         leg and retrying once.
      3. Submit a fresh plain STOP (never STOP_LIMIT — see
         submit_alpaca_trade()'s docstring for why: a stop-limit leg has
         gotten stuck in this exact same HELD state 3/3 times on this
         account).
      4. If a take-profit leg had to be cancelled in step 2, resubmit a
         fresh one at the tracked target1 once the stop is confirmed
         live — added 2026-08-16 after review found the original version
         of this function cancelled the take-profit leg and NOTHING ever
         put it back, permanently losing the position's automated exit
         despite the docstring's original claim that it "can be
         resubmitted once the stop is confirmed live."

    Rate-limited to one restore ATTEMPT per symbol per ALERT_COOLDOWN_MIN
    (30 min, the same rolling dedup mechanism the alert itself already
    uses) — added 2026-08-16 after review found this running on the 10s
    guard cadence with no throttle on the action itself (only on the
    Telegram alert). If a freshly-submitted stop lands HELD again (the
    exact failure this function exists to fix), the previous version
    cancelled and resubmitted it every single 10s tick indefinitely —
    leaving the position with zero live stop for a multi-second window on
    every cycle, forever, feeding the same 429 lockout risk
    _record_alpaca_429 was built to detect. This does not weaken
    detection: _check_stop_coverage()'s alert still fires (and this still
    auto-attempts on the FIRST pass), it just stops retrying faster than a
    human or a fixed underlying issue could plausibly resolve.
    """
    _cd_key = _STOP_RESTORE_COOLDOWN_KEY_FMT.format(ticker=ticker)
    if _is_duplicate_alert(_cd_key):
        return False, (f"restore attempted recently (within {ALERT_COOLDOWN_MIN}m) — "
                        "waiting on cooldown instead of retrying every cycle, needs manual review "
                        "if this persists")
    try:
        tracked = next((p for p in PositionTracker().positions
                        if p.ticker == ticker
                        and not p.setup.startswith(("Options Call ", "Options Put ", "Earnings "))),
                       None)
        if tracked is None or tracked.stop <= 0:
            return False, "not in PositionTracker (or no stop price on record) — can't safely auto-restore, needs manual review"

        from alpaca.trading.requests import (GetOrdersRequest as _GOReq, StopOrderRequest as _StopReq,
                                              LimitOrderRequest as _LimReq)
        from alpaca.trading.enums import (QueryOrderStatus as _QOS, OrderSide as _OSide,
                                           TimeInForce as _TIF, OrderType as _OType)

        _open = client.get_orders(filter=_GOReq(symbols=[ticker], status=_QOS.OPEN, limit=20))
        _stop_orders = [o for o in _open if o.side == _OSide.SELL
                        and o.order_type in (_OType.STOP, _OType.STOP_LIMIT, _OType.TRAILING_STOP)]
        _stop_ids    = {o.id for o in _stop_orders}
        _tp_orders   = [o for o in _open if o.side == _OSide.SELL and o.id not in _stop_ids]

        def _cancel_all(orders) -> bool:
            _any = False
            for _o in orders:
                try:
                    client.cancel_order_by_id(_o.id)
                    _any = True
                except Exception:
                    pass
            return _any

        stop_px = round(tracked.stop, 2)
        _tp_cancelled = False
        if _cancel_all(_stop_orders):
            time.sleep(2)   # let the cancel(s) actually process before resubmitting for the same shares
        try:
            order = client.submit_order(_StopReq(
                symbol=ticker, qty=qty, side=_OSide.SELL,
                time_in_force=_TIF.GTC, stop_price=stop_px,
            ))
        except Exception:
            # Broker rejected it (most likely: the take-profit leg still
            # holds the shares) — cancel it too and retry exactly once.
            if _cancel_all(_tp_orders):
                _tp_cancelled = True
                time.sleep(2)
            order = client.submit_order(_StopReq(
                symbol=ticker, qty=qty, side=_OSide.SELL,
                time_in_force=_TIF.GTC, stop_price=stop_px,
            ))
        _save_last_alert(_cd_key)

        _detail = f"resubmitted a plain stop at ${stop_px} (id {str(order.id)[:8]}…)"
        if _tp_cancelled and tracked.target1 and tracked.target1 > 0:
            try:
                tp_order = client.submit_order(_LimReq(
                    symbol=ticker, qty=qty, side=_OSide.SELL,
                    time_in_force=_TIF.GTC, limit_price=round(tracked.target1, 2),
                ))
                _detail += f"; take-profit re-submitted at ${round(tracked.target1, 2)} (id {str(tp_order.id)[:8]}…)"
            except Exception as _tp_exc:
                _detail += f"; ⚠️ take-profit re-submit FAILED ({_tp_exc}) — stop is live but no take-profit order exists"
        return True, _detail
    except Exception as exc:
        _save_last_alert(_cd_key)
        return False, f"auto-restore attempt failed: {exc}"


_stop_coverage_fetch_cache: dict = {"positions": None, "orders": None, "ts": 0.0}
_STOP_COVERAGE_CACHE_TTL_S = 30   # matches the already-vetted _REALTIME_PRICE_MAX_AGE_S /
                                  # _REALTIME_OPTION_QUOTE_MAX_AGE_S staleness convention

def _check_stop_coverage() -> Optional[dict]:
    """Orphan-position check + broker-side live-stop check, split out
    (2026-08-15) from _check_open_position_risk() so it can run on a tight
    cadence independent of regime computation. Originally only reached via
    run_pro_scanner()'s ~10min scan cadence — meaning a broken bracket order
    (e.g. the LITX stuck-HELD stop found live 2026-08-14) could sit
    undetected for most of a scan cycle. guard_loop() in dman_daemon.py now
    also calls this directly on its 10s cadence; _is_duplicate_alert()
    already caps actual alert volume, so calling this more often only
    tightens detection latency, it doesn't add alert spam.

    Returns the raw {symbol: Position} dict fetched from Alpaca (None if the
    fetch itself failed) so callers that also need "what do I actually hold
    right now" don't have to re-fetch — _check_open_position_risk() uses
    this to cross-reference pending signals against real positions.

    Found in the 2026-08-16 review: on the 10s guard cadence this made 2
    unconditional REST calls (get_all_positions, get_orders) every single
    tick with no caching -- ~4,800 calls/session, whether or not anything
    had changed. Both fetches are now cached for _STOP_COVERAGE_CACHE_TTL_S
    (30s, the same staleness this project already accepts for the
    real-time price/option-quote caches), a ~3x reduction that still keeps
    detection latency far tighter than the ~10min scan-only cadence this
    was originally built to improve on.

    Also found: the orphan check and the live-stop check used to share ONE
    exception boundary, so a failure partway through the orphan check
    silently skipped the live-stop check right after it -- the one that
    actually restores missing protection. Split into two independent
    try/except blocks so a failure in either is isolated and doesn't take
    the other down with it.
    """
    _client = get_alpaca_client()
    if _client is None:
        return None

    now = time.time()
    if (now - _stop_coverage_fetch_cache["ts"] < _STOP_COVERAGE_CACHE_TTL_S
            and _stop_coverage_fetch_cache["positions"] is not None):
        _alp_positions = _stop_coverage_fetch_cache["positions"]
        _open_orders   = _stop_coverage_fetch_cache["orders"]
    else:
        try:
            _alp_positions = {p.symbol: p for p in _client.get_all_positions()}
            _open_orders = _client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200))
            _stop_coverage_fetch_cache["positions"] = _alp_positions
            _stop_coverage_fetch_cache["orders"]    = _open_orders
            _stop_coverage_fetch_cache["ts"]        = now
        except Exception as _fe:
            print(f"  ⚠  Stop-coverage position/order fetch failed: {_fe}")
            # None (not {}) is the "couldn't verify" sentinel — an empty dict is
            # a legitimate "zero real positions" result, and the two must be
            # distinguishable by callers that filter against it (an empty dict
            # must filter pending down to nothing; a failed fetch must NOT
            # silently hide potentially-stale risk info by treating "unknown"
            # the same as "you hold nothing").
            return None

    # ── Orphan position check ──────────────────────────────────────────────
    # Any Alpaca open position not in dman_live_signals.json has no automated
    # stop management — flag immediately so manual action can be taken.
    # Own try/except (isolated from the live-stop check below): a failure
    # here must not silently skip the live-stop check right after it.
    try:
        # Build tracked set from BOTH sources:
        # 1. dman_live_signals.json — signals logged by run_pro_scanner()
        # 2. dman_positions.json   — positions logged by PositionTracker (covers pre-market fills)
        _tracked_tickers: set[str] = set()
        _tracked_occ:     set[str] = set()   # options: Alpaca reports OCC symbols, not ticker
        try:
            with open(LIVE_SIGNALS_FILE) as _f:
                _pf = json.load(_f)
            _tracked_tickers |= {p.get("ticker") for p in _pf.get("pending", [])}
        except Exception:
            pass
        try:
            _pt_pos = PositionTracker().positions
            for _p in _pt_pos:
                if _p.setup.startswith("Earnings "):
                    # A spread position has 2-4 real option legs, not a single
                    # symbol embedded in `setup` — without this, every leg
                    # would falsely alarm as an orphan (untracked) position.
                    _tracked_occ |= set(_p.legs)
                elif _p.setup.startswith("Options Call ") or _p.setup.startswith("Options Put "):
                    # _position_identity() already extracts the OCC symbol
                    # for exactly this purpose -- found in the 2026-08-16
                    # review: this used to reimplement the same parsing
                    # inline, one of three duplicated copies in the project.
                    _tracked_occ.add(_position_identity(_p.ticker, _p.setup))
                else:
                    _tracked_tickers.add(_position_identity(_p.ticker, _p.setup))
        except Exception:
            pass
        _orphans = [sym for sym in _alp_positions
                    if sym not in _tracked_tickers and sym not in _tracked_occ]
        if _orphans:
            _orphan_msgs = []
            for _sym in _orphans:
                _pos = _alp_positions[_sym]
                _qty  = float(_pos.qty)
                _ep   = float(_pos.avg_entry_price)
                _upnl = float(_pos.unrealized_pl)
                _upct = float(_pos.unrealized_plpc) * 100
                _orphan_msgs.append(
                    f"  {_sym}: {_qty:.0f}sh @ ${_ep:.2f}  "
                    f"unreal {'+' if _upnl >= 0 else ''}{_upnl:.2f} ({_upct:+.1f}%)"
                )
            _msg = (
                "⚠️ <b>ORPHAN POSITIONS — NO STOP COVERAGE</b>\n"
                "These are open in Alpaca but NOT in signal tracker:\n\n"
                + "\n".join(_orphan_msgs)
                + "\n\n<i>Close manually or add to watchlist to restore monitoring.</i>"
            )
            # Found in the 2026-08-16 review: this used to be one
            # single global key shared across every symbol, so a
            # cooldown started by orphan set {A} would suppress a
            # genuine alert for an unrelated later orphan set {B} for
            # the next ALERT_COOLDOWN_MIN+ minutes. Keying by the
            # sorted set of currently-orphaned symbols means the
            # SAME orphan set still dedupes normally (no re-alert
            # spam while nothing has changed), but a genuinely
            # different set of symbols always gets its own alert.
            _orphan_key = "__ORPHAN_POSITIONS__:" + ",".join(sorted(_orphans))
            if not _is_duplicate_alert(_orphan_key):
                send_telegram(_msg)
                _save_last_alert(_orphan_key)
                print(f"  ⚠  {len(_orphans)} orphan position(s) — sent alert: {_orphans}")
            else:
                print(f"  ⚠  {len(_orphans)} orphan position(s) — alert suppressed (sent recently)")
    except Exception as _oe:
        print(f"  ⚠  Orphan check failed: {_oe}")

    # ── Broker-side live-stop check ──────────────────────────────────────
    # A different question from the orphan check above: that one asks
    # "do WE know about this position," this asks "does ALPACA
    # actually have a LIVE protective stop working for it right now."
    # Confirmed live 2026-08-04: W was simultaneously an orphan (not
    # in either tracker) AND had its stop-limit leg stuck in HELD
    # status — its take-profit sibling had claimed all shares via
    # Alpaca's held_for_orders accounting, breaking the OCO link, so
    # the stop could never get shares allocated. Nothing caught this
    # until a manual Alpaca API query. Options positions are
    # excluded — they have no broker-side bracket by design and rely
    # entirely on the daemon's own 60s stop/T1/T2 loop instead.
    # Own try/except: isolated from the orphan check above so a failure
    # in either can't silently take down the other.
    try:
        from alpaca.trading.enums import OrderType, OrderStatus, AssetClass
        _live_stop_symbols = {
            _o.symbol for _o in _open_orders
            if _o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP)
            and _o.status != OrderStatus.HELD
        }
        _unprotected = [
            sym for sym, _pos in _alp_positions.items()
            if _pos.asset_class == AssetClass.US_EQUITY and sym not in _live_stop_symbols
        ]
        if _unprotected:
            # Auto-restore attempt (added 2026-08-15) BEFORE building the
            # alert, so the alert itself reports what actually happened
            # rather than always reading as "needs manual action" even
            # when this already fixed it. See _auto_restore_missing_stop()
            # for the full incident history and why this is safe.
            _up_msgs = []
            _any_restored = False
            for _sym in _unprotected:
                _pos = _alp_positions[_sym]
                _qty = float(_pos.qty)
                _restored, _detail = _auto_restore_missing_stop(_client, _sym, _qty)
                if _restored:
                    _any_restored = True
                    _up_msgs.append(f"  ✅ {_sym}: {_qty:.0f}sh — {_detail}")
                else:
                    _up_msgs.append(f"  🚨 {_sym}: {_qty:.0f}sh — no live stop, auto-restore failed: {_detail}")
            # Same fix as __ORPHAN_POSITIONS__ above: keyed by the
            # sorted set of currently-unprotected symbols instead of
            # one global key, so a benign "stop auto-restored" for
            # symbol A can no longer suppress a genuine "no live
            # stop" alert for an unrelated symbol B.
            _up_key = "__NO_LIVE_STOP__:" + ",".join(sorted(_unprotected))
            if not _is_duplicate_alert(_up_key):
                _header = ("🛠 <b>STOP AUTO-RESTORED</b>" if _any_restored and all(
                               "✅" in m for m in _up_msgs)
                           else "🚨 <b>NO LIVE STOP PROTECTION</b>")
                _footer = ("\n\n<i>Fresh stop(s) submitted automatically — verify in Alpaca when convenient.</i>"
                           if _any_restored else
                           "\n\n<i>Auto-restore failed — check for a stuck/HELD order or a broken OCO link, manual action needed.</i>")
                send_telegram(
                    f"{_header}\n"
                    "These equity positions had zero working stop-loss order on the exchange:\n\n"
                    + "\n".join(_up_msgs) + _footer
                )
                _save_last_alert(_up_key)
                print(f"  🚨 {len(_unprotected)} position(s) with NO live stop: {_unprotected}"
                      + (" (auto-restore attempted)" if _any_restored else ""))
            else:
                print(f"  🚨 {len(_unprotected)} position(s) with NO live stop — alert suppressed (sent recently): {_unprotected}")
    except Exception as _se:
        print(f"  ⚠  Live-stop check failed: {_se}")

    return _alp_positions


def _check_open_position_risk(regime: dict) -> None:
    """Read pending live signals, fetch current prices, alert if within 2% of stop."""
    _alp_positions = _check_stop_coverage()

    try:
        with open(LIVE_SIGNALS_FILE, "r") as f:
            data = json.load(f)
        pending = data.get("pending", [])
    except Exception:
        return

    # Confirmed live 2026-08-08: "pending" tracks every signal that was ever
    # ALERTED (_log_live_signal fires at alert time, independent of whether
    # --submit was used or the order actually filled), not signals that
    # became real positions. This risk check used to print/alert on every
    # pending entry as if it were an open position — FGL and AMZN both
    # showed up here with live stop-distance numbers despite neither ever
    # having filled a real order, which read as real open exposure that
    # didn't exist. Cross-referencing against real Alpaca positions (already
    # fetched above) keeps this check honest: only tickers you actually hold.
    #
    # `_alp_positions is None` (Alpaca unreachable) deliberately still
    # checks every pending entry rather than skipping — see
    # test_alpaca_unreachable_fails_open_still_shows_pending's docstring:
    # silently going dark on a position you genuinely hold, just because
    # Alpaca was briefly unreachable, is judged the worse failure mode
    # than an occasional false alarm on a phantom (alerted-but-never-
    # filled) signal. verified=False is threaded into the alert text
    # below instead, so a human reading it knows this specific alert
    # couldn't be cross-checked against real broker state this cycle —
    # reducing the FGL/AMZN false-CERTAINTY problem without reintroducing
    # a silent-outage blind spot on a real position.
    verified = _alp_positions is not None
    if verified:
        pending = [p for p in pending if p.get("ticker") in _alp_positions]

    if not pending:
        return

    vix_shock    = regime.get("vix_shock", False)
    def_rotation = regime.get("defensive_rotation", False)
    alerts       = []

    print("\n  📋 Open position risk check:")
    for pos in pending:
        ticker = pos.get("ticker", "?")
        bias   = pos.get("bias", "LONG")
        entry  = pos.get("entry", 0.0)
        stop   = pos.get("stop", 0.0)
        t1     = pos.get("target1", 0.0)
        score  = pos.get("score", 0)
        px     = get_live_price(ticker)
        if px is None or stop <= 0:
            print(f"    {ticker}: price unavailable — skipping")
            continue

        if bias == "LONG":
            dist_pct = (px - stop) / abs(stop) * 100
            t1_dist  = (t1 - px) / abs(px) * 100 if t1 > 0 else 0
            direction_ok = px > stop
        else:
            dist_pct = (stop - px) / abs(stop) * 100
            t1_dist  = (px - t1) / abs(px) * 100 if t1 > 0 else 0
            direction_ok = px < stop

        status_icon = "✅" if direction_ok and dist_pct > 5 else ("⚠️" if direction_ok else "🛑")
        print(f"    {status_icon} {ticker} ({bias}): px=${px:.2f}  stop=${stop:.2f}  "
              f"dist={dist_pct:+.1f}%  T1 dist={t1_dist:+.1f}%  score={score}")

        if not direction_ok or dist_pct < 2.0:
            extra = ""
            if vix_shock:
                extra += "  ⚡ VIX shock active — elevated gap risk"
            if def_rotation and bias == "LONG" and TICKER_SECTOR.get(ticker) == "Technology":
                extra += "  🔄 Defensive rotation — tech headwind"
            alerts.append(
                f"{'🛑' if not direction_ok else '⚠️'} <b>{ticker}</b> open {bias} "
                f"at risk\n"
                f"  Entry ${entry:.2f} | Stop ${stop:.2f} | Now ${px:.2f}\n"
                f"  Distance to stop: <b>{dist_pct:+.1f}%</b>{extra}"
            )

    if alerts:
        header = "⚠️ <b>OPEN POSITION RISK ALERT</b>\n\n"
        if not verified:
            header += ("⚠️ <i>Alpaca unreachable this cycle — could not confirm these are real "
                       "positions vs. an alerted-but-never-filled signal.</i>\n\n")
        send_telegram(header + "\n\n".join(alerts))
        print(f"  ⚠  Risk alert sent for {len(alerts)} position(s).")
    else:
        print(f"  ✅ All {len(pending)} open position(s) outside stop zone.")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.5 — SCAN RESULT LOG
# ═══════════════════════════════════════════════════════════════════════════

def _append_scan_log(entry: dict, max_entries: int = 20) -> None:
    """
    Append one scan result to the rolling scan log (keeps last max_entries
    by actual timestamp, not list position).

    Confirmed live 2026-08-12: this used a positional log[-max_entries:]
    slice, which sync_scan_log_with_remote()'s rebuild step was ALSO doing
    until fixed the night before (2026-08-11) to sort by ts first —  but
    that fix only touched the cross-process MERGE step, not this function,
    which runs first, on every single scan, before any merge happens. A
    real trading day produced this exact failure: entries from 6 days
    earlier (2026-08-06) were still present alongside only-through-11:47am
    entries from the CURRENT day, with every scan from 11:54am to market
    close silently missing — this function's own truncation dropped them
    the moment the on-disk file wasn't already perfectly sorted (e.g. after
    a git-level conflict resolution grabbed a stale snapshot), because it
    trusted list position instead of the ts field every entry already
    carries. Sorting here closes the gap regardless of whatever order the
    file was in when loaded.
    """
    log: list[dict] = []
    if os.path.exists(SCAN_LOG_FILE):
        try:
            with open(SCAN_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    log = sorted(log, key=lambda e: e.get("ts", ""), reverse=True)[:max_entries][::-1]
    _write_json_atomic(SCAN_LOG_FILE, log, indent=2)


def _log_scan_halt(reason: str, tickers: list, min_score: int) -> None:
    """
    Records a circuit-breaker halt (consecutive-loss / monthly-loss /
    daily-loss guard) to the same rolling scan log run_pro_scanner()
    writes to on a normal pass. Added 2026-08-13: those three guards
    return [] before EVER reaching run_pro_scanner()'s own
    _append_scan_log() call (which sits ~350 lines later, after regime
    computation) — meaning the entire scan log goes dark for the rest of
    the day the moment a limit trips, indistinguishable from the scanner
    silently being broken. Confirmed live the same day: dman_scan_log.json
    showed zero entries for all of 2026-08-13 despite 9+ real scanner runs,
    because the daily loss limit tripped ~10:40 AM ET and every run after
    that hit this exact silent gap. This is deliberately minimal (no
    regime/VIX lookup) — the only thing worth persisting here is THAT a
    halt happened and why, not a full market read for a run that never
    got that far.
    """
    try:
        _append_scan_log({
            "ts":            datetime.now(ET).isoformat(),
            "halted":        True,
            "halt_reason":   reason,
            "min_score":     min_score,
            "tickers_total": len(tickers),
            "signals":       0,
            "signal_tickers": [],
        })
    except Exception:
        pass


def print_scan_log() -> None:
    """Print a human-readable summary of the scan history log."""
    if not os.path.exists(SCAN_LOG_FILE):
        print("  No scan log found. Run at least one scan first.")
        return
    try:
        with open(SCAN_LOG_FILE) as f:
            log: list[dict] = json.load(f)
    except Exception as e:
        print(f"  Could not read scan log: {e}")
        return

    W = 68
    print(f"\n{'═'*W}")
    print(f"  DMan Scan Log — last {len(log)} run(s)")
    print(f"{'─'*W}")
    for entry in reversed(log):
        ts_raw = entry.get("ts", "?")
        try:
            from datetime import datetime as _dt
            ts = _dt.fromisoformat(ts_raw).strftime("%b %d %I:%M %p")
        except Exception:
            ts = ts_raw[:16]

        if entry.get("halted"):
            print(f"  {ts}  |  🛑 HALTED — {entry.get('halt_reason', '?')}  "
                  f"({entry.get('tickers_total', '?')} tickers, min={entry.get('min_score', '?')})")
            print(f"{'─'*W}")
            continue

        regime     = entry.get("regime", "?")
        rscore     = entry.get("regime_score", "?")
        vix        = entry.get("vix", 0)
        min_sc     = entry.get("min_score", "?")
        n_tickers  = entry.get("tickers_total", "?")
        n_signals  = entry.get("signals", 0)
        sig_ticks  = entry.get("signal_tickers", [])
        rej_none   = entry.get("rejected_no_signal", 0)
        rej_gate   = entry.get("rejected_hard_gate", 0)
        rej_score  = entry.get("rejected_low_score", 0)
        budget_hit = entry.get("budget_hit", False)
        universe   = entry.get("universe", "?")
        nb_pct     = entry.get("news_breadth_pct")
        nb_total   = entry.get("news_breadth_total", 0)

        sig_icon = "🟢" if n_signals > 0 else "❌"
        sig_str  = (f"{n_signals} signal(s): {', '.join(sig_ticks)}"
                    if sig_ticks else f"{n_signals} signals")
        budget_str = "  ⏱ TIME BUDGET HIT" if budget_hit else ""
        nb_str = (f"  📰 breadth {nb_pct:+.0f}% ({nb_total})" if nb_pct is not None
                  else f"  📰 breadth n/a ({nb_total})" if nb_total else "")

        print(f"  {ts}  |  {regime}({rscore}/19)  VIX {vix:.1f}  "
              f"min={min_sc}  [{universe}]{nb_str}")
        print(f"    {sig_icon} {n_tickers} tickers → {sig_str}{budget_str}")
        print(f"    Rejected: {rej_none} no-gap  {rej_gate} gate  {rej_score} low-score")
        print(f"{'─'*W}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.6 — SIGNAL EXPLAINABILITY  (/why TICKER)
# ═══════════════════════════════════════════════════════════════════════════

def explain_ticker(ticker: str, min_score: int = None) -> str:
    """
    Read-only diagnostic: runs the exact same detection → scoring → gate
    sequence run_pro_scanner() uses, but for one ticker on demand, and
    reports exactly where it stands instead of only showing up (or not) in
    the next scan pass. Never submits an order, writes to the win-rate
    tracker, or logs an alert — pure inspection. Added 2026-08-15 so a
    "why didn't/did TICKER trigger" question has a direct answer instead of
    re-deriving it by hand from scan log output, the way every prior
    incident in this session needed manual digging to explain.
    """
    ticker = ticker.upper().strip()
    lines = [f"🔍 <b>{ticker} — Signal Explain</b>"]
    try:
        raw = fetch_df(ticker)
        if raw is None or len(raw) < 60:
            lines.append(f"  ❌ No usable price data for {ticker} (fetch failed or too little history).")
            return "\n".join(lines)
        df = _compute_indicators_cached(ticker, raw)
        df.dropna(subset=["EMA50", "RSI", "MACD", "ATR"], inplace=True)
        if len(df) < 10:
            lines.append("  ❌ Not enough indicator history after warm-up.")
            return "\n".join(lines)

        sig = _raw_signals(df, ticker)
        if sig is None:
            r = df.iloc[-1]
            c = float(r["Close"])
            atr_val = float(r["ATR"]) if not pd.isna(r["ATR"]) else 0.0
            avg_vol = float(r["AvgVol20"]) if ("AvgVol20" in r.index and not pd.isna(r["AvgVol20"])) else 0.0
            atr_pct = atr_val / c * 100 if c > 0 else 0.0
            avg_dv  = c * avg_vol
            lines.append("  ❌ No qualifying pattern detected right now.")
            if atr_pct < ATR_PCT_MIN or avg_dv < AVG_DOLLAR_VOL_MIN:
                lines.append(f"  ⚠️ Quality gate failed: ATR% {atr_pct:.2f} (min {ATR_PCT_MIN}) | "
                              f"$vol {avg_dv:,.0f} (min {AVG_DOLLAR_VOL_MIN:,.0f})")
            else:
                lines.append("  Quality gates passed — none of the current LONG/SHORT "
                              "setups matched today's structure.")
            return "\n".join(lines)

        regime  = get_market_regime()
        tracker = WinRateTracker()
        try:
            _news_map = _fetch_alpaca_news([ticker], hours_back=20)
        except Exception:
            _news_map = {}
        sig.news_boost = _news_boost_after_sentiment_veto(bool(_news_map.get(ticker)), ticker)

        sig = score_signal(sig, df, regime, tracker)

        # Replicate run_pro_scanner()'s exact min-score escalation and gate
        # sequence — found 2026-08-16 review: this used to compute its own,
        # simpler version that could directly contradict what the live
        # scanner would actually do. Two gaps specifically: (1) Rel
        # Strength and Sector were reported as HARD blocking gates here,
        # but the live scanner only ever scores them as points — a signal
        # missing on RS/Sector can still trade live while this reported
        # "BLOCKED". (2) the VIX>25 / VIX-shock / seasonal-month / active
        # defensive-rotation adjustments the live scanner applies to the
        # score threshold (and, for defensive rotation, to the score
        # itself) were never applied here, so this could report "PASS" on
        # a signal the live system would reject on exactly the kind of
        # elevated-risk day someone is most likely to run /why on.
        min_score = min_score or tracker.adaptive_min_score()
        _vix_now = float(regime.get("details", {}).get("VIX", VIX_SIZE_BASE))
        if _vix_now > 25:
            min_score = max(min_score, 90)
        if regime.get("vix_shock"):
            min_score = max(min_score, min_score + 5)

        _def_rotation = regime.get("defensive_rotation", False)
        if _def_rotation and sig.bias == "LONG" and TICKER_SECTOR.get(sig.ticker, "") == "Technology":
            sig.confluence_score = max(0, sig.confluence_score - 5)

        _curr_month = datetime.today().month
        _seasonal_active = _curr_month in SEASONAL_WEAK_MONTHS
        _SEASONAL_EXEMPT = {"Gap & Hold", "Morning Runner"}

        effective_min = SETUP_MIN_CONFLUENCE.get(sig.setup, min_score)
        if sig.ticker in VOLATILE_TICKERS:
            effective_min = max(effective_min, VOLATILE_MIN_CONFLUENCE)
        if _seasonal_active and sig.setup not in _SEASONAL_EXEMPT:
            effective_min = max(effective_min, SEASONAL_MIN_SCORE)

        lines.append(f"  Setup   : <b>{sig.setup}</b> ({sig.bias})")
        lines.append(f"  Entry ${sig.entry:.2f} | Stop ${sig.stop:.2f} | T1 ${sig.target1:.2f} | RR {sig.rr:.2f}")
        lines.append(f"  Score   : {sig.confluence_score}/100  (need ≥{effective_min})")
        if _vix_now > 25 or regime.get("vix_shock") or _def_rotation or _seasonal_active:
            lines.append(f"  Session : " + ", ".join(filter(None, [
                f"VIX {_vix_now:.1f} (>25, floor raised)" if _vix_now > 25 else "",
                "VIX shock (+5 floor)" if regime.get("vix_shock") else "",
                "defensive rotation (-5 tech LONG penalty)" if _def_rotation else "",
                f"seasonal weak month (floor {SEASONAL_MIN_SCORE})" if _seasonal_active else "",
            ])))

        # Hard gates — must match run_pro_scanner()'s own list exactly.
        # Rel Strength / Sector are shown below for context but are SCORE
        # contributors only, never blocking on their own.
        hard_gates = {
            "Regime": sig.regime_ok, "MTF": sig.mtf_ok, "Earnings": sig.earnings_ok,
            "Macro": sig.macro_ok, "No Divergence": sig.divergence_free,
        }
        for name, ok in hard_gates.items():
            lines.append(f"  {'✅' if ok else '❌'} {name}")
        lines.append(f"  {'✅' if sig.rs_ok else 'ℹ️ '} Rel Strength (score only, not a hard gate)")
        lines.append(f"  {'✅' if sig.sector_ok else 'ℹ️ '} Sector (score only, not a hard gate)")

        hard_fail = [n for n, ok in hard_gates.items() if not ok]
        if hard_fail:
            lines.append(f"\n  🛑 Would be BLOCKED — failed hard gate(s): {', '.join(hard_fail)}")
        elif sig.confluence_score < effective_min:
            lines.append(f"\n  🛑 Would be BLOCKED — score {sig.confluence_score} < required {effective_min}")
        else:
            lines.append("\n  ✅ Would PASS all gates right now.")

        if sig.score_breakdown:
            lines.append("\n  Score breakdown:")
            for k, v in sig.score_breakdown.items():
                lines.append(f"    {k}: {v}")

        return "\n".join(lines)
    except Exception as e:
        lines.append(f"  ⚠️ Explain failed: {e}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 20 — PRO SCANNER
# ═══════════════════════════════════════════════════════════════════════════

_UNMAPPED_SECTOR = "Unmapped/Small-Cap"

def _apply_sector_concentration_cap(signals: list["ProSignal"]) -> list["ProSignal"]:
    """
    Max 2 signals per sector per scan, to avoid overweighting one sector.
    TICKER_SECTOR only curates large-caps -- found in the 2026-08-16
    review: TICKER_SECTOR.get(ticker, "") returns "" for anything outside
    that map, and the old check treated an empty sector as automatically
    EXEMPT from the cap, so it never applied to small-cap catalyst or
    dynamically-discovered signals at all, no matter how many fired in one
    scan. Bucketing every unmapped ticker under one shared sentinel sector
    isn't a real GICS classification, but it closes the actual hole: at
    most 2 unmapped-sector signals can now pass per scan, instead of an
    unbounded number.
    """
    sector_counts: dict[str, int] = {}
    concentrated: list[ProSignal] = []
    for sig in signals:
        sec   = TICKER_SECTOR.get(sig.ticker) or _UNMAPPED_SECTOR
        count = sector_counts.get(sec, 0)
        if count < 2:
            concentrated.append(sig)
            sector_counts[sec] = count + 1
        else:
            sys.stdout.write(
                f"  📊 Sector cap: {sig.ticker} skipped "
                f"({sec} already has {count} signal(s))\n"
            )
    return concentrated


def _finalize_and_alert_signals(signals: list["ProSignal"], regime: dict,
                                smallcap_extra: dict) -> None:
    """
    Alert only AFTER heat-cap + sector-cap have finalized the list. Both
    passes used to send the Telegram alert the moment a signal was
    detected, before those two caps ran — so a signal excluded by either
    cap still fired a "LONG XYZ" notification with no order ever
    submitted for it (the caller only submits `signals`, the list this
    function is handed after both caps have already run). This also kept
    phantom entries out of live-signal outcome tracking, which previously
    recorded a "signal" for tickers that were never traded.

    _log_live_signal() runs for EVERY signal here, dedup-suppressed or
    not -- found in the 2026-08-16 review: a signal inside the alert
    cooldown window is still tradeable (it's still in `signals`, what the
    caller actually submits to Alpaca), so skipping this call left a real
    fill with no entry in the live-outcome pending log to ever resolve
    win/loss against. Only the Telegram ALERT should be suppressed by
    dedup, not outcome tracking. Safe to call unconditionally:
    _log_live_signal() already dedupes internally by ticker+date, so a
    signal that keeps getting alert-suppressed across several scans in
    the same day still only logs once.
    """
    _alert_batch: list[str] = []
    for sig in signals:
        _log_live_signal(sig)
        if _is_duplicate_alert(sig.ticker):
            sys.stdout.write(f"       (dup suppressed — {sig.ticker} alerted <{ALERT_COOLDOWN_MIN}m ago)\n")
            continue
        if sig.ticker in smallcap_extra:
            fl_m, sh_pct, insider_pct, post_rs = smallcap_extra[sig.ticker]
            _alert_batch.append(format_smallcap_telegram(sig, fl_m, sh_pct, insider_pct, post_rs))
        else:
            _alert_batch.append(format_signal_telegram(sig, regime))
        _save_last_alert(sig.ticker)
    _send_signal_alert_batch(_alert_batch)


def run_pro_scanner(tickers: list[str] = WATCHLIST,
                    min_score: int = None,
                    use_ai: bool = False,
                    universe_label: str = "curated",
                    include_dynamic_smallcap: bool = True) -> list[ProSignal]:
    """
    Full pro-grade scanner with all 18 filters applied.
    Only returns signals that pass ALL hard gates AND score >= min_score.

    include_dynamic_smallcap=False skips ONLY the Finviz live-mover
    discovery step (a real yfinance-heavy call to fetch_dman_dynamic_tickers())
    — for callers that scan on a tight cadence (the daemon's 10-min loop)
    where the hourly cron's full "all"-universe scan already covers that
    broad net. DMan's own curated small-cap watchlist (DMAN_SMALLCAP_WATCHLIST)
    is NOT affected by this flag and is always scanned — that's the
    high-signal part worth checking frequently; Finviz's generic "any
    mover" discovery is the expensive, low-precision part worth throttling.
    """
    global MIN_CONFLUENCE
    tracker    = WinRateTracker()
    stats      = tracker.rolling_stats()
    min_score  = min_score or tracker.adaptive_min_score()

    # Resolve any pending live signals whose bars are now available
    resolved = resolve_live_outcomes(verbose=False)
    if resolved:
        print(f"  📊 {resolved} live trade(s) resolved — run --mode live-perf to see stats")

    # Consecutive loss guard — send Telegram once per session (dedup via alert cache)
    if stats["consec_losses"] >= MAX_CONSEC_LOSSES:
        print(f"\n  🛑 CONSECUTIVE LOSS GUARD: {stats['consec_losses']} losses in a row.")
        print(f"     Take a break. Reset your mind. Come back tomorrow.\n")
        if not _is_duplicate_alert("__CONSEC_LOSS__"):
            send_telegram(
                f"🛑 <b>DMan halted</b> — {stats['consec_losses']} consecutive losses.\n"
                f"Scanner paused for the day. Review your last trades."
            )
            _save_last_alert("__CONSEC_LOSS__")
        _log_scan_halt("consecutive_losses", tickers, min_score or 0)
        return []

    # Monthly loss circuit breaker — dedup so it fires at most once per 30-min window
    month_loss = get_this_month_loss()
    if month_loss <= -(MONTHLY_LOSS_LIMIT * 100):
        print(f"\n  🛑 MONTHLY LOSS LIMIT HIT: Down {month_loss:.1f}% this month "
              f"(limit: {MONTHLY_LOSS_LIMIT*100:.0f}%).")
        print(f"     Stop trading for the month. Review setups. Reset.\n")
        if not _is_duplicate_alert("__MONTHLY_LIMIT__"):
            send_telegram(f"🛑 <b>Monthly loss limit hit</b> — down {month_loss:.1f}% this month. Halted until next month.")
            _save_last_alert("__MONTHLY_LIMIT__")
        _log_scan_halt("monthly_loss_limit", tickers, min_score or 0)
        return []

    # Daily loss circuit breaker — dedup
    todays_loss = get_todays_loss()
    if todays_loss <= -(DAILY_LOSS_LIMIT * 100):
        print(f"\n  🛑 DAILY LOSS LIMIT HIT: Down {todays_loss:.1f}% today "
              f"(limit: {DAILY_LOSS_LIMIT*100:.0f}%).")
        print(f"     Stop trading. Protect your capital. Come back tomorrow.\n")
        if not _is_duplicate_alert("__DAILY_LIMIT__"):
            send_telegram(f"🛑 <b>Daily loss limit hit</b> — down {todays_loss:.1f}% today. Halted.")
            _save_last_alert("__DAILY_LIMIT__")
        _log_scan_halt("daily_loss_limit", tickers, min_score or 0)
        return []

    print(f"\n{'═'*68}")
    print(f"  D🔥man PRO Scanner v3  —  {datetime.today().strftime('%A %b %d, %Y')}")
    _cw = stats.get("consec_wins", 0)
    _cl = stats.get("consec_losses", 0)
    _streak_label = (f"🔥 {_cw}W streak" if _cw >= 2
                     else f"❄️ {_cl}L streak" if _cl >= 1 else "—")
    print(f"  Min score : {min_score}/100  |  AI scoring: {'ON' if use_ai else 'OFF'}")
    print(f"  Shorts    : {'ON' if ALLOW_SHORTS else 'OFF'}  |  "
          f"Rolling WR: {stats['win_rate']*100:.1f}%  ({stats['total']} trades)  |  Streak: {_streak_label}")
    print(f"{'═'*68}")

    # Get regime once (expensive call)
    print("  [1/2] Checking market regime & sectors...")
    regime   = get_market_regime()
    top_secs = get_top_sectors()
    vix_now  = float(regime['details'].get('VIX', 20))
    print(f"  Market : {regime['regime']} (score {regime['score']}/19)  VIX: {vix_now:.1f}")
    print(f"  Top sectors: {', '.join(top_secs)}")

    # VIX ≥ 40 hard halt — extreme tail-risk (COVID crash, flash crash, circuit breaker day)
    if vix_now >= 40:
        print(f"\n  🛑 VIX EXTREME: {vix_now:.1f} ≥ 40 — full halt. No orders in a crisis session.")
        if not _is_duplicate_alert("__VIX_EXTREME__"):
            send_telegram(
                f"🛑 <b>DMan HALTED — VIX Extreme</b>\n"
                f"VIX at {vix_now:.1f} (≥40). This is a crisis session.\n"
                f"All trading suspended. Protect capital. Come back when VIX < 35."
            )
            _save_last_alert("__VIX_EXTREME__")
        _log_scan_halt("vix_extreme", tickers, min_score or 0)
        return []

    # VIX regime scaling — tighten confluence floor in elevated-volatility markets
    if vix_now > 25:
        min_score = max(min_score, 90)
        print(f"  ⚠  VIX={vix_now:.1f} > 25 — min score raised to {min_score}/100")

    # VIX shock gate — single-session spike >20% means post-shock digestion period.
    # Even if VIX hasn't crossed 25 yet (e.g. 15→21), the spike itself signals
    # elevated intraday risk. Raise floor +5 pts to filter borderline setups.
    if regime.get("vix_shock"):
        min_score = max(min_score, min_score + 5)
        shock_note = regime["details"].get("VIX Shock", "")
        print(f"  ⚡ VIX SHOCK detected ({shock_note}) — min score raised to {min_score}/100")
        if not _is_duplicate_alert("__VIX_SHOCK__"):
            send_telegram(
                f"⚡ <b>VIX Shock active</b> — {shock_note}\n"
                f"Min score raised to {min_score}/100 for this session. Filters tighter."
            )
            _save_last_alert("__VIX_SHOCK__")

    # Defensive rotation flag — when XLP/XLU/XLV dominate XLK, warn about tech longs
    def_rotation = regime.get("defensive_rotation", False)
    if def_rotation:
        rot_note = regime["details"].get("Def Rotation", "")
        print(f"  🔄 DEFENSIVE ROTATION: {rot_note}")
        print(f"     Tech long signals will carry a -5 score penalty this session.")

    # Seasonal regime filter — weak months per backtest (25-38% WR combined).
    # Gap & Hold and Morning Runner are EXEMPT: the 38% Jul WR was driven by reversal/
    # mean-reversion setups that are now all disabled. These two setups have their own
    # technical gates (gap%, MACD, prior green, RVOL, regime score) that serve the same
    # purpose as the seasonal bar raise. Raising them to 92 would suppress most signals
    # while producing no protective benefit specific to those setups.
    # All other setups (if re-enabled) still get the full seasonal filter.
    _SEASONAL_EXEMPT  = {"Gap & Hold", "Morning Runner"}
    curr_month        = datetime.today().month
    _seasonal_active  = curr_month in SEASONAL_WEAK_MONTHS
    if _seasonal_active:
        month_name = datetime.today().strftime("%B")
        print(f"  📅  {month_name} seasonal filter — non-exempt setups raised to {SEASONAL_MIN_SCORE}/100 (Gap & Hold / Morning Runner exempt)")

    # Check open position risk — alert if any pending signal is within 2% of its stop
    _check_open_position_risk(regime)

    # Pre-fetch news for all tickers in one batch (much faster than per-ticker).
    # 20h (not 4h) to actually catch the catalyst that matters most: earnings
    # released after yesterday's close. A 4h window misses every after-hours
    # earnings reaction by the time the next morning's scan runs — confirmed
    # against GOOGL/TSLA's July 23 earnings gaps, whose news_boost would have
    # been False under the old window despite the catalyst being obvious.
    print(f"  [1.5/2] Pre-fetching news catalysts (last 20h)...", end=" ", flush=True)
    _scan_news_map: dict[str, list] = {}
    try:
        _scan_news_map = _fetch_alpaca_news(list(tickers), hours_back=20)
        _news_count = sum(1 for v in _scan_news_map.values() if v)
        print(f"{_news_count}/{len(tickers)} tickers have recent news")
        # Background knowledge-base log (2026-08-15) — this REST pre-fetch
        # is the only news pathway that runs during the cron scanner's own
        # windows (including premarket-early, before the daemon's
        # continuous news stream is even running for the day), so logging
        # here alongside the stream's own logging is what actually makes
        # coverage continuous across the full 4 AM-8 PM trading window
        # rather than just the hours the daemon happens to be up.
        # _log_news_event's own (symbols, headline) dedup keeps the same
        # story from re-logging every time this 20h-lookback fetch runs.
        # Sentiment looked up once per TICKER (not per headline) — it's
        # already a majority vote across that ticker's recent articles,
        # not headline-specific — and reused for every headline logged
        # for it this pass; _news_sentiment_verdict's own 10-min cache
        # keeps repeat cross-cycle lookups cheap.
        for _nt, _heads in _scan_news_map.items():
            if not _heads:
                continue
            _nt_sentiment = _news_sentiment_verdict(_nt)
            for _h in _heads:
                _log_news_event([_nt], _h, source="scan-prefetch", tag="watchlist",
                               sentiment=_nt_sentiment)
    except Exception as _ne:
        print(f"error ({str(_ne)[:60]})")

    # Batch-fetch daily bars via Alpaca SIP (Algo Trader Plus real-time feed)
    # before the ticker loop — a handful of chunked calls instead of one
    # yfinance request per ticker. This is what actually uses the paid
    # subscription for scan speed/reliability, not just as an emergency
    # fallback. Fail-safe: any ticker not warmed just falls through to
    # fetch_df()'s own per-ticker Alpaca-then-yfinance path unchanged.
    print(f"  [1.7/2] Pre-warming bars via Alpaca SIP (Algo Trader Plus)...", end=" ", flush=True)
    try:
        _warmed = prewarm_alpaca_bars(list(tickers))
        print(f"{_warmed}/{len(tickers)} tickers warmed")
    except Exception as _pe:
        print(f"error ({str(_pe)[:60]}) — falling back to per-ticker fetch")

    print(f"\n  [2/2] Scanning {len(tickers)} tickers...\n")

    signals = []
    rejected_counts = {"no_signal":0, "hard_gate":0, "low_score":0}
    _ticker_scan_start = time.monotonic()
    _ticker_scan_budget = 15 * 60  # 15-min cap on ticker loop (leaves room for universe build + persist)
    _budget_hit = False

    for i, ticker in enumerate(tickers, 1):
        if time.monotonic() - _ticker_scan_start > _ticker_scan_budget:
            _budget_hit = True
            remaining = len(tickers) - i + 1
            print(f"\n  ⏱  Scan time budget reached — {remaining} tickers skipped "
                  f"({i-1} scanned, {len(signals)} signal(s) found so far)")
            send_telegram(
                f"⏱ <b>DMan</b> scan time budget reached — {i-1}/{len(tickers)} tickers scanned, "
                f"{len(signals)} signal(s) found. Consider reducing universe size."
            )
            break
        sys.stdout.write(f"\r  {i:>3}/{len(tickers)}: {ticker:<8} ", )
        sys.stdout.flush()

        # Fetch + compute indicators
        raw = fetch_df(ticker)
        if raw is None or len(raw) < 60:
            rejected_counts["no_signal"] += 1
            continue
        df = _compute_indicators_cached(ticker, raw)
        df.dropna(subset=["EMA50","RSI","MACD","ATR"], inplace=True)
        if len(df) < 10:
            rejected_counts["no_signal"] += 1
            continue

        # Raw signal detection (v2 logic)
        sig = _raw_signals(df, ticker)
        if sig is None:
            rejected_counts["no_signal"] += 1
            continue

        # Tag news catalyst before scoring (adds +5 pts in score_signal, and
        # can override MTF for Gap & Hold / Bear Gap Hold — see score_signal).
        # See _news_boost_after_sentiment_veto's docstring for why a negative
        # headline doesn't earn the same credit a positive one does.
        sig.news_boost = _news_boost_after_sentiment_veto(bool(_scan_news_map.get(ticker)), ticker)

        # Apply all pro filters
        sig = score_signal(sig, df, regime, tracker)

        # Hard gates: regime + MTF + earnings + divergence (absolute stops)
        if not sig.regime_ok:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"REGIME BLOCKED ({sig.bias} in {regime['regime']})\n")
            continue
        if not sig.mtf_ok:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"MTF BLOCKED (weekly chart disagrees)\n")
            continue
        if not sig.earnings_ok:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"EARNINGS BLACKOUT\n")
            continue
        if not sig.macro_ok:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"MACRO BLACKOUT (FOMC/NFP)\n")
            continue
        if not sig.divergence_free:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"DIVERGENCE DETECTED\n")
            continue

        # Defensive rotation penalty — during active tech→defensive rotation sessions,
        # deduct 5 pts from tech long scores to reflect sector headwind.
        # TICKER_SECTOR maps tech names to "Technology"; penalty only on LONG signals.
        if def_rotation and sig.bias == "LONG":
            sig_sector = TICKER_SECTOR.get(sig.ticker, "")
            if sig_sector == "Technology":
                sig.confluence_score = max(0, sig.confluence_score - 5)
                sys.stdout.write(f"  [-5 def-rotation penalty → {sig.confluence_score}]  ")

        # Soft score gate (per-setup overrides + volatile ticker floor + seasonal)
        effective_min = SETUP_MIN_CONFLUENCE.get(sig.setup, min_score)
        if sig.ticker in VOLATILE_TICKERS:
            effective_min = max(effective_min, VOLATILE_MIN_CONFLUENCE)
        if _seasonal_active and sig.setup not in _SEASONAL_EXEMPT:
            effective_min = max(effective_min, SEASONAL_MIN_SCORE)
        if sig.confluence_score < effective_min:
            rejected_counts["low_score"] += 1
            sys.stdout.write(f"score {sig.confluence_score}/100 < {effective_min}\n")
            continue

        # Optional AI scoring. A None result means the call itself failed
        # (no key, timeout, rate limit, malformed response) — that must NOT
        # be treated as "the AI scored this low." Found 2026-08-16 review:
        # this used to always assign the return value (0 on failure)
        # straight into sig.ai_score and reject on `< 6`, so a rate-limited
        # or timed-out request silently rejected a signal that never
        # actually got an AI opinion, indistinguishable in the log from a
        # real low score.
        if use_ai and ANTHROPIC_API_KEY:
            _ai_result = ai_score_signal(sig, regime)
            if _ai_result is None:
                sys.stdout.write("AI score unavailable (API error) — falling back to confluence-only  ")
            else:
                sig.ai_score = _ai_result
                sig.final_score = round(sig.confluence_score*0.70 + sig.ai_score*10*0.30, 1)
                if sig.ai_score < 6:
                    rejected_counts["low_score"] += 1
                    sys.stdout.write(f"AI score {sig.ai_score}/10 too low\n")
                    continue

        # Gap & Hold: suppress alerts in first 15 min of session (9:30–9:44 AM ET).
        # Gaps that haven't survived initial selling pressure aren't proven holds yet.
        _now_et = datetime.now(ET)
        if (sig.setup == "Gap & Hold"
                and _now_et.hour == 9 and _now_et.minute < 45):
            sys.stdout.write("       (early-session gate — Gap & Hold held until 9:45 AM ET)\n")
            continue

        signals.append(sig)
        arrow = "🟢" if sig.bias == "LONG" else "🔴"
        sys.stdout.write(
            f"{arrow} PASS  score={sig.confluence_score}"
            + (f" AI={sig.ai_score}" if sig.ai_score else "")
            + "\n"
        )
        # NOTE: alerting is deferred until after the heat-cap and sector-cap
        # filters below finalize the signal list — see that block for why.

    # ── Small-cap / Low Float Catalyst pass ────────────────────────────────
    # _smallcap_extra holds per-ticker display data (float/short-interest/etc.)
    # needed by format_smallcap_telegram() at alert time below, since alerting
    # now happens after this pass has already returned/gone out of scope.
    _smallcap_extra: dict[str, tuple] = {}
    # Second pass over the same universe using Dman's micro-cap criteria.
    # Separate risk rules: 0.5% per trade, max $2,500 cost, lower score bar.
    if ENABLE_SMALLCAP:
        sc_rejected = 0
        sc_found    = 0
        # Dynamic Finviz discovery: low-float (<5M), price <$20, vol >500k
        _finviz_tickers: list[str] = []
        if ENABLE_DYNAMIC_SMALLCAP and include_dynamic_smallcap:
            print("  🔍  Fetching today's movers (Yahoo Finance day gainers + most actives)...", flush=True)
            _finviz_tickers = fetch_dman_dynamic_tickers()
            if _finviz_tickers:
                print(f"  🔍  Live movers: {len(_finviz_tickers)} candidates with RVOL ≥1.5x: "
                      f"{', '.join(_finviz_tickers[:10])}{'...' if len(_finviz_tickers) > 10 else ''}",
                      flush=True)
            else:
                print("  🔍  No live movers found (market closed or pre-market)", flush=True)
        # Merge: large-cap tickers (already cached) + curated watchlist + live movers
        sc_universe = list(dict.fromkeys(list(tickers) + DMAN_SMALLCAP_WATCHLIST + _finviz_tickers))
        for ticker in sc_universe:
            df = fetch_df(ticker)   # already cached from the large-cap pass
            if df is None or len(df) < 30:
                continue
            df = _compute_indicators_cached(ticker, df)
            sc_sig = detect_low_float_catalyst(df, ticker)
            if sc_sig is None:
                sc_rejected += 1
                continue
            # Simple hard gates for small-cap (skip MTF/RS/Sector — meaningless)
            macro_ok, _ = check_macro_safe()
            if not macro_ok:
                sc_rejected += 1
                continue
            earn_ok, _ = check_earnings_safe(ticker)
            if not earn_ok:
                sc_rejected += 1
                continue
            # Score with small-cap specific scorer
            # Watchlist tickers use lower threshold — Dman's curation is the edge signal
            sc_sig.confluence_score = score_smallcap_signal(sc_sig)
            _sc_threshold = DMAN_WATCHLIST_MIN_SCORE if ticker in DMAN_SMALLCAP_WATCHLIST else SMALLCAP_MIN_SCORE
            if sc_sig.confluence_score < _sc_threshold:
                sc_rejected += 1
                continue
            # Skip if same ticker already fired as large-cap signal
            if any(s.ticker == ticker for s in signals):
                continue
            fl_m, sh_pct, insider_pct, _cash_mc = _get_short_float_data(ticker)
            post_rs = _is_recent_reverse_split(ticker)
            sc_found += 1
            signals.append(sc_sig)
            sys.stdout.write(f"\r  🔥 SMALLCAP {ticker:<8} "
                             f"float={fl_m:.1f}M SI={sh_pct:.0f}%"
                             f"{' POST-RS' if post_rs else ''} "
                             f"score={sc_sig.confluence_score}\n")
            _smallcap_extra[ticker] = (fl_m, sh_pct, insider_pct, post_rs)
            # NOTE: alerting deferred to after heat-cap/sector-cap — see below.
        if sc_found or sc_rejected:
            print(f"  🔥  Small-cap pass: {sc_found} signal(s), {sc_rejected} rejected")

    signals.sort(key=lambda s: s.confluence_score, reverse=True)

    # Portfolio heat cap: admit signals until total open risk hits the limit.
    # Seed with EXISTING open positions so prior-scan risk is counted.
    # Found in the 2026-08-16 review: this used to call get_all_positions()
    # directly, a duplicate REST fetch of exactly what _check_open_position_risk()
    # (called earlier in this same scan, via _check_stop_coverage()) already
    # fetched. _check_stop_coverage() now caches its Alpaca fetch for 30s, so
    # calling it again here reuses that cache instead of a second round-trip
    # in the common case (this runs well within that window of the earlier call).
    heat_capped: list[ProSignal] = []
    eff_account = get_effective_account()
    total_risk_pct = 0.0
    try:
        if eff_account > 0:
            _heat_positions = _check_stop_coverage()
            if _heat_positions:
                for _hp in _heat_positions.values():
                    # Use (avg_entry_price - stop_price) × qty as risk, not full market_value.
                    # Alpaca doesn't expose stop_price on positions, so we approximate risk as
                    # 2% of account per existing position (matches SMALLCAP_RISK_PCT).
                    # Options legs are excluded here — confirmed live 2026-08-11: with 2
                    # equity swings (CELZ, CLRO) + 2 SMCI option legs open, this loop hit
                    # 8% against the 6% cap and would have silently heat-capped out ANY
                    # new equity signal, however good, regardless of the options' actual
                    # (much smaller, already-defined) premium risk. Options are already
                    # risk-managed separately — trailing stop, milestone alerts — so they
                    # shouldn't also consume the equity heat budget.
                    if getattr(_hp, "asset_class", None) == AssetClass.US_EQUITY:
                        total_risk_pct += SMALLCAP_RISK_PCT
    except Exception:
        pass   # if Alpaca unavailable, proceed without existing-position offset
    if total_risk_pct > 0:
        print(f"  🌡  Existing position heat: {total_risk_pct*100:.1f}% of account")
    for sig in signals:
        trade_risk_pct = sig.risk_usd / eff_account if eff_account > 0 else 0
        if total_risk_pct + trade_risk_pct <= PORTFOLIO_HEAT_LIMIT:
            heat_capped.append(sig)
            total_risk_pct += trade_risk_pct
        else:
            sys.stdout.write(
                f"  🌡  Heat cap {PORTFOLIO_HEAT_LIMIT*100:.0f}% reached "
                f"({total_risk_pct*100:.1f}% used) — {sig.ticker} excluded\n"
            )
    signals = heat_capped

    signals = _apply_sector_concentration_cap(signals)

    _finalize_and_alert_signals(signals, regime, _smallcap_extra)

    print(f"\n{'─'*68}")
    print(f"  ✅  {len(signals)} A+ setup(s) passed all filters")
    print(f"  ❌  Rejected: {rejected_counts['no_signal']} no signal, "
          f"{rejected_counts['hard_gate']} hard gate, "
          f"{rejected_counts['low_score']} low score")
    print(f"  🌡  Portfolio heat used: {total_risk_pct*100:.1f}% / {PORTFOLIO_HEAT_LIMIT*100:.0f}%")
    print(f"{'─'*68}\n")

    # Persist scan result to rolling log
    try:
        # News sentiment breadth snapshot (2026-08-15) — observation-only
        # per get_market_regime()'s own docstring; recorded here purely so
        # there's a reviewable per-scan trend to look back on before ever
        # deciding whether to wire it into scoring. None-safe: regime's
        # own news_breadth is None if that lookup itself failed.
        _nb = regime.get("news_breadth") or {}
        _append_scan_log({
            "ts":                  datetime.now(ET).isoformat(),
            "regime":              regime.get("regime", "?"),
            "regime_score":        regime.get("score", 0),
            "vix":                 round(float(regime["details"].get("VIX", 0)), 1),
            "min_score":           min_score,
            "universe":            universe_label,
            "tickers_total":       len(tickers),
            "signals":             len(signals),
            "signal_tickers":      [s.ticker for s in signals],
            "rejected_no_signal":  rejected_counts["no_signal"],
            "rejected_hard_gate":  rejected_counts["hard_gate"],
            "rejected_low_score":  rejected_counts["low_score"],
            "budget_hit":          _budget_hit,
            "news_breadth_pct":    _nb.get("breadth_pct"),
            "news_breadth_total":  _nb.get("total", 0),
        })
    except Exception:
        pass  # never let logging block the scan return

    # Near-miss collection — only when no signals fired; uses cached fetch_df() data (fast)
    # Covers the full scan universe (not just WATCHLIST) so Yahoo gainers are included.
    _near_misses: list[tuple[str, float, str]] = []
    _b_tier: list[dict] = []   # below-threshold setups for manual consideration
    if not signals:
        _nm_universe = list(dict.fromkeys(list(tickers)[:120] + list(WATCHLIST)))
        for _nm_t in _nm_universe:
            try:
                _nm_raw = fetch_df(_nm_t)
                if _nm_raw is None or len(_nm_raw) < 30:
                    continue
                _nm_df  = compute_indicators(_nm_raw.copy())
                _nm_r   = _nm_df.iloc[-1]
                _nm_p   = _nm_df.iloc[-2]
                _nm_gap = (float(_nm_r["Open"]) - float(_nm_p["Close"])) / float(_nm_p["Close"]) * 100
                if _nm_gap < 1.0:
                    continue
                _nm_macd     = float(_nm_r.get("MACD", 0) or 0)
                _nm_prn_grn  = float(_nm_p["Close"]) > float(_nm_p["Open"])
                _nm_sec_ok   = _sector_etf_above_ema50(_nm_t)
                # Hold% vs open: appended to MACD/prior-red blockers so the user
                # can see whether the price was above or below the gap open at scan time.
                try:
                    _nm_c_now = float(_nm_r["Close"].iloc[0]) if hasattr(_nm_r["Close"], "iloc") else float(_nm_r["Close"])
                    _nm_o_day = float(_nm_r["Open"].iloc[0])  if hasattr(_nm_r["Open"],  "iloc") else float(_nm_r["Open"])
                    _nm_hold_tag = f" ({(_nm_c_now - _nm_o_day) / _nm_o_day * 100:+.1f}%)"
                except Exception:
                    _nm_hold_tag = ""
                if not _nm_sec_ok:
                    _nm_blocker = "sector⚠️"
                elif _nm_macd <= 0:
                    _nm_blocker = f"MACD {_nm_macd:+.1f}{_nm_hold_tag}"
                elif not _nm_prn_grn:
                    _nm_blocker = f"prior red{_nm_hold_tag}"
                else:
                    # Primary filters all pass — run full pipeline to get exact blocker
                    try:
                        _nm_raw_sig = _raw_signals(_nm_df, _nm_t)
                        if _nm_raw_sig is None:
                            # Identify the specific _raw_signals sub-check that failed
                            _nm_rvol = float(_nm_r.get("RVOL", 0) or 0)
                            _nm_rsi  = float(_nm_r.get("RSI", 0) or 0)
                            _nm_c    = float(_nm_r["Close"])
                            _nm_o    = float(_nm_r["Open"])
                            if _nm_rvol < 1.5:
                                _nm_blocker = f"RVOL {_nm_rvol:.1f}x"
                            elif _nm_rsi <= 50:
                                _nm_blocker = f"RSI {_nm_rsi:.0f}"
                            elif _nm_c < _nm_o * 0.995:
                                _nm_blocker = f"not holding ({(_nm_c/_nm_o-1)*100:.1f}%)"
                            else:
                                _nm_blocker = "no setup pattern"
                        else:
                            _nm_scored = score_signal(_nm_raw_sig, _nm_df, regime, tracker)
                            _nm_sc     = _nm_scored.confluence_score
                            _nm_blocker = f"score {_nm_sc}/{min_score}"
                    except Exception:
                        _nm_blocker = "score short"
                # Collect actionable entry levels for near-miss Telegram
                try:
                    _nm_c_px  = float(_nm_r.get("Close", 0) or 0)
                    _nm_o_px  = float(_nm_r.get("Open",  0) or 0)
                    _nm_lo_px = float(_nm_r.get("Low",   0) or 0)
                    _nm_stop  = round(min(_nm_lo_px * 0.99, _nm_o_px * 0.985), 2) if _nm_lo_px > 0 else 0
                    _nm_risk  = (_nm_c_px - _nm_stop) if _nm_stop > 0 and _nm_c_px > _nm_stop else 0
                    _nm_t1    = round(_nm_c_px + 2.5 * _nm_risk, 2) if _nm_risk > 0 else 0
                    _nm_rvol  = float(_nm_r.get("RVOL", 0) or 0)
                    _nm_score_val = 0
                    if "score" in _nm_blocker:
                        try:
                            _nm_score_val = int(_nm_blocker.split()[1].split("/")[0])
                        except Exception:
                            pass
                    _near_misses.append((_nm_t, _nm_gap, _nm_blocker))
                    # B-tier: setup almost qualified (score within 15 of threshold, or
                    # only blocked by RVOL/RSI which could change intraday)
                    _b_tier_reason = ""
                    if _nm_score_val >= min_score - 15 and _nm_score_val > 0:
                        _b_tier_reason = f"score {_nm_score_val}/{min_score}"
                    elif "RVOL" in _nm_blocker and _nm_rvol >= 1.0:
                        _b_tier_reason = f"RVOL {_nm_rvol:.1f}x (needs ≥2.0x)"
                    if _b_tier_reason and _nm_c_px > 0 and _nm_stop > 0 and _nm_t1 > 0:
                        _b_tier.append({
                            "ticker": _nm_t, "gap": _nm_gap, "entry": _nm_c_px,
                            "stop": _nm_stop, "t1": _nm_t1, "rvol": _nm_rvol,
                            "reason": _b_tier_reason,
                        })
                except Exception:
                    _near_misses.append((_nm_t, _nm_gap, _nm_blocker))
            except Exception:
                continue
        _near_misses.sort(key=lambda x: x[1], reverse=True)
        _near_misses = _near_misses[:3]
        _b_tier.sort(key=lambda x: x["gap"], reverse=True)
        _b_tier = _b_tier[:2]

    # Expose scan metadata for the heartbeat in main()
    _last_scan_meta.update({
        "rejected":     rejected_counts,
        "near_misses":  _near_misses,
        "b_tier":       _b_tier,
        "tickers_total": len(tickers),
    })

    # Friday close-out advisory — fires on the 3:30 PM scan (30 min before bell)
    try:
        _now_co = datetime.now(ET)
        if _now_co.weekday() == 4:  # Friday
            _hhmm_co = _now_co.hour * 100 + _now_co.minute
            if 1530 <= _hhmm_co <= 1559:
                _pending_co = []
                if os.path.exists(LIVE_SIGNALS_FILE):
                    with open(LIVE_SIGNALS_FILE) as _fco:
                        _pending_co = json.load(_fco).get("pending", [])
                _mins_left = (16 * 60) - (_now_co.hour * 60 + _now_co.minute)
                # Find upcoming FOMC within 7 days
                _td_co = _now_co.date()
                _fomc_co = ""
                for _ev_co in sorted(_FOMC_DATES):
                    _d_co = (_ev_co - _td_co).days
                    if 1 <= _d_co <= 7:
                        _fomc_co = f" FOMC {_ev_co.strftime('%a %b %d')} in {_d_co}d."
                        break
                    if _d_co > 7:
                        break
                _pos_co = ""
                if _pending_co:
                    _pos_co = "\nOpen: " + ", ".join(p.get("ticker","?") for p in _pending_co)
                send_telegram(
                    f"⚠️ <b>FRIDAY — {_mins_left} min to close</b>\n"
                    f"Exit positions not at T1 to avoid weekend risk.{_fomc_co}"
                    f"{_pos_co}"
                )
                print(f"\n  ⚠️  Friday close-out advisory sent ({_mins_left} min to bell)")
    except Exception:
        pass

    return signals


def print_pro_signal(s: ProSignal):
    arrow  = "🟢 LONG " if s.bias == "LONG" else "🔴 SHORT"
    pct    = round(s.cost / get_effective_account() * 100, 1)
    kelly  = round(s.kelly_frac * 100, 2)
    beta_str = f"  Beta: {s.beta:.1f}x" if s.beta != 1.0 else ""
    print(f"{arrow}  {s.ticker}  ─  {s.setup}")
    print(f"   Entry   : ${s.entry:<10}  Stop   : ${s.stop}")
    print(f"   T1 (2R) : ${s.target1:<10}  T2 (3R): ${s.target2}")
    print(f"   R/R     : {s.rr}:1     RSI  : {s.rsi}   RVOL: {s.rvol}x")
    print(f"   Size    : {s.shares} shares  Cost: ${s.cost:,.0f} ({pct}% acct)")
    print(f"   Risk    : ${s.risk_usd:,.0f}  Kelly: {kelly}%{beta_str}")
    print(f"   Score   : {s.confluence_score}/100"
          + (f"  AI: {s.ai_score}/10" if s.ai_score else ""))
    print(f"   Gates   : {s.summary()}")
    print(f"   Candle  : {s.candle_pattern if s.candle_pattern else '—'}")
    print(f"   Why     : {s.reason}")

    # Trailing stop guidance
    if s.atr > 0:
        be_stop    = s.entry
        trail_stop = round(s.target1 - s.atr, 2) if s.bias == "LONG" else round(s.target1 + s.atr, 2)
        print(f"   Trail   : move stop → ${be_stop} (breakeven) after T1 | "
              f"then trail to ~${trail_stop} (1 ATR)")

    # Score breakdown
    bd = s.score_breakdown
    if bd:
        parts = [f"{k}:{v}" for k,v in bd.items() if v > 0]
        print(f"   Points  : {' | '.join(parts)}")

    # OpEx week warning
    if is_opex_week():
        print(f"   ⚠️  OpEx week — expect pinning/volatility; consider sizing down")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 21 — PRO BACKTESTER (walk-forward with all filters)
# ═══════════════════════════════════════════════════════════════════════════

def run_pro_backtest(tickers: list[str] = WATCHLIST,
                     years: int = 2, min_score: int = 85) -> dict:
    """
    Walk-forward backtest applying all pro filters on each historical window.
    More accurate than raw backtesting because regime/sector/RS are computed
    at each point in time (not with future data).
    """
    print(f"\n{'═'*68}")
    print(f"  D🔥man PRO Backtester v3 — {years}yr Walk-Forward")
    print(f"  Min score: {min_score}/100  |  Tickers: {len(tickers)}")
    print(f"  Slippage: 0.1%  |  Commission: $0")
    print(f"{'═'*68}\n")

    tracker  = WinRateTracker()
    all_trades = []
    equity   = ACCOUNT_SIZE

    end   = datetime.today()
    start = end - timedelta(days=365*years + 90)

    # Pre-load SPY once for historical regime computation (no look-ahead)
    spy_hist = None
    try:
        spy_hist = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        if spy_hist is not None and len(spy_hist) > 50:
            if isinstance(spy_hist.columns, pd.MultiIndex):
                spy_hist.columns = spy_hist.columns.droplevel(1)
        else:
            spy_hist = None
    except Exception:
        spy_hist = None

    for ticker in tickers:
        sys.stdout.write(f"\r  Backtesting {ticker:<8}...")
        sys.stdout.flush()

        try:
            raw = yf.download(ticker, start=start, end=end,
                              progress=False, auto_adjust=True)
            if raw is None or len(raw) < 100:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
        except Exception:
            continue

        in_trade    = False
        open_sig    = None
        entry_i     = 0
        entry_px    = 0.0

        for i in range(80, len(raw) - 1):
            window = compute_indicators(raw.iloc[:i+1].copy())
            window.dropna(subset=["EMA50","RSI","MACD","ATR"], inplace=True)
            if len(window) < 10:
                continue

            if not in_trade:
                sig = _raw_signals(window, ticker)
                if sig is None:
                    continue
                if sig.rr < MIN_RR:
                    continue

                # Historical regime from SPY data up to this bar's date (no look-ahead)
                current_date = raw.index[i]
                if spy_hist is not None:
                    spy_window = spy_hist[spy_hist.index <= current_date]
                    hist_regime = get_regime_from_window(spy_window)
                else:
                    hist_regime = {"regime": "CHOP", "score": 7}

                # Hard regime gate
                if sig.bias == "LONG"  and hist_regime["regime"] == "BEAR":
                    continue
                if sig.bias == "SHORT" and hist_regime["regime"] == "BULL":
                    continue

                r = window.iloc[-1]
                # RS: use 20-day price change as proxy
                pct20 = float(r["Chg20d"]) if "Chg20d" in r.index else 0
                if sig.bias == "LONG"  and pct20 < -5: continue
                if sig.bias == "SHORT" and pct20 >  5: continue
                # Divergence
                div_free, _ = check_divergence_free(window, sig.bias)
                if not div_free:
                    continue
                # Fibonacci, VWAP, POC, candlestick, 52wk prox
                _, fib_pts  = check_fibonacci(window, sig.entry)
                _, vwap_pts = check_vwap(window, sig.bias)
                _, poc_pts  = check_poc_alignment(window, sig.entry, sig.bias)
                _, candle_pts = detect_candle_pattern(window.iloc[-1], window.iloc[-2], sig.bias)
                try:
                    hi52 = float(window["High"].iloc[-252:].max()) if len(window) >= 252 else float(window["High"].max())
                    off_hi = (hi52 - sig.entry) / hi52 * 100
                    prox_pts = 10 if off_hi <= 5 else (7 if off_hi <= 15 else 0)
                except Exception:
                    prox_pts = 0

                # Backtest score using historical regime score instead of hardcoded 15
                regime_pts = min(15, hist_regime.get("score", 7))
                # Supertrend alignment
                try:
                    st_bull = bool(window["ST_bull"].iloc[-1])
                    st_pts  = 8 if (sig.bias == "LONG" and st_bull) or \
                                   (sig.bias == "SHORT" and not st_bull) else 0
                except Exception:
                    st_pts = 4
                # ADX strength
                try:
                    adx_val = float(window["ADX"].iloc[-1])
                    adx_pts = 5 if adx_val > 25 else (2 if adx_val > 20 else 0)
                except Exception:
                    adx_pts = 2
                # Divergence-free bonus
                div_pts = 5 if div_free else 0
                # ATR percentile score
                atr_pts = check_atr_percentile(window, sig.setup)
                # Regime-setup-type bonus (BULL→momentum, CHOP→reversal)
                momentum_setups = {"Vol Breakout", "Gap & Hold", "VCP", "EMA Pullback", "Morning Runner"}
                reversal_setups = {"OS Bounce", "OB Reversal", "MACD Cross", "MACD Bear",
                                   "Gap & Short", "EMA Breakdown", "Vol Breakdown"}
                cur_regime = hist_regime.get("regime", "CHOP")
                if cur_regime == "BULL" and sig.setup in momentum_setups:
                    regime_setup_pts = 8
                elif cur_regime in ("BEAR", "CHOP") and sig.setup in reversal_setups:
                    regime_setup_pts = 8
                else:
                    regime_setup_pts = 0
                bt_score = (
                    (10 if sig.rvol >= 2.0 else 5 if sig.rvol >= 1.5 else 0) +
                    (8 if sig.rr >= 2.5 else 5) +
                    fib_pts + vwap_pts + poc_pts + candle_pts + prox_pts +
                    (5 if 45 <= sig.rsi <= 62 else 0) +
                    regime_pts + st_pts + adx_pts + div_pts + atr_pts + regime_setup_pts
                )
                bt_min = SETUP_MIN_CONFLUENCE.get(sig.setup, min_score)
                if sig.ticker in VOLATILE_TICKERS:
                    bt_min = max(bt_min, VOLATILE_MIN_CONFLUENCE)
                if raw.index[i].month in SEASONAL_WEAK_MONTHS:
                    bt_min = max(bt_min, SEASONAL_MIN_SCORE)
                if bt_score < bt_min * 0.95:
                    continue

                entry_px = sig.entry * 1.001   # 0.1% slippage
                entry_i  = i
                in_trade = True
                open_sig = sig
                open_sig._t1_hit      = False
                open_sig._trail_stop  = sig.stop
                open_sig._remain_shares = 0
                open_sig._be1r_set    = False
                partial_pnl = 0.0
                eff_equity = max(equity, 5_000)
                risk_amt   = eff_equity * RISK_PER_TRADE
                sig.shares = max(1, int(risk_amt / sig.risk_per_share))

            else:
                bar   = raw.iloc[i]
                sig   = open_sig
                ep    = entry_px
                hold  = i - entry_i
                is_lo = sig.bias == "LONG"
                cur_stop = getattr(sig, "_trail_stop", sig.stop)
                partial_pnl = getattr(sig, "_partial_pnl", 0.0)

                hit_stop = float(bar["Low"]) <= cur_stop if is_lo else float(bar["High"]) >= cur_stop
                hit_t1   = (float(bar["High"]) >= sig.target1 if is_lo
                            else float(bar["Low"]) <= sig.target1) and not sig._t1_hit
                hit_t2   = (float(bar["High"]) >= sig.target2 if is_lo
                            else float(bar["Low"]) <= sig.target2) and sig._t1_hit
                hit_time = hold >= 15

                # BE@1R: move stop to entry once price reaches 1R profit (before T1 at 2.5R)
                be1r_px = (ep + (ep - sig.stop)) if is_lo else (ep - (sig.stop - ep))
                hit_be1r = (not getattr(sig, "_be1r_set", False)
                            and not sig._t1_hit
                            and (float(bar["High"]) >= be1r_px if is_lo
                                 else float(bar["Low"]) <= be1r_px))
                if hit_be1r:
                    sig._trail_stop = ep
                    sig._be1r_set   = True
                    continue  # don't exit — just protect the position

                # Stall exit: if no meaningful move after 3 bars, free the capital
                hit_stall = (hold >= 3 and not sig._t1_hit
                             and abs(float(bar["Close"]) - ep) / ep * 100 < 0.5)

                # Phase 1: T1 → scale ⅓ off, move stop to breakeven
                if hit_t1 and not sig._t1_hit:
                    t1px  = sig.target1 * 0.999
                    ps    = max(1, sig.shares // 3)
                    ppnl  = (t1px - ep)*ps if is_lo else (ep - t1px)*ps
                    equity += ppnl
                    sig._t1_hit         = True
                    sig._trail_stop     = ep         # stop to breakeven
                    sig._remain_shares  = sig.shares - ps
                    sig._partial_pnl    = ppnl
                    continue

                exit_px = exit_reason = None
                active  = getattr(sig, "_remain_shares", sig.shares) if sig._t1_hit else sig.shares

                if sig._t1_hit:
                    if hit_t2:     exit_px, exit_reason = sig.target2*0.999,  "T2"
                    elif hit_stop: exit_px, exit_reason = cur_stop*1.005, "STOP(BE)"
                    elif hit_time: exit_px, exit_reason = float(bar["Close"]), "TIME"
                else:
                    be_set = getattr(sig, "_be1r_set", False)
                    if hit_stop:
                        lbl = "STOP(BE)" if be_set else "STOP"
                        exit_px, exit_reason = cur_stop*1.005, lbl
                    elif hit_stall: exit_px, exit_reason = float(bar["Close"]), "STALL"
                    elif hit_time:  exit_px, exit_reason = float(bar["Close"]), "TIME"

                if exit_px is not None:
                    raw_pnl = ((exit_px-ep)*active if is_lo else (ep-exit_px)*active) + partial_pnl
                    equity += raw_pnl
                    outcome = "WIN" if raw_pnl > 0 else ("BE" if raw_pnl == 0 else "LOSS")
                    pnl_pct = raw_pnl / (ep * sig.shares) * 100

                    tracker.record(TradeRecord(
                        ticker=ticker, date=str(raw.index[i].date()),
                        bias=sig.bias, setup=sig.setup,
                        entry=round(ep,2), exit=round(exit_px,2),
                        outcome=outcome, pnl_pct=round(pnl_pct,2), score=bt_score,
                        is_live=False,   # backtest simulation, not a real fill
                    ))
                    all_trades.append({
                        "ticker":ticker,"setup":sig.setup,"bias":sig.bias,
                        "entry_date":str(raw.index[entry_i].date()),
                        "exit_date":str(raw.index[i].date()),
                        "pnl":round(raw_pnl,2),"pnl_pct":round(pnl_pct,2),
                        "outcome":outcome,"exit":exit_reason,"score":bt_score,
                    })
                    in_trade = False

    # Aggregate
    if not all_trades:
        print("\n  No trades generated. Try lowering --score.\n")
        return {}

    df_t   = pd.DataFrame(all_trades)
    wins   = df_t[df_t["outcome"]=="WIN"]
    losses = df_t[df_t["outcome"]=="LOSS"]
    wr     = len(wins)/len(df_t)*100
    gp     = wins["pnl"].sum()
    gl     = losses["pnl"].abs().sum()
    pf     = gp/gl if gl>0 else float("inf")
    eq_curve = ACCOUNT_SIZE + df_t["pnl"].cumsum()
    peak   = eq_curve.cummax()
    dd     = ((peak - eq_curve)/peak*100).max()
    ret    = (eq_curve.iloc[-1]-ACCOUNT_SIZE)/ACCOUNT_SIZE*100
    sharpe = (df_t["pnl_pct"].mean()/df_t["pnl_pct"].std()*(252**0.5)
              if df_t["pnl_pct"].std()>0 else 0)

    # Sortino ratio — penalises only downside volatility
    down_ret  = df_t[df_t["pnl_pct"] < 0]["pnl_pct"]
    down_std  = down_ret.std() if len(down_ret) > 1 else 1.0
    sortino   = df_t["pnl_pct"].mean() / down_std * (252**0.5) if down_std > 0 else 0

    # Calmar ratio — annualised return / max drawdown
    annual_ret = ret / years
    calmar     = annual_ret / dd if dd > 0 else float("inf")

    W = 68
    print(f"\n{'═'*W}")
    print(f"  PRO BACKTEST RESULTS — D🔥man v3")
    print(f"{'─'*W}")
    print(f"  Total Trades   : {len(df_t)}")
    print(f"  Win Rate       : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  Profit Factor  : {pf:.2f}x")
    print(f"  Total Return   : {'+' if ret>=0 else ''}{ret:.1f}%  ({'+' if annual_ret>=0 else ''}{annual_ret:.1f}%/yr)")
    print(f"  Max Drawdown   : -{dd:.1f}%")
    print(f"  Sharpe Ratio   : {sharpe:.2f}")
    print(f"  Sortino Ratio  : {sortino:.2f}")
    print(f"  Calmar Ratio   : {calmar:.2f}")
    print(f"  Avg Win %      : +{wins['pnl_pct'].mean():.2f}%")
    print(f"  Avg Loss %     : {losses['pnl_pct'].mean():.2f}%")
    print(f"{'─'*W}")

    by_setup = df_t.groupby("setup").agg(
        trades=("pnl","count"),
        wins=("outcome", lambda x: (x=="WIN").sum()),
        avg_pnl=("pnl_pct","mean")
    ).reset_index()
    print(f"\n  BY SETUP:")
    for _,row in by_setup.iterrows():
        w = row["wins"]/row["trades"]*100
        print(f"  {row['setup']:<20} {row['trades']:>4} trades | {w:>5.1f}% WR | "
              f"avg {row['avg_pnl']:+.2f}%")

    by_exit = df_t["exit"].value_counts()
    print(f"\n  EXIT BREAKDOWN:")
    for reason,cnt in by_exit.items():
        print(f"  {reason:<12}: {cnt} ({cnt/len(df_t)*100:.1f}%)")

    # Monthly P&L table
    df_t["month"] = pd.to_datetime(df_t["exit_date"]).dt.to_period("M")
    monthly = df_t.groupby("month").agg(
        trades=("pnl", "count"),
        pnl=("pnl", "sum"),
        wins=("outcome", lambda x: (x == "WIN").sum()),
    )
    print(f"\n  MONTHLY P&L:")
    print(f"  {'Month':<10} {'Trades':>6} {'WR':>6} {'P&L':>10}")
    print(f"  {'─'*36}")
    for month, row in monthly.iterrows():
        month_wr = row["wins"] / row["trades"] * 100 if row["trades"] > 0 else 0
        sign = "+" if row["pnl"] >= 0 else ""
        print(f"  {str(month):<10} {row['trades']:>6} {month_wr:>5.0f}%  {sign}${row['pnl']:>8,.0f}")
    print(f"{'═'*W}\n")

    # Save results
    csv_path = "dman_pro_backtest.csv"
    df_t.to_csv(csv_path, index=False)
    print(f"  💾 Full trade log saved to {csv_path}\n")

    return {"win_rate": wr, "profit_factor": pf, "total_return": ret,
            "max_drawdown": dd, "sharpe": sharpe, "trades": len(df_t)}


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 21.5 — WATCHLIST RANKING (near-signal leaderboard)
# ═══════════════════════════════════════════════════════════════════════════

def _missing_components(sig: ProSignal, gap: int) -> str:
    """Describe the most important failed gates or score gap for a near-signal."""
    if gap <= 0:
        return "✅ passes"
    failed = []
    if not sig.regime_ok:       failed.append("Regime")
    if not sig.mtf_ok:          failed.append("MTF")
    if not sig.rs_ok:           failed.append("RS")
    if not sig.sector_ok:       failed.append("Sector")
    if not sig.earnings_ok:     failed.append("Earnings!")
    if not sig.divergence_free: failed.append("Divergence!")
    return ", ".join(failed[:3]) if failed else f"score {gap}pts short"


def run_ranking(tickers: list[str] = WATCHLIST,
                min_score: int = None) -> None:
    """
    Score every ticker in the watchlist and print a leaderboard that includes
    near-signals (within 20 pts of threshold) — useful on flat scan days so
    you know which setups to watch for the next session.
    """
    tracker   = WinRateTracker()
    min_score = min_score or tracker.adaptive_min_score()
    near_band = min_score - 20

    regime = get_market_regime()

    print(f"\n{'═'*70}")
    print(f"  D🔥man Watchlist Ranking — {datetime.today().strftime('%A %b %d, %Y')}")
    print(f"  Threshold: {min_score}/100  |  Showing ≥{near_band} pts  |  "
          f"Market: {regime['regime']}")
    print(f"{'═'*70}\n")

    ranked: list[ProSignal] = []
    for i, ticker in enumerate(tickers, 1):
        sys.stdout.write(f"\r  Scoring {i}/{len(tickers)}: {ticker:<8}")
        sys.stdout.flush()

        raw = fetch_df(ticker)
        if raw is None or len(raw) < 60:
            continue
        df = compute_indicators(raw.copy())
        df.dropna(subset=["EMA50", "RSI", "MACD", "ATR"], inplace=True)
        if len(df) < 10:
            continue

        sig = _raw_signals(df, ticker)
        if sig is None:
            continue

        sig = score_signal(sig, df, regime, tracker)
        if sig.confluence_score >= near_band:
            ranked.append(sig)

    ranked.sort(key=lambda s: s.confluence_score, reverse=True)

    print(f"\n\n  {'TICKER':<8} {'BIAS':<6} {'SETUP':<18} {'SCORE':>6}  STATUS")
    print(f"  {'─'*60}")
    for s in ranked:
        gap    = min_score - s.confluence_score
        status = "✅ READY" if gap <= 0 else f"-{gap} pts  ({_missing_components(s, gap)})"
        arrow  = "▲" if s.bias == "LONG" else "▼"
        print(f"  {s.ticker:<8} {arrow} {s.bias:<5} {s.setup:<18} {s.confluence_score:>5}/100  {status}")
    print(f"\n  {len(ranked)} ticker(s) at or near threshold\n")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 21.9 — DAILY WATCHLIST + CLAUDE'S RADAR (Telegram)
# ═══════════════════════════════════════════════════════════════════════════

def _signal_health_label(entry: float, stop: float,
                          t1: float, t2: float, px: float) -> tuple[str, str]:
    """Return (emoji, one-line status) for a live signal given the current price."""
    if px <= stop:
        return "🛑", f"Below stop ${stop:.2f} — consider exiting"
    pct = (px - entry) / entry * 100
    if px >= t2:
        return "🚀🚀", f"T2 HIT! +{pct:.1f}% — take remaining profits"
    if px >= t1:
        return "🚀",   f"T1 hit! +{pct:.1f}% — trail stop to breakeven"
    if pct >= 3.0:
        return "⚡",   f"+{pct:.1f}% — move stop to breakeven"
    if pct >= 0:
        return "✅",   f"+{pct:.1f}% — holding above entry"
    if px > stop * 1.03:
        return "⚠️",  f"{pct:.1f}% — approaching stop"
    return "🟡", f"{pct:.1f}% below entry"


def get_pending_health() -> list[dict]:
    """
    Read dman_live_signals.json, fetch current price for each,
    and return health info for signals ≤ 7 calendar days old.
    """
    try:
        data = json.load(open(LIVE_SIGNALS_FILE))
    except Exception:
        return []
    today = date.today()
    results = []
    for p in data.get("pending", []):
        try:
            ticker   = p["ticker"]
            entry    = float(p["entry"])
            stop     = float(p["stop"])
            t1       = float(p.get("target1", entry * 1.25))
            t2       = float(p.get("target2", entry * 1.50))
            sig_date = date.fromisoformat(p["date"])
            days_old = (today - sig_date).days
            if days_old > 7:
                continue
            px = get_live_price(ticker)
            if not px:
                continue
            emoji, status = _signal_health_label(entry, stop, t1, t2, px)
            results.append({
                "ticker":   ticker,
                "setup":    p.get("setup", "Gap & Hold"),
                "entry":    entry,
                "stop":     stop,
                "t1":       t1,
                "t2":       t2,
                "current":  round(px, 2),
                "score":    p.get("score", 0),
                "days_old": days_old,
                "emoji":    emoji,
                "status":   status,
            })
        except Exception:
            continue
    return results


def get_radar_picks(pending_tickers: set, n: int = 3) -> list[ProSignal]:
    """
    Lightweight watchlist scan for near-miss Gap & Hold setups.
    Uses direct condition checks only — NO score_signal() calls (too slow for 51 tickers).
    Ranks by a simple heuristic: RVOL + gap size + momentum.
    Returns top-n, excluding tickers already in active plays.
    """
    radar: list[tuple[float, ProSignal]] = []   # (heuristic_score, sig)

    for ticker in WATCHLIST:
        if ticker in pending_tickers:
            continue
        try:
            df = fetch_df(ticker)
            if df is None or len(df) < 30:
                continue
            df = _compute_indicators_cached(ticker, df)

            r, p = df.iloc[-1], df.iloc[-2]
            c    = float(r["Close"])
            o    = float(r["Open"])
            pc   = float(p["Close"])

            # Gap & Hold conditions (lightweight — no API calls)
            gap_pct = (o - pc) / pc * 100
            if gap_pct < 1.0:                          # at least 1% gap
                continue
            if c < o * 0.993:                          # must be holding above open
                continue
            rvol = float(r.get("RVOL", 0) or 0)
            if rvol < 1.3:                             # need real volume
                continue
            rsi  = float(r.get("RSI",  50) or 50)
            macd = float(r.get("MACD",  0) or 0)
            if rsi < 45 or macd <= 0:                  # momentum check
                continue

            # Simple heuristic score: RVOL + gap size + RSI position
            h_score = rvol * 12 + gap_pct * 4 + (8 if 50 <= rsi <= 68 else 0)

            # Build a minimal ProSignal for display
            stop = round(min(float(r["Low"]) * 0.99, o * 0.985), 2)
            t1   = round(c + 2.5 * (c - stop), 2)
            t2   = round(c + 4.0 * (c - stop), 2)
            rr   = round((t1 - c) / (c - stop), 2) if c > stop else 0
            if rr < MIN_RR:
                continue

            reason = (f"Gap +{gap_pct:.1f}% holding | RVOL {rvol:.1f}x | "
                      f"RSI {rsi:.0f} | MACD {'▲' if macd > 0 else '▼'}")
            sig = ProSignal(
                ticker=ticker, setup="Gap & Hold", bias="LONG",
                entry=round(c, 2), stop=stop, target1=t1, target2=t2,
                rr=rr, rsi=round(rsi, 1), rvol=round(rvol, 2),
                reason=reason,
            )
            sig.confluence_score = min(84, int(h_score))   # cap below A+ threshold
            radar.append((h_score, sig))
        except Exception:
            continue

    radar.sort(key=lambda x: x[0], reverse=True)
    return [sig for _, sig in radar[:n]]


def format_watchlist_telegram(health: list[dict],
                               radar: list[ProSignal],
                               regime: dict) -> str:
    """Format the daily watchlist + Claude's Radar as a Telegram HTML message."""
    now_str  = datetime.now(ET).strftime("%a %b %d, %Y")
    vix      = regime["details"].get("VIX", "?")
    reg      = regime["regime"]

    lines = [
        f"🗒 <b>DMan Daily Watchlist</b> — {now_str}",
        f"📊 {reg}  |  VIX {vix}  |  "
        f"Min score: {WinRateTracker().adaptive_min_score()}/100",
        "",
    ]

    # ── Active plays ────────────────────────────────────────────────────────
    lines.append("━━━ <b>📌 ACTIVE PLAYS</b> ━━━")
    if health:
        for h in health:
            age_tag = f"  <i>({h['days_old']}d old)</i>" if h['days_old'] > 1 else ""
            pct     = (h['current'] - h['entry']) / h['entry'] * 100
            lines.append(
                f"\n{h['emoji']} <b>{h['ticker']}</b> — {h['setup']}"
                f"  score {h['score']}{age_tag}\n"
                f"   Entry <b>${h['entry']}</b>  →  Now <b>${h['current']}</b>"
                f"  ({pct:+.1f}%)\n"
                f"   T1 ${h['t1']}  |  T2 ${h['t2']}  |  Stop ${h['stop']}\n"
                f"   {h['status']}"
            )
    else:
        lines.append("📭 No active plays at the moment.")

    # ── Claude's Radar ───────────────────────────────────────────────────────
    if radar:
        lines += [
            "",
            "━━━ <b>👁 CLAUDE'S RADAR</b> — watch only, not signals ━━━",
            "<i>These setups are building but haven't hit the A+ bar yet.</i>",
        ]
        for i, sig in enumerate(radar, 1):
            reason_short = sig.reason.split("|")[0].strip() if "|" in sig.reason else sig.reason[:70]
            lines.append(
                f"\n{i}. <b>{sig.ticker}</b>  score {sig.confluence_score}/100"
                f"  |  RVOL {sig.rvol:.1f}x\n"
                f"   Entry ~${sig.entry:.2f}  Stop ${sig.stop:.2f}\n"
                f"   <i>{reason_short}</i>"
            )
    else:
        lines += ["", "👁 No radar picks right now — market is quiet."]

    lines.append("\n📡 <i>DMan PRO Algorithm v3 — auto-generated</i>")
    return "\n".join(lines)


def send_daily_watchlist() -> None:
    """Fetch health + radar, format, print, and send to Telegram. Never raises."""
    try:
        print("\n  [1/3] Reading active signals...")
        health = get_pending_health()

        pending_tickers = {h["ticker"] for h in health}
        print(f"  [2/3] Running radar scan ({len(WATCHLIST)} tickers, lightweight)...")
        regime = get_market_regime()
        radar  = get_radar_picks(pending_tickers)

        print(f"  [3/3] Sending watchlist to Telegram...")
        msg = format_watchlist_telegram(health, radar, regime)
        print(msg)
        send_telegram(msg)
        print("\n  ✅  Daily watchlist sent.\n")
    except Exception as _e:
        print(f"  ⚠️  Watchlist skipped — {_e}")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 22.5 — ALPACA PAPER TRADING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
#  pip install alpaca-py
#  Set env vars: ALPACA_API_KEY  ALPACA_SECRET_KEY
#  Paper keys are free at app.alpaca.markets

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, GetOrdersRequest,
        TakeProfitRequest, StopLossRequest,
    )
    from alpaca.trading.enums import (
        OrderSide, TimeInForce, OrderClass, QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

_alp_trade: Optional["TradingClient"]             = None
_alp_data:  Optional["StockHistoricalDataClient"] = None


def get_alpaca_client() -> Optional["TradingClient"]:
    global _alp_trade
    if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    if _alp_trade is None:
        _alp_trade = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
    return _alp_trade


def get_alpaca_data_client() -> Optional["StockHistoricalDataClient"]:
    global _alp_data
    if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    if _alp_data is None:
        _alp_data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    return _alp_data


def get_live_price(ticker: str) -> Optional[float]:
    """
    Mid-quote from Alpaca (real-time during market hours).
    Falls back to yfinance last close when Alpaca is unavailable.
    """
    try:
        dc = get_alpaca_data_client()
        if dc is not None:
            # Found in the 2026-08-16 review: this is the single hottest
            # price-check path in the file (behind T1/T2/early-lock
            # decisions and entry-price drift validation) and omitted
            # feed= entirely, unlike every other stock-data call site
            # (_fetch_alpaca_daily etc.) — a SIP entitlement lapse would
            # degrade this specific path silently, with no downgrade
            # alert, same class of gap as market_data_stream_loop()'s
            # hardcoded DataFeed.SIP.
            req   = StockLatestQuoteRequest(symbol_or_symbols=ticker, feed=_resolve_stock_feed())
            quote = dc.get_stock_latest_quote(req)[ticker]
            ask, bid = float(quote.ask_price), float(quote.bid_price)
            if ask > 0 and bid > 0:
                return round((ask + bid) / 2, 4)
    except Exception:
        pass
    return get_current_price(ticker)


def validate_entry_price(signal: ProSignal) -> tuple[bool, float]:
    """
    Verify the current price hasn't drifted too far from the computed entry.
    LONG : allow up to 1% dip (shakeout) but reject if it has gapped up >ENTRY_DRIFT_MAX.
    SHORT: inverse.
    Returns (still_valid, current_price).
    """
    cur = get_live_price(signal.ticker)
    if cur is None:
        return True, signal.entry       # can't check — give benefit of doubt
    drift = (cur - signal.entry) / signal.entry
    if signal.bias == "LONG":
        valid = -0.01 <= drift <= ENTRY_DRIFT_MAX
    else:
        valid = -ENTRY_DRIFT_MAX <= drift <= 0.01
    return valid, round(cur, 2)


def _load_sync_state() -> dict:
    try:
        with open(ALPACA_SYNC_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_sync": None, "recorded_ids": []}


def _save_sync_state(state: dict) -> None:
    _write_json_atomic(ALPACA_SYNC_FILE, state, indent=2)


# See merge_positions_snapshots()'s docstring for the full incident this
# guards against (FGL resurrecting every git_sync() cycle for 35+ minutes,
# 2026-08-13). 6 hours is a generous margin over any realistic sync-cycle
# gap — this only needs to survive long enough for the close to actually
# reach a committed origin/main, typically one cycle (SYNC_EVERY_S=300s).
_CLOSED_IDENTITY_TOMBSTONE_S = 6 * 3600


def _recent_closed_identities() -> set:
    """_position_identity() values sync_alpaca_fills() has confirmed closed
    at Alpaca within the last _CLOSED_IDENTITY_TOMBSTONE_S seconds — read
    by sync_positions_with_remote() so merge_positions_snapshots() never
    resurrects a position this process already knows, with certainty
    (a real Alpaca fill), is actually closed."""
    try:
        state = _load_sync_state()
        closed = state.get("closed_identities", {})
        now = time.time()
        return {ident for ident, ts in closed.items()
                if now - float(ts) < _CLOSED_IDENTITY_TOMBSTONE_S}
    except Exception:
        return set()


def _mark_identity_closed(state: dict, ticker: str, setup: str) -> None:
    """Records `ticker`/`setup`'s _position_identity() into `state`'s
    closed_identities tombstone (mutates in place — caller still owns
    _save_sync_state()). Also prunes anything past the TTL so this dict
    doesn't grow unbounded across a long-running daemon session."""
    now = time.time()
    closed = state.setdefault("closed_identities", {})
    closed[_position_identity(ticker, setup)] = now
    for ident, ts in list(closed.items()):
        if now - float(ts) >= _CLOSED_IDENTITY_TOMBSTONE_S:
            del closed[ident]


def _get_pdt_status() -> dict:
    """
    Query Alpaca for current PDT day-trade count.
    Returns dict with keys: used, remaining, swing_mode, equity.
    swing_mode=True when equity < $25k AND remaining <= 1 (save last trade for emergencies).

    Fails CLOSED (remaining=0, swing_mode=True) if the real PDT budget
    can't be verified. Found in the 2026-08-16 review: this used to fail
    OPEN ("remaining": 3, swing_mode=False, "use the normal path") on
    both a missing Alpaca client AND any exception fetching account
    state -- the unsafe direction on a sub-$25k account. A transient API
    hiccup at the exact moment the REAL budget was already exhausted
    would let a same-day round-trip trade through unblocked, risking an
    actual PDT-rule violation flagged by the broker. Failing closed costs
    nothing but an unnecessary overnight hold in the false-positive case
    (assumed zero budget when real budget existed); it never risks a
    real violation.
    """
    try:
        _client = get_alpaca_client()
        if _client is None:
            return {"used": 0, "remaining": 0, "swing_mode": True, "equity": 0.0}
        _acct     = _client.get_account()
        _equity   = float(getattr(_acct, "equity", 0) or 0)
        _dt_count = int(getattr(_acct, "daytrade_count", 0) or 0)
        if _equity >= 25_000:
            return {"used": _dt_count, "remaining": 99, "swing_mode": False, "equity": _equity}
        _remaining = max(0, 3 - _dt_count)
        _swing     = _remaining <= 1   # at 1 or 0 remaining: go swing to protect the budget
        return {"used": _dt_count, "remaining": _remaining, "swing_mode": _swing, "equity": _equity}
    except Exception:
        return {"used": 0, "remaining": 0, "swing_mode": True, "equity": 0.0}


def submit_alpaca_trade(signal: ProSignal) -> tuple[Optional[str], Optional[str]]:
    """
    Place a bracket order on Alpaca (paper or live).
    Normal mode  (signal.swing_mode=False):
      Entry  — GTC limit order, full bracket (entry + stop + T1 take-profit OCO).
      GTC (not DAY) so the stop-loss survives past a single session — see the
      inline comment at the order-submission call for the live incident that
      made this necessary (a DAY bracket silently loses its stop at close if
      the trade hasn't resolved yet, which for Gap & Hold is common, not rare).
    Swing trade mode (signal.swing_mode=True, PDT budget ≤ 1):
      Entry  — GTC limit order, OTO with stop only (no T1 TP — momentum-watch manages exit)
      This ensures the position is held overnight and does NOT consume a day-trade count.

    Stop-loss leg is a plain STOP, not STOP_LIMIT — confirmed live 2026-08-06:
    on this account, a bracket's STOP_LIMIT child leg reliably gets stuck in
    OrderStatus.HELD after the entry fills (the take-profit LIMIT sibling
    activates fine and reserves the shares; the stop-limit leg never gets
    released into a live working order) — reproduced 3/3 times (W, CLRO,
    CELZ), each one left with real money completely unprotected until
    manually caught and repaired. limit_price is optional on Alpaca's own
    StopLossRequest spec; dropping it is a supported construction, not a
    workaround. Trade-off: a plain stop fills at market once triggered (no
    floor on a violent gap), but that's strictly safer than a stop-limit
    that may never activate at all.

    Returns (order_id, None) on success, (None, error_message) on failure. The
    error message is surfaced to Telegram by the caller — a prior version only
    printed it to the GitHub Actions log, which is not somewhere the user
    actually looks; a stretch of silent 401s from a stale Alpaca key went
    undiagnosed because "Order FAILED — check GitHub Actions logs" told the
    user nothing actionable.
    """
    client = get_alpaca_client()
    if client is None:
        _err = "Alpaca unavailable — ALPACA_API_KEY / ALPACA_SECRET_KEY not set"
        print(f"  ⚠️  {_err}")
        return None, _err
    if signal.shares < 1:
        _err = f"{signal.ticker}: 0 shares computed — skipping submission"
        print(f"  ⚠️  {_err}")
        return None, _err
    if signal.bias == "LONG":
        _cash_ok, _cash_msg = _cash_available_for(signal.cost)
        if not _cash_ok:
            _err = f"{signal.ticker}: {_cash_msg}"
            print(f"  ⚠️  {_err}")
            return None, _err

    side      = OrderSide.BUY  if signal.bias == "LONG" else OrderSide.SELL
    limit_px  = round(signal.entry,   2)
    stop_px   = round(signal.stop,    2)
    target_px = round(signal.target1, 2)
    label     = "PAPER" if ALPACA_PAPER else "LIVE"

    try:
        if signal.swing_mode:
            # PDT budget ≤ 1 — GTC entry + stop only (no T1 TP).
            # Position held overnight = not a day trade. Momentum-watch manages the exit.
            order = client.submit_order(LimitOrderRequest(
                symbol        = signal.ticker,
                qty           = signal.shares,
                side          = side,
                limit_price   = limit_px,
                time_in_force = TimeInForce.GTC,
                order_class   = OrderClass.OTO,
                stop_loss     = StopLossRequest(stop_price=stop_px),
            ))
            oid = str(order.id)
            print(f"  📤 [{label}] 🔄 SWING {signal.ticker} {signal.bias} {signal.shares}sh  "
                  f"GTC limit=${limit_px}  stop=${stop_px}  (no T1 — held overnight)  id={oid[:8]}…")
        else:
            # Full bracket with T1 take-profit. GTC (not DAY) — confirmed live
            # 2026-08-03: FERG and AMZN both entered, neither hit stop nor T1
            # same day, and at close the ENTIRE bracket (including the
            # stop-loss) expired since it was a DAY order — both positions
            # sat with zero broker-side protection overnight, silently, with
            # nothing re-establishing a stop (momentum-watch only sends
            # Telegram alerts asking the user to manually manage the exit,
            # it doesn't place orders). "Gap & Hold" holds for multiple days
            # as a matter of course (real trade history shows 2-4 day
            # holds), so assuming same-day resolution here was the actual
            # bug. If the trade resolves same-day, behavior is unchanged —
            # a fill is a fill regardless of tif. If it doesn't, the stop
            # and target now persist until hit instead of evaporating at
            # 4pm. swing_mode's separate OTO path above is untouched.
            order = client.submit_order(LimitOrderRequest(
                symbol        = signal.ticker,
                qty           = signal.shares,
                side          = side,
                limit_price   = limit_px,
                time_in_force = TimeInForce.GTC,
                order_class   = OrderClass.BRACKET,
                take_profit   = TakeProfitRequest(limit_price=target_px),
                stop_loss     = StopLossRequest(stop_price=stop_px),
            ))
            oid = str(order.id)
            print(f"  📤 [{label}] {signal.ticker} {signal.bias} {signal.shares}sh  "
                  f"limit=${limit_px}  stop=${stop_px}  T1=${target_px}  id={oid[:8]}…")
        return oid, None
    except Exception as exc:
        print(f"  ❌ Alpaca order failed ({signal.ticker}): {exc}")
        return None, str(exc)


def sync_alpaca_fills(tracker: WinRateTracker) -> int:
    """
    Detect positions that closed in Alpaca since the last sync.

    Strategy:
      1. Compare our PositionTracker against Alpaca's current open positions.
      2. Any ticker we track that is NO LONGER in Alpaca = it was closed.
      3. Find the most-recent filled closing order for that ticker.
      4. Auto-record the trade, update the daily P&L file, and fire a Telegram alert.

    Returns the count of newly recorded trades.
    """
    client = get_alpaca_client()
    if client is None:
        return 0

    pt    = PositionTracker()
    state = _load_sync_state()
    recorded_ids: set[str] = set(state.get("recorded_ids", []))
    new_count = 0

    # — current Alpaca open positions ————————————————————————————————————
    try:
        alp_open = {p.symbol for p in client.get_all_positions()}
    except Exception as exc:
        print(f"  ⚠️  Alpaca sync: could not fetch positions — {exc}")
        return 0

    for pos in list(pt.positions):
        ticker = pos.ticker
        # Earnings spreads have 2-4 real option legs, not one symbol — the rest
        # of this loop (single-symbol "still open?" check, single-order exit
        # lookup) is built around exactly one Alpaca symbol per position and
        # isn't a safe fit for a multi-leg position. Their full lifecycle
        # (fill confirmation, DTE-based close, outcome recording) is already
        # handled explicitly by _monitor_earnings_spread_position() /
        # _close_earnings_spread() — skip them here rather than force a
        # partial, likely-wrong adaptation of this single-symbol logic.
        if pos.setup.startswith("Earnings "):
            continue
        # Options positions: Alpaca reports the OCC symbol (e.g. "SMCI260724C00027500"),
        # not the underlying. _position_identity() already extracts OCC
        # from the setup string for exactly this purpose -- found in the
        # 2026-08-16 review: this used to reimplement the same parsing
        # inline (and, unlike _position_identity(), fell back to plain
        # `ticker` instead of ticker.upper() for the non-options case).
        _is_options = pos.setup.startswith("Options Call ") or pos.setup.startswith("Options Put ")
        _alp_sym = _position_identity(ticker, pos.setup)
        _occ_sym = _alp_sym if _is_options else None
        if _alp_sym in alp_open:
            continue    # still open — nothing to record yet

        # Position is gone from Alpaca — find the exit fill
        try:
            orders = client.get_orders(filter=GetOrdersRequest(
                symbols=[_alp_sym],
                status=QueryOrderStatus.CLOSED,
                limit=20,
            ))
        except Exception:
            continue

        is_lo = pos.bias == "LONG"
        # Options always buy-to-open / sell-to-close regardless of put/call
        closing_side = OrderSide.SELL if (is_lo or _occ_sym) else OrderSide.BUY

        from alpaca.trading.enums import OrderStatus as _OrderStatus
        for order in orders:
            oid = str(order.id)
            if oid in recorded_ids:
                continue
            # Confirmed live 2026-08-07: str(order.status) on a real filled
            # order is "OrderStatus.FILLED", never the bare "filled" this
            # used to compare against — meaning this check was true for
            # EVERY order regardless of actual status, and this whole
            # function has never once auto-recorded a real trade close.
            # IOTR, ARTL, AMZN, FERG, W all needed manual --mode record
            # intervention because of this single line.
            if order.status != _OrderStatus.FILLED:
                continue
            if order.side != closing_side:
                # Entry fill or unrelated order — mark seen and skip
                recorded_ids.add(oid)
                continue

            fill_px = float(order.filled_avg_price or 0)
            if fill_px <= 0:
                continue

            qty = int(float(getattr(order, "filled_qty", None) or order.qty or pos.shares))

            # Options: premium P&L = always (exit - entry) * qty regardless of put/call.
            # Both calls and puts are bought-to-open; exit price > entry = WIN.
            # qty here is order.filled_qty, which Alpaca reports in CONTRACTS
            # for an options order (see the qty=contracts submission at
            # _submit_options_call, ~line 14981) -- one contract controls 100
            # shares of premium, so the dollar P&L needs the *100 real-money
            # multiplier. Missing it (found 2026-08-16 review) meant a real
            # options loss/gain was recorded as 1/100th of its actual size in
            # dman_daily_pnl.json, making DAILY_LOSS_LIMIT effectively blind
            # to options P&L -- a real -20% options loss on this account was
            # recorded as roughly -0.04% for circuit-breaker purposes.
            if _occ_sym:
                pnl_pct    = (fill_px - pos.entry) / pos.entry * 100 if pos.entry > 0 else 0
                dollar_pnl = (fill_px - pos.entry) * qty * 100
            else:
                pnl_pct    = ((fill_px - pos.entry) / pos.entry * 100 if is_lo
                               else (pos.entry - fill_px) / pos.entry * 100)
                dollar_pnl = (fill_px - pos.entry) * qty * (1 if is_lo else -1)
            outcome    = ("WIN" if pnl_pct > 0.1 else
                          "LOSS" if pnl_pct < -0.1 else "BE")
            # get_effective_account() (live equity), not the static
            # ACCOUNT_SIZE secret -- the account has grown/shrunk from
            # whatever ACCOUNT_SIZE was last set to, so dividing by the
            # stale constant understates or overstates every recorded %
            # relative to what DAILY_LOSS_LIMIT/MONTHLY_LOSS_LIMIT actually
            # compare it against on this same live account.
            acct_pct   = dollar_pnl / get_effective_account() * 100

            fill_date = (order.filled_at.strftime("%Y-%m-%d")
                         if getattr(order, "filled_at", None) else
                         datetime.today().strftime("%Y-%m-%d"))

            # Independent safety net against duplicate recording, on top of
            # (not instead of) the recorded_ids check above. Confirmed live
            # 2026-08-11: _restore_corrupted_json()'s `git checkout -- file`
            # fallback (dman_daemon.py, on a stash-pop conflict) can revert
            # dman_alpaca_sync.json to whatever was in the LAST COMMIT at
            # that exact moment — if that commit predates when an order's
            # id was safely pushed, the restore silently un-records it,
            # letting an already-closed position (CLRO) get re-detected and
            # re-recorded 5 times in one day, each one adding its full P&L
            # again to both the win-rate history and dman_daily_pnl.json —
            # the latter accumulated to a phantom -24.76% "today" that
            # would have tripped DAILY_LOSS_LIMIT (3%) and halted all new
            # entries had the corruption finished accumulating during
            # market hours instead of after close. recorded_ids is a
            # cache, not a ledger — this checks the ledger itself
            # (tracker.records, i.e. dman_win_rate.json) directly, so a
            # lost cache entry can no longer cause a duplicate no matter
            # what upstream git race caused it.
            # Found in the 2026-08-16 review: this match had no date
            # component -- ticker + setup + exit-price-within-half-a-cent
            # alone can coincidentally match two genuinely SEPARATE trades
            # weeks apart that both happened to close near the same
            # round-number level (a common occurrence, not a rare edge
            # case). The real re-detection this guards against (see the
            # CLRO incident above) is always same-day or adjacent-day, so
            # requiring the matched record's date to be within a tight
            # window keeps the actual protection while no longer silently
            # dropping an unrelated real trade's P&L from win-rate history.
            _DUPE_DATE_WINDOW_DAYS = 2
            def _dates_close(r_date: str) -> bool:
                try:
                    return abs((date.fromisoformat(r_date) - date.fromisoformat(fill_date)).days) \
                           <= _DUPE_DATE_WINDOW_DAYS
                except (TypeError, ValueError):
                    return False
            _dupe = any(r.ticker == ticker and r.setup == pos.setup
                        and abs(r.exit - round(fill_px, 2)) < 0.005
                        and _dates_close(r.date)
                        for r in tracker.records)
            if _dupe:
                pt.close(ticker, occ_symbol=_occ_sym)
                _mark_identity_closed(state, ticker, pos.setup)
                recorded_ids.add(oid)
                print(f"  ⏭️  Skipped duplicate close record: {ticker} "
                      f"${pos.entry}→${fill_px:.2f} already in win-rate history")
                break

            tracker.record(TradeRecord(
                ticker  = ticker,
                date    = fill_date,
                bias    = pos.bias,
                setup   = pos.setup,
                entry   = round(pos.entry, 2),
                exit    = round(fill_px, 2),
                outcome = outcome,
                pnl_pct = round(pnl_pct, 2),
                score   = getattr(pos, "score", 0),
                is_live = True,   # a real Alpaca fill, not a simulation
            ))
            record_daily_pnl(acct_pct)
            record_monthly_pnl(acct_pct)
            pt.close(ticker, occ_symbol=_occ_sym)
            _mark_identity_closed(state, ticker, pos.setup)

            sign = "+" if dollar_pnl >= 0 else ""
            print(f"  📋 Synced: {ticker} {pos.bias}  "
                  f"${pos.entry}→${fill_px:.2f}  {outcome}  "
                  f"{sign}{pnl_pct:.1f}%  ({sign}${dollar_pnl:,.0f})")
            send_telegram(
                f"📋 <b>Trade Closed</b> — {ticker} {pos.bias}\n"
                f"${pos.entry} → ${fill_px:.2f}  |  {outcome}  {'+' if pnl_pct>=0 else ''}{pnl_pct:.1f}%\n"
                f"P&L: {sign}${dollar_pnl:,.0f}"
            )
            recorded_ids.add(oid)
            new_count += 1
            break   # one exit record per position
        else:
            # Inner loop found no valid closing fill to record. Two distinct
            # cases land here:
            #  1. Entry order was cancelled/expired (never filled) — a ghost.
            #  2. The closing order was already recorded in a PRIOR cycle (its
            #     id is already in recorded_ids), but the tracked entry still
            #     exists anyway — e.g. resurrected by merge_positions_snapshots()'s
            #     union-of-tickers rule pulling in a stale snapshot that
            #     predates the original close. Confirmed live 2026-08-11: CLRO
            #     sat as a phantom "open" position for 5 days this way — its
            #     closing stop order (filled 2026-08-06) was already in
            #     recorded_ids, so every cycle kept silently `continue`-ing
            #     past it, because removal used to live ONLY inside the
            #     "found a NEW closing fill" branch above, never running for
            #     an already-recorded one.
            # _alp_sym is confirmed absent from alp_open (checked above) in
            # both cases — always clear the stale tracker entry here, and
            # only alert/re-record for the genuine "never filled" ghost case
            # so an already-reported close doesn't re-notify.
            # Options always buy-to-open; equity: BUY for LONG, SELL for SHORT.
            _entry_side = OrderSide.BUY if (is_lo or _occ_sym) else OrderSide.SELL
            # Same class of bug as the FILLED check above, plus a spelling
            # mismatch on top: Alpaca's real enum value is "canceled" (one
            # L), this compared against "cancelled" (two L) — doubly never
            # matched regardless of the str()-vs-enum issue.
            _GHOST_STATUSES = (_OrderStatus.CANCELED, _OrderStatus.EXPIRED, _OrderStatus.REPLACED)
            _ghost_order = next((_eo for _eo in orders
                                  if _eo.side == _entry_side and _eo.status in _GHOST_STATUSES), None)
            pt.close(ticker, occ_symbol=_occ_sym)
            _mark_identity_closed(state, ticker, pos.setup)
            _lbl = _occ_sym if _occ_sym else ticker
            if _ghost_order is not None:
                print(f"  🗑️  Ghost cleared: {_lbl} — entry limit {_ghost_order.status}, never filled")
                send_telegram(
                    f"🗑️ <b>Ghost cleared</b>: {_lbl} — entry limit order "
                    f"{str(_ghost_order.status)} (never filled). Removed from position tracker."
                )
            else:
                print(f"  🧹 Stale entry cleared: {_lbl} — not open at Alpaca, "
                      f"already-recorded close (likely resurrected by a state merge)")

    state["last_sync"]    = datetime.utcnow().isoformat()
    state["recorded_ids"] = list(recorded_ids)[-500:]
    _save_sync_state(state)
    return new_count


def show_alpaca_account() -> None:
    """Print Alpaca account summary + current open positions + pending orders."""
    client = get_alpaca_client()
    if client is None:
        print("  ⚠️  Alpaca not configured — set ALPACA_API_KEY + ALPACA_SECRET_KEY.")
        return
    try:
        acct      = client.get_account()
        positions = client.get_all_positions()
        orders    = client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=20))

        W = 70
        mode_label = "PAPER" if ALPACA_PAPER else "LIVE ⚠️"
        print(f"\n{'═'*W}")
        print(f"  Alpaca [{mode_label}] Account  —  "
              f"{datetime.today().strftime('%A %b %d, %Y')}")
        print(f"{'─'*W}")
        print(f"  Portfolio Value : ${float(acct.portfolio_value):>12,.2f}")
        print(f"  Cash            : ${float(acct.cash):>12,.2f}")
        print(f"  Buying Power    : ${float(acct.buying_power):>12,.2f}")
        unreal = float(getattr(acct, "unrealized_pl", 0) or 0)
        sign   = "+" if unreal >= 0 else ""
        print(f"  Unrealized P&L  : {sign}${unreal:>11,.2f}")
        print(f"{'─'*W}")

        if positions:
            print(f"  {'TICKER':<8} {'SIDE':<6} {'QTY':>5} "
                  f"{'ENTRY':>10} {'NOW':>10} {'UNREAL P&L':>13}")
            print(f"  {'─'*58}")
            for p in positions:
                side_str = "LONG" if float(p.qty) > 0 else "SHORT"
                cur_px   = float(p.current_price   or 0)
                avg_px   = float(p.avg_entry_price or 0)
                unr      = float(p.unrealized_pl   or 0)
                s        = "+" if unr >= 0 else ""
                print(f"  {p.symbol:<8} {side_str:<6} "
                      f"{abs(int(float(p.qty))):>5} "
                      f"${avg_px:>9.2f} ${cur_px:>9.2f} "
                      f"{s}${unr:>11,.2f}")
        else:
            print("  No open positions.")

        if orders:
            print(f"\n  Pending Orders ({len(orders)}):")
            for o in orders:
                lmt = f"lmt=${o.limit_price}" if o.limit_price else ""
                stp = f"stp=${o.stop_price}"  if o.stop_price  else ""
                print(f"    {o.symbol:<8} {str(o.side):<5} qty={o.qty:<5} "
                      f"{str(o.type):<12} {lmt} {stp}")
        # ── Options access check ──────────────────────────────────────────
        print(f"{'─'*W}")
        try:
            _acct_raw = acct.__dict__ if hasattr(acct, "__dict__") else {}
            # Alpaca returns options_approved_level / options_trading_level on the account object
            _opt_approved = getattr(acct, "options_approved_level", None)
            _opt_trading  = getattr(acct, "options_trading_level",  None)
            # Also try raw dict keys in case SDK doesn't expose them as attributes
            if _opt_approved is None:
                _opt_approved = _acct_raw.get("options_approved_level")
            if _opt_trading is None:
                _opt_trading  = _acct_raw.get("options_trading_level")

            _level_desc = {
                0: "None — not approved for options trading",
                1: "Level 1 — covered calls only",
                2: "Level 2 — long calls + puts (algo uses this)",
                3: "Level 3 — spreads approved",
            }
            _approved_int = int(_opt_approved) if _opt_approved is not None else -1
            _trading_int  = int(_opt_trading)  if _opt_trading  is not None else -1

            if _approved_int < 0:
                print("  Options Level  : ⚠️  unable to read — check app.alpaca.markets")
                print("                   Account → Trading → Options → Options Level")
            elif _approved_int == 0:
                print("  Options Level  : ❌ NOT APPROVED")
                print("                   Apply at: app.alpaca.markets → Account → Trading → Options")
                print("                   Algo needs Level 2 (long calls/puts) to buy calls")
            elif _approved_int == 1:
                print(f"  Options Level  : ⚠️  Level 1 (covered calls only)")
                print("                   Upgrade to Level 2 to buy long calls with the algo")
            else:
                _enabled = "✅ ENABLED" if ENABLE_OPTIONS_TRADING else "⏸  set ENABLE_OPTIONS_TRADING=True to activate"
                print(f"  Options Level  : ✅ Level {_approved_int} — {_level_desc.get(_approved_int, 'approved')}")
                print(f"  Algo Options   : {_enabled}")
        except Exception as _oe:
            print(f"  Options Level  : ⚠️  check failed ({_oe})")
            print("                   Verify at app.alpaca.markets → Account → Trading")
        print(f"{'═'*W}\n")
    except Exception as exc:
        print(f"  ❌ Alpaca account fetch failed: {exc}")


_DAY_START_EQUITY_FILE = "dman_day_start_equity.json"

def _get_day_start_equity(current_equity: float) -> float:
    """
    Alpaca's own last_equity field is the prior TRADING DAY's closing
    equity — it does not account for a deposit/withdrawal that lands
    overnight, so a naive (equity - last_equity) treats new capital as
    trading profit. Confirmed live 2026-08-06: a same-day $2,000 deposit
    inflated the EOD P&L alert to "+66.26%" for what was actually a real
    loss day. This tracks our own baseline instead — the first equity
    reading seen today, set once per calendar day — so capital already in
    the account by the time this first runs is correctly excluded from
    "today's" P&L rather than misread as a gain.
    """
    today_str = date.today().isoformat()
    try:
        with open(_DAY_START_EQUITY_FILE) as f:
            data = json.load(f)
        if data.get("date") == today_str:
            return float(data["equity"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        pass
    _write_json_atomic(_DAY_START_EQUITY_FILE, {"date": today_str, "equity": current_equity})
    return current_equity


def send_account_pnl_telegram(label: str = "EOD") -> None:
    """
    Fetch live Alpaca account snapshot and send a P&L summary to Telegram.
    Called automatically at 4 PM EOD and available via --mode pnl for on-demand use.
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("  Alpaca credentials not configured — skipping P&L summary")
        return
    try:
        from alpaca.trading.client import TradingClient
        tc    = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        acct  = tc.get_account()
        equity     = float(acct.equity)
        cash       = float(acct.cash)
        last_eq    = _get_day_start_equity(equity)
        day_pl     = equity - last_eq
        day_pl_pct = day_pl / last_eq * 100 if last_eq > 0 else 0.0
        bp         = float(getattr(acct, "buying_power", cash))

        arrow = "🟢" if day_pl >= 0 else "🔴"
        lines = [
            f"{arrow} <b>DMan {label} Account Summary</b>",
            f"Equity  : <b>${equity:,.2f}</b>",
            f"Day P&L : <b>${day_pl:+.2f}  ({day_pl_pct:+.2f}%)</b>",
            f"Cash    : ${cash:,.2f}   BP: ${bp:,.2f}",
        ]

        positions = tc.get_all_positions()
        if positions:
            lines.append(f"\n<b>Open Positions ({len(positions)})</b>")
            for p in positions:
                sym    = p.symbol
                qty    = p.qty
                avg_px = float(p.avg_entry_price)
                pl     = float(p.unrealized_pl)
                pl_pct = float(p.unrealized_plpc) * 100
                p_arrow = "🟢" if pl >= 0 else "🔴"
                lines.append(f"  {p_arrow} <b>{sym}</b>  {qty}sh @ ${avg_px:.2f}  "
                              f"P&L ${pl:+.2f} ({pl_pct:+.1f}%)")
        else:
            lines.append("\nNo open positions.")

        send_telegram("\n".join(lines))
        print(f"  📊 P&L summary sent — equity ${equity:,.2f}  day {day_pl_pct:+.2f}%")
    except Exception as e:
        print(f"  ⚠️  P&L summary failed: {e}")

    # Per-setup live win-rate drift — piggybacks on this function's existing
    # once/day EOD cron cadence rather than adding a new schedule. Separate
    # Telegram message (not appended above) so it's easy to scan/search on
    # its own, and deduped like every other alert so a manual `--mode pnl`
    # run doesn't repeat it within the same cooldown window.
    try:
        _drift = WinRateTracker().setup_performance_drift()
        if _drift and not _is_duplicate_alert("__SETUP_PERF_DRIFT__"):
            _lines = ["📉 <b>Setup Performance Drift</b>",
                      "Live win rate has dropped below floor for:\n"]
            for _d in _drift:
                _lines.append(
                    f"  <b>{_d['setup']}</b>: {_d['win_rate']*100:.0f}% WR over last "
                    f"{_d['total']} live trade(s) ({_d['wins']}W/{_d['losses']}L, "
                    f"avg loss {_d['avg_loss_pct']:.1f}%)"
                )
            send_telegram("\n".join(_lines))
            _save_last_alert("__SETUP_PERF_DRIFT__")
            print(f"  📉 Setup drift alert sent for {len(_drift)} setup(s)")
    except Exception as e:
        print(f"  ⚠️  Setup drift check failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  Options engine — Greeks-aware contract selection + L2 context + submission
# ═══════════════════════════════════════════════════════════════════════════

_RATE_LIMIT_EVENTS_FILE = "dman_rate_limit_events.json"


def _record_alpaca_429(source: str) -> None:
    """
    Confirmed live 2026-07-30: a runaway websocket reconnect storm elsewhere
    (dman_daemon.py:stream_loop) rate-limited Alpaca's options data for
    hours, and every downstream caller (_get_option_snapshot -> earnings
    spreads, ITM calls) silently treated the resulting 429s as "no data" —
    indistinguishable from a genuinely quiet options chain. run_watchdog()'s
    existing checks (sync/scan freshness) had no way to catch this, since
    the daemon otherwise kept running and syncing normally. This gives
    watchdog a direct signal instead: today's 429 count, so a repeat is
    visible within 30 min instead of discovered after the fact.
    """
    try:
        try:
            with open(_RATE_LIMIT_EVENTS_FILE) as _f:
                _d = json.load(_f)
        except Exception:
            _d = {}
        _today = date.today().isoformat()
        _day = _d.get(_today, {})
        _day[source] = _day.get(source, 0) + 1
        _d = {_today: _day}   # only keep today — this is a same-day signal, not history
        with open(_RATE_LIMIT_EVENTS_FILE, "w") as _f:
            json.dump(_d, _f)
    except Exception:
        pass


_options_feed_state: dict = {"feed": None, "checked_at": 0.0}
_OPTIONS_FEED_RECHECK_S = 3600   # re-probe hourly — entitlement can also come back
_OPTIONS_FEED_STATE_FILE = "dman_options_feed_state.json"

def _load_options_feed_state() -> None:
    """See _load_feed_state()'s docstring for the full incident this fixes
    (confirmed live 2026-08-08 on this exact state dict)."""
    _load_feed_state(_options_feed_state, _OPTIONS_FEED_STATE_FILE)


def _save_options_feed_state() -> None:
    _save_feed_state(_options_feed_state, _OPTIONS_FEED_STATE_FILE)


def _resolve_options_feed() -> str:
    """
    OPRA entitlement on this account has flipped on/off before with zero code
    change on our end (403'd 2026-07-29, fixed by falling back to
    "indicative", confirmed OPRA working again 2026-07-30, then confirmed
    403 AGAIN 2026-08-06 — a full week where every options signal silently
    failed and fell through to skip/equity, only caught by manually testing
    the raw endpoint). Hardcoding OPTIONS_DATA_FEED trusts that entitlement
    never changes, which has already been false twice. This probes once
    (cached for _OPTIONS_FEED_RECHECK_S, persisted across processes — see
    _load_options_feed_state) and actually alerts on a fallback instead of
    failing silently — the whole point is never losing a week to this again
    without anyone knowing, and not spamming a repeat alert every run either.
    """
    _load_options_feed_state()
    now = time.time()
    if _options_feed_state["feed"] is not None and \
       (now - _options_feed_state["checked_at"]) < _OPTIONS_FEED_RECHECK_S:
        return _options_feed_state["feed"]

    _options_feed_state["checked_at"] = now
    # Default to whatever was cached before THIS probe, not the preferred
    # feed — same fix as _resolve_stock_feed(). Found 2026-08-16 review:
    # this used to always start as OPTIONS_DATA_FEED, and neither the
    # early "couldn't get a probe symbol" return nor the except-branch
    # touched it, so any transient error (or a probe-symbol miss) silently
    # reverted a real, still-active downgrade (cached "indicative") back
    # to the preferred feed for up to _OPTIONS_FEED_RECHECK_S (1hr) —
    # every options snapshot call in that window then hit real 403s and
    # returned None, reproducing the exact week-long silent-failure
    # incident this mechanism exists to prevent, just bounded to an hour
    # instead of indefinite.
    resolved = _options_feed_state["feed"] or OPTIONS_DATA_FEED
    try:
        # The snapshot endpoint needs a real OCC-format contract symbol —
        # a bare ticker 400s (invalid format) rather than 403ing, which
        # would silently defeat this whole probe (confirmed live
        # 2026-08-06: testing with "AAPL" always returned 400, never
        # revealing the real 403 entitlement error). AAPL always has a
        # liquid, currently-valid chain, and the contracts endpoint itself
        # isn't gated by the OPRA subscription — only the snapshot is.
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType
        client = get_alpaca_client()
        probe_symbol = None
        if client is not None:
            contracts = client.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=["AAPL"], type=ContractType.CALL, limit=1,
            ))
            items = getattr(contracts, "option_contracts", None) or []
            if items:
                probe_symbol = items[0].symbol
        if probe_symbol is None:
            _save_options_feed_state()   # persist the checked_at bump even on an incomplete probe
            return resolved   # couldn't get a probe symbol this cycle — try again next time

        r = requests.get(
            "https://data.alpaca.markets/v1beta1/options/snapshots",
            headers={"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY},
            params={"symbols": probe_symbol, "feed": OPTIONS_DATA_FEED},
            timeout=8,
        )
        if r.status_code == 403:
            resolved = "indicative"
            if _options_feed_state["feed"] != resolved:   # only alert on a real state change
                print(f"  ⚠️  {OPTIONS_DATA_FEED} options feed not entitled — falling back to indicative")
                send_telegram(
                    f"⚠️ <b>Options data feed downgraded</b>\n"
                    f"{OPTIONS_DATA_FEED} returned 403 (not entitled) — falling back to indicative feed. "
                    f"Options trading still works, just on delayed/indicative quotes. "
                    f"Check the Algo Trader Plus subscription if this is unexpected."
                )
        elif r.status_code == 200:
            resolved = OPTIONS_DATA_FEED
            if _options_feed_state["feed"] == "indicative":
                # Was on fallback, preferred feed just came back — worth knowing.
                send_telegram(f"✅ <b>{OPTIONS_DATA_FEED} options feed entitlement restored</b> — back to real-time quotes.")
    except Exception:
        pass   # network hiccup — keep whatever we resolved last, don't flap on a transient error

    _options_feed_state["feed"] = resolved
    _save_options_feed_state()
    return resolved


def _get_option_snapshot(occ_symbol: str) -> dict | None:
    """
    Full options snapshot from Alpaca Data API including Greeks, IV, bid-ask depth.
    Returns comprehensive dict or None on failure.

    Greeks decoded:
      delta  — P&L per $1 stock move (0-1 for calls; target 0.40-0.60)
      gamma  — rate delta changes per $1 stock move (high near expiry/ATM)
      theta  — daily time decay in $ per share (always negative for longs)
      vega   — P&L per 1% change in implied vol (higher = more IV-sensitive)
      iv     — implied volatility (annualized %; compare to hist vol for edge)
    """
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    _headers = {
        "APCA-API-KEY-ID":     ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    try:
        r = requests.get(
            "https://data.alpaca.markets/v1beta1/options/snapshots",
            headers=_headers,
            params={"symbols": occ_symbol, "feed": _resolve_options_feed()},
            timeout=8,
        )
        if r.status_code == 429:
            _record_alpaca_429("options_snapshot")
            return None
        if r.status_code != 200:
            return None
        raw = r.json().get("snapshots", {}).get(occ_symbol, {})
        if not raw:
            return None

        q    = raw.get("latestQuote", {})
        bid  = float(q.get("bp", 0) or 0)
        ask  = float(q.get("ap", 0) or 0)
        bsz  = int(q.get("bs", 0) or 0)     # contracts available at best bid
        asz  = int(q.get("as", 0) or 0)     # contracts available at best ask
        if ask <= 0:
            return None

        mid        = round((bid + ask) / 2, 2)
        spread_pct = (ask - bid) / mid if mid > 0 else 1.0

        g     = raw.get("greeks", {}) or {}
        delta = float(g.get("delta", 0) or 0)
        gamma = float(g.get("gamma", 0) or 0)
        theta = float(g.get("theta", 0) or 0)   # $/share/day (negative)
        vega  = float(g.get("vega",  0) or 0)   # $/share per 1% IV move

        iv = float(raw.get("impliedVolatility", 0) or 0)
        oi = int(raw.get("openInterest", 0) or 0)

        return {
            "bid": round(bid, 2), "ask": round(ask, 2), "mid": mid,
            "bid_size": bsz, "ask_size": asz,
            "spread_pct": round(spread_pct, 3),
            "delta": round(delta, 3),
            "gamma": round(gamma, 4),
            "theta": round(theta, 4),   # $/share/day — negative (decay enemy)
            "vega":  round(vega,  4),
            "iv":    round(iv, 3),      # e.g. 0.45 = 45% IV
            "oi":    oi,
        }
    except Exception:
        return None


def _merge_contract_oi(snap: dict, contract) -> dict:
    """
    _get_option_snapshot()'s "oi" is always 0 — the options snapshot endpoint
    doesn't carry openInterest at all, on any feed (confirmed live 2026-07-30;
    this was previously misattributed to OPTIONS_DATA_FEED="indicative", but
    "opra" doesn't return it either). Real OI is already present on the
    contract object returned by client.get_option_contracts() at every call
    site (confirmed live: e.g. AAPL 340C showed open_interest=4344) — this
    merges it in so _score_option_contract() and spread-leg selection see
    real numbers instead of always-0.
    """
    try:
        raw_oi = getattr(contract, "open_interest", None)
        if raw_oi is not None:
            snap["oi"] = int(float(raw_oi))
    except Exception:
        pass
    return snap


def _score_option_contract(snap: dict, current_price: float) -> tuple[int, str]:
    """
    Score a contract 0-100 optimized for DMan ITM strategy (delta 0.60-0.80).
      Delta  (40 pts): 0.60-0.80 ideal — ITM calls move with the stock
      Spread (35 pts): tight spread = cheap entry/exit
      OI     (25 pts): open interest confirms chain liquidity
    """
    score  = 0
    parts: list[str] = []

    # Delta: ITM sweet spot 0.60-0.80 (target ≈ delta 0.70 per OPTIONS_ITM_TARGET_PCT)
    delta = abs(snap.get("delta", 0))
    if 0.60 <= delta <= 0.80:
        score += 40; parts.append(f"delta {delta:.2f} ✅ ITM")
    elif 0.80 < delta <= 0.95:
        score += 30; parts.append(f"delta {delta:.2f} deep ITM")
    elif 0.45 <= delta < 0.60:
        score += 20; parts.append(f"delta {delta:.2f} ATM")
    elif delta >= 0.40:
        score += 10; parts.append(f"delta {delta:.2f} OTM")
    else:
        parts.append(f"delta {delta:.2f} ❌ too OTM")

    # Spread (as % of mid — more accurate than % of ask)
    sp = snap.get("spread_pct", 1.0)
    if sp <= 0.08:
        score += 35; parts.append(f"spread {sp*100:.0f}% tight ✅")
    elif sp <= 0.15:
        score += 22; parts.append(f"spread {sp*100:.0f}% ok")
    elif sp <= 0.25:
        score += 10; parts.append(f"spread {sp*100:.0f}% wide")
    else:
        parts.append(f"spread {sp*100:.0f}% ❌ skip")

    # Open interest
    oi = snap.get("oi", 0)
    if oi >= 500:
        score += 25; parts.append(f"OI {oi:,} ✅")
    elif oi >= 200:
        score += 18; parts.append(f"OI {oi:,} good")
    elif oi >= 50:
        score += 10; parts.append(f"OI {oi:,} thin")
    else:
        parts.append(f"OI {oi} ❌ illiquid")

    return score, "  ".join(parts)


def _get_options_market_context(ticker: str, expiry_str: str) -> dict:
    """
    Level 2 market context for options:
      - Put/call ratio at this expiry (from yfinance option chain)
      - Stock bid-ask spread (underlying liquidity affects option fills)
      - Dominant OI strike (the market's "gravity" — price tends to pin here)

    Put/call < 0.7 = call-heavy flow, bullish sentiment for calls
    Put/call > 1.3 = put-heavy flow, hedging demand = caution
    """
    result = {"pc_ratio": 1.0, "dominant_call_strike": 0.0,
              "stock_spread_pct": 0.0, "flow_label": "neutral"}
    try:
        tk = yf.Ticker(ticker)
        # Stock bid-ask spread
        fi = tk.fast_info
        try:
            _ask_s = float(getattr(fi, "ask", 0) or 0)
            _bid_s = float(getattr(fi, "bid", 0) or 0)
            if _ask_s > 0:
                result["stock_spread_pct"] = round((_ask_s - _bid_s) / _ask_s * 100, 2)
        except Exception:
            pass

        # Option chain at target expiry
        try:
            chain = tk.option_chain(expiry_str)
            calls = chain.calls
            puts  = chain.puts
            total_call_oi = calls["openInterest"].sum()
            total_put_oi  = puts["openInterest"].sum()
            if total_call_oi > 0:
                pc = round(total_put_oi / total_call_oi, 2)
                result["pc_ratio"] = pc
                if pc < 0.70:
                    result["flow_label"] = "bullish (call-heavy flow)"
                elif pc > 1.30:
                    result["flow_label"] = "cautious (put-heavy hedge flow)"
                else:
                    result["flow_label"] = "neutral"
            # Dominant call strike = highest OI call (market's gravity)
            if not calls.empty:
                idx = calls["openInterest"].idxmax()
                result["dominant_call_strike"] = float(calls.loc[idx, "strike"])
        except Exception:
            pass
    except Exception:
        pass
    return result


_REALIZED_VOL_CACHE: dict[str, tuple[float, float]] = {}
_REALIZED_VOL_CACHE_TTL_S = 3600   # 1 hr — realized vol doesn't meaningfully shift within a session

def _realized_vol_estimate(ticker: str, lookback: int = 20) -> float:
    """
    Annualized realized volatility from recent daily closes — the sigma
    input for _estimate_bs_delta() below. Returns 0.40 (a reasonable
    generic momentum-stock default) on any data failure, since this only
    feeds an approximate delta for ITM strike selection, never priced
    risk directly.
    """
    cached = _REALIZED_VOL_CACHE.get(ticker)
    if cached and (time.time() - cached[0]) < _REALIZED_VOL_CACHE_TTL_S:
        return cached[1]
    vol = 0.40
    try:
        hist = yf.Ticker(ticker).history(period=f"{lookback + 10}d")
        closes = hist["Close"].dropna().values
        if len(closes) >= lookback // 2 + 2:
            rets = np.diff(np.log(closes[-(lookback + 1):]))
            if len(rets) >= 3:
                daily_std = float(np.std(rets, ddof=1))
                vol = max(0.15, min(2.0, daily_std * math.sqrt(252)))
    except Exception:
        pass
    _REALIZED_VOL_CACHE[ticker] = (time.time(), vol)
    return vol


def _estimate_bs_delta(current_price: float, strike: float, dte: int, is_call: bool, sigma: float) -> float:
    """
    Black-Scholes delta estimate — used only when the broker-supplied
    delta is unavailable (exactly 0.0, the placeholder value on a snapshot
    with no real Greeks). Confirmed live 2026-08-08: this account has
    options TRADING approved (level 3, real buying power) but NOT the
    OPRA options market-DATA subscription — a separate Alpaca product.
    Alpaca's "indicative" fallback feed (data.alpaca.markets/v1beta1/
    options/snapshots?feed=indicative) returns a bare latestQuote with NO
    greeks/impliedVolatility/openInterest fields at all — not degraded,
    literally absent from the JSON. _get_option_snapshot() defaults every
    missing Greek to 0.0, and the ITM hard filter downstream
    (`delta < 0.40 -> reject`) then rejected every single contract on
    every single scan, unconditionally — the actual root cause of zero
    options fills despite full trading approval. This estimate lets
    contract SELECTION proceed on realized-vol-implied Greeks instead of
    real ones; order PLACEMENT itself never needed OPRA to begin with.
    """
    try:
        T = max(dte, 1) / 365.0
        r = 0.045
        d1 = (math.log(current_price / strike) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
        n_d1 = 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        return round(n_d1 if is_call else n_d1 - 1, 3)
    except Exception:
        return 0.0


def _find_best_call_contract(client, ticker: str, current_price: float) -> dict | None:
    """
    Greeks-aware contract selection:
      1. Query 5 strikes around ATM (ATM-1 to ATM+3 increments)
      2. Get full snapshot with Greeks for each
      3. Score each contract (delta, spread, OI)
      4. Return highest-scoring contract that passes MIN_SCORE=30

    Returns full contract dict with Greeks, L2 context, score, or None.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    # Gate: underlying must have ≥5M avg daily volume — ensures liquid options chain.
    # Fail closed: if volume data unavailable, skip rather than enter illiquid contract.
    try:
        _fi = yf.Ticker(ticker).fast_info
        _adv = float(getattr(_fi, "three_month_average_volume", 0) or 0)
        if _adv < OPTIONS_MIN_UNDERLYING_VOL:
            print(f"  ⚠️  {ticker} options skipped — ADV {_adv/1e6:.1f}M < {OPTIONS_MIN_UNDERLYING_VOL/1e6:.0f}M floor")
            return None
    except Exception:
        print(f"  ⚠️  {ticker} options skipped — ADV check failed (fail-closed)")
        return None

    today = date.today()
    # Pick the Friday nearest to OPTIONS_TARGET_DTE — not just the first available Friday.
    target_expiry = None
    _best_diff = float("inf")
    for offset in range(OPTIONS_DTE_MIN, OPTIONS_DTE_MAX + 8):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 4:   # Friday
            _diff = abs(offset - OPTIONS_TARGET_DTE)
            if _diff < _best_diff:
                _best_diff = _diff
                target_expiry = candidate
    if not target_expiry:
        return None

    incr = 1.0 if current_price < 25 else (2.5 if current_price < 200 else 5.0)
    atm  = round(round(current_price / incr) * incr, 2)
    # Call ITM strike scan: start 4 increments below ATM (deep ITM), scan toward ATM.
    # Targeting delta ~0.70 per OPTIONS_ITM_TARGET_PCT=0.04 (4% ITM).
    strikes_to_scan = [atm + i * incr for i in range(-4, 2)]

    best_score    = -1
    best_contract: dict | None = None

    for strike in strikes_to_scan:
        strike = round(strike, 2)
        if strike <= 0:
            continue
        try:
            raw = client.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=[ticker],
                expiration_date=target_expiry,
                type=ContractType.CALL,
                # strike_price_gte/lte require STRING values (GetOptionContractsRequest
                # field type is Optional[str]) — passing floats raises a pydantic
                # ValidationError on every call, silently caught below, meaning this
                # function always returned None and every options signal silently
                # fell back to an equity order. Confirmed live 2026-07-29.
                strike_price_gte=str(round(strike - 0.01, 2)),
                strike_price_lte=str(round(strike + 0.01, 2)),
                limit=1,
            ))
            items = getattr(raw, "option_contracts", None) or (
                raw if isinstance(raw, list) else []
            )
            if not items:
                continue
            occ  = items[0].symbol
            snap = _get_option_snapshot(occ)
            if not snap:
                continue
            snap = _merge_contract_oi(snap, items[0])
            # No real delta from the broker (indicative feed / no OPRA
            # subscription) -- estimate one via Black-Scholes rather than
            # letting the hard filter below reject every contract on a
            # placeholder 0.0. See _estimate_bs_delta() docstring.
            if snap["delta"] == 0.0:
                snap["delta"] = _estimate_bs_delta(
                    current_price, strike, (target_expiry - today).days,
                    True, _realized_vol_estimate(ticker))
                snap["delta_estimated"] = True
            # Hard filter: reject OTM (delta < 0.40) and wide-spread contracts
            if snap["delta"] < 0.40 or snap["spread_pct"] > OPTIONS_MAX_SPREAD_PCT:
                continue
            score, reason = _score_option_contract(snap, current_price)
            _delta_tag = "~" if snap.get("delta_estimated") else ""
            print(f"    {occ}  Δ{_delta_tag}{snap['delta']:.2f}  θ{snap['theta']:.3f}/d  "
                  f"IV{snap['iv']*100:.0f}%  OI{snap['oi']}  score={score}  [{reason[:60]}]")
            if score > best_score:
                best_score = score
                best_contract = {
                    "occ_symbol": occ,
                    "strike":     float(getattr(items[0], "strike_price", strike)),
                    "expiry":     target_expiry.isoformat(),
                    "dte":        (target_expiry - today).days,
                    **snap,
                    "score":      score,
                    "score_reason": reason,
                }
        except Exception as _e:
            print(f"    ⚠️  Strike ${strike} lookup error: {_e}")
            continue

    if best_contract and best_score >= 30:
        return best_contract
    return None


def _find_best_put_contract(client, ticker: str, current_price: float) -> dict | None:
    """
    Select the best ITM put contract for a bearish play.
    Mirrors _find_best_call_contract but uses PUT type and scans strikes ABOVE
    current price (ITM for puts = strike > current price, targeting delta -0.65 to -0.75).
    Same 5M underlying volume gate, same scoring.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    # Same 5M ADV gate — puts on illiquid underlyings are equally worthless.
    # Fail closed: data failure → skip rather than enter illiquid contract.
    try:
        _fi  = yf.Ticker(ticker).fast_info
        _adv = float(getattr(_fi, "three_month_average_volume", 0) or 0)
        if _adv < OPTIONS_MIN_UNDERLYING_VOL:
            print(f"  ⚠️  {ticker} puts skipped — ADV {_adv/1e6:.1f}M < {OPTIONS_MIN_UNDERLYING_VOL/1e6:.0f}M floor")
            return None
    except Exception:
        print(f"  ⚠️  {ticker} puts skipped — ADV check failed (fail-closed)")
        return None

    today = date.today()
    target_expiry = None
    _best_diff = float("inf")
    for offset in range(OPTIONS_DTE_MIN, OPTIONS_DTE_MAX + 8):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 4:
            _diff = abs(offset - OPTIONS_TARGET_DTE)
            if _diff < _best_diff:
                _best_diff = _diff
                target_expiry = candidate
    if not target_expiry:
        return None

    incr = 1.0 if current_price < 25 else (2.5 if current_price < 200 else 5.0)
    atm  = round(round(current_price / incr) * incr, 2)
    # Put ITM strike scan: start 2 increments above ATM (targeting delta ~0.70 for puts)
    strikes_to_scan = [atm + i * incr for i in range(2, 7)]

    best_score    = -1
    best_contract: dict | None = None

    for strike in strikes_to_scan:
        strike = round(strike, 2)
        if strike <= 0:
            continue
        try:
            raw = client.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=[ticker],
                expiration_date=target_expiry,
                type=ContractType.PUT,
                # see _find_best_call_contract — strike_price_gte/lte require strings.
                strike_price_gte=str(round(strike - 0.01, 2)),
                strike_price_lte=str(round(strike + 0.01, 2)),
                limit=1,
            ))
            items = getattr(raw, "option_contracts", None) or (
                raw if isinstance(raw, list) else []
            )
            if not items:
                continue
            occ  = items[0].symbol
            snap = _get_option_snapshot(occ)
            if not snap:
                continue
            snap = _merge_contract_oi(snap, items[0])
            # See _find_best_call_contract — same missing-Greeks fallback.
            if snap["delta"] == 0.0:
                snap["delta"] = _estimate_bs_delta(
                    current_price, strike, (target_expiry - today).days,
                    False, _realized_vol_estimate(ticker))
                snap["delta_estimated"] = True
            delta_abs = abs(snap.get("delta", 0))
            if delta_abs < 0.40 or snap["spread_pct"] > OPTIONS_MAX_SPREAD_PCT:
                continue
            score, reason = _score_option_contract(snap, current_price)
            _delta_tag = "~" if snap.get("delta_estimated") else ""
            print(f"    PUT {occ}  |Δ|{_delta_tag}{delta_abs:.2f}  θ{snap['theta']:.3f}/d  "
                  f"IV{snap['iv']*100:.0f}%  OI{snap['oi']}  score={score}  [{reason[:60]}]")
            if score > best_score:
                best_score = score
                best_contract = {
                    "occ_symbol": occ,
                    "strike":     float(getattr(items[0], "strike_price", strike)),
                    "expiry":     target_expiry.isoformat(),
                    "dte":        (target_expiry - today).days,
                    "option_type": "PUT",
                    **snap,
                    "score":      score,
                    "score_reason": reason,
                }
        except Exception as _e:
            print(f"    ⚠️  PUT strike ${strike} lookup error: {_e}")
            continue

    if best_contract and best_score >= 30:
        return best_contract
    return None


def _fetch_available_expiries(client, ticker: str, max_days: int = 90) -> list[date]:
    """
    All distinct expiration dates actually listed for `ticker` within the
    next `max_days`, sorted ascending — backs /options TICKER's "browse
    other expiries" list.

    Confirmed live 2026-08-12: GetOptionContractsRequest WITHOUT explicit
    expiration_date_gte/lte bounds silently narrows to just the single
    nearest expiry (68 contracts, 1 expiry for SMCI) instead of the real
    listed ladder — passing an explicit range surfaces it correctly (344
    contracts, 7 weekly expiries out to late September). Every other
    options function in this file only ever needs ONE target expiry
    (computed directly, never enumerated) and so never hit this; it only
    surfaced once /options TICKER needed the real expiry list to browse.
    Returns [] on any failure — a discovery helper, not a hard dependency.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType
    try:
        raw = client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], type=ContractType.CALL,
            expiration_date_gte=date.today(),
            expiration_date_lte=date.today() + timedelta(days=max_days),
            limit=1000,
        ))
        items = getattr(raw, "option_contracts", None) or (raw if isinstance(raw, list) else [])
        expiries = set()
        for i in items:
            _e = i.expiration_date
            expiries.add(_e if isinstance(_e, date) else date.fromisoformat(str(_e)))
        return sorted(expiries)
    except Exception:
        return []


def _fetch_option_chain_for_display(client, ticker: str, current_price: float,
                                     expiry: Optional[date] = None,
                                     num_strikes: int = 8) -> Optional[dict]:
    """
    Broader, human-browsable options chain for the Telegram /options command.

    Unlike _find_best_call_contract/_find_best_put_contract (each of which
    picks ONE ITM-biased "best" contract for the automated scanner, scanning
    only strikes below/above current price respectively), this returns a
    band of `num_strikes` strikes around ATM for BOTH calls and puts, live
    bid/ask/delta per contract, so a human can actually see and choose from
    a menu instead of only ever getting the algo's single automatic pick.
    Not filtered by the automated path's delta/spread hard-gates — browsing
    should show what's actually available; the human (or, for an automated
    entry, the normal scan/score pipeline) is the filter here.

    `expiry` picks a specific listed date (from _fetch_available_expiries);
    if omitted, defaults to the nearest Friday to OPTIONS_TARGET_DTE — same
    default the automated path uses, so a plain "/options TICKER" behaves
    consistently with what the scanner would pick.

    Returns {"ticker", "expiry", "dte", "underlying_price", "items": [...]}
    or None if the underlying fails the same liquidity gate the automated
    path uses, or no expiry/contracts can be found. Each item is a dict:
    {"type": "CALL"/"PUT", "occ_symbol", "strike", "bid", "ask", "delta",
    "delta_estimated"}. Never raises — a single strike's lookup failure is
    skipped, not fatal to the whole chain.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    try:
        _fi  = yf.Ticker(ticker).fast_info
        _adv = float(getattr(_fi, "three_month_average_volume", 0) or 0)
        if _adv < OPTIONS_MIN_UNDERLYING_VOL:
            return None
    except Exception:
        return None

    today = date.today()
    target_expiry = expiry
    if target_expiry is None:
        _best_diff = float("inf")
        for offset in range(OPTIONS_DTE_MIN, OPTIONS_DTE_MAX + 8):
            candidate = today + timedelta(days=offset)
            if candidate.weekday() == 4:   # Friday
                _diff = abs(offset - OPTIONS_TARGET_DTE)
                if _diff < _best_diff:
                    _best_diff = _diff
                    target_expiry = candidate
    if not target_expiry:
        return None

    incr = 1.0 if current_price < 25 else (2.5 if current_price < 200 else 5.0)
    atm  = round(round(current_price / incr) * incr, 2)
    # num_strikes strikes centered near ATM, slightly ITM-biased (one more
    # below than above) to match the same ITM lean _find_best_call_contract
    # uses elsewhere, rather than a purely symmetric OTM/ITM split.
    _lo = -(num_strikes // 2)
    _hi = num_strikes + _lo
    strikes = [round(atm + i * incr, 2) for i in range(_lo, _hi) if atm + i * incr > 0]

    items: list[dict] = []
    for _otype, _contract_type in (("CALL", ContractType.CALL), ("PUT", ContractType.PUT)):
        for strike in strikes:
            try:
                raw = client.get_option_contracts(GetOptionContractsRequest(
                    underlying_symbols=[ticker],
                    expiration_date=target_expiry,
                    type=_contract_type,
                    strike_price_gte=str(round(strike - 0.01, 2)),
                    strike_price_lte=str(round(strike + 0.01, 2)),
                    limit=1,
                ))
                found = getattr(raw, "option_contracts", None) or (
                    raw if isinstance(raw, list) else []
                )
                if not found:
                    continue
                occ  = found[0].symbol
                snap = _get_option_snapshot(occ)
                if not snap or snap.get("bid", 0) <= 0 or snap.get("ask", 0) <= 0:
                    continue
                delta_estimated = False
                if snap["delta"] == 0.0:
                    delta_estimated = True
                    snap["delta"] = _estimate_bs_delta(
                        current_price, strike, (target_expiry - today).days,
                        _otype == "CALL", _realized_vol_estimate(ticker))
                items.append({
                    "type": _otype, "occ_symbol": occ, "strike": strike,
                    "bid": snap["bid"], "ask": snap["ask"],
                    "delta": snap["delta"], "delta_estimated": delta_estimated,
                })
            except Exception:
                continue

    if not items:
        return None
    return {
        "ticker": ticker, "expiry": target_expiry.isoformat(),
        "dte": (target_expiry - today).days, "underlying_price": current_price,
        "items": items,
    }


def _find_spread_legs(client, ticker: str, current_price: float, side: str,
                      target_debit: float) -> Optional[dict]:
    """
    Select a long OTM leg + short OTM leg for an earnings vertical debit spread,
    on the Friday nearest EARNINGS_SPREAD_TARGET_DTE. Starts at
    EARNINGS_SPREAD_LONG_OTM_PCT/EARNINGS_SPREAD_SHORT_OTM_PCT and narrows the
    width toward EARNINGS_SPREAD_MIN_WIDTH_PCT if the net debit exceeds
    target_debit — a real account this size can't always afford the "textbook"
    width (tonight's manual META spread cost $834; 5% of this account is ~$150).

    Liquidity-only leg filter (bid>0, ask>0, spread_pct <= MAX_SPREAD_PCT) —
    NOT _score_option_contract()'s delta-0.60-0.80 scoring, which is tuned for
    directional ITM calls and is the wrong shape for OTM spread legs. Open
    interest is recorded (via _merge_contract_oi, real numbers as of
    2026-07-30) but still NOT hard-gated — OTM earnings-week strikes can be
    thin without being illiquid, and the bid/ask/spread checks already screen
    for that directly.

    side: "CALL" or "PUT". Returns a dict with long/short OCC symbols, strikes,
    net debit, expiry/DTE — or None if no liquid pair is found at any width.
    """
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums import ContractType

    try:
        _fi = yf.Ticker(ticker).fast_info
        _adv = float(getattr(_fi, "three_month_average_volume", 0) or 0)
        if _adv < OPTIONS_MIN_UNDERLYING_VOL:
            return None
    except Exception:
        return None

    today = date.today()
    target_expiry = None
    best_diff = float("inf")
    for offset in range(EARNINGS_SPREAD_MIN_DTE, EARNINGS_SPREAD_MAX_DTE + 8):
        candidate = today + timedelta(days=offset)
        if candidate.weekday() == 4:   # Friday
            diff = abs(offset - EARNINGS_SPREAD_TARGET_DTE)
            if diff < best_diff:
                best_diff = diff
                target_expiry = candidate
    if not target_expiry:
        return None

    is_call = side == "CALL"
    contract_type = ContractType.CALL if is_call else ContractType.PUT

    # Band wide enough to cover the long target through the widest possible
    # short target, plus a small buffer since listed strikes won't land
    # exactly on the computed targets.
    if is_call:
        lo = current_price * (1 + EARNINGS_SPREAD_LONG_OTM_PCT - 0.02)
        hi = current_price * (1 + EARNINGS_SPREAD_LONG_OTM_PCT + EARNINGS_SPREAD_MAX_WIDTH_PCT + 0.02)
    else:
        lo = current_price * (1 - EARNINGS_SPREAD_LONG_OTM_PCT - EARNINGS_SPREAD_MAX_WIDTH_PCT - 0.02)
        hi = current_price * (1 - EARNINGS_SPREAD_LONG_OTM_PCT + 0.02)

    try:
        raw = client.get_option_contracts(GetOptionContractsRequest(
            underlying_symbols=[ticker], expiration_date=target_expiry,
            type=contract_type,
            strike_price_gte=str(round(max(lo, 0.01), 2)),
            strike_price_lte=str(round(max(hi, lo + 0.01), 2)),
            limit=50,
        ))
        items = getattr(raw, "option_contracts", None) or (raw if isinstance(raw, list) else [])
        items = sorted(items, key=lambda c: float(c.strike_price))
    except Exception as exc:
        print(f"  ⚠️  {ticker} {side} spread band lookup failed: {exc}", file=sys.stderr)
        return None
    if len(items) < 2:
        return None

    long_target = (current_price * (1 + EARNINGS_SPREAD_LONG_OTM_PCT) if is_call
                   else current_price * (1 - EARNINGS_SPREAD_LONG_OTM_PCT))
    long_c = min(items, key=lambda c: abs(float(c.strike_price) - long_target))

    width_pct = EARNINGS_SPREAD_SHORT_OTM_PCT - EARNINGS_SPREAD_LONG_OTM_PCT
    while width_pct >= EARNINGS_SPREAD_MIN_WIDTH_PCT - 1e-9:
        long_strike = float(long_c.strike_price)
        short_target = long_strike * (1 + width_pct) if is_call else long_strike * (1 - width_pct)
        others = [c for c in items if float(c.strike_price) != long_strike]
        if not others:
            width_pct -= 0.01
            continue
        short_c = min(others, key=lambda c: abs(float(c.strike_price) - short_target))

        long_snap  = _get_option_snapshot(long_c.symbol)
        short_snap = _get_option_snapshot(short_c.symbol)
        width_pct -= 0.01
        if not long_snap or not short_snap:
            continue
        long_snap  = _merge_contract_oi(long_snap, long_c)
        short_snap = _merge_contract_oi(short_snap, short_c)
        if long_snap["ask"] <= 0 or short_snap["bid"] <= 0:
            continue
        if (long_snap["spread_pct"] > EARNINGS_SPREAD_MAX_SPREAD_PCT
                or short_snap["spread_pct"] > EARNINGS_SPREAD_MAX_SPREAD_PCT):
            continue

        net_debit = round(long_snap["ask"] - short_snap["bid"], 2)
        if net_debit <= 0:
            continue

        result = {
            "long_occ": long_c.symbol, "short_occ": short_c.symbol,
            "long_strike": float(long_c.strike_price), "short_strike": float(short_c.strike_price),
            "long_oi": long_snap.get("oi", 0), "short_oi": short_snap.get("oi", 0),
            "net_debit": net_debit, "expiry": target_expiry.isoformat(),
            "dte": (target_expiry - today).days,
        }
        # Good enough to fit budget, or we're already at the narrowest allowed
        # width — return either way; build_earnings_spread_plan() decides
        # whether the final cost is still acceptable.
        if net_debit * 100 <= target_debit or width_pct < EARNINGS_SPREAD_MIN_WIDTH_PCT - 1e-9:
            return result
    return None


def _submit_options_put(
    client, ticker: str, current_price: float, risk_dollars: float, signal: "ProSignal"
) -> tuple[str | None, dict | None]:
    """
    Greeks-aware ITM put submission for bearish plays (Bear Gap Hold, etc.).
    Confirms bullish P/C flow is absent before buying puts — if market is
    call-heavy (P/C < 0.3) the move may already be priced in, skip it.
    Otherwise mirrors _submit_options_call logic exactly.
    """
    print(f"  🎯 PUT Options: scanning contracts for {ticker} (price=${current_price:.2f}  budget=${risk_dollars:.0f})")
    contract = _find_best_put_contract(client, ticker, current_price)
    if not contract:
        print(f"  ⚠️  No suitable put contract found for {ticker}")
        return None, None

    expiry_str = contract["expiry"]
    l2 = _get_options_market_context(ticker, expiry_str)
    contract["pc_ratio"]            = l2["pc_ratio"]
    contract["flow_label"]          = l2["flow_label"]
    contract["dominant_call_strike"]= l2["dominant_call_strike"]
    contract["stock_spread_pct"]    = l2["stock_spread_pct"]

    # Skip if call flow dominates — market is positioned bullish, puts won't pay
    if l2["pc_ratio"] < 0.3:
        msg = (f"🚫 <b>Put skipped — bullish flow</b>: {ticker}\n"
               f"P/C ratio {l2['pc_ratio']:.2f} (<0.3 = call-heavy). "
               "Market positioned bullish — put entry poor risk/reward.")
        send_telegram(msg)
        print(f"  🚫 P/C={l2['pc_ratio']:.2f} < 0.3 — bullish flow, aborting puts")
        return None, None

    _put_mid = (contract.get("bid", contract["ask"]) + contract["ask"]) / 2
    premium_per_contract = _put_mid * 100
    if premium_per_contract <= 0:
        return None, None

    # ITM puts (delta ≥ 0.60): full budget — high delta = efficient capital use.
    # Near-ATM puts (delta < 0.60): no delta adjustment (flat sizing).
    raw_contracts = int(risk_dollars / premium_per_contract)
    contracts     = max(1, min(raw_contracts, 10))
    _limit_px     = round(_put_mid * 1.03, 2)   # mid + 3% buffer for fill
    total_cost    = round(contracts * _put_mid * 100, 2)

    if _put_mid * 100 > risk_dollars * 1.5:
        print(f"  ⚠️  Too expensive: 1 put contract=${_put_mid*100:.0f}  budget=${risk_dollars:.0f} — skip")
        return None, None

    _cash_ok, _cash_msg = _cash_available_for(total_cost)
    if not _cash_ok:
        print(f"  ⚠️  {ticker} put: {_cash_msg}")
        return None, None

    try:
        order = client.submit_order(LimitOrderRequest(
            symbol        = contract["occ_symbol"],
            qty           = contracts,
            side          = OrderSide.BUY,
            limit_price   = _limit_px,
            time_in_force = TimeInForce.DAY,
        ))
        label = "PAPER" if ALPACA_PAPER else "LIVE"
        _put_dabs = abs(contract.get("delta", 0))
        print(f"  📤 [{label}] OPTIONS {ticker} PUT  "
              f"strike=${contract['strike']}  exp={expiry_str}  "
              f"{contracts}x @ ${_limit_px} (mid+3%)  total=${total_cost:.0f}  "
              f"|Δ|{_put_dabs:.2f}  id={str(order.id)[:8]}…")
        contract["ask"]        = _limit_px   # record actual limit used
        contract["contracts"]  = contracts
        contract["total_cost"] = total_cost
        contract["option_type"] = "PUT"
        return str(order.id), contract
    except Exception as exc:
        print(f"  ❌ Put options order failed ({ticker}): {exc}")
        return None, None


def _submit_earnings_spread(client, plan: dict) -> tuple[str | None, str | None]:
    """
    Submits the whole earnings play (2 or 4 legs) as ONE atomic multi-leg
    (MLEG) order — the first use of OrderClass.MLEG in this codebase. Every
    prior options function here (_submit_options_call/_put/_close) submits
    one leg at a time; a debit spread done as two separate single-leg orders
    would carry real leg-imbalance risk (fill one side, not the other ->
    naked, undefined risk instead of the defined-risk spread intended).
    Confirmed against the installed alpaca-py: LimitOrderRequest.legs accepts
    up to 4 OptionLegRequest entries for options, and limit_price is positive
    for a debit / negative for a credit (both from the library's own
    docstring, not assumed).

    plan must have "sets" and "net_debit", plus a "call" and/or "put" key,
    each with "long_occ"/"short_occ" (as built by build_earnings_spread_plan()).
    Returns (order_id, None) on success or (None, error_text) on failure —
    error text is always surfaced, never swallowed, matching
    _submit_alpaca_trade's existing error-surfacing convention.
    """
    from alpaca.trading.requests import OptionLegRequest
    from alpaca.trading.enums import PositionIntent

    legs = []
    if plan.get("call"):
        legs += [
            OptionLegRequest(symbol=plan["call"]["long_occ"],  ratio_qty=1,
                              side=OrderSide.BUY,  position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=plan["call"]["short_occ"], ratio_qty=1,
                              side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
        ]
    if plan.get("put"):
        legs += [
            OptionLegRequest(symbol=plan["put"]["long_occ"],  ratio_qty=1,
                              side=OrderSide.BUY,  position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=plan["put"]["short_occ"], ratio_qty=1,
                              side=OrderSide.SELL, position_intent=PositionIntent.SELL_TO_OPEN),
        ]
    if not legs:
        return None, "plan has no call or put legs"
    if len(legs) > 4:
        return None, f"plan has {len(legs)} legs — Alpaca allows at most 4 for options"

    _spread_cost = round(plan["net_debit"] * plan["sets"] * 100, 2)
    _cash_ok, _cash_msg = _cash_available_for(_spread_cost)
    if not _cash_ok:
        return None, _cash_msg

    try:
        order = client.submit_order(LimitOrderRequest(
            qty           = plan["sets"],
            order_class   = OrderClass.MLEG,
            time_in_force = TimeInForce.DAY,
            limit_price   = round(plan["net_debit"], 2),   # positive = debit
            legs          = legs,
        ))
        label = "PAPER" if ALPACA_PAPER else "LIVE"
        print(f"  📤 [{label}] EARNINGS SPREAD {plan['ticker']}  {len(legs)} legs  "
              f"{plan['sets']}x @ ${plan['net_debit']} debit  id={str(order.id)[:8]}…")
        return str(order.id), None
    except Exception as exc:
        print(f"  ❌ Earnings spread order failed ({plan.get('ticker')}): {exc}")
        return None, str(exc)


def _submit_options_call(
    client, ticker: str, current_price: float, risk_dollars: float, signal: "ProSignal"
) -> tuple[str | None, dict | None]:
    """
    Greeks-aware options order submission:
      1. Find best contract (multi-strike scan, scored by delta/spread/OI)
      2. Get L2 market context (P/C ratio, dominant strike, stock spread)
      3. Size by premium budget — adjust for delta (low delta = need more contracts)
      4. Submit limit at ask, record full Greeks in PositionTracker
      5. Abort if P/C ratio > 1.5 (heavy put hedging = bearish flow, skip the call)

    Returns (order_id, contract_dict) or (None, None).
    """
    print(f"  🎯 Options: scanning contracts for {ticker} (price=${current_price:.2f}  budget=${risk_dollars:.0f})")
    contract = _find_best_call_contract(client, ticker, current_price)
    if not contract:
        print(f"  ⚠️  No suitable call contract found for {ticker}")
        return None, None

    expiry_str = contract["expiry"]
    print(f"  📊 Fetching L2 market context ({ticker} chain {expiry_str})…")
    l2 = _get_options_market_context(ticker, expiry_str)
    contract["pc_ratio"]            = l2["pc_ratio"]
    contract["flow_label"]          = l2["flow_label"]
    contract["dominant_call_strike"]= l2["dominant_call_strike"]
    contract["stock_spread_pct"]    = l2["stock_spread_pct"]

    # Abort if put flow is overwhelming — market is heavily hedged against us
    if l2["pc_ratio"] > 1.5:
        msg = (f"🚫 <b>Options skipped — bearish flow</b>: {ticker}\n"
               f"P/C ratio {l2['pc_ratio']:.2f} (>1.5 = put-heavy hedge demand)\n"
               f"Falling back to equity order.")
        send_telegram(msg)
        print(f"  🚫 P/C={l2['pc_ratio']:.2f} > 1.5 — bearish flow, aborting options")
        return None, None

    _call_mid = (contract.get("bid", contract["ask"]) + contract["ask"]) / 2
    premium_per_contract = _call_mid * 100
    if premium_per_contract <= 0:
        return None, None

    # ITM calls (delta ≥ 0.60): full budget — high delta means efficient capital use.
    # Flat sizing — no penalty for being ITM (we WANT high-delta contracts).
    raw_contracts = int(risk_dollars / premium_per_contract)
    contracts     = max(1, min(raw_contracts, 10))
    _limit_px     = round(_call_mid * 1.03, 2)   # mid + 3% buffer for fill
    total_cost    = round(contracts * _call_mid * 100, 2)

    if _call_mid * 100 > risk_dollars * 1.5:
        print(f"  ⚠️  Too expensive: 1 contract=${_call_mid*100:.0f}  budget=${risk_dollars:.0f} — skip")
        return None, None

    _cash_ok, _cash_msg = _cash_available_for(total_cost)
    if not _cash_ok:
        print(f"  ⚠️  {ticker} call: {_cash_msg}")
        return None, None

    try:
        order = client.submit_order(LimitOrderRequest(
            symbol        = contract["occ_symbol"],
            qty           = contracts,
            side          = OrderSide.BUY,
            limit_price   = _limit_px,
            time_in_force = TimeInForce.DAY,
        ))
        label = "PAPER" if ALPACA_PAPER else "LIVE"
        _call_delta = contract.get("delta", 0)
        print(f"  📤 [{label}] OPTIONS {ticker} CALL  "
              f"strike=${contract['strike']}  exp={expiry_str}  "
              f"{contracts}x @ ${_limit_px} (mid+3%)  total=${total_cost:.0f}  "
              f"Δ{_call_delta:.2f}  id={str(order.id)[:8]}…")
        contract["ask"]        = _limit_px   # record actual limit used
        contract["contracts"]  = contracts
        contract["total_cost"] = total_cost
        return str(order.id), contract
    except Exception as exc:
        print(f"  ❌ Options order failed ({ticker}): {exc}")
        return None, None


def _submit_manual_options_buy(client, pending: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Submit the user-confirmed BUY-to-open from the Telegram
    /options -> /buy -> YES flow, and register it in PositionTracker using
    the SAME setup-string convention and target/stop formula
    _submit_options_call/_submit_options_put use for automated entries
    (target1 = ask*1.5, target2 = ask*2.5, stop = ask*0.5), so the existing
    monitoring machinery (_monitor_option_position, trailing exit,
    milestone alerts, DTE alerts) treats a manually-opened position
    identically to an automated one regardless of how it was opened.

    Re-fetches a live quote right before submitting — the /options menu or
    /buy confirmation could be several minutes old — and aborts if the ask
    has moved past MANUAL_BUY_MAX_PRICE_DRIFT_PCT since confirmation, same
    staleness philosophy as _handle_earnings_approval_reply's drift check.
    This is purely a "has the market moved" gate; the price actually
    submitted is always pending["limit_price"] (the user's own chosen
    price from /buy, defaulting to the ask shown at menu time) — unlike
    automated entries, this is never silently re-priced to ask+3%, since
    the whole point of /buy taking an explicit price is control over
    exactly what gets paid. Respects the same MAX_POSITIONS tracking-slot
    cap automated entries do (pt.open() below); deliberately does NOT
    re-check the portfolio heat cap from run_pro_scanner()'s admission
    loop — that cap exists to stop automated signal admission from
    silently over-concentrating, whereas this is one deliberate,
    human-confirmed action at a time.

    Returns (order_id, error_message) — error_message is None on success.
    """
    occ = pending["occ_symbol"]
    snap = _get_option_snapshot(occ)
    if not snap or snap.get("ask", 0) <= 0:
        return None, f"{occ}: no live quote available right now — not submitted."

    _ask_then = float(pending.get("ask_at_confirm", 0) or 0)
    _ask_now  = float(snap["ask"])
    if _ask_then > 0:
        _drift = abs(_ask_now - _ask_then) / _ask_then * 100
        if _drift > MANUAL_BUY_MAX_PRICE_DRIFT_PCT:
            return None, (f"{occ}: ask moved {_drift:.1f}% since you confirmed "
                          f"(${_ask_then:.2f} → ${_ask_now:.2f}), past the "
                          f"{MANUAL_BUY_MAX_PRICE_DRIFT_PCT:.0f}% staleness limit — "
                          f"not submitted. Run /options again for a fresh quote.")

    contracts  = int(pending["contracts"])
    limit_px   = round(float(pending["limit_price"]), 2)   # the user's own chosen price, not ask+3%
    total_cost = round(contracts * limit_px * 100, 2)

    _cash_ok, _cash_msg = _cash_available_for(total_cost)
    if not _cash_ok:
        return None, f"{occ}: {_cash_msg}"

    try:
        order = client.submit_order(LimitOrderRequest(
            symbol=occ, qty=contracts, side=OrderSide.BUY,
            limit_price=limit_px, time_in_force=TimeInForce.DAY,
        ))
    except Exception as exc:
        return None, f"{occ}: order submission failed — {exc}"

    strike_dir = "P" if pending["option_type"] == "PUT" else "C"
    t1 = round(limit_px * 1.5, 2)
    t2 = round(limit_px * 2.5, 2)
    sl = round(limit_px * 0.50, 2)
    tracked = PositionTracker().open(OpenPosition(
        ticker     = pending["ticker"],
        bias       = "LONG" if pending["option_type"] == "CALL" else "SHORT",
        setup      = (f"Options {pending['option_type'].title()} {occ} "
                      f"(${pending['strike']:g}{strike_dir} exp {pending['expiry']})"),
        entry      = limit_px, stop = sl, target1 = t1, target2 = t2,
        shares     = contracts * 100, entry_date = datetime.today().strftime("%Y-%m-%d"),
        atr        = float(snap.get("delta", 0)), score = 0,
    ))
    if not tracked:
        try:
            client.cancel_order_by_id(str(order.id))
        except Exception:
            pass
        return None, f"{occ}: MAX_POSITIONS reached — order cancelled, no tracking slot available."
    return str(order.id), None


def _shares_fallback_allowed(ticker: str) -> bool:
    """
    True only for DMan's own curated small-cap watchlist. Policy set
    2026-08-05: grow the account on options, not on buying expensive
    large-cap shares outright when an options fill isn't available —
    DMan's calls are the deliberate exception, since those are cheap,
    thin-float gap-ups where "buy a lot of shares" is the actual play,
    and where listed options usually don't exist or aren't liquid anyway.
    """
    return ticker in DMAN_SMALLCAP_WATCHLIST


def _shares_fallback_budget_shares(current_price: float, budget: float) -> int:
    """
    How many whole shares fit inside `budget` at `current_price`. User
    decision 2026-08-08: when an options-eligible signal has no execution
    path (illiquid/too wide/too expensive) and isn't on the small-cap
    watchlist, buy this many shares instead of skipping entirely — capped
    to the SAME options-equivalent budget, not the signal's normal
    full-size equity sizing, so this still honors "grow on options, not
    expensive full-size shares." Returns 0 if the price is non-positive
    or exceeds the whole budget (can't afford even 1 share) — the caller
    treats 0 as "skip, budget too small for this name."
    """
    if current_price <= 0 or budget <= 0:
        return 0
    return int(budget / current_price)


def _submit_signals_to_alpaca(signals: list[ProSignal]) -> None:
    """
    Validate entry prices and submit passing signals to Alpaca (paper or live).
    Re-anchors each signal's stop and target to the live price so bracket legs
    are always correct relative to the actual fill price.
    Automatically adds each submitted trade to PositionTracker.
    Called after a scan when --submit flag is set.
    """
    if not signals:
        return
    if not ALPACA_API_KEY:
        print("  ⚠️  --submit requires ALPACA_API_KEY to be set.")
        return

    # Belt-and-suspenders: re-check circuit breakers here in case this function
    # is called directly (e.g. --mode alpaca, manual workflow_dispatch after close).
    if not is_market_open():
        print("  ⏸️  Market is closed — no orders submitted.")
        return
    if is_halted():
        _hr = ""
        try:
            with open(HALT_FILE) as _hf:
                _hr = json.load(_hf).get("reason", "")
        except Exception:
            pass
        print(f"  🛑 Manual halt active{(' — ' + _hr) if _hr else ''} — no orders submitted (/resume to re-enable).")
        return
    _tracker_cb = WinRateTracker()
    _stats_cb   = _tracker_cb.rolling_stats()
    if _stats_cb["consec_losses"] >= MAX_CONSEC_LOSSES:
        print(f"  🛑 Consecutive loss guard active ({_stats_cb['consec_losses']} losses) — no orders.")
        return
    if get_todays_loss() <= -(DAILY_LOSS_LIMIT * 100):
        print(f"  🛑 Daily loss limit active — no orders.")
        return
    if get_this_month_loss() <= -(MONTHLY_LOSS_LIMIT * 100):
        print(f"  🛑 Monthly loss limit active — no orders.")
        return

    mode_label = "PAPER" if ALPACA_PAPER else "LIVE"

    # ── Live-mode safety warnings ──────────────────────────────────────────
    if not ALPACA_PAPER:
        # Warn if ACCOUNT_SIZE was not explicitly configured
        if not os.getenv("ACCOUNT_SIZE"):
            msg = ("⚠️ <b>DMan LIVE mode</b>: ACCOUNT_SIZE secret not set — "
                   f"sizing uses ${ACCOUNT_SIZE:,.0f} default. "
                   "Set the real value in GitHub secrets → ACCOUNT_SIZE.")
            send_telegram(msg)
            print(f"  ⚠️  LIVE: ACCOUNT_SIZE not set — defaulting to ${ACCOUNT_SIZE:,.0f}")

        # PDT (Pattern Day Trader) check — runs before any order is submitted.
        # Accounts < $25k: max 3 day trades per rolling 5-day window.
        # At 0 remaining  → HALT: no new orders at all.
        # At 1 remaining  → SWING MODE: submit GTC entry + stop only (no T1 TP).
        #                   Position held overnight = NOT a day trade.
        # At 2+ remaining → normal day trade bracket.
        _pdt = {"used": 0, "remaining": 3, "swing_mode": False, "equity": 0.0}
        try:
            _pdt = _get_pdt_status()
            _equity    = _pdt["equity"]
            _dt_count  = _pdt["used"]
            _remaining = _pdt["remaining"]
            if _equity < 25_000:
                if _remaining == 0:
                    msg = ("🚫 <b>DMan LIVE — PDT HALT</b>: account equity "
                           f"${_equity:,.0f} &lt; $25k — day-trade limit reached "
                           f"({_dt_count}/3 used). No new orders will be placed. "
                           "Deposit funds to $25k+ or wait for the rolling window to reset.")
                    send_telegram(msg)
                    print(f"  🚫 PDT HALT: {_dt_count}/3 day trades used, equity ${_equity:,.0f} — skipping all submissions")
                    return
                elif _pdt["swing_mode"]:
                    # 1 day trade remaining — switch all new entries to swing mode
                    for _s in signals:
                        _s.swing_mode = True
                    msg = (f"🔄 <b>DMan LIVE — SWING MODE</b>: {_dt_count}/3 day trades used — "
                           f"1 remaining. Submitting as GTC swing trades (held overnight) "
                           "to preserve the last day-trade budget.")
                    send_telegram(msg)
                    print(f"  🔄 SWING MODE: {_dt_count}/3 day trades used — all new entries go GTC")
                else:
                    msg = (f"⚠️ <b>DMan LIVE — PDT</b>: {_dt_count}/3 used — "
                           f"{_remaining} day trade(s) remaining. Orders proceeding normally.")
                    send_telegram(msg)
                    print(f"  ⚠️  PDT: {_remaining} day trade(s) remaining (equity ${_equity:,.0f})")
        except Exception as _pdt_exc:
            _pdt_msg = (f"🚫 <b>DMan LIVE — PDT CHECK FAILED</b>: Cannot verify day-trade "
                        f"count ({_pdt_exc}). No orders submitted to prevent ghost positions.")
            send_telegram(_pdt_msg)
            print(f"  🚫 PDT check failed — halting to prevent ghost positions: {_pdt_exc}")
            return

        # Warn (but do not block) if FOMC is within 7 days — 12h dedup so it fires
        # at most twice per day, not on every signal submission.
        _today_live = date.today()
        for _ev_live in sorted(_FOMC_DATES):
            _d_live = (_ev_live - _today_live).days
            if 0 <= _d_live <= 7:
                if not _is_duplicate_alert("__FOMC_WARN__"):
                    msg = (f"⚠️ <b>DMan LIVE mode</b>: FOMC {_ev_live.strftime('%a %b %d')} "
                           f"in {_d_live}d — elevated risk. "
                           "Recommend paper mode until after FOMC + OPEX clear. "
                           "Proceeding anyway — monitor positions closely.")
                    send_telegram(msg)
                    _save_last_alert("__FOMC_WARN__")
                    print(f"  ⚠️  LIVE: FOMC in {_d_live}d — high-risk week warning sent")
                break
            if _d_live > 7:
                break

        # Major unscheduled macro events (tariff deadlines, etc.) — check_macro_safe()
        # already HARD-blocks new entries on these days; this alert exists so the
        # reason is visible on the phone instead of only in scan logs nobody checks.
        for _ev_mm in sorted(_MAJOR_MACRO_EVENT_DATES):
            _d_mm = (_ev_mm - _today_live).days
            if -1 <= _d_mm <= MACRO_BLACKOUT:
                if not _is_duplicate_alert("__MACRO_EVENT_WARN__"):
                    send_telegram(
                        f"🚫 <b>DMan LIVE mode</b>: major macro event "
                        f"{_ev_mm.strftime('%a %b %d')} — new entries BLOCKED today "
                        f"(hard gate, not advisory). Existing positions still monitored/exited normally."
                    )
                    _save_last_alert("__MACRO_EVENT_WARN__")
                    print(f"  🚫 LIVE: major macro event {_ev_mm} — entries blocked, alert sent")
                break

    # Adaptive risk multiplier — full global context (replaces SPY-only check).
    # Reads futures, VIX, DXY, BTC, Asia overnight, IWM/SPY ratio.
    # Score -4 → 0.35x sizing  |  Score +4 → 1.30x sizing.
    print("  🌍 Fetching global context for adaptive sizing...", flush=True)
    _ctx = _fetch_global_context()
    _risk_off_mult = _ctx["risk_mult"]
    _ctx_tone = _ctx["tone"]
    if _risk_off_mult != 1.0:
        _dir = "reduced" if _risk_off_mult < 1.0 else "boosted"
        _pct = abs(1 - _risk_off_mult) * 100
        send_telegram(
            f"{_ctx['summary']}\n"
            f"Position sizes <b>{_dir} {_pct:.0f}%</b> (mult {_risk_off_mult:.2f}x)"
        )
        print(f"  🌍 {_ctx_tone}  score={_ctx['score']:+d}  → sizing {_risk_off_mult:.2f}x")

    # Hot streak press — 3+ consecutive wins → 1.25x sizing (compounding the edge)
    # 1 consecutive loss → 0.85x (early caution before the 3-loss halt kicks in)
    _streak_stats_live = WinRateTracker().rolling_stats()
    _consec_wins_live  = _streak_stats_live.get("consec_wins", 0)
    _consec_loss_live  = _streak_stats_live.get("consec_losses", 0)
    if _consec_wins_live >= 3:
        _risk_off_mult = min(1.50, _risk_off_mult * 1.25)
        print(f"  🔥 HOT STREAK: {_consec_wins_live} wins in a row — sizing ×{_risk_off_mult:.2f}")
        if not _is_duplicate_alert("__HOT_STREAK__"):
            send_telegram(
                f"🔥 <b>DMan HOT STREAK</b> — {_consec_wins_live} consecutive wins\n"
                f"Sizing boosted to {_risk_off_mult:.2f}× to press the edge."
            )
            _save_last_alert("__HOT_STREAK__")
    elif _consec_loss_live == 1:
        _risk_off_mult = min(_risk_off_mult, 0.85)
        print(f"  ⚠️  1 consecutive loss — early caution, sizing →0.85×")

    pt        = PositionTracker()
    submitted = 0

    # Guard: skip tickers PositionTracker already knows about — prevents duplicate
    # bracket orders if the same signal fires across two consecutive scans.
    already_tracked = {p.ticker for p in pt.positions}

    print(f"\n  {'─'*68}")
    print(f"  [{mode_label}] Validating {len(signals)} signal(s) for Alpaca submission…")

    for sig in signals:
        if sig.ticker in already_tracked:
            print(f"  ⏭️  {sig.ticker:<8} already in open positions — skipping duplicate")
            continue

        # size_position_kelly() reports shares=0 when even 1 share would
        # risk meaningfully more than the sized Kelly budget (a wide-stop
        # or higher-priced name against a small risk fraction) -- see its
        # docstring. Must be skipped here, not submitted: a qty=0 order
        # would either be rejected by the broker or, worse, silently
        # round up somewhere downstream and re-blow the exact budget this
        # was meant to protect.
        if sig.shares <= 0:
            print(f"  ⏭️  {sig.ticker:<8} sizing failed — even 1 share exceeds the "
                  f"risk budget for this stop distance — skipping")
            continue

        valid, cur = validate_entry_price(sig)
        drift_pct  = (cur - sig.entry) / sig.entry * 100
        if not valid:
            print(f"  ⚡ {sig.ticker:<8} entry stale  "
                  f"signal=${sig.entry}  now=${cur}  drift={drift_pct:+.1f}%  — skipped")
            send_telegram(
                f"⚡ <b>Signal skipped — stale entry</b>: {sig.ticker} {sig.bias}\n"
                f"Detected ${sig.entry} → now ${cur:.2f} ({drift_pct:+.1f}% drift)\n"
                f"Entry re-validation failed — no order placed."
            )
            continue

        # Apply sizing multiplier — down for risk-off/cold streak, up for hot streak/risk-on
        if _risk_off_mult != 1.0:
            sig.shares = max(1, int(sig.shares * _risk_off_mult))
            sig.cost   = round(sig.shares * sig.entry, 2)

        # Re-anchor bracket to live price — preserves the signal's ACTUAL
        # target ratio (whatever score_signal()/_raw_signals() set — a
        # setup-specific multiplier or a gap-echo level, no longer a
        # hardcoded number here) and shifts the whole entry/stop/target
        # structure together to the real fill price, so a small amount of
        # drift between signal-time and submission-time doesn't change
        # what trade this actually is. Found 2026-08-16 review: this used
        # to hardcode a 2.5x/4.0x multiplier here — a THIRD number,
        # different from score_signal()'s (then also-hardcoded) 2.0x/3.0x
        # — so what got submitted to the broker never matched what was
        # logged/alerted even before score_signal()'s own overwrite bug is
        # counted separately.
        _orig_risk = round(sig.entry - sig.stop, 4)  # risk-per-share from signal detection
        if _orig_risk > 0:
            _t1_mult = abs(sig.target1 - sig.entry) / _orig_risk
            _t2_mult = abs(sig.target2 - sig.entry) / _orig_risk
            if sig.bias == "LONG":
                _live_entry = round(cur * 1.001, 2)    # 0.1% buffer → improves fill odds
                sig.entry   = _live_entry
                sig.stop    = round(_live_entry - _orig_risk, 2)
                sig.target1 = round(_live_entry + _t1_mult * _orig_risk, 2)
                sig.target2 = round(_live_entry + _t2_mult * _orig_risk, 2)
            else:  # SHORT
                _live_entry = round(cur * 0.999, 2)
                sig.entry   = _live_entry
                sig.stop    = round(_live_entry + _orig_risk, 2)
                sig.target1 = round(_live_entry - _t1_mult * _orig_risk, 2)
                sig.target2 = round(_live_entry - _t2_mult * _orig_risk, 2)

        # ── Options branch: calls (LONG) or puts (SHORT) ──────────────────────
        # Calls: WATCHLIST membership OR a setup already trusted for options
        # (OPTIONS_SETUPS = Gap & Hold / Morning Runner) — mirrors the puts
        # relaxation directly below, and for the same reason: the curated
        # ~90-name WATCHLIST was never the real quality gate for whether a
        # SETUP deserves options, it's the ADV/delta/spread checks inside
        # _find_best_call_contract (5M ADV floor) that actually screen
        # liquidity. Confirmed live 2026-07-31: the "all tickers" scan
        # (--universe all, ~1,200 names/day) generates real Gap & Hold/
        # Morning Runner signals outside WATCHLIST that had no options path
        # before this — same dead-end class as the MBLY puts incident below,
        # just on the call side. Setups NOT in OPTIONS_SETUPS (e.g. Vol
        # Breakdown) still fall through to equity regardless of ticker,
        # same as before — this only widens eligibility for setups already
        # proven in backtest, not a blanket "any LONG signal" gate.
        _use_options = (ENABLE_OPTIONS_TRADING and sig.bias == "LONG"
                        and (sig.ticker in WATCHLIST or sig.setup in OPTIONS_SETUPS))
        # Puts: WATCHLIST membership OR a Bear Gap Hold signal — that setup is
        # one of the two most strictly-gated patterns in the system (gap%,
        # RVOL, RSI, MACD-confirmed, held-below-open, MTF+news-catalyst
        # checked) and shouldn't dead-end just because the ticker isn't in
        # the ~83-name curated large-cap list. The options pipeline's own
        # liquidity gate (5M ADV floor in _find_best_put_contract) is the
        # real safety check here, not list membership. Confirmed dead-end in
        # production: MBLY scored 100/100 Bear Gap Hold on 2026-07-23, was
        # correctly alerted, but had NO execution path — SHORT bias blocks
        # equity (ALLOW_SHORTS=False) and it wasn't in WATCHLIST (blocked
        # puts too) — a valid signal that could never become a trade.
        _use_puts = (OPTIONS_ENABLE_PUTS and sig.bias == "SHORT"
                     and (sig.ticker in WATCHLIST or sig.setup == "Bear Gap Hold"))
        _opt_contract: dict | None = None
        oid: str | None = None
        _submit_err: str | None = None
        _options_was_attempted = _use_options or _use_puts

        if _use_options:
            _opt_client = get_alpaca_client()
            if _opt_client:
                _opt_risk = round(OPTIONS_MAX_POSITION_COST * _risk_off_mult, 2)
                print(f"  🎯 Options mode: finding call for {sig.ticker}  budget=${_opt_risk:.0f}")
                try:
                    oid, _opt_contract = _submit_options_call(
                        _opt_client, sig.ticker, cur, _opt_risk, sig
                    )
                except Exception as _opt_exc:
                    print(f"  ⚠️  Options error ({sig.ticker}): {_opt_exc} — falling back to shares")
                    oid = None
                if oid is None:
                    print(f"  ↩️  Options unavailable for {sig.ticker} — falling back to shares")
                    _use_options = False

        elif _use_puts:
            _opt_client = get_alpaca_client()
            if _opt_client:
                _opt_risk = round(OPTIONS_MAX_POSITION_COST * _risk_off_mult, 2)
                print(f"  🐻 Put options mode: finding put for {sig.ticker}  budget=${_opt_risk:.0f}")
                try:
                    oid, _opt_contract = _submit_options_put(
                        _opt_client, sig.ticker, cur, _opt_risk, sig
                    )
                except Exception as _opt_exc:
                    print(f"  ⚠️  Put options error ({sig.ticker}): {_opt_exc} — skipping")
                    oid = None
                if oid is None:
                    print(f"  ↩️  Put options unavailable for {sig.ticker} — SHORT signal skipped (ALLOW_SHORTS=False)")
                    _use_puts = False

        if not _use_options and not _use_puts:
            if sig.bias == "SHORT" and not ALLOW_SHORTS:
                print(f"  ⏭️  {sig.ticker} {sig.setup} SHORT skipped — ALLOW_SHORTS=False, "
                      f"not in WATCHLIST, and not a Bear Gap Hold signal")
                continue
            # Shares-fallback policy (2026-08-05): grow the account on options,
            # not on buying expensive large-cap shares outright. DMan's own
            # curated small-cap watchlist is the one deliberate exception —
            # those are exactly the cheap, thin-float gap-ups where "buy a lot
            # of shares" IS the play (and where options usually aren't liquid
            # enough to exist anyway).
            if not _shares_fallback_allowed(sig.ticker):
                if not _options_was_attempted:
                    # Never options-eligible in the first place (e.g. a Vol
                    # Breakdown signal, or a setup outside OPTIONS_SETUPS) —
                    # the original 2026-08-05 policy stands unchanged: skip
                    # quietly, no shares purchase, no alert.
                    print(f"  ⏭️  {sig.ticker} {sig.setup} skipped — options unavailable/ineligible "
                          f"and not a DMan watchlist ticker (shares reserved for DMan picks only)")
                    continue
                # Confirmed live 2026-08-08: a signal that WAS options-eligible
                # (WATCHLIST/OPTIONS_SETUPS) and got a real attempt -- not just
                # never-eligible -- silently produced zero trade, the same
                # "valid signal, no execution path" dead-end already fixed once
                # for puts (MBLY, see the OPTIONS_ENABLE_PUTS comment above).
                # User decision 2026-08-08: fall back to a BUDGET-CAPPED shares
                # position (same $ risk as the options attempt would have used,
                # not the signal's normal full-size equity sizing) rather than
                # skip entirely -- this still honors "grow on options, not
                # expensive full-size shares," it just guarantees SOMETHING
                # trades instead of a signal quietly vanishing after being
                # alerted.
                _fallback_budget = round(OPTIONS_MAX_POSITION_COST * _risk_off_mult, 2)
                _fallback_shares = _shares_fallback_budget_shares(cur, _fallback_budget)
                if _fallback_shares < 1:
                    print(f"  ⏭️  {sig.ticker} {sig.setup} skipped — options unavailable and "
                          f"price ${cur} too high for the ${_fallback_budget:.0f} shares-fallback budget")
                    send_telegram(
                        f"⏭️ <b>Signal alerted but not executed</b>: {sig.ticker} {sig.setup}\n"
                        f"Options attempted and unavailable. Price ${cur:.2f} exceeds the "
                        f"${_fallback_budget:.0f} shares-fallback budget for even 1 share. No trade placed."
                    )
                    continue
                sig.shares = _fallback_shares
                sig.cost   = round(_fallback_shares * cur, 2)
                print(f"  ↩️  {sig.ticker} {sig.setup}: options unavailable — falling back to "
                      f"{_fallback_shares} budget-capped share(s) (${sig.cost:.0f} of ${_fallback_budget:.0f})")
                oid, _submit_err = submit_alpaca_trade(sig)
                if oid:
                    send_telegram(
                        f"↩️ <b>Options unavailable — capped shares fallback</b>: {sig.ticker} {sig.setup}\n"
                        f"{_fallback_shares} sh @ ~${cur:.2f} = ${sig.cost:.0f} "
                        f"(capped to the ${_fallback_budget:.0f} options-equivalent budget)"
                    )
            else:
                oid, _submit_err = submit_alpaca_trade(sig)

        if oid:
            if (_use_options or _use_puts) and _opt_contract:
                _occ     = _opt_contract.get("occ_symbol", "")
                _exp_str = _opt_contract.get("expiry", "?")
                _strike  = _opt_contract.get("strike", 0)
                _ask     = _opt_contract.get("ask", 0)
                _ctrs    = _opt_contract.get("contracts", 1)
                _delta   = _opt_contract.get("delta", 0)
                _theta   = _opt_contract.get("theta", 0)
                _gamma   = _opt_contract.get("gamma", 0)
                _vega    = _opt_contract.get("vega", 0)
                _iv      = _opt_contract.get("iv", 0)
                _oi      = _opt_contract.get("oi", 0)
                _bsz     = _opt_contract.get("bid_size", 0)
                _asz     = _opt_contract.get("ask_size", 0)
                _pc      = _opt_contract.get("pc_ratio", 1.0)
                _flow    = _opt_contract.get("flow_label", "")
                _dom_k   = _opt_contract.get("dominant_call_strike", 0)
                _opt_type = _opt_contract.get("option_type", "CALL")   # CALL or PUT
                _t1_prem = round(_ask * 1.5, 2)   # +50% premium = T1 (realistic for ITM delta 0.70)
                _t2_prem = round(_ask * 2.5, 2)   # +150% premium = T2 (full runner)
                _sl_prem = round(_ask * 0.50, 2)  # -50% stop
                _theta_pct_day = abs(_theta / _ask * 100) if _ask > 0 else 0
                _strike_dir = "P" if _opt_type == "PUT" else "C"

                _tracked = pt.open(OpenPosition(
                    ticker     = sig.ticker,
                    bias       = "LONG" if _use_options else "SHORT",
                    setup      = f"Options {_opt_type.title()} {_occ} (${_strike}{_strike_dir} exp {_exp_str})",
                    entry      = _ask,
                    stop       = _sl_prem,
                    target1    = _t1_prem,
                    target2    = _t2_prem,
                    shares     = _ctrs * 100,
                    entry_date = datetime.today().strftime("%Y-%m-%d"),
                    atr        = _delta,
                    score      = sig.confluence_score,
                ))
                if not _tracked:
                    print(f"  ⚠️  MAX_POSITIONS reached — cancelling {sig.ticker} options order {oid[:8]}")
                    send_telegram(f"⚠️ <b>MAX POSITIONS</b> — {sig.ticker} options order {oid[:8]} cancelled (portfolio full, no tracking slot available)")
                    try:
                        _cc = get_alpaca_client()
                        if _cc:
                            _cc.cancel_order_by_id(oid)
                    except Exception as _ce:
                        send_telegram(f"🚨 <b>CANCEL FAILED</b> — {sig.ticker} {oid[:8]}: {_ce}. Cancel manually in Alpaca!")
                    continue
                submitted += 1
                # OPRA data subscription not entitled on this account (confirmed
                # live 2026-08-08) -- when the broker can't supply real Greeks,
                # _find_best_call/put_contract fills delta from a Black-Scholes
                # estimate instead of blocking the trade. Flagged here so this
                # never reads as a real broker-quoted delta.
                _delta_note = " (Δ est. — OPRA not entitled)" if _opt_contract.get("delta_estimated") else ""
                _greek_str = (
                    f"Δ {_delta:.2f}  Γ {_gamma:.4f}  θ {_theta:.3f}/d  "
                    f"ν {_vega:.3f}  IV {_iv*100:.0f}%  OI {_oi:,}{_delta_note}"
                )
                _l2_str = (
                    f"L2: bid {_bsz}×{_opt_contract.get('bid',0):.2f}  "
                    f"ask {_asz}×{_ask:.2f}  "
                    f"P/C {_pc:.2f} ({_flow})"
                    + (f"  Dominant strike ${_dom_k}" if _dom_k else "")
                )
                _icon = "🐻" if _opt_type == "PUT" else "🎯"
                send_telegram(
                    f"{_icon} <b>Options order placed</b> [{mode_label}] — {sig.ticker} {_opt_type}\n"
                    f"Contract: <b>{_occ}</b>\n"
                    f"Strike ${_strike}  Exp {_exp_str}  ({_opt_contract.get('dte','?')}d)\n"
                    f"Premium: ${_ask}/sh × {_ctrs}ct = <b>${_opt_contract['total_cost']:.0f}</b>  "
                    f"(θ decay {_theta_pct_day:.1f}%/day)\n"
                    f"T1 (+50%): ${_t1_prem}  T2 (+150%): ${_t2_prem}  Stop (-50%): ${_sl_prem}\n"
                    f"{_greek_str}\n"
                    f"{_l2_str}\n"
                    f"Underlying: ${cur:.2f}  Score: {sig.confluence_score}/100  ID: {oid[:8]}…"
                )
            else:
                _setup_tag = ("SWING — " + sig.setup) if sig.swing_mode else sig.setup
                _tracked = pt.open(OpenPosition(
                    ticker     = sig.ticker,
                    bias       = sig.bias,
                    setup      = _setup_tag,
                    entry      = sig.entry,
                    stop       = sig.stop,
                    target1    = sig.target1,
                    target2    = sig.target2,
                    shares     = sig.shares,
                    entry_date = datetime.today().strftime("%Y-%m-%d"),
                    atr        = sig.atr,
                    score      = sig.confluence_score,
                ))
                if not _tracked:
                    print(f"  ⚠️  MAX_POSITIONS reached — cancelling {sig.ticker} order {oid[:8]}")
                    send_telegram(f"⚠️ <b>MAX POSITIONS</b> — {sig.ticker} order {oid[:8]} cancelled (portfolio full)")
                    try:
                        _cc = get_alpaca_client()
                        if _cc:
                            _cc.cancel_order_by_id(oid)
                    except Exception as _ce:
                        send_telegram(f"🚨 <b>CANCEL FAILED</b> — {sig.ticker} {oid[:8]}: {_ce}. Cancel manually in Alpaca!")
                    continue
                submitted += 1
                if sig.swing_mode:
                    send_telegram(
                        f"🔄 <b>SWING Order placed</b> [{mode_label}] — {sig.ticker} {sig.bias}\n"
                        f"GTC Limit ${sig.entry}  Stop ${sig.stop}  T1 ${sig.target1} (monitor tomorrow)\n"
                        f"Shares: {sig.shares}  Score: {sig.confluence_score}/100  ID: {oid[:8]}…\n"
                        f"<i>PDT budget preserved — position held overnight, managed by momentum-watch</i>"
                    )
                else:
                    send_telegram(
                        f"✅ <b>Order placed</b> [{mode_label}] — {sig.ticker} {sig.bias}\n"
                        f"Limit ${sig.entry}  Stop ${sig.stop}  T1 ${sig.target1}\n"
                        f"Shares: {sig.shares}  Score: {sig.confluence_score}/100  ID: {oid[:8]}…"
                    )
        else:
            send_telegram(
                f"❌ <b>Order FAILED</b> [{mode_label}] — {sig.ticker} {sig.bias}\n"
                f"{_submit_err or 'Alpaca rejected the order — check GitHub Actions logs immediately.'}"
            )

    print(f"  📤 {submitted}/{len(signals)} signal(s) submitted [{mode_label}]\n")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 22 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="D🔥man Algorithm v3 PRO — targeting 80%+ win rate"
    )
    parser.add_argument("--mode", default="scan",
        choices=["scan","backtest","performance","regime","record",
                 "watch","rank","open","positions","alpaca","sync",
                 "live-outcomes","live-perf","premarket","premarket-early",
                 "momentum-watch","watchlist","scan-log","readiness","pnl",
                 "stocktwits","guard","merge-positions","watchdog","earnings-scan",
                 "fallback-guard"],
        help=("scan         : run pro scanner with all filters\n"
              "backtest     : walk-forward backtest\n"
              "performance  : win rate tracker report\n"
              "regime       : today's market regime only\n"
              "record       : log a completed trade outcome\n"
              "watch        : re-scan on a timer during market hours\n"
              "rank         : near-signal leaderboard for the watchlist\n"
              "open         : log a new open position\n"
              "positions    : view open positions with live P&L\n"
              "alpaca       : Alpaca account dashboard + scan + submit\n"
              "sync         : sync Alpaca fills → auto-record closed trades\n"
              "live-outcomes: resolve pending live signals → CSV log\n"
              "live-perf    : print live-trade ground-truth WR stats"))
    parser.add_argument("--score",   type=int, default=None,
                        help="Min confluence score override (default: adaptive)")
    parser.add_argument("--ai",      action="store_true",
                        help="Enable Claude AI scoring (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--tickers",  nargs="+", help="Override watchlist")
    parser.add_argument("--universe", choices=["curated","all"], default="curated",
                        help="curated: use 49-ticker watchlist (default); all: scan all US-listed stocks")
    parser.add_argument("--years",   type=int,  default=2,
                        help="Backtest lookback in years")
    parser.add_argument("--no-shorts", action="store_true",
                        help="Disable short signals")
    parser.add_argument("--export",  action="store_true",
                        help="Save scan signals to dman_signals_YYYY-MM-DD.json")
    parser.add_argument("--submit",  action="store_true",
                        help="Auto-submit scan signals to Alpaca paper trading")
    # --mode record args
    parser.add_argument("--ticker",     help="Ticker symbol (for --mode record)")
    parser.add_argument("--entry",      type=float, help="Entry price (for --mode record)")
    parser.add_argument("--exit-price", type=float, dest="exit_price",
                        help="Exit price (for --mode record)")
    parser.add_argument("--bias",       help="LONG or SHORT (for --mode record)")
    parser.add_argument("--outcome",    help="WIN, LOSS, or BE (for --mode record)")
    parser.add_argument("--setup-name", dest="setup_name", default="Manual",
                        help="Setup name (for --mode record, default: Manual)")
    parser.add_argument("--interval",   type=int, default=30,
                        help="Minutes between scans in --mode watch (default: 30)")
    # --mode open args
    parser.add_argument("--shares",     type=int,   help="Shares held (for --mode open)")
    parser.add_argument("--stop-price", type=float, dest="stop_price",
                        help="Stop loss price (for --mode open)")
    parser.add_argument("--target1",    type=float, help="T1 target price (for --mode open)")
    parser.add_argument("--target2",    type=float, help="T2 target price (for --mode open)")
    args = parser.parse_args()

    global ALLOW_SHORTS
    if args.no_shorts:
        ALLOW_SHORTS = False

    # Process any pending Telegram commands (/halt, /close, …) before acting —
    # a /halt sent overnight must take effect before this run submits anything.
    try:
        _tg_n = _process_telegram_commands()
        if _tg_n:
            print(f"  📱 Processed {_tg_n} Telegram command(s)")
    except Exception:
        pass

    # Lightweight, no-ticker modes exit before the universe/watchlist loading
    # below — merge-positions runs once per persist step (every scan) and has
    # no use for a ticker list; loading one would cost an unnecessary Yahoo
    # Finance round-trip on every single commit. watchdog is the same shape:
    # runs on its own frequent schedule and never touches a ticker universe.
    if args.mode == "merge-positions":
        # Name kept for backward compat with the existing workflow call —
        # now pre-merges every append-heavy state file that can be written
        # by more than one process (cron scanner + daemon), not just
        # positions. See merge_json_lists() for why: dman_scan_log.json
        # went a full trading day (2026-07-27) with zero new entries
        # despite 8+ real successful scans, because it's rewritten so
        # often (every cron scan AND every 10-min daemon scan) that a
        # whole-file conflict-resolution silently discarded whichever
        # side's entry lost the race, every single time.
        sync_positions_with_remote()
        sync_scan_log_with_remote()
        sync_win_rate_with_remote()
        sync_live_signals_with_remote()
        sync_alpaca_sync_state_with_remote()
        sync_news_log_with_remote()
        return
    if args.mode == "watchdog":
        run_watchdog()
        return

    if args.tickers:
        tickers = args.tickers
    elif args.universe == "all":
        tickers = build_scan_universe()
    else:
        # "curated" mode: load today's pre-built universe cache if available.
        # The 9 AM premarket briefing writes dman_universe_cache.json so the
        # 9:45 AM Gap & Hold scan gets full 500-ticker coverage without
        # spending 7 minutes rebuilding the universe at the gate.
        _cache_loaded = False
        try:
            if os.path.exists("dman_universe_cache.json"):
                with open("dman_universe_cache.json") as _ucf:
                    _cached = json.load(_ucf)
                _cache_date = _cached.get("date", "")
                _today_str  = datetime.today().strftime("%Y-%m-%d")
                if _cache_date == _today_str and _cached.get("tickers"):
                    tickers = _cached["tickers"]
                    print(f"  📦 Loaded pre-built universe: {len(tickers)} tickers "
                          f"(cached at 9 AM — full Gap & Hold coverage active)")
                    _cache_loaded = True
        except Exception as _ce:
            print(f"  ⚠️  Universe cache read failed ({_ce}) — using curated list")
        if not _cache_loaded:
            tickers = WATCHLIST
            print(f"  📋 Using curated watchlist: {len(tickers)} tickers "
                  f"(run premarket briefing first to enable full universe)")
        # Always inject today's live movers so the 9:45 AM scan sees real volume
        # even when the pre-market briefing cache is stale or missing.
        if ENABLE_DYNAMIC_SMALLCAP:
            try:
                _live_movers = fetch_dman_dynamic_tickers(max_tickers=30)
                if _live_movers:
                    _before = len(tickers)
                    tickers = list(dict.fromkeys(list(tickers) + _live_movers))
                    _added  = len(tickers) - _before
                    if _added:
                        print(f"  📈  +{_added} live movers added (Yahoo gainers/actives): "
                              f"{', '.join(_live_movers[:8])}{'...' if _added > 8 else ''}")
            except Exception:
                pass
        # Watchlist-independent earnings movers — see fetch_earnings_mover_tickers()
        # docstring: CRWV beat EPS +30.9% and gapped +15.7% (2026-08-12) and was
        # never once considered because it was never on WATCHLIST. This closes
        # that blind spot the same way the live-movers injection above closes it
        # for pure price/volume movers.
        if ENABLE_EARNINGS_MOVER_SCAN:
            try:
                _earn_movers = fetch_earnings_mover_tickers(max_tickers=15)
                if _earn_movers:
                    _before = len(tickers)
                    tickers = list(dict.fromkeys(list(tickers) + _earn_movers))
                    _added  = len(tickers) - _before
                    if _added:
                        print(f"  🚀  +{_added} earnings movers added (beat + real gap, "
                              f"off-watchlist): {', '.join(_earn_movers[:8])}"
                              f"{'...' if _added > 8 else ''}")
            except Exception:
                pass

    print("""
  ██████╗ ███╗   ███╗ █████╗ ███╗   ██╗
  ██╔══██╗████╗ ████║██╔══██╗████╗  ██║   🔥 PRO v3
  ██║  ██║██╔████╔██║███████║██╔██╗ ██║   18 filters
  ██║  ██║██║╚██╔╝██║██╔══██║██║╚██╗██║   Target: 80%+ WR
  ██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝  @ProfessorDman1
    """)

    if args.mode == "record":
        missing = [f for f, v in [("--ticker", args.ticker), ("--entry", args.entry),
                                   ("--exit-price", args.exit_price), ("--bias", args.bias),
                                   ("--outcome", args.outcome)] if v is None]
        if missing:
            print(f"  --mode record requires: {', '.join(missing)}")
            sys.exit(1)
        outcome = args.outcome.upper()
        if outcome not in ("WIN", "LOSS", "BE"):
            print("  --outcome must be WIN, LOSS, or BE")
            sys.exit(1)
        bias = args.bias.upper()
        pnl_pct = ((args.exit_price - args.entry) / args.entry * 100
                   if bias == "LONG"
                   else (args.entry - args.exit_price) / args.entry * 100)
        tracker = WinRateTracker()
        tracker.record(TradeRecord(
            ticker=args.ticker.upper(),
            date=datetime.today().strftime("%Y-%m-%d"),
            bias=bias,
            setup=args.setup_name,
            entry=args.entry,
            exit=args.exit_price,
            outcome=outcome,
            pnl_pct=round(pnl_pct, 2),
            score=0,
            is_live=True,   # --mode record is for logging real executed trades
        ))
        # Close position first so we can read the share count for account-level P&L
        closed = PositionTracker().close(args.ticker)
        shares_used = closed.shares if closed else (args.shares or 0)
        if shares_used > 0:
            dollar_pnl   = (args.exit_price - args.entry) * shares_used * (1 if bias == "LONG" else -1)
            acct_pnl_pct = dollar_pnl / ACCOUNT_SIZE * 100
            record_daily_pnl(acct_pnl_pct)   # account-level %, not stock price %
        print(f"\n  ✅ Recorded: {args.ticker.upper()} {bias} "
              f"${args.entry} → ${args.exit_price} | {outcome} ({pnl_pct:+.2f}%)\n")
        tracker.print_report()

    elif args.mode == "stocktwits":
        run_stocktwits_monitor()
        # Meta-watchdog belt-and-suspenders — see _check_and_heal_watchdog()'s
        # docstring for the 2026-08-17 incident this closes. StockTwits runs
        # far more frequently than the scanner, so this is the tighter of
        # the two redundant checks. Never let this block or fail the
        # primary stocktwits run.
        try:
            _check_and_heal_watchdog()
        except Exception:
            pass

    elif args.mode == "watchdog":
        run_watchdog()

    elif args.mode == "fallback-guard":
        run_fallback_guard()

    elif args.mode == "earnings-scan":
        # Cron-dispatched redundancy for the daemon's earnings_loop — the cron
        # scanner re-checks-out fresh code every run (hourly), unlike the
        # daemon which checks out once per session and doesn't hot-reload.
        # Idempotent: safe to run every hour, only sends a new offer once.
        run_earnings_spread_scan()
        expire_earnings_spread_offers()

    elif args.mode == "guard":
        # One-shot options guard — the daemon loops this every 60s
        _ga = run_options_guard()
        print(f"  🛡  Options guard: {len(_ga)} position(s) checked")

    elif args.mode == "regime":
        print("  Checking market regime...\n")
        regime = get_market_regime()
        r = regime["regime"]; s = regime["score"]
        d = regime["details"]
        print(f"  Market Regime : {r}  (score {s}/19)")
        for k, v in d.items():
            print(f"  {k:<18}: {v}")
        print(f"\n  Top sectors   : {', '.join(get_top_sectors())}")

    elif args.mode == "performance":
        tracker = WinRateTracker()
        tracker.print_report()

    elif args.mode == "backtest":
        run_pro_backtest(tickers, years=args.years,
                         min_score=args.score or MIN_CONFLUENCE)

    elif args.mode == "positions":
        PositionTracker().show()

    elif args.mode == "open":
        missing = [f for f, v in [("--ticker", args.ticker), ("--entry", args.entry),
                                   ("--bias", args.bias)] if v is None]
        if missing:
            print(f"  --mode open requires: {', '.join(missing)}")
            sys.exit(1)
        pt  = PositionTracker()
        pos = OpenPosition(
            ticker     = args.ticker.upper(),
            bias       = args.bias.upper(),
            setup      = args.setup_name or "Manual",
            entry      = args.entry,
            stop       = args.stop_price or 0.0,
            target1    = args.target1    or 0.0,
            target2    = args.target2    or 0.0,
            shares     = args.shares     or 1,
            entry_date = datetime.today().strftime("%Y-%m-%d"),
        )
        if pt.open(pos):
            print(f"\n  ✅ Position logged: {pos.ticker} {pos.bias} "
                  f"{pos.shares}sh @ ${pos.entry}  stop ${pos.stop}\n")

    elif args.mode == "rank":
        run_ranking(tickers, min_score=args.score)

    elif args.mode == "watch":
        interval = args.interval
        print(f"\n  👁  Watch mode — scanning every {interval} min during ET market hours\n"
              f"       Press Ctrl+C to stop.\n")
        try:
            while True:
                now          = datetime.now(ET)
                market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
                market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)

                if now.weekday() >= 5:          # weekend
                    print(f"  Weekend — sleeping 1h...")
                    time.sleep(3600)
                    continue

                if now < market_open:
                    wait = int((market_open - now).total_seconds())
                    print(f"  Pre-market — {now.strftime('%H:%M ET')} — "
                          f"waiting {wait//60}m for open...")
                    time.sleep(min(wait, 600))
                    continue

                if now > market_close:
                    print(f"  Market closed for today. Exiting watch mode.\n")
                    break

                print(f"\n  ── Scan at {now.strftime('%H:%M ET')} ──")
                _cache.clear()             # force fresh data
                _indicator_cache.clear()   # stale indicators computed off the old raw data must not survive the clear
                # Sync any fills that came in since last cycle
                n_fills = sync_alpaca_fills(WinRateTracker())
                if n_fills:
                    print(f"  📋 {n_fills} trade(s) auto-recorded from Alpaca fills")
                signals = run_pro_scanner(tickers, min_score=args.score, use_ai=args.ai)
                if not signals:
                    print("  No new A+ setups this scan.")
                else:
                    print(f"{'─'*68}\n  A+ SIGNALS\n{'─'*68}\n")
                    for s in signals:
                        print_pro_signal(s)
                    if args.submit:
                        _submit_signals_to_alpaca(signals)

                next_scan = now + timedelta(minutes=interval)
                if next_scan > market_close:
                    print("  Next scan would be after close — done for today.\n")
                    break
                wait_sec = max(0, int((next_scan - datetime.now(ET)).total_seconds()))
                print(f"  Next scan: {next_scan.strftime('%H:%M ET')}  "
                      f"(sleeping {interval}m)\n")
                time.sleep(wait_sec)
        except KeyboardInterrupt:
            print("\n  Watch mode stopped.\n")

    elif args.mode == "sync":
        tracker  = WinRateTracker()
        n_fills  = sync_alpaca_fills(tracker)
        print(f"\n  📋 Alpaca sync complete — {n_fills} new trade(s) recorded.\n")
        if n_fills:
            tracker.print_report()

    elif args.mode == "alpaca":
        # 1. Show account dashboard
        show_alpaca_account()
        # 2. Sync pending fills
        tracker = WinRateTracker()
        n_fills = sync_alpaca_fills(tracker)
        if n_fills:
            print(f"  📋 {n_fills} trade(s) auto-recorded\n")
            tracker.print_report()
        # 3. Scan + validate + submit
        signals = run_pro_scanner(tickers, min_score=args.score, use_ai=args.ai)
        if not signals:
            print("  No A+ setups — nothing to submit.\n")
        else:
            print(f"{'─'*68}\n  A+ SIGNALS\n{'─'*68}\n")
            for s in signals:
                print_pro_signal(s)
            if args.submit:
                _submit_signals_to_alpaca(signals)
            else:
                print("  [dry-run] Pass --submit to place live orders.\n")
            if args.export:
                fname = f"dman_signals_{datetime.today().strftime('%Y-%m-%d')}.json"
                _write_json_atomic(fname, [asdict(s) for s in signals], indent=2)
                print(f"  💾 Signals exported to {fname}\n")

    elif args.mode == "live-outcomes":
        print(f"\n{'═'*60}")
        print(f"  Resolving pending live signals...")
        print(f"{'─'*60}")
        n = resolve_live_outcomes(verbose=True)
        if not n:
            print("  Nothing new to resolve — all signals are still open or already resolved.")
        print(f"{'═'*60}\n")

    elif args.mode == "live-perf":
        print_live_performance()

    elif args.mode == "pnl":
        send_account_pnl_telegram(label="On-Demand")

    elif args.mode == "premarket":
        run_premarket_briefing()

    elif args.mode == "premarket-early":
        run_premarket_early_scan()

    elif args.mode == "momentum-watch":
        run_momentum_watch()

    elif args.mode == "watchlist":
        send_daily_watchlist()

    elif args.mode == "scan-log":
        print_scan_log()

    elif args.mode == "readiness":
        run_readiness_scan()

    elif args.mode == "scan":
        # Sync Alpaca fills first so PositionTracker is current before we submit
        if args.submit and ALPACA_API_KEY:
            _sync_tracker = WinRateTracker()
            n_fills = sync_alpaca_fills(_sync_tracker)
            if n_fills:
                print(f"  📋 {n_fills} trade(s) auto-recorded from Alpaca fills")

        signals = run_pro_scanner(tickers,
                                   min_score=args.score,
                                   use_ai=args.ai,
                                   universe_label=args.universe)
        if not signals:
            print("  No A+ setups today. The filters are working —")
            print("  D🔥man waits for the PERFECT setup, not just any setup.\n")
        else:
            print(f"{'─'*68}\n  A+ SIGNALS\n{'─'*68}\n")
            for s in signals:
                print_pro_signal(s)
            if args.submit:
                _submit_signals_to_alpaca(signals)
            if args.export:
                fname = f"dman_signals_{datetime.today().strftime('%Y-%m-%d')}.json"
                _write_json_atomic(fname, [asdict(s) for s in signals], indent=2)
                print(f"  💾 Signals exported to {fname}\n")

        # Safety EOD P&L — fires on the 3:30 PM scan as a belt-and-suspenders backup
        # in case the dedicated 4 PM cron is delayed past the market-hours gate.
        _eod_t = datetime.now(ET).hour * 100 + datetime.now(ET).minute
        if 1525 <= _eod_t <= 1600:
            print("\n  [EOD] Final scan window — sending P&L summary...")
            send_account_pnl_telegram("EOD")

        # Meta-watchdog belt-and-suspenders — see _check_and_heal_watchdog()'s
        # docstring for the 2026-08-17 incident this closes. Never let this
        # block or fail the actual scan.
        try:
            _check_and_heal_watchdog()
        except Exception:
            pass

        # Scan heartbeat — include regime context so user knows why it's quiet
        # get_market_regime() is cheap here because fetch_df() hits the in-memory cache
        t_str = datetime.now(ET).strftime("%I:%M %p")
        _hb_regime = get_market_regime()
        _hb_r  = _hb_regime.get("regime", "?")
        _hb_rs = _hb_regime.get("score", "?")
        _hb_meta  = _last_scan_meta
        _hb_rej   = _hb_meta.get("rejected", {})
        _hb_total = _hb_meta.get("tickers_total", 0)
        _hb_gate  = _hb_rej.get("hard_gate", 0)
        _hb_score = _hb_rej.get("low_score", 0)
        _hb_nm_list = _hb_meta.get("near_misses", [])
        _hb_bt_list = _hb_meta.get("b_tier", [])
        _hb_counts = (f"{_hb_total} scanned"
                      + (f" | {_hb_gate} gate-blocked" if _hb_gate else "")
                      + (f" | {_hb_score} score-short" if _hb_score else ""))
        _hb_nm_str = ""
        if _hb_nm_list:
            _hb_nm_str = "\nNear-miss: " + " | ".join(
                f"<b>{_t}</b> +{_g:.1f}% → {_b}" for _t, _g, _b in _hb_nm_list
            )
        _hb_bt_str = ""
        if _hb_bt_list:
            _bt_lines = []
            for _bt in _hb_bt_list:
                _bt_lines.append(
                    f"📋 <b>{_bt['ticker']}</b> +{_bt['gap']:.1f}%  RVOL {_bt['rvol']:.1f}x  "
                    f"({_bt['reason']})\n"
                    f"   Manual: entry ~${_bt['entry']}  stop ${_bt['stop']}  T1 ${_bt['t1']}"
                )
            _hb_bt_str = "\n\n<b>WATCH — manual entries available:</b>\n" + "\n".join(_bt_lines)
        if signals:
            send_telegram(
                f"🔍 <b>DMan</b> {t_str} — {len(signals)} signal(s) fired\n"
                f"Regime: {_hb_r} ({_hb_rs}/19)"
            )
        else:
            _fomc_bkout = any(abs((ev - date.today()).days) <= MACRO_BLACKOUT
                              for ev in _FOMC_DATES)
            _hb_hhmm = datetime.now(ET).hour * 100 + datetime.now(ET).minute
            if _fomc_bkout and 1425 <= _hb_hhmm <= 1500:
                # Post-FOMC 2:30 PM reaction wrap — fires once, covers the window right
                # after the 2 PM ET announcement when initial reaction has settled
                _lift_day2 = "soon"
                for _doff2 in range(1, 8):
                    _ck2 = date.today() + timedelta(days=_doff2)
                    if _ck2.weekday() >= 5 or _ck2 in _MARKET_HOLIDAYS:
                        continue
                    if all(abs((ev - _ck2).days) > MACRO_BLACKOUT for ev in _FOMC_DATES):
                        _lift_day2 = _ck2.strftime("%a %b %d")
                        break
                _rs_summary = ""
                try:
                    _spy_df2 = fetch_df("SPY")
                    if _spy_df2 is not None and len(_spy_df2) >= 1:
                        _spy_row2 = _spy_df2.iloc[-1]
                        _spy_day_chg = (float(_spy_row2["Close"]) - float(_spy_row2["Open"])) / float(_spy_row2["Open"]) * 100
                    else:
                        _spy_day_chg = 0.0
                    _rs_all2: list[tuple[str, float, float]] = []
                    for _rs_t2 in WATCHLIST[:35]:
                        try:
                            _rs_df2 = fetch_df(_rs_t2)
                            if _rs_df2 is None or len(_rs_df2) < 1:
                                continue
                            _rs_row3 = _rs_df2.iloc[-1]
                            _rs_chg2 = (float(_rs_row3["Close"]) - float(_rs_row3["Open"])) / float(_rs_row3["Open"]) * 100
                            _rs_all2.append((_rs_t2, _rs_chg2, _rs_chg2 - _spy_day_chg))
                        except Exception:
                            continue
                    _rs_all2.sort(key=lambda x: x[2], reverse=True)
                    _ldr = " | ".join(f"<b>{t}</b> {c:+.1f}%" for t, c, r in _rs_all2[:3]) or "—"
                    _lag = " | ".join(f"<b>{t}</b> {c:+.1f}%" for t, c, r in _rs_all2[-3:][::-1]) if len(_rs_all2) >= 3 else "—"
                    _rs_summary = (
                        f"\nSPY: {_spy_day_chg:+.1f}% today"
                        f"\nRS leaders → watch {_lift_day2}: {_ldr}"
                        f"\nRS laggards: {_lag}"
                    )
                except Exception:
                    pass
                send_telegram(
                    f"📊 <b>DMan</b> {t_str} — FOMC reaction wrap 🔒\n"
                    f"Blackout lifts: <b>{_lift_day2}</b>"
                    f"{_rs_summary}"
                )
            elif _fomc_bkout:
                send_telegram(
                    f"🔒 <b>DMan</b> {t_str} — FOMC blackout\n"
                    f"Regime: {_hb_r} ({_hb_rs}/19) | {_hb_counts}"
                    f"{_hb_nm_str}"
                )
            else:
                # Down-day context — warn when market is selling off so user
                # knows silence is intentional, not a scanner issue
                _spy_ctx = ""
                try:
                    _spy_hb = fetch_df("SPY")
                    if _spy_hb is not None and len(_spy_hb) >= 2:
                        _spy_c2  = float(_spy_hb.iloc[-1]["Close"].iloc[0]) if hasattr(_spy_hb.iloc[-1]["Close"], "iloc") else float(_spy_hb.iloc[-1]["Close"])
                        _spy_pc2 = float(_spy_hb.iloc[-2]["Close"].iloc[0]) if hasattr(_spy_hb.iloc[-2]["Close"], "iloc") else float(_spy_hb.iloc[-2]["Close"])
                        _spy_net2 = (_spy_c2 - _spy_pc2) / _spy_pc2 * 100
                        if _spy_net2 <= -1.0:
                            _xlk_net2 = 0.0
                            try:
                                _xlk_hb = fetch_df("XLK")
                                if _xlk_hb is not None and len(_xlk_hb) >= 2:
                                    _xlk_c2  = float(_xlk_hb.iloc[-1]["Close"].iloc[0]) if hasattr(_xlk_hb.iloc[-1]["Close"], "iloc") else float(_xlk_hb.iloc[-1]["Close"])
                                    _xlk_pc2 = float(_xlk_hb.iloc[-2]["Close"].iloc[0]) if hasattr(_xlk_hb.iloc[-2]["Close"], "iloc") else float(_xlk_hb.iloc[-2]["Close"])
                                    _xlk_net2 = (_xlk_c2 - _xlk_pc2) / _xlk_pc2 * 100
                            except Exception:
                                pass
                            _spy_ctx = f"\n📉 SPY {_spy_net2:+.1f}%"
                            if _xlk_net2 <= -1.5:
                                _spy_ctx += f" | XLK {_xlk_net2:+.1f}% — sector selloff, standing down on longs"
                            else:
                                _spy_ctx += " — market weak, no long setups"
                except Exception:
                    pass

                # End-of-day scan — 4 PM close only
                # • Recovery watch: names down >5% today → potential bounce candidates tomorrow
                # • Intraday momentum: names up >5% intraday AND held into close → watch for
                #   follow-through gap next morning (captures TSLA-style no-gap run days)
                _eod_watch = ""
                if 1550 <= _hb_hhmm <= 1615:
                    try:
                        _eod_losers:  list[tuple[str, float]] = []
                        _eod_runners: list[tuple[str, float]] = []
                        for _eod_t in WATCHLIST[:35]:
                            try:
                                _eod_df = fetch_df(_eod_t)
                                if _eod_df is None or len(_eod_df) < 2:
                                    continue
                                _eod_row = _eod_df.iloc[-1]
                                _eod_prv = _eod_df.iloc[-2]
                                _eod_c   = float(_eod_row["Close"].iloc[0]) if hasattr(_eod_row["Close"], "iloc") else float(_eod_row["Close"])
                                _eod_o   = float(_eod_row["Open"].iloc[0])  if hasattr(_eod_row["Open"],  "iloc") else float(_eod_row["Open"])
                                _eod_pc  = float(_eod_prv["Close"].iloc[0]) if hasattr(_eod_prv["Close"], "iloc") else float(_eod_prv["Close"])
                                _eod_net   = (_eod_c - _eod_pc) / _eod_pc * 100
                                _eod_intra = (_eod_c - _eod_o)  / _eod_o  * 100
                                if _eod_net <= -5.0:
                                    _eod_losers.append((_eod_t, _eod_net))
                                if _eod_intra >= 4.0 and _eod_net >= 3.0:
                                    _eod_runners.append((_eod_t, _eod_intra))
                            except Exception:
                                continue
                        _eod_losers.sort(key=lambda x: x[1])
                        _eod_runners.sort(key=lambda x: x[1], reverse=True)
                        if len(_eod_losers) >= 3:
                            _eod_watch += (
                                f"\n⚠️ <b>Sector flush</b> — {len(_eod_losers)} names down 5%+: "
                                f"watch for gap-down continuation or reversal bounce tomorrow"
                            )
                        if _eod_losers:
                            _eod_watch += "\n👀 Recovery watch tomorrow: " + " | ".join(
                                f"<b>{t}</b> {c:+.1f}%" for t, c in _eod_losers[:4]
                            )
                        if _eod_runners:
                            _eod_watch += "\n🔥 Intraday momentum — watch for gap tomorrow: " + " | ".join(
                                f"<b>{t}</b> {c:+.1f}%" for t, c in _eod_runners[:3]
                            )
                    except Exception:
                        pass

                send_telegram(
                    f"🔍 <b>DMan</b> {t_str} — quiet ✅\n"
                    f"Regime: {_hb_r} ({_hb_rs}/19) | {_hb_counts}"
                    f"{_hb_nm_str}"
                    f"{_hb_bt_str}"
                    f"{_spy_ctx}"
                    f"{_eod_watch}"
                )

                # 4 PM only: send live account P&L summary to Telegram
                if 1550 <= _hb_hhmm <= 1615 and not ALPACA_PAPER:
                    send_account_pnl_telegram(label="EOD")


if __name__ == "__main__":
    try:
        main()
    except Exception as _top_exc:
        import traceback as _tb
        print(_tb.format_exc(), file=sys.stderr)
        try:
            send_telegram(
                f"🚨 <b>DMan CRASHED</b>\n"
                f"<code>{str(_top_exc)[:200]}</code>\n"
                f"Check GitHub Actions logs for full traceback."
            )
        except Exception:
            pass
        sys.exit(1)  # exit 1 — show workflow red AND send detailed Telegram (no generic duplicate)