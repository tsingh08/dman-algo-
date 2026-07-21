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

import os, sys, json, time, math, re, argparse, warnings, traceback, requests
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
MIN_CONFLUENCE     = 75          # 0-100 score; raise to 80 for extra caution
SETUP_MIN_CONFLUENCE = {         # per-setup overrides for historically weak setups
    "Gap & Short":    82,
    "Vol Breakdown":  85,
    "MACD Bear":      82,
    "OS Bounce":      82,
    "Morning Runner": 85,
}

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
SMALLCAP_RISK_PCT      = 0.02     # 2% account risk per trade — aggressive sizing for small account
SMALLCAP_MAX_COST      = 2_500    # hard cap on position cost — micro-caps are high risk
SMALLCAP_MIN_SCORE          = 55   # minimum score for dynamically discovered tickers
DMAN_WATCHLIST_MIN_SCORE    = 45   # lower bar for personally curated watchlist tickers
SMALLCAP_T1_MULT       = 0.30     # T1 at +30% (Dman targets 50-200% — partial at 30%)
SMALLCAP_T2_MULT       = 0.75     # T2 at +75%
SMALLCAP_STOP_PCT      = 0.18     # 18% stop — penny stocks are volatile; wider needed
SMALLCAP_52WK_LOW_PCT  = 0.30     # "bottom chart" = within 30% of 52-week low

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
OPTIONS_RISK_PCT            = 0.12   # 12% of account per trade (~$360 at $3k) — aggressive ITM sizing
OPTIONS_DTE_MIN             = 5      # minimum 5 DTE — allows weekly for fast gap plays
OPTIONS_DTE_MAX             = 28     # max 4 weeks
OPTIONS_MAX_SPREAD_PCT      = 0.15   # skip contract if bid-ask spread > 15% of mid (was 20% of ask)
OPTIONS_MIN_UNDERLYING_VOL  = 5_000_000   # underlying must avg ≥5M shares/day (liquid options)
OPTIONS_ENABLE_PUTS         = True   # buy ITM puts on bearish/Bear Gap Hold signals
OPTIONS_DATA_FEED           = "opra"        # Alpaca Algo Trader Plus — real-time OPRA feed
STOCK_DATA_FEED             = "sip"         # Alpaca Algo Trader Plus — consolidated real-time SIP feed

# Dman's curated small-cap watch — always scanned regardless of dollar-volume threshold.
# These are tickers Dman actively calls on Twitter (ultra-low float, catalyst-driven).
# Finviz dynamic discovery supplements this list every scan via ENABLE_DYNAMIC_SMALLCAP.
DMAN_SMALLCAP_WATCHLIST = [
    # Core — original Dman calls (ultra-low float, sub-$20)
    "APVO",   # 1.25M float — conference catalyst Mar 2026, $10+ target
    "MASK",   # 1.13M float — low-float momentum play
    # UGRO removed Jul 13 2026 — no price data returned, likely suspended/halted
    "ONCO",   # 1.15M float — bounce candidate
    # FCHL removed Jul 15 2026 — avg vol 37K (too thin, spreads untradeable)
    "ARTL",   # 3.49M float — mentioned in 100-600% runner week
    "ELAB",   # 4.54M float — confirmed multi-day runner
    # Dman July 2026 tweet thread — GDC, ADTX, CDT, LNKS, CAST, SILO
    # IOTR removed Jul 13 2026 — Day 6 post-spike, 0.0x RVOL, -8.8% Mon; effectively dead momentum
    # GDC removed — price $0.02, below SMALLCAP_MIN_PRICE ($0.10); cannot generate signal
    # ADTX removed Jul 9 2026 — stock at $0.004, -25% on Jul 8, effectively dead
    # LNKS removed Jul 15 2026 — avg vol 45K (too thin, spreads untradeable)
    # SILO removed Jul 15 2026 — Day 4 fade played out, -24.3% 5d; momentum exhausted
    # FCHL removed Jul 15 2026 — avg vol 37K (too thin, spreads untradeable)
    "CDT",    # 200% swing runner — Dman low-float call
    "CAST",   # micro-cap catalyst — Dman watch
    # Broader low-float universe — Dman-style patterns
    "ATOS",   # perpetual short-squeeze candidate, micro-float
    "IMPP",   # shipping penny — repeated Dman-style RVOL spikes
    # HPNN removed — price $0.0008, permanently below SMALLCAP_MIN_PRICE ($0.10)
    "GFAI",   # AI penny stock — high retail interest
    "BFRI",   # biotech with low float — squeeze setup
    # AITX removed — price $0.009, permanently below SMALLCAP_MIN_PRICE ($0.10)
    "TRVI",   # low-float biotech — MACD curling plays
    # MRIN, CJET, ACST, ATNF removed Jul 10 2026 — delisted/no data
    # Dman Jul 2 2026 tweets — fake-offering bounce + penny OTC runners
    "LABT",   # biotech, Dman loaded <$3 on fake-offering scare, target $10-20
    "YHC",    # penny OTC, Dman loaded .04s → now $2.22 (+5500%), +5900% 5d; watch for VWAP reclaim on pullback
    "LIMN",   # +125% from 52-week low .09s (Jul 2 call) — now $0.10, at SMALLCAP_MIN_PRICE floor; watch only
]

# ── Options layer (Dman style: ITM calls on large-cap signals) ───────────────
# Dman buys ITM calls at support bottoms on large-caps (SPY, QQQ, TSLA, NVDA, PLTR).
# Advisory mode by default — alerts show exact contract, premium, Greeks, stop/target.
# Set OPTIONS_AUTO_EXECUTE = True to route orders through Alpaca paper/live.
ENABLE_OPTIONS          = True
OPTIONS_SETUPS          = {"Gap & Hold", "Morning Runner"}  # setups that also get options
OPTIONS_MIN_SCORE       = 80            # match adaptive min score threshold
OPTIONS_MIN_PRICE       = 10.0          # no options on sub-$10 stocks (illiquid/nonexistent)
OPTIONS_MAX_PRICE       = 500.0         # skip very expensive stocks (options too costly)
ADVISORY_OPTIONS_RISK_PCT = 0.25        # 25% of account — used by old advisory size_options_trade()
OPTIONS_MAX_PREMIUM_USD = 500           # cap at $500/trade — 1-2 contracts at $3k account
OPTIONS_TARGET_DTE      = 14            # target 2-week DTE — DMan gap plays resolve in 1-5 days
OPTIONS_MIN_DTE         = 5             # below this: theta burns too fast
OPTIONS_MAX_DTE         = 28            # beyond this: premium too expensive for gap plays
OPTIONS_ITM_TARGET_PCT  = 0.04          # target 4% ITM (≈ delta 0.70) — "ITM calls"
OPTIONS_STOP_LOSS_PCT   = 0.50          # exit if premium drops 50% (Dman's mental stop)
OPTIONS_PROFIT1_PCT     = 0.50          # take 50% profit at +50% gain
OPTIONS_CLOSE_DTE       = 7             # close or roll when DTE ≤ 7 (theta risk)
OPTIONS_AUTO_EXECUTE    = False         # True = place orders via Alpaca; False = advisory only
OPTIONS_SHORT_SETUPS    = {"Vol Breakdown", "EMA Breakdown", "Gap & Short"}  # → ITM puts

# Pre-event strangles (direction-neutral, buys both call + put before big catalysts)
STRANGLE_TICKERS    = ["SPY", "QQQ"]   # always liquid enough for two-legged plays
STRANGLE_OTM_PCT    = 0.04             # 4% OTM per leg (keeps premium reasonable)
STRANGLE_TARGET_DTE = 7                # weekly options — captures the move, limits theta
STRANGLE_MIN_DTE    = 5                # ≥5 DTE avoids same-week expiry (gamma/theta bleed)
STRANGLE_MAX_DTE    = 14
STRANGLE_RISK_PCT   = 0.01             # 1% of account per strangle event

MONTHLY_LOSS_LIMIT = 0.04          # halt for the month when down ≥4% of account
MONTHLY_PNL_FILE   = "dman_monthly_pnl.json"

# ── VIX-adjusted position sizing ─────────────────────────────────────────────
# Scales share count down proportionally when VIX exceeds baseline.
# VIX 20 (baseline) → 1.0x  |  VIX 30 → 0.67x  |  VIX 40 → 0.50x
# Never sizes UP when VIX is below baseline (floor at 1.0 — risk control only).
VIX_SIZE_BASE = 20.0   # baseline VIX; size is 1.0x at or below this level

# ── Sector ETF map — used for momentum confirmation in score_signal() ─────────
# When a stock's sector ETF is green on the day, Gap & Hold conviction is higher.
SECTOR_ETF: dict[str, str] = {
    "Technology":            "XLK",
    "Communication Services":"XLC",
    "Consumer Discretionary":"XLY",
    "Consumer Staples":      "XLP",
    "Energy":                "XLE",
    "Financials":            "XLF",
    "Health Care":           "XLV",
    "Industrials":           "XLI",
    "Materials":             "XLB",
    "Real Estate":           "XLRE",
    "Utilities":             "XLU",
}

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

# Telegram alerts (optional — set via env vars or hardcode)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Alpaca Paper Trading (optional — set via env vars or hardcode)
# Get keys at app.alpaca.markets → Paper Trading → API Keys
ALPACA_API_KEY    = os.getenv("APCA_API_KEY_ID",    "")   # standard Alpaca env var name
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY", "")
BENZINGA_API_KEY  = os.getenv("BENZINGA_API_KEY", "")     # Benzinga Basic — real-time news
ALPACA_PAPER      = False     # LIVE — real brokerage, real money
ENTRY_DRIFT_MAX   = 0.02      # reject signal if price drifted >2% from computed entry
ALPACA_SYNC_FILE   = "dman_alpaca_sync.json"
LAST_ALERTS_FILE   = "dman_last_alerts.json"
ALERT_COOLDOWN_MIN = 30          # suppress duplicate Telegram alert for same ticker within N min
LIVE_SIGNALS_FILE  = "dman_live_signals.json"   # pending live signals awaiting outcome
LIVE_OUTCOMES_FILE = "dman_live_outcomes.csv"    # ground-truth live trade log
SCAN_LOG_FILE      = "dman_scan_log.json"        # rolling log of each scan run (last 20)

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

TICKER_SECTOR = {
    # Mega-cap tech
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "META":"Technology","GOOGL":"Comm Services","AMZN":"Consumer Disc","TSLA":"Consumer Disc",
    "NFLX":"Comm Services","CRM":"Technology","SNOW":"Technology","PLTR":"Technology",
    "SMCI":"Technology","AVGO":"Technology",
    "AMAT":"Technology","MU":"Technology",
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
    """Return True if US market is currently open (9:30am–4:00pm ET, Mon–Fri)."""
    now = datetime.now(ET)
    if now.weekday() >= 5:
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
        all_syms = list(set(nasdaq_syms + other_syms) - set(WATCHLIST))
    except Exception as e:
        print(f"  [universe] Symbol fetch failed ({e}), using curated + extended list.", flush=True)
        # Fallback: curated watchlist + extended universe (RVOL filter still applied below)
        all_syms = list(set(EXTENDED_UNIVERSE) - set(WATCHLIST))

    print(f"  [universe] {len(all_syms):,} symbols → filtering by price/volume...", flush=True)

    # Batch-download 5-day snapshot to score RVOL cheaply
    active: list[tuple[str, float]] = []
    batch_size = 400
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]
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


def fetch_dman_dynamic_tickers(max_tickers: int = 60) -> list[str]:
    """
    Fetch today's actual movers from Yahoo Finance (day gainers + most actives).
    Replaces Finviz, which is consistently blocked in GitHub Actions.
    Returns tickers that are genuinely moving with volume RIGHT NOW — not a
    static screener but a live snapshot of what has institutional interest today.
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
            f"?formatted=false&lang=en-US&region=US&scrIds={scr_id}&count=25"
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
                # Price $2–$100, gap ≥ 2%, RVOL ≥ 1.5x, avg vol ≥ 100K
                if (sym and sym.isalpha() and 1 <= len(sym) <= 5
                        and 2.0 <= price <= 100.0
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


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

_cache: dict[str, pd.DataFrame] = {}

def fetch_df(ticker: str, period_days: int = 430,
             interval: str = "1d") -> Optional[pd.DataFrame]:
    """Download OHLCV data with in-memory caching to avoid redundant API calls."""
    key = f"{ticker}_{interval}"
    if key in _cache:
        return _cache[key]
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
    # ── Alpaca daily bars fallback (SIP feed — Algo Trader Plus) ─────────────
    try:
        if ALPACA_AVAILABLE:
            dc = get_alpaca_data_client()
            if dc is not None:
                from datetime import timezone as _tz
                _alp_req = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=TimeFrame.Day,
                    start=(datetime.today() - timedelta(days=period_days + 5)).replace(tzinfo=_tz.utc),
                    feed=STOCK_DATA_FEED,
                )
                _alp_resp = dc.get_stock_bars(_alp_req)
                _alp_bars = _alp_resp[ticker] if ticker in _alp_resp else []
                if len(_alp_bars) >= 20:
                    _records = [
                        {"Open": b.open, "High": b.high, "Low": b.low,
                         "Close": b.close, "Volume": b.volume, "Date": b.timestamp}
                        for b in _alp_bars
                    ]
                    _df_alp = pd.DataFrame(_records).set_index("Date")
                    _df_alp.index = pd.to_datetime(_df_alp.index, utc=True).tz_convert(None)
                    _cache[key] = _df_alp
                    print(f"  [fetch_df] {ticker} — Alpaca SIP fallback ({len(_df_alp)} bars)", file=sys.stderr)
                    return _df_alp
    except Exception:
        pass
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
        + _fomc_note
        + _friday_note
    )


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
        with open(LAST_ALERTS_FILE, "w") as f:
            json.dump(alerts, f)
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
        with open(LIVE_SIGNALS_FILE, "w") as f:
            json.dump(data, f, indent=2)
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

        # BE@1R: move stop to entry once 1R profit is reached (before T1)
        if not be1r_set and not t1_hit:
            if (is_long and H >= be1r_px) or (not is_long and L <= be1r_px):
                trail_stop = entry
                be1r_set   = True

        # Stop check
        stopped = (is_long and L <= trail_stop) or (not is_long and H >= trail_stop)
        if stopped:
            exit_px     = trail_stop
            exit_reason = "STOP(BE)" if be1r_set else "STOP"
            pnl_pct     = ((exit_px - entry) / entry * 100) if is_long else ((entry - exit_px) / entry * 100)
            return {"exit_date": exit_bar_date, "exit_px": round(exit_px, 2),
                    "exit_reason": exit_reason,
                    "outcome": "WIN" if pnl_pct >= 0 else "LOSS",
                    "pnl_pct": round(pnl_pct, 2), "hold_bars": hold}

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

    # Write CSV header if file doesn't exist
    write_header = not os.path.exists(LIVE_OUTCOMES_FILE)
    csv_rows: list[str] = []
    if write_header:
        csv_rows.append("ticker,setup,bias,entry_date,exit_date,entry,stop,"
                        "target1,target2,exit_px,exit_reason,pnl_pct,outcome,score,hold_bars")

    for p in pending:
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
    with open(LIVE_SIGNALS_FILE, "w") as f:
        json.dump(data, f, indent=2)

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


def _stocktwits_inject_tickers(tickers_to_add: list[str]) -> list[str]:
    """
    Insert new tickers into DMAN_SMALLCAP_WATCHLIST in dman_algo.py by
    finding the closing ] of the list and inserting before it.
    Returns the list of tickers actually written (skips already-present ones).
    """
    _algo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dman_algo.py")
    try:
        with open(_algo_path, encoding="utf-8") as _f:
            _src = _f.read()
    except Exception as _e:
        print(f"  ⚠️  Could not read dman_algo.py: {_e}")
        return []

    _today = date.today().isoformat()
    _added: list[str] = []
    _insertions = ""
    for _t in tickers_to_add:
        if f'"{_t}"' in _src:
            print(f"  📡 {_t} already in algo — skipping")
            continue
        _insertions += f'    "{_t}",   # StockTwits DMan call {_today}\n'
        _added.append(_t)

    if not _insertions:
        return []

    # Bracket-count to find the closing ] of DMAN_SMALLCAP_WATCHLIST
    _marker = "DMAN_SMALLCAP_WATCHLIST = ["
    _start = _src.find(_marker)
    if _start == -1:
        print("  ⚠️  DMAN_SMALLCAP_WATCHLIST not found in dman_algo.py")
        return []

    _depth = 0
    _i = _start + len(_marker) - 1  # position of the opening [
    while _i < len(_src):
        if _src[_i] == "[":
            _depth += 1
        elif _src[_i] == "]":
            _depth -= 1
            if _depth == 0:
                _new_src = _src[:_i] + _insertions + _src[_i:]
                try:
                    with open(_algo_path, "w", encoding="utf-8") as _f:
                        _f.write(_new_src)
                    return _added
                except Exception as _e:
                    print(f"  ⚠️  Could not write dman_algo.py: {_e}")
                    return []
        _i += 1

    print("  ⚠️  Could not locate end of DMAN_SMALLCAP_WATCHLIST")
    return []


def run_stocktwits_monitor() -> None:
    """
    Fetch ProfessorDman1's recent StockTwits messages.
    Auto-adds new ticker calls to DMAN_SMALLCAP_WATCHLIST in dman_algo.py.
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

    # Telegram — confirm what was added
    _lines = []
    for _c in _new_calls:
        _tag = "✅ added to scanner" if _c["ticker"] in _added else "already tracked"
        _lines.append(f"  <b>{_c['ticker']}</b> [{_tag}]\n  \"{_c['body']}\"")

    send_telegram(
        f"📡 <b>DMan StockTwits — {len(_new_calls)} call(s) detected</b>\n\n"
        + "\n\n".join(_lines)
        + (f"\n\n✅ {len(_added)} ticker(s) added to DMAN_SMALLCAP_WATCHLIST." if _added else "")
    )
    print(f"  📡 Added: {_added}  |  Already known: {[c['ticker'] for c in _new_calls if c['ticker'] not in _added]}")


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
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            articles = resp.json() if isinstance(resp.json(), list) else resp.json().get("result", [])
            for art in articles:
                title = art.get("title", "")
                if not title:
                    continue
                for tk_obj in art.get("tickers", []):
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
            timeout=10,
        )
        if resp.status_code != 200:
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


def _fetch_alpaca_news(tickers: list[str], hours_back: int = 18) -> dict[str, list[str]]:
    """
    Fetch recent news headlines for a list of tickers.
    Priority: Benzinga real-time API → Alpaca News API → yfinance.
    Returns {ticker: [headline, ...]}.
    """
    from datetime import timezone
    result: dict[str, list[str]] = {t: [] for t in tickers}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    # Primary: Benzinga real-time news (requires BENZINGA_API_KEY)
    if BENZINGA_API_KEY:
        bz = _fetch_benzinga_ticker_news(tickers, hours_back=hours_back)
        filled = sum(1 for v in bz.values() if v)
        print(f"  [news] Benzinga returned headlines for {filled}/{len(tickers)} tickers")
        # Merge — keep yfinance fallback only for tickers with no Benzinga results
        for t, headlines in bz.items():
            if headlines:
                result[t] = headlines
        # If Benzinga covered everything, return early
        if all(result[t] for t in tickers):
            return result

    # Secondary: Alpaca News API (free tier, delayed)
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
    print(f"  Scanning {len(DMAN_SMALLCAP_WATCHLIST)} small-cap names...\n")

    # Pull global context first — drives pre-market sizing and aggression
    print("  🌍 Global context...", flush=True)
    _early_ctx  = _fetch_global_context()
    _early_news = _fetch_breaking_news_rss(hours_back=8)

    news_map = _fetch_alpaca_news(DMAN_SMALLCAP_WATCHLIST, hours_back=20)

    mover_blocks:    list[tuple[float, str]] = []   # (abs_gap, telegram_block)
    news_alerts:     list[str] = []
    pm_auto_entries: list[dict] = []               # moon-shot auto-submit candidates

    for ticker in DMAN_SMALLCAP_WATCHLIST:
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
                entry_parts = ["  🎯 <b>ENTRY PLAN</b>"]
                if tier != "D":
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
                entry_block = "\n".join(entry_parts)

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
            try:
                _pm_acct  = _client.get_account()
                _pm_eq    = float(getattr(_pm_acct, "equity", 0) or 0)
                _pm_dt    = int(getattr(_pm_acct, "daytrade_count", 0) or 0)
                if _pm_eq < 25_000 and _pm_dt >= 3:
                    send_telegram(f"🚫 <b>PDT HALT (pre-market)</b>: {_pm_dt}/3 day trades used — no pre-market orders placed.")
                    pm_auto_entries.clear()
            except Exception as _pdt_pm:
                send_telegram(f"🚫 <b>PDT check failed (pre-market)</b>: {_pdt_pm} — skipping pre-market orders.")
                pm_auto_entries.clear()

            from alpaca.trading.requests import LimitOrderRequest as _LimReq
            from alpaca.trading.enums   import OrderSide as _Side, TimeInForce as _TIF
            _acct    = get_effective_account()
            _base_risk = _acct * SMALLCAP_RISK_PCT * MOONSHOT_RISK_MULT * _early_ctx["risk_mult"]
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
                    feed=STOCK_DATA_FEED,
                )
                _alp_resp = dc.get_stock_bars(_alp_req)
                _alp_bars = _alp_resp[ticker] if ticker in _alp_resp else []
                if len(_alp_bars) >= 3:
                    _records = [
                        {"Open": b.open, "High": b.high, "Low": b.low,
                         "Close": b.close, "Volume": b.volume, "Timestamp": b.timestamp}
                        for b in _alp_bars
                    ]
                    _df = pd.DataFrame(_records).set_index("Timestamp")
                    _df.index = pd.to_datetime(_df.index, utc=True)
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
                    # Options positions: track via underlying price + live Greek snapshot
                    if setup.startswith("Options Call "):
                        # Parse OCC symbol from setup: "Options Call SMCI260724C00027500 ($27.5C exp ...)"
                        parts = setup.split()
                        _occ  = parts[2] if len(parts) >= 3 else ""
                        _entry_prem = e   # premium paid per share
                        _stop_prem  = float(pos.get("stop",    _entry_prem * 0.5))
                        _t1_prem    = float(pos.get("target1", _entry_prem * 2.0))
                        _delta_entry= float(pos.get("atr",     0.45))  # stored in atr field
                        _ctrs       = max(1, int(pos.get("shares", 100)) // 100)
                        if not _occ:
                            continue
                        # Live option snapshot
                        _snap = _get_option_snapshot(_occ)
                        if not _snap:
                            options_alerts.append(
                                f"⚠️ <b>{t}</b> CALL {_occ}\n"
                                f"   Cannot fetch live quote — check position manually"
                            )
                            continue
                        _cur_prem  = _snap["mid"]   # display P&L at mid
                        _exit_prem = _snap.get("bid", _cur_prem)  # exit at bid (conservative)
                        _pnl_pct   = (_cur_prem - _entry_prem) / _entry_prem * 100 if _entry_prem > 0 else 0
                        _theta_now = _snap.get("theta", 0)
                        _delta_now = _snap.get("delta", _delta_entry)
                        _iv_now    = _snap.get("iv", 0)
                        _theta_pct = abs(_theta_now / _cur_prem * 100) if _cur_prem > 0 else 0
                        _t2_prem   = float(pos.get("target2", _entry_prem * 2.5))
                        # DTE estimate from OCC symbol (YY MM DD in positions 6-11 for 21-char OCC)
                        try:
                            _occ_exp = _occ[-15:-9] if len(_occ) >= 15 else ""
                            _exp_dt  = datetime.strptime(_occ_exp, "%y%m%d").date() if _occ_exp else None
                            _dte_now = (_exp_dt - date.today()).days if _exp_dt else 99
                        except Exception:
                            _dte_now = 99

                        # Determine alert type — deduped per day
                        _opt_t1k   = f"{t}_OPT_T1_{date.today().isoformat()}"
                        _opt_t2k   = f"{t}_OPT_T2_{date.today().isoformat()}"
                        _opt_stopk = f"{t}_OPT_STOP_{date.today().isoformat()}"
                        _opt_dtek  = f"{t}_OPT_DTE_{date.today().isoformat()}"
                        # Use bid for stop check — that's what we'd actually receive on exit
                        if _exit_prem <= _stop_prem:
                            _action = "🔴 EXIT — STOP HIT"
                            _msg = (f"Bid ${_exit_prem:.2f} ≤ stop ${_stop_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell to limit loss")
                            if not _is_alerted_today(_opt_stopk):
                                send_telegram(f"🔴 <b>OPTIONS STOP</b> — {t} CALL {_occ}\n{_msg}")
                                _mark_alerted(_opt_stopk)
                        elif _cur_prem >= _t2_prem:
                            _action = "🚀 T2 HIT — FULL EXIT"
                            _msg = (f"Premium ${_cur_prem:.2f} ≥ T2 ${_t2_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell all, full runner achieved")
                            if not _is_alerted_today(_opt_t2k):
                                send_telegram(f"🚀 <b>OPTIONS T2 HIT</b> — {t} CALL {_occ}\n{_msg}")
                                _mark_alerted(_opt_t2k)
                        elif _cur_prem >= _t1_prem:
                            _action = "🟢 T1 HIT — TAKE ½ PROFIT"
                            _msg = (f"Premium ${_cur_prem:.2f} ≥ T1 ${_t1_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell ½, raise stop to breakeven")
                            if not _is_alerted_today(_opt_t1k):
                                send_telegram(f"🟢 <b>OPTIONS T1 HIT</b> — {t} CALL {_occ}\n{_msg}")
                                _mark_alerted(_opt_t1k)
                        elif _dte_now <= OPTIONS_CLOSE_DTE:
                            _action = f"⏳ DTE ALERT — {_dte_now}d left, consider close"
                            _msg = (f"Only {_dte_now}d to expiry — theta burning fast. "
                                    f"P&L: {_pnl_pct:+.0f}%. Close or roll now.")
                            if not _is_alerted_today(_opt_dtek):
                                send_telegram(f"⏳ <b>DTE WARNING</b> — {t} CALL {_occ}\n{_msg}")
                                _mark_alerted(_opt_dtek)
                        elif _theta_pct > 5.0 and _pnl_pct < 10:
                            _action = "⏰ THETA ALERT — consider exit"
                            _msg = (f"Decaying {_theta_pct:.1f}%/day with only "
                                    f"{_pnl_pct:+.0f}% gain — time is working against you")
                        else:
                            _action = "✅ HOLDING"
                            _msg = f"P&L: <b>{_pnl_pct:+.0f}%</b>  θ decay {_theta_pct:.1f}%/day  DTE {_dte_now}d"

                        options_alerts.append(
                            f"{_action}  <b>{t}</b> CALL {_occ}\n"
                            f"   Prem: entry ${_entry_prem:.2f} → now ${_cur_prem:.2f} (bid ${_exit_prem:.2f})  "
                            f"({_pnl_pct:+.0f}%)  {_ctrs}ct × 100 = ${_cur_prem*_ctrs*100:.0f} mkt val\n"
                            f"   {_msg}\n"
                            f"   Δ {_delta_now:.2f}  θ {_theta_now:.3f}/d  IV {_iv_now*100:.0f}%  "
                            f"T1 ${_t1_prem:.2f}  T2 ${_t2_prem:.2f}  Stop ${_stop_prem:.2f}  DTE {_dte_now}d"
                        )
                        continue   # options positions don't go into equity active_plays

                    elif setup.startswith("Options Put "):
                        # Parse OCC symbol from setup: "Options Put SMCI260724P00027500 ($27.5P exp ...)"
                        parts = setup.split()
                        _occ  = parts[2] if len(parts) >= 3 else ""
                        _entry_prem = e
                        _stop_prem  = float(pos.get("stop",    _entry_prem * 0.5))
                        _t1_prem    = float(pos.get("target1", _entry_prem * 2.0))
                        _delta_entry= float(pos.get("atr",     0.45))
                        _ctrs       = max(1, int(pos.get("shares", 100)) // 100)
                        if not _occ:
                            continue
                        _snap = _get_option_snapshot(_occ)
                        if not _snap:
                            options_alerts.append(
                                f"⚠️ <b>{t}</b> PUT {_occ}\n"
                                f"   Cannot fetch live quote — check position manually"
                            )
                            continue
                        _cur_prem  = _snap["mid"]
                        _exit_prem = _snap.get("bid", _cur_prem)
                        _pnl_pct   = (_cur_prem - _entry_prem) / _entry_prem * 100 if _entry_prem > 0 else 0
                        _theta_now = _snap.get("theta", 0)
                        _delta_now = _snap.get("delta", _delta_entry)
                        _iv_now    = _snap.get("iv", 0)
                        _theta_pct = abs(_theta_now / _cur_prem * 100) if _cur_prem > 0 else 0
                        _t2_prem   = float(pos.get("target2", _entry_prem * 2.5))
                        try:
                            _occ_exp = _occ[-15:-9] if len(_occ) >= 15 else ""
                            _exp_dt  = datetime.strptime(_occ_exp, "%y%m%d").date() if _occ_exp else None
                            _dte_now = (_exp_dt - date.today()).days if _exp_dt else 99
                        except Exception:
                            _dte_now = 99

                        _opt_t1k   = f"{t}_PUT_T1_{date.today().isoformat()}"
                        _opt_t2k   = f"{t}_PUT_T2_{date.today().isoformat()}"
                        _opt_stopk = f"{t}_PUT_STOP_{date.today().isoformat()}"
                        _opt_dtek  = f"{t}_PUT_DTE_{date.today().isoformat()}"
                        if _exit_prem <= _stop_prem:
                            _action = "🔴 EXIT — STOP HIT"
                            _msg = (f"Bid ${_exit_prem:.2f} ≤ stop ${_stop_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell to limit loss")
                            if not _is_alerted_today(_opt_stopk):
                                send_telegram(f"🔴 <b>OPTIONS STOP</b> — {t} PUT {_occ}\n{_msg}")
                                _mark_alerted(_opt_stopk)
                        elif _cur_prem >= _t2_prem:
                            _action = "🚀 T2 HIT — FULL EXIT"
                            _msg = (f"Premium ${_cur_prem:.2f} ≥ T2 ${_t2_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell all, full runner achieved")
                            if not _is_alerted_today(_opt_t2k):
                                send_telegram(f"🚀 <b>OPTIONS T2 HIT</b> — {t} PUT {_occ}\n{_msg}")
                                _mark_alerted(_opt_t2k)
                        elif _cur_prem >= _t1_prem:
                            _action = "🟢 T1 HIT — TAKE ½ PROFIT"
                            _msg = (f"Premium ${_cur_prem:.2f} ≥ T1 ${_t1_prem:.2f} "
                                    f"({_pnl_pct:+.0f}%) — sell ½, raise stop to breakeven")
                            if not _is_alerted_today(_opt_t1k):
                                send_telegram(f"🟢 <b>OPTIONS T1 HIT</b> — {t} PUT {_occ}\n{_msg}")
                                _mark_alerted(_opt_t1k)
                        elif _dte_now <= OPTIONS_CLOSE_DTE:
                            _action = f"⏳ DTE ALERT — {_dte_now}d left, consider close"
                            _msg = (f"Only {_dte_now}d to expiry. P&L: {_pnl_pct:+.0f}%. Close or roll now.")
                            if not _is_alerted_today(_opt_dtek):
                                send_telegram(f"⏳ <b>DTE WARNING</b> — {t} PUT {_occ}\n{_msg}")
                                _mark_alerted(_opt_dtek)
                        elif _theta_pct > 5.0 and _pnl_pct < 10:
                            _action = "⏰ THETA ALERT — consider exit"
                            _msg = (f"Decaying {_theta_pct:.1f}%/day with only "
                                    f"{_pnl_pct:+.0f}% gain — time working against you")
                        else:
                            _action = "🐻 HOLDING"
                            _msg = f"P&L: <b>{_pnl_pct:+.0f}%</b>  θ decay {_theta_pct:.1f}%/day  DTE {_dte_now}d"

                        options_alerts.append(
                            f"{_action}  <b>{t}</b> PUT {_occ}\n"
                            f"   Prem: entry ${_entry_prem:.2f} → now ${_cur_prem:.2f} (bid ${_exit_prem:.2f})  "
                            f"({_pnl_pct:+.0f}%)  {_ctrs}ct × 100 = ${_cur_prem*_ctrs*100:.0f} mkt val\n"
                            f"   {_msg}\n"
                            f"   Δ {_delta_now:.2f}  θ {_theta_now:.3f}/d  IV {_iv_now*100:.0f}%  "
                            f"T1 ${_t1_prem:.2f}  T2 ${_t2_prem:.2f}  Stop ${_stop_prem:.2f}  DTE {_dte_now}d"
                        )
                        continue

                    if e > 0:
                        # Automatic T1/T2 exit alerts for equity positions — fires once
                        # per target per day (dedup prevents repeat on every 30-min cycle)
                        _cur_eq = get_live_price(t)
                        if _cur_eq is not None:
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
                                send_telegram(
                                    f"✅ <b>T1 HIT</b> — {t} LONG\n"
                                    f"Entry ${e} → Now ${_cur_eq:.2f} (+{_pnle}%)  T1 ${_t1e}\n"
                                    f"<b>Sell 50% HERE.</b>  Move stop to ${e} (breakeven).  "
                                    f"Let rest ride to T2 ${_t2e}."
                                )
                                _mark_alerted(_t1k)
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

    # ── 6.5. Upcoming earnings on watchlist ───────────────────────────
    print("  [6.5/7] Scanning earnings calendar...")  # noqa: label kept for operator readability
    earnings_section = ""
    try:
        upcoming = get_upcoming_earnings(WATCHLIST, days_ahead=5)
        if upcoming:
            lines = []
            for item in upcoming:
                d = item["days_away"]
                da = item["days_away"]
                if da == 0:
                    tag = "today"
                elif da == 1:
                    tag = "tomorrow"
                else:
                    tag = f"in {da}d"
                in_bl = "" if da > EARNINGS_BLACKOUT else " 🚫 BLACKOUT"
                lines.append(f"  {item['ticker']:<6} — {item['earn_date'].strftime('%a %b %d')} ({tag}){in_bl}")
            earnings_section = (
                "\n\n📆 <b>EARNINGS THIS WEEK (watchlist)</b>\n"
                + "\n".join(lines)
                + "\n<i>Tickers marked BLACKOUT are skipped until 5d after report</i>"
            )
        else:
            earnings_section = "\n\n📆 <b>EARNINGS</b>: No watchlist tickers report in next 5 days."
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

        result.update({
            "regime":     regime,
            "score":      score,
            "spy_trend":  spy_above_50,
            "adx_strong": adx_strong,
            "vix_ok":     vix_mid,
            "vix_shock":  vix_shock,
            "defensive_rotation": defensive_rotation,
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
    """
    global _sector_cache, _sector_cache_ts

    if (_sector_cache is not None and _sector_cache_ts is not None
            and (datetime.now() - _sector_cache_ts).total_seconds() < 14400):
        return _sector_cache

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
    _sector_cache    = ranked[:n]
    _sector_cache_ts = datetime.now()
    return _sector_cache


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

def check_earnings_safe(ticker: str) -> tuple[bool, int]:
    """
    Blocks signals if earnings are within EARNINGS_BLACKOUT days.
    Uses yfinance calendar data.
    Returns (safe, score 0-5).
    """
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return True, 5   # no data = assume safe

        # calendar has 'Earnings Date' column
        if "Earnings Date" in cal.columns:
            earn_dates = pd.to_datetime(cal["Earnings Date"]).dropna()
        elif isinstance(cal.index, pd.DatetimeIndex):
            earn_dates = cal.index
        else:
            return True, 5

        today = pd.Timestamp.today().normalize()
        for ed in earn_dates:
            days_away = (ed.normalize() - today).days
            if -1 <= days_away <= EARNINGS_BLACKOUT:
                return False, 0   # too close to earnings
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
    today = pd.Timestamp.today().normalize()
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None or cal.empty:
                continue
            if "Earnings Date" in cal.columns:
                earn_dates = pd.to_datetime(cal["Earnings Date"]).dropna()
            elif isinstance(cal.index, pd.DatetimeIndex):
                earn_dates = cal.index
            else:
                continue
            for ed in earn_dates:
                days_away = (ed.normalize() - today).days
                if 0 <= days_away <= days_ahead:
                    results.append({
                        "ticker":     ticker,
                        "earn_date":  ed.date(),
                        "days_away":  days_away,
                    })
                    break  # one entry per ticker
        except Exception:
            continue
    results.sort(key=lambda x: x["days_away"])
    return results


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
_PPI_DATES: set[date] = {
    # 2026 — estimated as CPI+1 business day (verify from bls.gov/schedule)
    date(2026,  1, 15), date(2026,  2, 12), date(2026,  3, 12),
    date(2026,  4, 11), date(2026,  5, 14), date(2026,  6, 11),
    date(2026,  7, 15), date(2026,  8, 13), date(2026,  9, 10),
    date(2026, 10, 15), date(2026, 11, 13), date(2026, 12, 11),
    # 2027 — estimated; verify from bls.gov each December
    date(2027,  1, 14), date(2027,  2, 11), date(2027,  3, 11),
    date(2027,  4, 10), date(2027,  5, 13), date(2027,  6, 10),
    date(2027,  7, 15), date(2027,  8, 12), date(2027,  9,  9),
    date(2027, 10, 14), date(2027, 11, 11), date(2027, 12, 10),
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

        # NFP / CPI / PPI: pre-market 8:30 AM release — block only before 10:00 AM ET
        for ev in (_nfp_dates() | _CPI_DATES | _PPI_DATES):
            days_away = (ev - today).days
            if days_away == 0 and now_et.hour < 10:
                return False, 0

        return True, 5
    except Exception:
        return True, 5


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
    """
    try:
        atr   = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0
        if atr == 0:
            return raw_stop

        window = df.iloc[-15:]

        if bias == "LONG":
            swing_low  = float(window["Low"].min())
            atr_stop   = entry - 1.5 * atr
            # Use the HIGHER of raw and swing-low stop (tighter but real)
            best_stop  = max(swing_low - 0.3*atr, raw_stop, atr_stop)
            # Never tighten so much it invalidates 2R
            max_risk   = (entry - raw_stop) * 1.3
            best_stop  = max(best_stop, entry - max_risk)
            return round(best_stop, 2)

        else:  # SHORT
            swing_high = float(window["High"].max())
            atr_stop   = entry + 1.5 * atr
            best_stop  = min(swing_high + 0.3*atr, raw_stop, atr_stop)
            max_risk   = (raw_stop - entry) * 1.3
            best_stop  = min(best_stop, entry + max_risk)
            return round(best_stop, 2)

    except Exception:
        return raw_stop


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 15 — FILTER 11: AI SETUP SCORER (Claude API)
# ═══════════════════════════════════════════════════════════════════════════

def ai_score_signal(signal: ProSignal, regime: dict) -> int:
    """
    Send signal details to Claude and get a 1-10 confidence score.
    Returns integer score (0 if API unavailable or key not set).

    Prompt is concise to minimise tokens; macro env added for context-aware scoring.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "":
        return 0   # AI scoring skipped — no key

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
                "model":      "claude-sonnet-4-6",
                "max_tokens": 10,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        text = resp.json()["content"][0]["text"].strip()
        m = re.search(r'\b(10|[1-9])\b', text)
        score = int(m.group(1)) if m else 5
        return max(1, min(10, score))
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 16 — FILTER 12: KELLY CRITERION POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════

def get_todays_loss() -> float:
    """
    Return today's realized P&L as a signed percentage of account size.
    Negative = loss. Returns 0.0 if no trades recorded today.
    """
    try:
        with open(DAILY_PNL_FILE) as f:
            data = json.load(f)
        if data.get("date") != date.today().isoformat():
            return 0.0
        return float(data.get("pnl_pct", 0.0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.0


def record_daily_pnl(pnl_pct: float) -> None:
    """Add pnl_pct (signed %) to today's running P&L file."""
    today_str = date.today().isoformat()
    try:
        with open(DAILY_PNL_FILE) as f:
            data = json.load(f)
        if data.get("date") != today_str:
            data = {"date": today_str, "pnl_pct": 0.0}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"date": today_str, "pnl_pct": 0.0}
    data["pnl_pct"] = round(data["pnl_pct"] + pnl_pct, 4)
    with open(DAILY_PNL_FILE, "w") as f:
        json.dump(data, f)


def get_this_month_loss() -> float:
    """Return this calendar month's realized P&L as a signed % of account. 0.0 if none."""
    try:
        with open(MONTHLY_PNL_FILE) as f:
            data = json.load(f)
        if data.get("month") != date.today().strftime("%Y-%m"):
            return 0.0
        return float(data.get("pnl_pct", 0.0))
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.0


def record_monthly_pnl(pnl_pct: float) -> None:
    """Add pnl_pct (signed %) to this month's running P&L file."""
    month_str = date.today().strftime("%Y-%m")
    try:
        with open(MONTHLY_PNL_FILE) as f:
            data = json.load(f)
        if data.get("month") != month_str:
            data = {"month": month_str, "pnl_pct": 0.0}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"month": month_str, "pnl_pct": 0.0}
    data["pnl_pct"] = round(data["pnl_pct"] + pnl_pct, 4)
    with open(MONTHLY_PNL_FILE, "w") as f:
        json.dump(data, f)


def get_effective_account() -> float:
    """ACCOUNT_SIZE reduced by today's realized losses from the circuit-breaker file."""
    pnl_pct = get_todays_loss()   # signed %; 0 or negative on loss days
    adjusted = ACCOUNT_SIZE * (1 + pnl_pct / 100)
    return max(adjusted, ACCOUNT_SIZE * 0.5)   # floor at 50% of configured size


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
                         vix: float = VIX_SIZE_BASE) -> ProSignal:
    """Apply Kelly-optimal, beta-adjusted, VIX-scaled position sizing to a signal."""
    kf  = kelly_fraction(win_rate, avg_win_r)

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
    adj_shares  = max(1, int(raw_shares * vix_mult))

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
        with open(self.filepath, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def record(self, trade: TradeRecord):
        self.records.append(trade)
        self._save()

    def rolling_stats(self, n: int = 50) -> dict:
        """Stats over the last n closed trades."""
        recent = self.records[-n:] if len(self.records) >= n else self.records
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
        Falls back to aggregate stats if fewer than 5 trades exist for this setup.
        """
        recent = [r for r in self.records[-200:] if r.setup == setup][-n:]
        if len(recent) < 5:
            return self.rolling_stats()   # not enough data — use aggregate
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
        with open(self.filepath, "w") as f:
            json.dump([asdict(p) for p in self.positions], f, indent=2)

    def open(self, pos: OpenPosition) -> bool:
        self.positions = [p for p in self.positions if p.ticker != pos.ticker]
        if len(self.positions) >= MAX_POSITIONS:
            print(f"  ⚠️  MAX_POSITIONS ({MAX_POSITIONS}) reached — cannot open {pos.ticker}.")
            return False
        self.positions.append(pos)
        self._save()
        return True

    def close(self, ticker: str) -> Optional[OpenPosition]:
        found = next((p for p in self.positions if p.ticker == ticker.upper()), None)
        self.positions = [p for p in self.positions if p.ticker != ticker.upper()]
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


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 18 — CORE SIGNAL DETECTORS (from v2, inline)
# ═══════════════════════════════════════════════════════════════════════════

RVOL_MIN       = 1.3
RVOL_MIN_SHORT = 1.2

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
            if (d1_gap >= 4.0 and d1_held >= 0.6 and d2_above
                    and d2_rvol and d2_rsi
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
            _bg_gap_pct  = (float(p["Close"]) - c) / float(p["Close"]) * 100   # gap down %
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
    _sector_etf  = SECTOR_ETF.get(_sector_name, "")
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

    # 5. Earnings safety (5 pts)
    earn_ok, earn_score = check_earnings_safe(signal.ticker)
    signal.earnings_ok  = earn_ok
    breakdown["Earnings"] = earn_score

    # 5b. Macro calendar safety (5 pts) — FOMC / NFP blackout
    macro_ok, macro_score = check_macro_safe()
    signal.macro_ok = macro_ok
    breakdown["Macro"] = macro_score

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

    # Recalculate targets after stop adjustment
    if signal.bias == "LONG":
        signal.target1 = round(signal.entry + 2.0*(signal.entry - signal.stop), 2)
        signal.target2 = round(signal.entry + 3.0*(signal.entry - signal.stop), 2)
    else:
        signal.target1 = round(signal.entry - 2.0*(signal.stop - signal.entry), 2)
        signal.target2 = round(signal.entry - 3.0*(signal.stop - signal.entry), 2)
    rps = abs(signal.entry - signal.stop)
    if rps > 0:
        signal.rr = round(abs(signal.target1 - signal.entry) / rps, 2)

    # Kelly sizing — prefer setup-specific win rate when enough data exists
    setup_s  = tracker.setup_stats(signal.setup)
    _vix_now = float(regime.get("details", {}).get("VIX", VIX_SIZE_BASE))
    signal = size_position_kelly(
        signal,
        account   = get_effective_account(),
        win_rate  = max(0.5, setup_s["win_rate"]),
        avg_win_r = max(1.5, setup_s["avg_win_r"]),
        vix       = _vix_now,
    )

    # Final weighted score: confluence (70%) + AI score (30%)
    signal.final_score = round(signal.confluence_score * 0.70 +
                                signal.ai_score * 10 * 0.30, 1)

    return signal


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.5 — SMALL-CAP / LOW FLOAT CATALYST MODULE (Dman style)
# ═══════════════════════════════════════════════════════════════════════════

def _is_recent_reverse_split(ticker: str, days: int = 45) -> bool:
    """
    Return True if ticker had a reverse split within `days` days.
    Post-RS plays are Professor Dman's #1 category: float collapses, shorts get trapped.
    """
    try:
        splits = yf.Ticker(ticker).splits
        if splits is None or splits.empty:
            return False
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        recent = splits[splits.index >= cutoff]
        return bool((recent < 1.0).any())   # ratio < 1.0 = reverse split
    except Exception:
        return False


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
            near_bottom = True  # data unavailable — don't block

        # Must be either near 52wk low OR have very high RVOL (catalyst-driven spike)
        if not (near_bottom or rvol >= 5.0):
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
        try:
            today_gap_pct = (float(df.iloc[-1]["Open"]) - float(df.iloc[-2]["Close"])) / \
                             float(df.iloc[-2]["Close"]) * 100
        except Exception:
            today_gap_pct = 0.0

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

        # Position sizing — Moon Shot tier gets 3x allocation; normal uses 0.5% risk
        acct      = get_effective_account()
        risk_pct  = SMALLCAP_RISK_PCT * (MOONSHOT_RISK_MULT if moonshot else 1.0)
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
        f"{sig.reason}"
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


def fetch_option_chain(ticker: str, side: str = "calls",
                       min_dte: int = None,
                       max_dte: int = None,
                       target_dte: int = None) -> Optional[tuple[pd.DataFrame, str, int]]:
    """
    Fetch the calls or puts chain for the expiration closest to target DTE.
    Returns (chain_df, expiration_str, dte) or None on failure.
    side: "calls" | "puts"
    """
    if not ENABLE_OPTIONS:
        return None
    _min_dte    = min_dte    or OPTIONS_MIN_DTE
    _max_dte    = max_dte    or OPTIONS_MAX_DTE
    _target_dte = target_dte or OPTIONS_TARGET_DTE
    try:
        tk          = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None

        today = date.today()
        def _dte(exp: str) -> int:
            return (date.fromisoformat(exp) - today).days

        candidates = [(e, _dte(e)) for e in expirations
                      if _min_dte <= _dte(e) <= _max_dte]
        if not candidates:
            return None

        best_exp, best_dte = min(candidates,
                                 key=lambda x: abs(x[1] - _target_dte))
        chain = tk.option_chain(best_exp)
        df = chain.calls.copy() if side == "calls" else chain.puts.copy()
        df["_expiration"] = best_exp
        df["_dte"]        = best_dte
        return df, best_exp, best_dte
    except Exception:
        return None


def select_itm_call(calls: pd.DataFrame, current_price: float,
                    expiration: str, dte: int, ticker: str = "") -> Optional[dict]:
    """
    Select the best ITM call using Dman's criteria:
    - Target 4% ITM (OPTIONS_ITM_TARGET_PCT) ≈ delta 0.70
    - Minimum liquidity (volume or open interest > 0)
    - Use bid/ask midpoint as premium
    Returns a contract dict or None.
    """
    try:
        target_strike = current_price * (1 - OPTIONS_ITM_TARGET_PCT)
        # Search ±5% around target strike (still ITM)
        itm = calls[
            (calls["strike"] <= current_price) &
            (calls["strike"] >= current_price * 0.90)
        ].copy()

        if itm.empty:
            return None

        # Require some liquidity
        itm = itm[(itm["volume"].fillna(0) > 0) | (itm["openInterest"].fillna(0) > 10)]
        if itm.empty:
            return None

        # Rank: closest to target strike, then best volume
        itm["_dist"]  = (itm["strike"] - target_strike).abs()
        itm["_vol"]   = itm["volume"].fillna(0).astype(float)
        itm["_oi"]    = itm["openInterest"].fillna(0).astype(float)
        itm = itm.sort_values(["_dist", "_vol"], ascending=[True, False])

        best    = itm.iloc[0]
        bid     = float(best.get("bid", 0) or 0)
        ask     = float(best.get("ask", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        # Skip contracts where bid-ask spread > 15% of mid (illiquid — bad fill)
        mid = (bid + ask) / 2
        if (ask - bid) / mid > 0.15:
            return None

        premium      = round(mid, 2)
        iv           = float(best.get("impliedVolatility", 0) or 0)
        moneyness    = round((current_price - float(best["strike"])) / current_price * 100, 1)
        # Approximate delta: 0.50 ATM → 0.70 at 4% ITM → 0.95 at 9%+ ITM
        est_delta    = round(min(0.95, max(0.50, 0.50 + moneyness * 0.05)), 2)
        occ_symbol   = _build_occ_symbol(ticker=ticker,
                                         strike=float(best["strike"]),
                                         expiration=expiration)

        return {
            "strike":       float(best["strike"]),
            "expiration":   expiration,
            "dte":          dte,
            "premium":      premium,
            "bid":          round(bid, 2),
            "ask":          round(ask, 2),
            "iv_pct":       round(iv * 100, 1),
            "est_delta":    est_delta,
            "moneyness_pct": moneyness,
            "volume":       int(best.get("volume",       0) or 0),
            "open_interest":int(best.get("openInterest", 0) or 0),
            "occ_symbol":   occ_symbol,
        }
    except Exception:
        return None


def size_options_trade(premium: float) -> int:
    """
    Number of contracts to buy based on OPTIONS_RISK_PCT and OPTIONS_MAX_PREMIUM_USD.
    Each contract = 100 shares. Cost = premium * 100 * contracts.
    """
    acct        = get_effective_account()
    budget      = min(acct * ADVISORY_OPTIONS_RISK_PCT, OPTIONS_MAX_PREMIUM_USD)
    cost_per    = premium * 100            # 1 contract = 100 shares
    if cost_per <= 0:
        return 0
    contracts   = int(budget / cost_per)
    return contracts  # 0 = premium too expensive for budget; caller skips


def size_strangle_trade(total_premium: float) -> int:
    """
    Number of strangles (each = 1 call + 1 put contract) to buy.
    Allocates STRANGLE_RISK_PCT of account. Returns 0 if too expensive.
    """
    acct     = get_effective_account()
    budget   = acct * STRANGLE_RISK_PCT
    cost_per = total_premium * 100   # 1 strangle = 100 shares per leg
    if cost_per <= 0:
        return 0
    return max(0, int(budget / cost_per))


def format_options_telegram(sig: "ProSignal", contract: dict, contracts: int) -> str:
    """Telegram alert for an ITM call suggestion alongside a large-cap stock signal."""
    exp_fmt  = date.fromisoformat(contract["expiration"]).strftime("%b %d, %Y")
    total    = round(contract["premium"] * 100 * contracts, 2)
    stop_px  = round(contract["premium"] * (1 - OPTIONS_STOP_LOSS_PCT), 2)
    target1  = round(contract["premium"] * (1 + OPTIONS_PROFIT1_PCT),  2)
    exec_tag = "✅ AUTO-SUBMITTED to Alpaca" if OPTIONS_AUTO_EXECUTE else "📋 Advisory — enter manually"
    stock_plan = (
        f"\n📌 <b>Stock plan</b>: Entry ${sig.entry:.2f}  "
        f"Stop ${sig.stop:.2f}  T1 ${sig.target1:.2f}  T2 ${sig.target2:.2f}"
    )
    return (
        f"📊 <b>DMan OPTIONS Alert</b> — {sig.ticker}\n"
        f"🟢 CALL  <b>{sig.ticker}</b>  "
        f"Strike ${contract['strike']:.0f}  Exp {exp_fmt}  ({contract['dte']}d)\n"
        f"Premium: <b>${contract['premium']}</b>  "
        f"(bid ${contract['bid']} / ask ${contract['ask']})\n"
        f"Moneyness: {contract['moneyness_pct']:.1f}% ITM  "
        f"|  IV: {contract['iv_pct']:.0f}%  "
        f"|  δ ≈ {contract['est_delta']}\n"
        f"Contracts: <b>{contracts}</b>  |  Total cost: ${total:,.0f}  "
        f"|  Vol: {contract['volume']:,}  OI: {contract['open_interest']:,}\n"
        f"Stop: exit if premium ≤ ${stop_px} (-50%)\n"
        f"Target 1: exit half at ${target1} (+50%)  |  let rest run\n"
        f"Close/roll when DTE ≤ {OPTIONS_CLOSE_DTE} days\n"
        f"Symbol: <code>{contract['occ_symbol']}</code>\n"
        f"{exec_tag}"
        f"{stock_plan}\n"
        f"Based on: {sig.setup} score {sig.confluence_score}/100"
    )


def generate_options_signal(sig: "ProSignal") -> None:
    """
    If ENABLE_OPTIONS and this signal qualifies:
    - LONG signals  → ITM call (Gap & Hold, Morning Runner)
    - SHORT signals → ITM put  (Vol Breakdown, EMA Breakdown, Gap & Short)
    Optionally auto-submits to Alpaca if OPTIONS_AUTO_EXECUTE = True.
    """
    if not ENABLE_OPTIONS:
        return
    if sig.confluence_score < OPTIONS_MIN_SCORE:
        return
    if sig.entry <= OPTIONS_MIN_PRICE:
        return
    if sig.entry > OPTIONS_MAX_PRICE:
        return

    is_long  = sig.bias == "LONG"  and sig.setup in OPTIONS_SETUPS
    is_short = sig.bias == "SHORT" and sig.setup in OPTIONS_SHORT_SETUPS
    if not is_long and not is_short:
        return

    if is_long:
        result = fetch_option_chain(sig.ticker, side="calls")
        if result is None:
            return
        chain_df, expiration, dte = result
        contract = select_itm_call(chain_df, sig.entry, expiration, dte, sig.ticker)
        if contract is None:
            return
        contracts = size_options_trade(contract["premium"])
        if contracts == 0:
            return
        total_cost = contract["premium"] * 100 * contracts
        print(f"  📊 OPTIONS: {sig.ticker} ${contract['strike']:.0f}C "
              f"exp {expiration} ({dte}d)  "
              f"premium ${contract['premium']}  "
              f"{contracts} contract(s)  cost ${total_cost:,.0f}")
        send_telegram(format_options_telegram(sig, contract, contracts))
        if OPTIONS_AUTO_EXECUTE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            _submit_options_alpaca(sig.ticker, contract["strike"], expiration,
                                   contracts, contract["ask"])

    else:  # SHORT → ITM put
        result = fetch_option_chain(sig.ticker, side="puts")
        if result is None:
            return
        chain_df, expiration, dte = result
        contract = select_itm_put(chain_df, sig.entry, expiration, dte, sig.ticker)
        if contract is None:
            return
        contracts = size_options_trade(contract["premium"])
        if contracts == 0:
            return
        total_cost = contract["premium"] * 100 * contracts
        print(f"  📊 OPTIONS: {sig.ticker} ${contract['strike']:.0f}P "
              f"exp {expiration} ({dte}d)  "
              f"premium ${contract['premium']}  "
              f"{contracts} contract(s)  cost ${total_cost:,.0f}")
        send_telegram(format_put_telegram(sig, contract, contracts))
        if OPTIONS_AUTO_EXECUTE and ALPACA_API_KEY and ALPACA_SECRET_KEY:
            _submit_options_alpaca(sig.ticker, contract["strike"], expiration,
                                   contracts, contract["ask"], call_put="P")


def _submit_options_alpaca(ticker: str, strike: float, expiration: str,
                            contracts: int, limit_price: float,
                            call_put: str = "C") -> None:
    """
    Place a limit buy order for an ITM call or put via Alpaca Options API.
    Uses the ask price as limit (slightly aggressive fill, avoids missing the trade).
    Only called when OPTIONS_AUTO_EXECUTE = True.
    """
    try:
        occ = _build_occ_symbol(ticker, strike, expiration, call_put)
        if not occ:
            print(f"  ⚠️  OPTIONS: could not build OCC symbol for {ticker}")
            return
        base = ("https://paper-api.alpaca.markets" if ALPACA_PAPER
                else "https://api.alpaca.markets")
        headers = {
            "APCA-API-KEY-ID":     ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type":        "application/json",
        }
        payload = {
            "symbol":        occ,
            "qty":           str(contracts),
            "side":          "buy",
            "type":          "limit",
            "time_in_force": "day",
            "limit_price":   str(round(limit_price, 2)),
        }
        resp = requests.post(f"{base}/v2/orders", json=payload,
                             headers=headers, timeout=10)
        if resp.status_code in (200, 201):
            print(f"  ✅ OPTIONS order submitted: {occ} x{contracts} @ ${limit_price}")
        else:
            print(f"  ❌ OPTIONS order failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        print(f"  ❌ OPTIONS Alpaca error: {e}")


# ── ITM Puts (SHORT signals) ──────────────────────────────────────────────

def select_itm_put(puts: pd.DataFrame, current_price: float,
                   expiration: str, dte: int, ticker: str = "") -> Optional[dict]:
    """
    Select best ITM put: target strike 4% above current price (puts with strike > price are ITM).
    Mirrors select_itm_call() logic exactly, just for the other side.
    """
    try:
        target_strike = current_price * (1 + OPTIONS_ITM_TARGET_PCT)
        itm = puts[
            (puts["strike"] >= current_price) &
            (puts["strike"] <= current_price * 1.10)
        ].copy()
        if itm.empty:
            return None
        itm = itm[(itm["volume"].fillna(0) > 0) | (itm["openInterest"].fillna(0) > 10)]
        if itm.empty:
            return None
        itm["_dist"] = (itm["strike"] - target_strike).abs()
        itm["_vol"]  = itm["volume"].fillna(0).astype(float)
        itm = itm.sort_values(["_dist", "_vol"], ascending=[True, False])
        best = itm.iloc[0]
        bid  = float(best.get("bid", 0) or 0)
        ask  = float(best.get("ask", 0) or 0)
        if bid <= 0 or ask <= 0:
            return None
        mid = (bid + ask) / 2
        if (ask - bid) / mid > 0.15:
            return None
        premium   = round(mid, 2)
        iv        = float(best.get("impliedVolatility", 0) or 0)
        moneyness = round((float(best["strike"]) - current_price) / current_price * 100, 1)
        # Approximate delta: -0.50 ATM → -0.70 at 4% ITM → -0.95 at 9%+ ITM (more negative = deeper ITM)
        est_delta = round(max(-0.95, min(-0.50, -(0.50 + moneyness * 0.05))), 2)
        occ_sym   = _build_occ_symbol(ticker, float(best["strike"]), expiration, "P")
        return {
            "strike":        float(best["strike"]),
            "expiration":    expiration,
            "dte":           dte,
            "premium":       premium,
            "bid":           round(bid, 2),
            "ask":           round(ask, 2),
            "iv_pct":        round(iv * 100, 1),
            "est_delta":     est_delta,
            "moneyness_pct": moneyness,
            "volume":        int(best.get("volume",       0) or 0),
            "open_interest": int(best.get("openInterest", 0) or 0),
            "occ_symbol":    occ_sym,
        }
    except Exception:
        return None


def format_put_telegram(sig: "ProSignal", contract: dict, contracts: int) -> str:
    """Telegram alert for an ITM put alongside a short signal."""
    exp_fmt  = date.fromisoformat(contract["expiration"]).strftime("%b %d, %Y")
    total    = round(contract["premium"] * 100 * contracts, 2)
    stop_px  = round(contract["premium"] * (1 - OPTIONS_STOP_LOSS_PCT), 2)
    target1  = round(contract["premium"] * (1 + OPTIONS_PROFIT1_PCT),  2)
    exec_tag = "✅ AUTO-SUBMITTED to Alpaca" if OPTIONS_AUTO_EXECUTE else "📋 Advisory — enter manually"
    stock_plan = (
        f"\n📌 <b>Stock plan</b>: Entry ${sig.entry:.2f}  "
        f"Stop ${sig.stop:.2f}  T1 ${sig.target1:.2f}  T2 ${sig.target2:.2f}"
    )
    return (
        f"📊 <b>DMan OPTIONS Alert</b> — {sig.ticker}\n"
        f"🔴 PUT  <b>{sig.ticker}</b>  "
        f"Strike ${contract['strike']:.0f}  Exp {exp_fmt}  ({contract['dte']}d)\n"
        f"Premium: <b>${contract['premium']}</b>  "
        f"(bid ${contract['bid']} / ask ${contract['ask']})\n"
        f"Moneyness: {contract['moneyness_pct']:.1f}% ITM  "
        f"|  IV: {contract['iv_pct']:.0f}%  "
        f"|  δ ≈ {contract['est_delta']}\n"
        f"Contracts: <b>{contracts}</b>  |  Total cost: ${total:,.0f}  "
        f"|  Vol: {contract['volume']:,}  OI: {contract['open_interest']:,}\n"
        f"Stop: exit if premium ≤ ${stop_px} (-50%)\n"
        f"Target 1: exit half at ${target1} (+50%)  |  let rest run\n"
        f"Close/roll when DTE ≤ {OPTIONS_CLOSE_DTE} days\n"
        f"Symbol: <code>{contract['occ_symbol']}</code>\n"
        f"{exec_tag}"
        f"{stock_plan}\n"
        f"Based on: {sig.setup} score {sig.confluence_score}/100"
    )


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


def _check_open_position_risk(regime: dict) -> None:
    """Read pending live signals, fetch current prices, alert if within 2% of stop.
    Also alerts on orphan Alpaca positions that have no stop coverage in our tracker."""
    # ── Orphan position check ──────────────────────────────────────────────
    # Any Alpaca open position not in dman_live_signals.json has no automated
    # stop management — flag immediately so manual action can be taken.
    try:
        _client = get_alpaca_client()
        if _client is not None:
            _alp_positions = {p.symbol: p for p in _client.get_all_positions()}
            # Build tracked set from BOTH sources:
            # 1. dman_live_signals.json — signals logged by run_pro_scanner()
            # 2. dman_positions.json   — positions logged by PositionTracker (covers pre-market fills)
            _tracked_tickers: set[str] = set()
            try:
                with open(LIVE_SIGNALS_FILE) as _f:
                    _pf = json.load(_f)
                _tracked_tickers |= {p.get("ticker") for p in _pf.get("pending", [])}
            except Exception:
                pass
            try:
                _pt_pos = PositionTracker().positions
                _tracked_tickers |= {p.ticker for p in _pt_pos}
            except Exception:
                pass
            _orphans = [sym for sym in _alp_positions if sym not in _tracked_tickers]
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
                send_telegram(_msg)
                print(f"  ⚠  {len(_orphans)} orphan position(s) — sent alert: {_orphans}")
    except Exception as _oe:
        print(f"  ⚠  Orphan check failed: {_oe}")

    try:
        with open(LIVE_SIGNALS_FILE, "r") as f:
            data = json.load(f)
        pending = data.get("pending", [])
    except Exception:
        return

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
        send_telegram(header + "\n\n".join(alerts))
        print(f"  ⚠  Risk alert sent for {len(alerts)} position(s).")
    else:
        print(f"  ✅ All {len(pending)} open position(s) outside stop zone.")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 19.5 — SCAN RESULT LOG
# ═══════════════════════════════════════════════════════════════════════════

def _append_scan_log(entry: dict, max_entries: int = 20) -> None:
    """Append one scan result to the rolling scan log (keeps last max_entries)."""
    log: list[dict] = []
    if os.path.exists(SCAN_LOG_FILE):
        try:
            with open(SCAN_LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(entry)
    log = log[-max_entries:]  # keep rolling window
    with open(SCAN_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


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

        sig_icon = "🟢" if n_signals > 0 else "❌"
        sig_str  = (f"{n_signals} signal(s): {', '.join(sig_ticks)}"
                    if sig_ticks else f"{n_signals} signals")
        budget_str = "  ⏱ TIME BUDGET HIT" if budget_hit else ""

        print(f"  {ts}  |  {regime}({rscore}/19)  VIX {vix:.1f}  "
              f"min={min_sc}  [{universe}]")
        print(f"    {sig_icon} {n_tickers} tickers → {sig_str}{budget_str}")
        print(f"    Rejected: {rej_none} no-gap  {rej_gate} gate  {rej_score} low-score")
        print(f"{'─'*W}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 20 — PRO SCANNER
# ═══════════════════════════════════════════════════════════════════════════

def run_pro_scanner(tickers: list[str] = WATCHLIST,
                    min_score: int = None,
                    use_ai: bool = False,
                    universe_label: str = "curated") -> list[ProSignal]:
    """
    Full pro-grade scanner with all 18 filters applied.
    Only returns signals that pass ALL hard gates AND score >= min_score.
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

    # Pre-fetch news for all tickers in one batch (much faster than per-ticker)
    print(f"  [1.5/2] Pre-fetching news catalysts (last 4h)...", end=" ", flush=True)
    _scan_news_map: dict[str, list] = {}
    try:
        _scan_news_map = _fetch_alpaca_news(list(tickers), hours_back=4)
        _news_count = sum(1 for v in _scan_news_map.values() if v)
        print(f"{_news_count}/{len(tickers)} tickers have recent news")
    except Exception as _ne:
        print(f"error ({str(_ne)[:60]})")

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
        df = compute_indicators(raw.copy())
        df.dropna(subset=["EMA50","RSI","MACD","ATR"], inplace=True)
        if len(df) < 10:
            rejected_counts["no_signal"] += 1
            continue

        # Raw signal detection (v2 logic)
        sig = _raw_signals(df, ticker)
        if sig is None:
            rejected_counts["no_signal"] += 1
            continue

        # Tag news catalyst before scoring (adds +5 pts in score_signal)
        sig.news_boost = bool(_scan_news_map.get(ticker))

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

        # Optional AI scoring
        if use_ai and ANTHROPIC_API_KEY:
            sig.ai_score = ai_score_signal(sig, regime)
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
        if _is_duplicate_alert(sig.ticker):
            sys.stdout.write(f"       (dup suppressed — {sig.ticker} alerted <{ALERT_COOLDOWN_MIN}m ago)\n")
        else:
            send_telegram(format_signal_telegram(sig, regime))
            _save_last_alert(sig.ticker)
            _log_live_signal(sig)   # record for live outcome tracking
            generate_options_signal(sig)   # ITM call advisory if eligible

    # ── Small-cap / Low Float Catalyst pass ────────────────────────────────
    # Second pass over the same universe using Dman's micro-cap criteria.
    # Separate risk rules: 0.5% per trade, max $2,500 cost, lower score bar.
    if ENABLE_SMALLCAP:
        sc_rejected = 0
        sc_found    = 0
        # Dynamic Finviz discovery: low-float (<5M), price <$20, vol >500k
        _finviz_tickers: list[str] = []
        if ENABLE_DYNAMIC_SMALLCAP:
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
            df = compute_indicators(df.copy()) if "MACD" not in df.columns else df
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
            if _is_duplicate_alert(ticker):
                sys.stdout.write(f"       (dup suppressed)\n")
            else:
                send_telegram(format_smallcap_telegram(sc_sig, fl_m, sh_pct,
                                                       insider_pct, post_rs))
                _save_last_alert(ticker)
                _log_live_signal(sc_sig)
        if sc_found or sc_rejected:
            print(f"  🔥  Small-cap pass: {sc_found} signal(s), {sc_rejected} rejected")

    signals.sort(key=lambda s: s.confluence_score, reverse=True)

    # Portfolio heat cap: admit signals until total open risk hits the limit.
    # Seed with EXISTING open positions so prior-scan risk is counted.
    heat_capped: list[ProSignal] = []
    eff_account = get_effective_account()
    total_risk_pct = 0.0
    try:
        _heat_client = get_alpaca_client()
        if _heat_client is not None and eff_account > 0:
            for _hp in _heat_client.get_all_positions():
                # Use (avg_entry_price - stop_price) × qty as risk, not full market_value.
                # Alpaca doesn't expose stop_price on positions, so we approximate risk as
                # 2% of account per existing position (matches SMALLCAP_RISK_PCT).
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

    # Sector concentration cap: max 2 signals per sector to avoid overweighting
    sector_counts: dict[str, int] = {}
    concentrated: list[ProSignal] = []
    for sig in signals:
        sec   = TICKER_SECTOR.get(sig.ticker, "")
        count = sector_counts.get(sec, 0)
        if not sec or count < 2:
            concentrated.append(sig)
            sector_counts[sec] = count + 1
        else:
            sys.stdout.write(
                f"  📊 Sector cap: {sig.ticker} skipped "
                f"({sec} already has {count} signal(s))\n"
            )
    signals = concentrated

    print(f"\n{'─'*68}")
    print(f"  ✅  {len(signals)} A+ setup(s) passed all filters")
    print(f"  ❌  Rejected: {rejected_counts['no_signal']} no signal, "
          f"{rejected_counts['hard_gate']} hard gate, "
          f"{rejected_counts['low_score']} low score")
    print(f"  🌡  Portfolio heat used: {total_risk_pct*100:.1f}% / {PORTFOLIO_HEAT_LIMIT*100:.0f}%")
    print(f"{'─'*68}\n")

    # Persist scan result to rolling log
    try:
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
            df = compute_indicators(df.copy()) if "MACD" not in df.columns else df

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
            req   = StockLatestQuoteRequest(symbol_or_symbols=ticker)
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
    with open(ALPACA_SYNC_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _get_pdt_status() -> dict:
    """
    Query Alpaca for current PDT day-trade count.
    Returns dict with keys: used, remaining, swing_mode, equity.
    swing_mode=True when equity < $25k AND remaining <= 1 (save last trade for emergencies).
    Falls back to safe defaults (swing_mode=False, remaining=3) if Alpaca is unavailable.
    """
    try:
        _client = get_alpaca_client()
        if _client is None:
            return {"used": 0, "remaining": 3, "swing_mode": False, "equity": 0.0}
        _acct     = _client.get_account()
        _equity   = float(getattr(_acct, "equity", 0) or 0)
        _dt_count = int(getattr(_acct, "daytrade_count", 0) or 0)
        if _equity >= 25_000:
            return {"used": _dt_count, "remaining": 99, "swing_mode": False, "equity": _equity}
        _remaining = max(0, 3 - _dt_count)
        _swing     = _remaining <= 1   # at 1 or 0 remaining: go swing to protect the budget
        return {"used": _dt_count, "remaining": _remaining, "swing_mode": _swing, "equity": _equity}
    except Exception:
        return {"used": 0, "remaining": 3, "swing_mode": False, "equity": 0.0}


def submit_alpaca_trade(signal: ProSignal) -> Optional[str]:
    """
    Place a bracket order on Alpaca (paper or live).
    Day trade mode  (signal.swing_mode=False):
      Entry  — DAY limit order, full bracket (entry + stop + T1 take-profit OCO)
    Swing trade mode (signal.swing_mode=True, PDT budget ≤ 1):
      Entry  — GTC limit order, OTO with stop only (no T1 TP — momentum-watch manages exit)
      This ensures the position is held overnight and does NOT consume a day-trade count.

    Returns Alpaca order ID on success, None on failure.
    """
    client = get_alpaca_client()
    if client is None:
        print("  ⚠️  Alpaca unavailable — check ALPACA_API_KEY / ALPACA_SECRET_KEY.")
        return None
    if signal.shares < 1:
        print(f"  ⚠️  {signal.ticker}: 0 shares computed — skipping submission.")
        return None

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
                stop_loss     = StopLossRequest(stop_price=stop_px,
                                                limit_price=round(max(stop_px * 0.85, 0.01), 2)),
            ))
            oid = str(order.id)
            print(f"  📤 [{label}] 🔄 SWING {signal.ticker} {signal.bias} {signal.shares}sh  "
                  f"GTC limit=${limit_px}  stop=${stop_px}  (no T1 — held overnight)  id={oid[:8]}…")
        else:
            # Normal day trade — full bracket with T1 take-profit
            order = client.submit_order(LimitOrderRequest(
                symbol        = signal.ticker,
                qty           = signal.shares,
                side          = side,
                limit_price   = limit_px,
                time_in_force = TimeInForce.DAY,
                order_class   = OrderClass.BRACKET,
                take_profit   = TakeProfitRequest(limit_price=target_px),
                stop_loss     = StopLossRequest(stop_price=stop_px,
                                                limit_price=round(max(stop_px * 0.85, 0.01), 2)),
            ))
            oid = str(order.id)
            print(f"  📤 [{label}] {signal.ticker} {signal.bias} {signal.shares}sh  "
                  f"limit=${limit_px}  stop=${stop_px}  T1=${target_px}  id={oid[:8]}…")
        return oid
    except Exception as exc:
        print(f"  ❌ Alpaca order failed ({signal.ticker}): {exc}")
        return None


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
        # Options positions: Alpaca reports the OCC symbol (e.g. "SMCI260724C00027500"),
        # not the underlying. Extract OCC from setup string for correct lookup.
        _occ_sym = None
        if pos.setup.startswith("Options Call ") or pos.setup.startswith("Options Put "):
            _sp = pos.setup.split()
            _occ_sym = _sp[2] if len(_sp) >= 3 else None

        _alp_sym = _occ_sym if _occ_sym else ticker
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

        for order in orders:
            oid = str(order.id)
            if oid in recorded_ids:
                continue
            if str(order.status) != "filled":
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
            if _occ_sym:
                pnl_pct    = (fill_px - pos.entry) / pos.entry * 100 if pos.entry > 0 else 0
                dollar_pnl = (fill_px - pos.entry) * qty
            else:
                pnl_pct    = ((fill_px - pos.entry) / pos.entry * 100 if is_lo
                               else (pos.entry - fill_px) / pos.entry * 100)
                dollar_pnl = (fill_px - pos.entry) * qty * (1 if is_lo else -1)
            outcome    = ("WIN" if pnl_pct > 0.1 else
                          "LOSS" if pnl_pct < -0.1 else "BE")
            acct_pct   = dollar_pnl / ACCOUNT_SIZE * 100

            fill_date = (order.filled_at.strftime("%Y-%m-%d")
                         if getattr(order, "filled_at", None) else
                         datetime.today().strftime("%Y-%m-%d"))

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
            ))
            record_daily_pnl(acct_pct)
            pt.close(ticker)

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
            # Inner loop found no valid closing fill. Entry order may have been
            # cancelled or expired (limit never filled at open) — remove the ghost.
            # Options always buy-to-open; equity: BUY for LONG, SELL for SHORT.
            _entry_side = OrderSide.BUY if (is_lo or _occ_sym) else OrderSide.SELL
            for _eo in orders:
                if (_eo.side == _entry_side and
                        str(_eo.status).lower() in ("cancelled", "expired", "replaced")):
                    pt.close(ticker)
                    _ghost_lbl = _occ_sym if _occ_sym else ticker
                    print(f"  🗑️  Ghost cleared: {_ghost_lbl} — entry limit {_eo.status}, never filled")
                    send_telegram(
                        f"🗑️ <b>Ghost cleared</b>: {_ghost_lbl} — entry limit order "
                        f"{str(_eo.status)} (never filled). Removed from position tracker."
                    )
                    break

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
        last_eq    = float(acct.last_equity)
        cash       = float(acct.cash)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Options engine — Greeks-aware contract selection + L2 context + submission
# ═══════════════════════════════════════════════════════════════════════════

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
            params={"symbols": occ_symbol, "feed": OPTIONS_DATA_FEED},
            timeout=8,
        )
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
                strike_price_gte=strike - 0.01,
                strike_price_lte=strike + 0.01,
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
            # Hard filter: reject OTM (delta < 0.40) and wide-spread contracts
            if snap["delta"] < 0.40 or snap["spread_pct"] > OPTIONS_MAX_SPREAD_PCT:
                continue
            score, reason = _score_option_contract(snap, current_price)
            print(f"    {occ}  Δ{snap['delta']:.2f}  θ{snap['theta']:.3f}/d  "
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
                strike_price_gte=strike - 0.01,
                strike_price_lte=strike + 0.01,
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
            delta_abs = abs(snap.get("delta", 0))
            if delta_abs < 0.40 or snap["spread_pct"] > OPTIONS_MAX_SPREAD_PCT:
                continue
            score, reason = _score_option_contract(snap, current_price)
            print(f"    PUT {occ}  |Δ|{delta_abs:.2f}  θ{snap['theta']:.3f}/d  "
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

        # Re-anchor bracket to live price — preserves original R multiple but
        # uses the actual price we'll fill at, so stop and target are correct.
        _orig_risk = round(sig.entry - sig.stop, 4)  # risk-per-share from signal detection
        if _orig_risk > 0:
            if sig.bias == "LONG":
                _live_entry = round(cur * 1.001, 2)    # 0.1% buffer → improves fill odds
                sig.entry   = _live_entry
                sig.stop    = round(_live_entry - _orig_risk, 2)
                sig.target1 = round(_live_entry + 2.5 * _orig_risk, 2)
                sig.target2 = round(_live_entry + 4.0 * _orig_risk, 2)
            else:  # SHORT
                _live_entry = round(cur * 0.999, 2)
                sig.entry   = _live_entry
                sig.stop    = round(_live_entry + _orig_risk, 2)
                sig.target1 = round(_live_entry - 2.5 * _orig_risk, 2)
                sig.target2 = round(_live_entry - 4.0 * _orig_risk, 2)

        # ── Options branch: calls (LONG) or puts (SHORT) on WATCHLIST tickers ─
        _use_options = ENABLE_OPTIONS_TRADING and sig.ticker in WATCHLIST and sig.bias == "LONG"
        _use_puts    = OPTIONS_ENABLE_PUTS     and sig.ticker in WATCHLIST and sig.bias == "SHORT"
        _opt_contract: dict | None = None
        oid: str | None = None

        if _use_options:
            _opt_client = get_alpaca_client()
            if _opt_client:
                _opt_risk = round(get_effective_account() * OPTIONS_RISK_PCT * _risk_off_mult, 2)
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
                _opt_risk = round(get_effective_account() * OPTIONS_RISK_PCT * _risk_off_mult, 2)
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
                print(f"  ⏭️  {sig.ticker} SHORT skipped — ALLOW_SHORTS=False and not in options universe")
                continue
            oid = submit_alpaca_trade(sig)

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
                _greek_str = (
                    f"Δ {_delta:.2f}  Γ {_gamma:.4f}  θ {_theta:.3f}/d  "
                    f"ν {_vega:.3f}  IV {_iv*100:.0f}%  OI {_oi:,}"
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
                f"Alpaca rejected the order — check GitHub Actions logs immediately."
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
                 "momentum-watch","watchlist","scan-log","readiness","pnl"],
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
                _cache.clear()   # force fresh data
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
                with open(fname, "w") as f:
                    json.dump([asdict(s) for s in signals], f, indent=2)
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
                with open(fname, "w") as f:
                    json.dump([asdict(s) for s in signals], f, indent=2)
                print(f"  💾 Signals exported to {fname}\n")

        # Safety EOD P&L — fires on the 3:30 PM scan as a belt-and-suspenders backup
        # in case the dedicated 4 PM cron is delayed past the market-hours gate.
        _eod_t = datetime.now(ET).hour * 100 + datetime.now(ET).minute
        if 1525 <= _eod_t <= 1600:
            print("\n  [EOD] Final scan window — sending P&L summary...")
            send_account_pnl_telegram("EOD")

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