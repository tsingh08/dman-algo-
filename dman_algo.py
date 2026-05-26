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

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ACCOUNT_SIZE       = 25_000
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

MONTHLY_LOSS_LIMIT = 0.04          # halt for the month when down ≥4% of account
MONTHLY_PNL_FILE   = "dman_monthly_pnl.json"

# Seasonal regime — backtest shows Jan(38% WR), Aug(25%), Sep(29%) are chronic losers
SEASONAL_WEAK_MONTHS = {1, 8, 9}
SEASONAL_MIN_SCORE   = 92          # raised bar during weak months

# ADX trend-strength gate — skip directionless/choppy stocks before any pattern check
ADX_TREND_MIN = 20                 # <20 = ranging market; patterns fail more often

ALLOW_SHORTS       = True
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
ALPACA_PAPER      = True      # flip to False only when ready for live brokerage
ENTRY_DRIFT_MAX   = 0.02      # reject signal if price drifted >2% from computed entry
ALPACA_SYNC_FILE   = "dman_alpaca_sync.json"
LAST_ALERTS_FILE   = "dman_last_alerts.json"
ALERT_COOLDOWN_MIN = 30          # suppress duplicate Telegram alert for same ticker within N min

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
    "SHOP":"Technology","SMCI":"Technology","AVGO":"Technology",
    "AMAT":"Technology","MU":"Technology",
    # Semis
    "QCOM":"Technology","MRVL":"Technology","KLAC":"Technology","ON":"Technology",
    # High-beta fintech (ARM, COIN, MARA, PYPL, UPST removed — 0% WR in backtest)
    "RIOT":"Technology","HOOD":"Financials","SOFI":"Financials",
    "AFRM":"Financials","XYZ":"Financials",
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
    "SOUN":"Technology","RKLB":"Industrials",
    # Market regime ETFs
    "SPY":"","QQQ":"","IWM":"",
}

WATCHLIST = list(TICKER_SECTOR.keys())

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
                        top_n: int        = 300) -> list[str]:
    """
    Fetch all NASDAQ + NYSE listed symbols, filter by price/volume,
    sort by today's RVOL, return top N movers combined with curated WATCHLIST.
    Falls back to WATCHLIST if any step fails.
    """
    import io
    print("  [universe] Fetching NASDAQ + NYSE symbol list...", flush=True)
    try:
        def _fetch_syms(url: str, sym_col: str) -> list[str]:
            r = requests.get(url, timeout=20)
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            raw = df[sym_col].dropna().astype(str).tolist()
            return [s.strip() for s in raw if s.strip().isalpha() and 1 <= len(s.strip()) <= 5]

        nasdaq_syms = _fetch_syms(
            "https://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt", "Symbol")
        other_syms  = _fetch_syms(
            "https://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt", "ACT Symbol")
        all_syms = list(set(nasdaq_syms + other_syms) - set(WATCHLIST))
    except Exception as e:
        print(f"  [universe] Symbol fetch failed ({e}), using curated list.", flush=True)
        return WATCHLIST

    print(f"  [universe] {len(all_syms):,} symbols → filtering by price/volume...", flush=True)

    # Batch-download 5-day snapshot to score RVOL cheaply
    active: list[tuple[str, float]] = []
    batch_size = 400
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]
    for idx, batch in enumerate(batches, 1):
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

    # Sort by RVOL descending, take top N, prepend curated list
    active.sort(key=lambda x: x[1], reverse=True)
    dynamic = [sym for sym, _ in active[:top_n]]
    combined = list(dict.fromkeys(WATCHLIST + dynamic))  # curated first, then movers, deduped
    print(f"  [universe] Final universe: {len(combined)} tickers ({len(WATCHLIST)} curated + {len(dynamic)} dynamic)", flush=True)
    return combined

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
    try:
        end   = datetime.today()
        start = end - timedelta(days=period_days)
        raw   = yf.download(ticker, start=start, end=end,
                            interval=interval, progress=False, auto_adjust=True)
        if raw is None or len(raw) < 20:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        _cache[key] = raw
        return raw
    except Exception:
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
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception:
        return False


def format_signal_telegram(s: "ProSignal", regime: dict) -> str:
    """Format a ProSignal as a Telegram-ready HTML message."""
    arrow = "🟢 LONG" if s.bias == "LONG" else "🔴 SHORT"
    opex  = " ⚠️ OpEx week" if is_opex_week() else ""
    return (
        f"<b>D🔥man Signal{opex}</b>\n"
        f"{arrow} <b>{s.ticker}</b> — {s.setup}\n"
        f"Entry: <b>${s.entry}</b>  Stop: ${s.stop}\n"
        f"T1 (2R): ${s.target1}   T2 (3R): ${s.target2}\n"
        f"R/R: {s.rr}:1   RSI: {s.rsi}   RVOL: {s.rvol}x\n"
        f"Score: {s.confluence_score}/100"
        + (f"  AI: {s.ai_score}/10" if s.ai_score else "")
        + f"\nMarket: {regime.get('regime','?')}\n{s.reason}"
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


_float_cache: dict[str, tuple[float, float]] = {}  # ticker → (float_shares, short_pct)

def _get_short_float_data(ticker: str) -> tuple[float, float]:
    """
    Return (float_shares_millions, short_pct_of_float) for a ticker.
    Uses yf.Ticker.info; cached per session. Returns (0, 0) on failure.
    """
    if ticker in _float_cache:
        return _float_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        float_shares = info.get("floatShares") or info.get("sharesOutstanding") or 0
        short_pct    = info.get("shortPercentOfFloat") or 0.0
        result = (float_shares / 1_000_000, float(short_pct) * 100)
    except Exception:
        result = (0.0, 0.0)
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
    shares:     int   = 0
    cost:       float = 0.0
    risk_usd:   float = 0.0
    kelly_frac: float = 0.0

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

    Returns dict with regime, score (0-15), and details.
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
            iwm_ret = (float(iwm_df["Close"].iloc[-1]) / float(iwm_df["Close"].iloc[-21]) - 1) * 100
            spy_ret = (float(spy_df["Close"].iloc[-1]) / float(spy_df["Close"].iloc[-21]) - 1) * 100
            if iwm_ret > spy_ret - 5:   # IWM not lagging SPY by more than 5 pts
                score += 1
            breadth_note = f"IWM {iwm_ret:+.1f}% vs SPY {spy_ret:+.1f}%"

        # Regime classification
        if score >= 10 and bull_di:
            regime = "BULL"
        elif score <= 4 or (not bull_di and vix_val > 30):
            regime = "BEAR"
        else:
            regime = "CHOP"

        result.update({
            "regime":     regime,
            "score":      score,
            "spy_trend":  spy_above_50,
            "adx_strong": adx_strong,
            "vix_ok":     vix_mid,
            "details": {
                "SPY vs EMA20": spy_above_20,
                "SPY vs EMA50": spy_above_50,
                "SPY vs SMA200": spy_above_200,
                "ADX": round(adx_val, 1),
                "VIX": round(vix_val, 1),
                "+DI > -DI": bull_di,
                "IWM Breadth": breadth_note,
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
        if r == "BULL":   return True,  min(15, s)
        if r == "CHOP":   return True,  min(8,  s)    # allowed but reduced score
        return False, 0    # BEAR — no longs

    else:  # SHORT
        if r == "BEAR":   return True,  min(15, 15 - s)
        if r == "CHOP":   return True,  min(8,  8  - s//2)
        return False, 0    # BULL — no shorts


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
    Block signals within MACRO_BLACKOUT days of FOMC decisions, NFP, or CPI releases.
    These events create gap risk and intraday whipsaws that blow through stops regardless
    of setup quality. Returns (safe, score 0-5).
    """
    try:
        events = _FOMC_DATES | _nfp_dates() | _CPI_DATES
        today  = date.today()
        if today.year > _FOMC_LAST_CONFIRMED_YEAR:
            import sys as _sys
            print(f"  ⚠️  FOMC dates beyond {_FOMC_LAST_CONFIRMED_YEAR} are estimated — "
                  f"update _FOMC_DATES from federalreserve.gov", file=_sys.stderr)
        for ev in events:
            days_away = (ev - today).days
            if -1 <= days_away <= MACRO_BLACKOUT:
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

    Prompt is kept concise so it uses minimal tokens.
    """
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "":
        return 0   # AI scoring skipped — no key

    prompt = f"""You are an expert technical analyst evaluating a swing trade setup.
Score this setup from 1-10 based on quality and probability of success.
Return ONLY a single integer (1-10), nothing else.

Setup Details:
- Ticker     : {signal.ticker}
- Bias       : {signal.bias}
- Pattern    : {signal.setup}
- Entry      : ${signal.entry}
- Stop Loss  : ${signal.stop}  (risk: ${signal.risk_per_share:.2f}/share)
- Target 1   : ${signal.target1}  (2R)
- Target 2   : ${signal.target2}  (3R)
- R/R Ratio  : {signal.rr}:1
- RSI (14)   : {signal.rsi}
- Rel Volume : {signal.rvol}x
- Market     : {regime.get('regime','?')} (SPY regime score {regime.get('score',0)}/15)
- Setup Reason: {signal.reason}
- Confluence : {signal.confluence_score}/100

Score 8-10 only if: setup is textbook, in a trending market, strong volume, clear R/R.
Score 5-7 for borderline or mixed signals.
Score 1-4 for weak setups with red flags."""

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
                         win_rate: float, avg_win_r: float) -> ProSignal:
    """Apply Kelly-optimal, beta-adjusted position sizing to a signal."""
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

    risk_budget = account * kf
    signal.shares     = max(1, int(risk_budget / rps))
    signal.cost       = round(signal.shares * signal.entry, 2)
    signal.risk_usd   = round(signal.shares * rps, 2)
    signal.kelly_frac = kf
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
                    "total": 0}

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

        return {
            "win_rate":     round(win_rate, 3),
            "avg_win_r":    round(avg_win_r, 2),
            "avg_loss_r":   round(avg_loss_r, 2),
            "consec_losses":consec,
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
        return {
            "win_rate":   round(wr, 3),
            "avg_win_r":  round(avg_win_r, 2),
            "avg_loss_r": round(avg_loss_r, 2),
            "total":      len(recent),
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
            cur = get_current_price(p.ticker)
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

            if   t2_hit:  status = "🎯 T2 HIT — consider full exit"
            elif t1_hit:  status = "✅ T1 HIT — trail stop to breakeven"
            elif stopped:
                status = "🛑 AT STOP — exit immediately"
                send_telegram(
                    f"🛑 <b>STOP HIT</b> — {p.ticker} {p.bias}\n"
                    f"Entry ${p.entry} → Now ${cur:.2f} | Stop ${p.stop}\n"
                    f"P&L: {'+' if pnl_pct>=0 else ''}{pnl_pct:.1f}%  Exit NOW."
                )
            else:         status = "⏳ Active"

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

    # L3: Oversold Bounce — true oversold only, uptrend context, volume-confirmed turn
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

    # L4: MACD Cross Bull — fresh cross only; require trend + volume + momentum alignment
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
        if (gap_pct >= 1.5 and c >= float(r["Open"]) * 0.995
                and float(r["RVOL"]) >= 1.5 and float(r["RSI"]) > 50
                and float(r["MACD"]) > float(r["MACD_sig"])
                and _sector_etf_above_ema50(ticker)):
            gap_stop = min(float(r["Low"]) * 0.99, float(r["Open"]) * 0.985)
            sig = _long("Gap & Hold", gap_stop, 2.5, 4.0,
                        reason=f"Gap up +{gap_pct:.1f}% from prior close, holding, RVOL {float(r['RVOL']):.1f}x")
            if sig.rr >= MIN_RR:
                candidates.append(sig)
    except Exception:
        pass

    # L7: Morning Runner — Dman news-catalyst style: gap ≥5%, RVOL ≥5x, holding above open
    try:
        gap_up = (float(r["Open"]) - float(p["Close"])) / float(p["Close"]) * 100
        if (gap_up >= 5.0 and c >= float(r["Open"]) * 0.97
                and float(r["RVOL"]) >= 5.0
                and 50 <= float(r["RSI"]) <= 72
                and float(r["MACD"]) > float(r["MACD_sig"])):
            mr_stop = min(float(r["Low"]) * 0.99, float(r["Open"]) * 0.96)
            fl_m, sh_pct = _get_short_float_data(ticker)
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

        # S2: Volume Breakdown — tightened: RVOL 3.5x, RSI <32, range >1.5x ATR
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

        # S4: MACD Bear Cross — cross below zero, RSI in neutral zone, volume confirmed
        if (float(p["MACD"]) > float(p["MACD_sig"]) and float(r["MACD"]) < float(r["MACD_sig"])
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

    # 19. Short float / squeeze potential (0-10 pts) — Morning Runner only
    if signal.setup == "Morning Runner":
        fl_m, sh_pct = _get_short_float_data(signal.ticker)
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
    setup_s = tracker.setup_stats(signal.setup)
    signal = size_position_kelly(
        signal,
        account   = get_effective_account(),
        win_rate  = max(0.5, setup_s["win_rate"]),
        avg_win_r = max(1.5, setup_s["avg_win_r"]),
    )

    # Final weighted score: confluence (70%) + AI score (30%)
    signal.final_score = round(signal.confluence_score * 0.70 +
                                signal.ai_score * 10 * 0.30, 1)

    return signal


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 20 — PRO SCANNER
# ═══════════════════════════════════════════════════════════════════════════

def run_pro_scanner(tickers: list[str] = WATCHLIST,
                    min_score: int = None,
                    use_ai: bool = False) -> list[ProSignal]:
    """
    Full pro-grade scanner with all 18 filters applied.
    Only returns signals that pass ALL hard gates AND score >= min_score.
    """
    global MIN_CONFLUENCE
    tracker    = WinRateTracker()
    stats      = tracker.rolling_stats()
    min_score  = min_score or tracker.adaptive_min_score()

    # Consecutive loss guard
    if stats["consec_losses"] >= MAX_CONSEC_LOSSES:
        print(f"\n  🛑 CONSECUTIVE LOSS GUARD: {stats['consec_losses']} losses in a row.")
        print(f"     Take a break. Reset your mind. Come back tomorrow.\n")
        return []

    # Monthly loss circuit breaker
    month_loss = get_this_month_loss()
    if month_loss <= -(MONTHLY_LOSS_LIMIT * 100):
        print(f"\n  🛑 MONTHLY LOSS LIMIT HIT: Down {month_loss:.1f}% this month "
              f"(limit: {MONTHLY_LOSS_LIMIT*100:.0f}%).")
        print(f"     Stop trading for the month. Review setups. Reset.\n")
        send_telegram(f"🛑 <b>Monthly loss limit hit</b> — down {month_loss:.1f}% this month. Halted until next month.")
        return []

    # Daily loss circuit breaker
    todays_loss = get_todays_loss()
    if todays_loss <= -(DAILY_LOSS_LIMIT * 100):
        print(f"\n  🛑 DAILY LOSS LIMIT HIT: Down {todays_loss:.1f}% today "
              f"(limit: {DAILY_LOSS_LIMIT*100:.0f}%).")
        print(f"     Stop trading. Protect your capital. Come back tomorrow.\n")
        send_telegram(f"🛑 <b>Daily loss limit hit</b> — down {todays_loss:.1f}% today. Halted.")
        return []

    print(f"\n{'═'*68}")
    print(f"  D🔥man PRO Scanner v3  —  {datetime.today().strftime('%A %b %d, %Y')}")
    print(f"  Min score : {min_score}/100  |  AI scoring: {'ON' if use_ai else 'OFF'}")
    print(f"  Shorts    : {'ON' if ALLOW_SHORTS else 'OFF'}  |  "
          f"Rolling WR: {stats['win_rate']*100:.1f}%  ({stats['total']} trades)")
    print(f"{'═'*68}")

    # Get regime once (expensive call)
    print("  [1/2] Checking market regime & sectors...")
    regime   = get_market_regime()
    top_secs = get_top_sectors()
    print(f"  Market : {regime['regime']} (score {regime['score']}/15)  "
          f"VIX: {regime['details'].get('VIX','?')}")
    print(f"  Top sectors: {', '.join(top_secs)}")

    # VIX regime scaling — tighten confluence floor in elevated-volatility markets
    vix_now = float(regime['details'].get('VIX', 20))
    if vix_now > 25:
        min_score = max(min_score, 90)
        print(f"  ⚠  VIX={vix_now:.1f} > 25 — min score raised to {min_score}/100")

    # Seasonal regime filter — Jan/Aug/Sep are chronic losers in backtest (25-38% WR)
    curr_month = datetime.today().month
    if curr_month in SEASONAL_WEAK_MONTHS:
        min_score = max(min_score, SEASONAL_MIN_SCORE)
        month_name = datetime.today().strftime("%B")
        print(f"  📅  {month_name} seasonal filter — min score raised to {min_score}/100")

    print(f"\n  [2/2] Scanning {len(tickers)} tickers...\n")

    signals = []
    rejected_counts = {"no_signal":0, "hard_gate":0, "low_score":0}

    for i, ticker in enumerate(tickers, 1):
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

        # Apply all 15 pro filters
        sig = score_signal(sig, df, regime, tracker)

        # Hard gates: regime + earnings + divergence (absolute stops)
        if not sig.regime_ok:
            rejected_counts["hard_gate"] += 1
            sys.stdout.write(f"REGIME BLOCKED ({sig.bias} in {regime['regime']})\n")
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

        # Soft score gate (per-setup overrides + volatile ticker floor)
        effective_min = SETUP_MIN_CONFLUENCE.get(sig.setup, min_score)
        if sig.ticker in VOLATILE_TICKERS:
            effective_min = max(effective_min, VOLATILE_MIN_CONFLUENCE)
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

    signals.sort(key=lambda s: s.confluence_score, reverse=True)

    # Portfolio heat cap: admit signals until total open risk hits the limit
    heat_capped: list[ProSignal] = []
    total_risk_pct = 0.0
    eff_account = get_effective_account()
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
                if bt_score < bt_min * 0.85:
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
#  SECTION 22.5 — ALPACA PAPER TRADING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
#  pip install alpaca-py
#  Set env vars: ALPACA_API_KEY  ALPACA_SECRET_KEY
#  Paper keys are free at app.alpaca.markets

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, GetOrdersRequest,
        TakeProfitRequest, StopLossRequest,
    )
    from alpaca.trading.enums import (
        OrderSide, TimeInForce, OrderClass, QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest
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


def submit_paper_trade(signal: ProSignal) -> Optional[str]:
    """
    Place a bracket order on Alpaca paper:
      Entry  — market order (fills at open / current price)
      Stop   — stop order at signal.stop
      Target — limit order at signal.target1 (2R exit)

    The broker handles stop/target execution automatically; dman just syncs
    the fill back via sync_alpaca_fills().

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
    stop_px   = round(signal.stop,    2)
    target_px = round(signal.target1, 2)

    try:
        order = client.submit_order(MarketOrderRequest(
            symbol        = signal.ticker,
            qty           = signal.shares,
            side          = side,
            time_in_force = TimeInForce.DAY,
            order_class   = OrderClass.BRACKET,
            take_profit   = TakeProfitRequest(limit_price=target_px),
            stop_loss     = StopLossRequest(stop_price=stop_px),
        ))
        oid = str(order.id)
        label = "PAPER" if ALPACA_PAPER else "LIVE"
        print(f"  📤 [{label}] {signal.ticker} {signal.bias} {signal.shares}sh  "
              f"stop=${stop_px}  T1=${target_px}  id={oid[:8]}…")
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
        if ticker in alp_open:
            continue    # still open — nothing to record yet

        # Position is gone from Alpaca — find the exit fill
        try:
            orders = client.get_orders(filter=GetOrdersRequest(
                symbols=[ticker],
                status=QueryOrderStatus.CLOSED,
                limit=20,
            ))
        except Exception:
            continue

        is_lo       = pos.bias == "LONG"
        closing_side = OrderSide.SELL if is_lo else OrderSide.BUY

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

            pnl_pct     = ((fill_px - pos.entry) / pos.entry * 100 if is_lo
                           else (pos.entry - fill_px) / pos.entry * 100)
            outcome     = ("WIN" if pnl_pct > 0.1 else
                           "LOSS" if pnl_pct < -0.1 else "BE")
            dollar_pnl  = (fill_px - pos.entry) * qty * (1 if is_lo else -1)
            acct_pct    = dollar_pnl / ACCOUNT_SIZE * 100

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
                score   = 0,
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
        print(f"{'═'*W}\n")
    except Exception as exc:
        print(f"  ❌ Alpaca account fetch failed: {exc}")


def _submit_signals_to_alpaca(signals: list[ProSignal]) -> None:
    """
    Validate entry prices and submit passing signals to Alpaca paper.
    Automatically adds each submitted trade to PositionTracker.
    Called after a scan when --submit flag is set.
    """
    if not signals:
        return
    if not ALPACA_API_KEY:
        print("  ⚠️  --submit requires ALPACA_API_KEY to be set.")
        return

    pt        = PositionTracker()
    submitted = 0

    print(f"\n  {'─'*68}")
    print(f"  Validating {len(signals)} signal(s) for Alpaca submission…")

    for sig in signals:
        valid, cur = validate_entry_price(sig)
        drift_pct  = (cur - sig.entry) / sig.entry * 100
        if not valid:
            print(f"  ⚡ {sig.ticker:<8} entry stale  "
                  f"signal=${sig.entry}  now=${cur}  drift={drift_pct:+.1f}%  — skipped")
            continue

        oid = submit_paper_trade(sig)
        if oid:
            pt.open(OpenPosition(
                ticker     = sig.ticker,
                bias       = sig.bias,
                setup      = sig.setup,
                entry      = sig.entry,
                stop       = sig.stop,
                target1    = sig.target1,
                target2    = sig.target2,
                shares     = sig.shares,
                entry_date = datetime.today().strftime("%Y-%m-%d"),
                atr        = sig.atr,
            ))
            submitted += 1

    print(f"  📤 {submitted}/{len(signals)} signal(s) submitted to Alpaca paper\n")


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 22 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="D🔥man Algorithm v3 PRO — targeting 80%+ win rate"
    )
    parser.add_argument("--mode", default="scan",
        choices=["scan","backtest","performance","regime","record",
                 "watch","rank","open","positions","alpaca","sync"],
        help=("scan        : run pro scanner with all filters\n"
              "backtest    : walk-forward backtest\n"
              "performance : win rate tracker report\n"
              "regime      : today's market regime only\n"
              "record      : log a completed trade outcome\n"
              "watch       : re-scan on a timer during market hours\n"
              "rank        : near-signal leaderboard for the watchlist\n"
              "open        : log a new open position\n"
              "positions   : view open positions with live P&L\n"
              "alpaca      : Alpaca account dashboard + scan + submit\n"
              "sync        : sync Alpaca fills → auto-record closed trades"))
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
        tickers = WATCHLIST

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

    elif args.mode == "regime":
        print("  Checking market regime...\n")
        regime = get_market_regime()
        r = regime["regime"]; s = regime["score"]
        d = regime["details"]
        print(f"  Market Regime : {r}  (score {s}/15)")
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
            _submit_signals_to_alpaca(signals)
            if args.export:
                fname = f"dman_signals_{datetime.today().strftime('%Y-%m-%d')}.json"
                with open(fname, "w") as f:
                    json.dump([asdict(s) for s in signals], f, indent=2)
                print(f"  💾 Signals exported to {fname}\n")

    elif args.mode == "scan":
        signals = run_pro_scanner(tickers,
                                   min_score=args.score,
                                   use_ai=args.ai)
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


if __name__ == "__main__":
    main()