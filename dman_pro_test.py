"""
DMan Algo — Alpaca Pro Readiness Test
Runs before Monday open to confirm every system is live and healthy.
"""
import os, sys, json, requests
from datetime import datetime, date, timedelta

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(label, ok, detail=""):
    icon = PASS if ok else FAIL
    results.append((ok, label, detail))
    print(f"  {icon}  {label}" + (f"  →  {detail}" if detail else ""))

def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Pull keys from env ────────────────────────────────────────
ALPACA_KEY    = os.getenv("APCA_API_KEY_ID", "")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY", "")
TELEGRAM_TOK  = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CID  = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ACCOUNT_SIZE  = os.getenv("ACCOUNT_SIZE", "")

# ════════════════════════════════════════════════════════════════
section("1 / 7  —  SECRETS & ENVIRONMENT")
# ════════════════════════════════════════════════════════════════

check("APCA_API_KEY_ID set",    bool(ALPACA_KEY),    ALPACA_KEY[:8] + "…" if ALPACA_KEY else "MISSING")
check("APCA_API_SECRET_KEY set", bool(ALPACA_SECRET), "present" if ALPACA_SECRET else "MISSING")
check("TELEGRAM_TOKEN set",     bool(TELEGRAM_TOK),  "present" if TELEGRAM_TOK else "MISSING")
check("TELEGRAM_CHAT_ID set",   bool(TELEGRAM_CID),  TELEGRAM_CID if TELEGRAM_CID else "MISSING")
check("ANTHROPIC_API_KEY set",  bool(ANTHROPIC_KEY), "present" if ANTHROPIC_KEY else "MISSING")
check("ACCOUNT_SIZE set",       bool(ACCOUNT_SIZE),  f"${ACCOUNT_SIZE}" if ACCOUNT_SIZE else "MISSING")

if not ALPACA_KEY or not ALPACA_SECRET:
    print("\n  Cannot continue without Alpaca credentials. Set env vars and retry.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
section("2 / 7  —  ALPACA ACCOUNT & PRO TIER")
# ════════════════════════════════════════════════════════════════

try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=False)
    acct = client.get_account()
    equity   = float(acct.equity or 0)
    buying   = float(acct.buying_power or 0)
    dt_count = int(getattr(acct, "daytrade_count", 0) or 0)
    status   = getattr(acct, "status", "unknown")
    check("Account active",      status == "ACTIVE",     f"status={status}")
    check("Equity > $0",         equity > 0,             f"${equity:,.2f}")
    check("Buying power > $0",   buying > 0,             f"${buying:,.2f}")
    check("Day trades used",     True,                   f"{dt_count}/3 used")
    check("Not PDT-halted",      getattr(acct, "pattern_day_trader", False) == False or equity >= 25000,
          "OK" if equity >= 25000 else f"PDT — {dt_count}/3 used, swing mode at ≥2")
except Exception as e:
    check("Alpaca connection",   False, str(e))
    print("  Cannot continue without Alpaca. Check credentials.")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════
section("3 / 7  —  OPTIONS DATA FEED (OPRA)")
# ════════════════════════════════════════════════════════════════

# Test real-time options snapshot on AAPL (ultra-liquid — always has OPRA data)
TEST_OCC = None
try:
    from alpaca.trading.requests import GetOptionContractsRequest
    from alpaca.trading.enums   import ContractType
    expiry = date.today() + timedelta(days=21)
    # Find the nearest expiry Friday after 14 days
    for offset in range(14, 40):
        d = date.today() + timedelta(days=offset)
        if d.weekday() == 4:  # Friday
            expiry = d
            break
    req = GetOptionContractsRequest(
        underlying_symbols=["AAPL"],
        expiration_date_lte=str(expiry),
        expiration_date_gte=str(date.today() + timedelta(days=7)),
        contract_type=ContractType.CALL,
        limit=5,
    )
    contracts = client.get_option_contracts(req)
    items = getattr(contracts, "option_contracts", []) or []
    if items:
        TEST_OCC = items[0].symbol
        check("Options chain accessible", True, f"AAPL contracts found ({len(items)} returned)")
    else:
        check("Options chain accessible", False, "No contracts returned — market may be closed")
except Exception as e:
    check("Options chain query",  False, str(e))

# Now test the OPRA snapshot endpoint directly
if TEST_OCC:
    try:
        r = requests.get(
            "https://data.alpaca.markets/v1beta1/options/snapshots",
            headers={"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET},
            params={"symbols": TEST_OCC, "feed": "opra"},
            timeout=10,
        )
        if r.status_code == 200:
            snap = r.json().get("snapshots", {}).get(TEST_OCC, {})
            if snap:
                q = snap.get("latestQuote", {})
                bid, ask = float(q.get("bp", 0) or 0), float(q.get("ap", 0) or 0)
                greeks = snap.get("greeks", {})
                delta  = float(greeks.get("delta", 0) or 0)
                check("OPRA real-time feed works",  True,
                      f"{TEST_OCC}  bid=${bid:.2f}  ask=${ask:.2f}  Δ={delta:.2f}")
            else:
                check("OPRA snapshot data",  False,
                      "Empty snapshot — market closed or contract expired (test during market hours)")
        elif r.status_code == 403:
            check("OPRA feed access",  False,
                  "403 Forbidden — Pro subscription may not be active yet (allow 5-10 min after upgrade)")
        else:
            check("OPRA snapshot endpoint",  False, f"HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        check("OPRA snapshot request",  False, str(e))
else:
    print(f"  {WARN} Skipping OPRA test — no contract OCC symbol available")

# ════════════════════════════════════════════════════════════════
section("4 / 7  —  NEWS API")
# ════════════════════════════════════════════════════════════════

try:
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest
    from datetime import timezone
    nc  = NewsClient(api_key=ALPACA_KEY, secret_key=ALPACA_SECRET)
    req = NewsRequest(symbols=["AAPL"], start=datetime.now(timezone.utc) - timedelta(hours=48), limit=3)
    news = nc.get_news(req)
    items = news.data.get("AAPL", []) if hasattr(news, "data") else []
    check("Alpaca News API",  len(items) > 0, f"{len(items)} headlines returned for AAPL (48h)")
except Exception as e:
    check("Alpaca News API",  False, str(e))

# ════════════════════════════════════════════════════════════════
section("5 / 7  —  MARKET DATA (yfinance)")
# ════════════════════════════════════════════════════════════════

try:
    import yfinance as yf
    df = yf.download("SPY", period="5d", progress=False, auto_adjust=True)
    last_date = df.index[-1].date()
    staleness = (date.today() - last_date).days
    check("yfinance SPY data",    df is not None and len(df) >= 2, f"last bar: {last_date}")
    check("Data freshness",       staleness <= 3,                  f"{staleness} days old")
except Exception as e:
    check("yfinance",  False, str(e))

# ════════════════════════════════════════════════════════════════
section("6 / 7  —  TELEGRAM")
# ════════════════════════════════════════════════════════════════

if TELEGRAM_TOK and TELEGRAM_CID:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOK}/sendMessage",
            json={
                "chat_id":    TELEGRAM_CID,
                "text":       "🧪 <b>DMan Pro Readiness Test</b>\nAlpaca Algo Trader Plus activated — OPRA feed live. Monday ready.",
                "parse_mode": "HTML",
            },
            timeout=8,
        )
        ok = r.status_code == 200
        check("Telegram message sent",  ok,  "delivered" if ok else f"HTTP {r.status_code}")
    except Exception as e:
        check("Telegram",  False, str(e))
else:
    check("Telegram credentials",  False, "TELEGRAM_TOKEN or TELEGRAM_CHAT_ID not set in env")

# ════════════════════════════════════════════════════════════════
section("7 / 7  —  GITHUB ACTIONS READINESS")
# ════════════════════════════════════════════════════════════════

import subprocess, pathlib
try:
    result = subprocess.run(["git", "log", "--oneline", "-5"], capture_output=True, text=True,
                            cwd="C:/Users/singh")
    check("Git repo accessible",  result.returncode == 0, "")
    if result.returncode == 0:
        commits = result.stdout.strip().split("\n")
        print(f"\n  Recent commits:")
        for c in commits:
            print(f"    {c}")
except Exception as e:
    check("Git",  False, str(e))

# Check workflow file has BENZINGA_API_KEY
wf = pathlib.Path("C:/Users/singh/.github/workflows/dman_scanner.yml")
if wf.exists():
    content = wf.read_text()
    check("BENZINGA_API_KEY in workflow",  "BENZINGA_API_KEY" in content,  "dman_scanner.yml")
else:
    check("Workflow file exists",  False, str(wf))

# ════════════════════════════════════════════════════════════════
print(f"\n{'═'*60}")
passed = sum(1 for ok, _, _ in results if ok)
failed = sum(1 for ok, _, _ in results if not ok)
total  = len(results)
print(f"  RESULT: {passed}/{total} checks passed  |  {failed} failed")
if failed == 0:
    print("  🚀 ALL SYSTEMS GO — algo is ready for Monday")
else:
    print("  ⚠️  Fix failed checks before market open")
    for ok, label, detail in results:
        if not ok:
            print(f"     ❌  {label}: {detail}")
print(f"{'═'*60}\n")
