"""Rebuild dman_smallcap_watchlist.json from a liquidity screen, weekly.

    python tools/smallcap_universe_refresh.py [--dry-run]

The list used to be every name DMan ever called, appended forever. Measured
2026-10-26, point-in-time and with delisted names included (111 weeks,
2024-08..2026-10, next-5-session return):

    DMan list (107/wk)         mean -1.73%  median -2.79%  vs SPY -2.00  (halves -0.89 / -3.10)
    $1-5, $5M+/day, no funds   mean +0.15%  median -1.07%  vs SPY -0.24  (halves +0.30 / -0.78; top 150 by $vol)
    $5+,  $5M+/day             mean +0.09%  median +0.10%  vs SPY -0.25

A price FLOOR on the old list made it worse (2026-10-16: >=$10 -> -8.09%);
the handicap was the names, not their price. So this keeps the low-price
profile and replaces the names: US-listed $1-$5 stocks averaging $5M+ a day
over 20 sessions, not on WATCHLIST, plus any DMan call from the last 14 days
that passed the play rule (dman_play_log.json), so his live calls still scan.

Fails safe: fewer than MIN_NAMES survivors means a data problem, and the
existing list is left untouched rather than replaced by a stub.
"""
import contextlib, io, json, os, re, sys, datetime as dt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT); sys.path.insert(0, _ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a
import requests
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment, DataFeed

PRICE_MIN, PRICE_MAX, ADV_MIN, ADV_DAYS = 1.0, 5.0, 5_000_000, 20
DMAN_CALL_DAYS, MIN_NAMES, MAX_NAMES, CHUNK = 14, 50, 150, 400
EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS"}
# Funds by name. The first run filled the screen with leveraged single-stock
# ETFs (NVD, TSLG, MSTZ, PLTD, CONL) -- $1-$5 and liquid, but not companies.
# Re-measured without them: still -0.24 vs SPY at the 150 cap (vs -2.00).
FUND = re.compile(r"\b(ETF|ETN|ETP|Fund|Trust|Leveraged|Inverse|Ultra|ProShares|Direxion|"
                  r"GraniteShares|Tradr|Defiance|T-Rex|YieldMax|Roundhill|iShares|Invesco|SPDR|"
                  r"2X|3X|1\.5X|Daily|Bull|Bear|Notes?)\b", re.I)


def _keys():
    return (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY"),
            os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY"))


def _symbols(k, s):
    # Raw JSON, not alpaca-py's models: those reject any exchange code newer
    # than the library (TXSE did exactly that on 2026-10-26).
    r = requests.get("https://api.alpaca.markets/v2/assets",
                     params={"status": "active", "asset_class": "us_equity"},
                     headers={"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}, timeout=60)
    r.raise_for_status()
    return sorted(x["symbol"] for x in r.json()
                  if x.get("tradable") and x.get("exchange") in EXCHANGES
                  and str(x.get("symbol", "")).isalpha()
                  and not FUND.search(str(x.get("name") or "")))


def screen() -> list[str]:
    k, s = _keys()
    sc = StockHistoricalDataClient(k, s)
    syms = _symbols(k, s)
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=ADV_DAYS * 2)
    keep = []
    for i in range(0, len(syms), CHUNK):
        try:
            bars = sc.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=syms[i:i + CHUNK], timeframe=TimeFrame.Day, start=start,
                adjustment=Adjustment.SPLIT, feed=DataFeed(a._resolve_stock_feed()))).data
        except Exception as e:
            print(f"  chunk {i}: {type(e).__name__}: {str(e)[:80]}")
            continue
        for sym, bs in bars.items():
            bs = bs[-ADV_DAYS:]
            if len(bs) < ADV_DAYS * 3 // 4:
                continue
            adv = sum(float(b.close) * float(b.volume) for b in bs) / len(bs)
            if PRICE_MIN <= float(bs[-1].close) < PRICE_MAX and adv >= ADV_MIN:
                keep.append((adv, sym))
    watch = set(a.TICKER_SECTOR)
    # Most liquid first, capped: every name here is also polled by the
    # market-hours monitor, and the old list was 128.
    return [sym for _, sym in sorted(keep, reverse=True) if sym not in watch][:MAX_NAMES]


def recent_dman_calls() -> list[str]:
    try:
        log = json.load(open(a.DMAN_PLAY_LOG_FILE if hasattr(a, "DMAN_PLAY_LOG_FILE")
                             else "dman_play_log.json", encoding="utf-8"))
    except Exception:
        return []
    cut = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DMAN_CALL_DAYS)
    out = []
    for r in log if isinstance(log, list) else []:
        try:
            if dt.datetime.fromisoformat(str(r["posted"]).replace("Z", "+00:00")) >= cut:
                out.append(str(r["ticker"]).upper())
        except Exception:
            continue
    return out


def main(dry: bool) -> int:
    names = screen()
    called = sorted({t for t in recent_dman_calls() if t not in a.TICKER_SECTOR})
    calls = [t for t in called if t not in names]
    print(f"screen: {len(names)} names ${PRICE_MIN:g}-${PRICE_MAX:g}, ${ADV_MIN/1e6:g}M+/day; "
          f"+{len(calls)} recent DMan call(s)")
    if len(names) < MIN_NAMES:
        print(f"only {len(names)} < {MIN_NAMES} -- data problem, list left unchanged")
        return 1
    try:
        old = json.load(open(a.SMALLCAP_WATCHLIST_FILE, encoding="utf-8"))
    except Exception:
        old = {}
    old_t = set(old.get("tickers", []) if isinstance(old, dict) else old)
    new_t = names + calls
    print(f"  kept {len(old_t & set(new_t))}, added {len(set(new_t) - old_t)}, dropped {len(old_t - set(new_t))}")
    if dry:
        print("  dry run -- not written"); return 0
    doc = {"_readme": (f"Rebuilt weekly by tools/smallcap_universe_refresh.py: US-listed "
                       f"${PRICE_MIN:g}-${PRICE_MAX:g} stocks averaging ${ADV_MIN/1e6:g}M+/day over "
                       f"{ADV_DAYS} sessions, not on WATCHLIST, plus DMan calls from the last "
                       f"{DMAN_CALL_DAYS} days. StockTwits mode still appends new calls in between."),
           "refreshed": dt.date.today().isoformat(), "tickers": new_t,
           # Only these earn the "his call IS the catalyst" gates -- see
           # dman_algo._load_dman_called().
           "dman_calls": called}
    with open(a.SMALLCAP_WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main("--dry-run" in sys.argv))
