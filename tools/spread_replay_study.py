"""Momentum call spread replay on REAL Alpaca hourly option bars, live exit rules.

Result 2026-10-26 (24 signals, LONG score>=95 $100+: 13 live 2026-09/10, 11 from
the 2y point-in-time backtest): mean +17.7% of debit (se 20.4), median -2.2%,
12/24 win, bimodal; the same signals as SHARES -2.28%. PLTR 2026-10-08 replays as
-72%: its +102% was the manual exit. Optimistic fills (hourly closes, $0.05/leg).
Re-run at the MOMENTUM_SPREAD_REVIEW_N verdict.

Usage: py -3 tools/spread_replay_study.py signals.json out.json
"""
import json, os, sys, time, datetime as dt
from zoneinfo import ZoneInfo
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest
from alpaca.trading.enums import AssetStatus, ContractType
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import OptionBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo("America/New_York")
K, S = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY"), os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
tc = TradingClient(K, S, paper=False); oc = OptionHistoricalDataClient(K, S); sc = StockHistoricalDataClient(K, S)
BUDGET, SLIP, TP, DECAY, MINW, STARTW = 400.0, 0.05, 0.70, 0.50, 0.005, 0.03

def contracts(tk, d0):
    out = []
    for st in (AssetStatus.INACTIVE, AssetStatus.ACTIVE):
        tok = None
        while True:
            r = tc.get_option_contracts(GetOptionContractsRequest(
                underlying_symbols=[tk], type=ContractType.CALL, status=st,
                expiration_date_gte=d0 + dt.timedelta(days=3), expiration_date_lte=d0 + dt.timedelta(days=14),
                limit=10000, page_token=tok))
            out += r.option_contracts or []
            tok = r.next_page_token
            if not tok: break
    return out

def hourly(sym, d0, d1):
    r = oc.get_option_bars(OptionBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Hour,
                                             start=dt.datetime.combine(d0, dt.time(9, 0), ET),
                                             end=dt.datetime.combine(d1, dt.time(17, 0), ET)))
    return {b.timestamp.astimezone(ET): b for b in r.data.get(sym, [])}

def spot(tk, d0):
    r = sc.get_stock_bars(StockBarsRequest(symbol_or_symbols=tk, timeframe=TimeFrame.Hour,
                                           start=dt.datetime.combine(d0, dt.time(9, 30), ET),
                                           end=dt.datetime.combine(d0, dt.time(11, 0), ET)))
    bs = [b for b in r.data.get(tk, []) if b.timestamp.astimezone(ET).hour >= 10]
    return float(bs[0].vwap) if bs else None

def replay(sig):
    tk = sig["ticker"]; d0 = dt.date.fromisoformat(sig["entry_date"])
    px = spot(tk, d0)
    if not px: return {"skip": "no spot"}
    if px < 100: return {"skip": "under $100"}
    cs = contracts(tk, d0)
    if not cs: return {"skip": "no chain"}
    exps = sorted({c.expiration_date for c in cs}, key=lambda e: abs((e - d0).days - 7))
    exp = exps[0]
    ks = sorted({float(c.strike_price): c.symbol for c in cs if c.expiration_date == exp}.items())
    if len(ks) < 2: return {"skip": "thin chain"}
    lk, lsym = min(ks, key=lambda kv: abs(kv[0] - px))
    shorts = [kv for kv in ks if kv[0] > lk and kv[0] - lk >= MINW * px and kv[0] <= lk + STARTW * px + 1e-9]
    if not shorts: return {"skip": "no short strike"}
    d1 = exp
    L = hourly(lsym, d0, d1)
    for sk, ssym in sorted(shorts, reverse=True):          # widest first, narrow to fit
        Sb = hourly(ssym, d0, d1)
        t0 = [t for t in sorted(set(L) & set(Sb)) if t.date() == d0 and t.hour >= 10]
        if not t0: continue
        debit = float(L[t0[0]].vwap) - float(Sb[t0[0]].vwap) + 2 * SLIP
        width = sk - lk
        if debit <= 0 or debit >= width or debit * 100 > BUDGET: continue
        maxg = width - debit
        lv = sv = None; res = None
        for t in sorted(set(L) | set(Sb)):
            if t < t0[0]: continue
            lv = float(L[t].close) if t in L else lv
            sv = float(Sb[t].close) if t in Sb else sv
            if lv is None or sv is None: continue
            val = min(max(lv - sv, 0.0), width)
            dte = (exp - t.date()).days
            if dte <= 1:
                res = ("DTE", val - 2 * SLIP); break
            if val - debit >= TP * maxg:
                res = ("TP", val - 2 * SLIP); break
            if t.date() > d0 and t.hour >= 15 and val < DECAY * debit:
                res = ("DECAY", val - 2 * SLIP); break
        if res is None:
            res = ("END", (val if lv is not None else 0) - 2 * SLIP)
        out_val = max(res[1], 0.0)
        return {"exit": res[0], "debit": round(debit, 2), "width": width, "exp": str(exp),
                "long": lsym, "short": ssym, "pnl_pct": round((out_val - debit) / debit * 100, 1),
                "pnl_usd": round((out_val - debit) * 100, 0)}
    return {"skip": "no spread fits budget / no fills"}

if __name__ == "__main__":
    sigs = json.load(open(sys.argv[1])); rows = []
    for s in sigs:
        try: r = replay(s)
        except Exception as e: r = {"skip": f"error {type(e).__name__}: {str(e)[:80]}"}
        rows.append({**s, **r}); time.sleep(0.2)
        print(s["ticker"], s["entry_date"], r.get("exit", r.get("skip")), r.get("pnl_pct", ""), flush=True)
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
