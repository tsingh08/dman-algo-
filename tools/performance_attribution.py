"""Desk-style tear sheet from the broker's own fills. Read-only.

What a strategy made is judged against what the same dollars would have made
in SPY over the same holding window. Percent averages across a -94% option and
a -3% share trade are not P&L; dollars are.

  1. FILL activities -> FIFO round trips per symbol (options x100).
  2. Each round trip is tagged with its strategy from dman_win_rate.json
     (ticker/underlying + exit date), else "unlabelled".
  3. Benchmark: SPY hourly close at entry and exit -> what the trade's cost
     basis would have returned in SPY. alpha$ = pnl$ - cost * spy_ret.
  4. Account: daily portfolio history (cash flows excluded via profit_loss)
     regressed on SPY -> beta, annualised alpha, Sharpe, max drawdown.

Usage: py -3 tools/performance_attribution.py [--json out.json]
Needs APCA_API_KEY_ID / APCA_API_SECRET_KEY (live account).
"""
import json, math, os, re, sys, statistics as st
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import requests

TRADE = "https://api.alpaca.markets"
DATA = "https://data.alpaca.markets"
HDRS = {"APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY", "")}
OCC = re.compile(r"^([A-Z.]{1,6})(\d{6})([CP])(\d{8})$")
LEDGER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "dman_win_rate.json")


def _get(url, params=None):
    r = requests.get(url, headers=HDRS, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def fills():
    out, tok = [], None
    while True:
        p = {"direction": "asc", "page_size": 100}
        if tok:
            p["page_token"] = tok
        page = _get(f"{TRADE}/v2/account/activities/FILL", p)
        if not page:
            return out
        out.extend(page)
        tok = page[-1]["id"]
        if len(page) < 100:
            return out


def family(setup: str) -> str:
    s = (setup or "").replace("SWING — ", "")
    for key, fam in (("Breakout Zone", "Breakout Zone"), ("Momentum Call Spread", "Momentum spread"),
                     ("Earnings", "Earnings options"), ("Options ", "Single options"),
                     ("Momentum Watch", "Momentum Watch"), ("Low Float", "Low Float Catalyst"),
                     ("Gap & Hold", "Gap & Hold"), ("Bear Gap", "Bear Gap Hold"),
                     ("Day 2", "Day 2 Continuation"), ("Dip", "Dip in Uptrend"),
                     ("Morning Runner", "Morning Runner"), ("Adopted", "Adopted/orphan"),
                     ("Untracked", "Adopted/orphan")):
        if key in s:
            return fam
    return s.split(" +")[0][:30] or "unlabelled"


def round_trips(fs):
    """FIFO-match fills per symbol into closed round trips (long or short)."""
    lots = defaultdict(deque)          # sym -> deque of [signed_qty, px, ts]
    trips = []
    for f in fs:
        sym, ts = f["symbol"], f["transaction_time"]
        q = float(f["qty"]) * (1 if f["side"] == "buy" else -1)
        px = float(f["price"])
        mult = 100 if OCC.match(sym) else 1
        book = lots[sym]
        while q and book and (book[0][0] > 0) != (q > 0):
            lot = book[0]
            m = min(abs(q), abs(lot[0]))
            sign = 1 if lot[0] > 0 else -1            # +1 long lot, -1 short lot
            trips.append({"symbol": sym, "entry_ts": lot[2], "exit_ts": ts, "qty": m * sign,
                          "entry_px": lot[1], "exit_px": px, "mult": mult,
                          "cost": m * lot[1] * mult,
                          "pnl": sign * m * (px - lot[1]) * mult})
            lot[0] -= sign * m
            q += sign * m
            if abs(lot[0]) < 1e-9:
                book.popleft()
        if abs(q) > 1e-9:
            book.append([q, px, ts])
    open_lots = {s: [list(x) for x in b] for s, b in lots.items() if b}
    return trips, open_lots


def spy_hourly(start, end):
    bars, tok = [], None
    while True:
        p = {"timeframe": "1Hour", "start": start, "end": end, "adjustment": "all",
             "feed": "sip", "limit": 10000}
        if tok:
            p["page_token"] = tok
        j = _get(f"{DATA}/v2/stocks/SPY/bars", p)
        bars.extend(j.get("bars") or [])
        tok = j.get("next_page_token")
        if not tok:
            break
    return [(datetime.fromisoformat(b["t"].replace("Z", "+00:00")), float(b["c"])) for b in bars]


def spy_at(series, ts):
    t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    last = series[0][1]
    for bt, c in series:
        if bt > t:
            break
        last = c
    return last


def tstat(xs):
    if len(xs) < 2:
        return float("nan")
    sd = st.stdev(xs)
    return st.mean(xs) / (sd / math.sqrt(len(xs))) if sd > 0 else float("nan")


def main():
    if not HDRS["APCA-API-KEY-ID"]:
        sys.exit("APCA_API_KEY_ID not set")
    fs = fills()
    trips, open_lots = round_trips(fs)
    ledger = [r for r in json.load(open(LEDGER, encoding="utf-8")) if r.get("is_live")]
    by_key = defaultdict(list)
    for r in ledger:
        by_key[(str(r["ticker"]).upper(), str(r["date"])[:10])].append(r["setup"])

    spy = spy_hourly(fs[0]["transaction_time"][:10] + "T00:00:00Z",
                     (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat())
    for t in trips:
        m = OCC.match(t["symbol"])
        und = m.group(1) if m else t["symbol"]
        day = t["exit_ts"][:10]
        tags = by_key.get((und, day)) or by_key.get((t["symbol"], day))
        if not tags:          # ledger date can be the ET date of a late-UTC exit
            prev = (datetime.fromisoformat(day) - timedelta(days=1)).date().isoformat()
            tags = by_key.get((und, prev))
        t["family"] = family(tags[0]) if tags else "unlabelled"
        s0, s1 = spy_at(spy, t["entry_ts"]), spy_at(spy, t["exit_ts"])
        t["spy_ret"] = s1 / s0 - 1
        t["bench"] = t["cost"] * t["spy_ret"] * (1 if t["qty"] > 0 else -1)
        t["alpha"] = t["pnl"] - t["bench"]
        t["ret"] = t["pnl"] / t["cost"] if t["cost"] else 0.0
        t["xret"] = t["ret"] - t["spy_ret"] * (1 if t["qty"] > 0 else -1)

    print(f"{len(fs)} fills -> {len(trips)} closed round trips "
          f"({fs[0]['transaction_time'][:10]} .. {fs[-1]['transaction_time'][:10]}); "
          f"{len(open_lots)} symbols still open")
    fams = defaultdict(list)
    for t in trips:
        fams[t["family"]].append(t)
    print(f"\n{'strategy':22s} {'n':>3s} {'P&L $':>9s} {'SPY $':>8s} {'alpha $':>9s} "
          f"{'mean xret':>9s} {'t':>5s} {'beat SPY':>8s} {'win':>5s} {'avg W/L $':>13s}")
    rows = sorted(fams.items(), key=lambda kv: sum(x["alpha"] for x in kv[1]))
    for fam, ts in rows:
        xr = [x["xret"] * 100 for x in ts]
        w = [x["pnl"] for x in ts if x["pnl"] > 0]
        l = [x["pnl"] for x in ts if x["pnl"] <= 0]
        print(f"{fam:22s} {len(ts):3d} {sum(x['pnl'] for x in ts):9.0f} "
              f"{sum(x['bench'] for x in ts):8.0f} {sum(x['alpha'] for x in ts):9.0f} "
              f"{st.mean(xr):+8.2f}% {tstat(xr):5.1f} "
              f"{sum(1 for v in xr if v > 0) / len(xr):8.0%} {len(w) / len(ts):5.0%} "
              f"{(st.mean(w) if w else 0):6.0f}/{(st.mean(l) if l else 0):6.0f}")
    tp = sum(t["pnl"] for t in trips)
    tb = sum(t["bench"] for t in trips)
    print(f"{'TOTAL':22s} {len(trips):3d} {tp:9.0f} {tb:8.0f} {tp - tb:9.0f}")

    months = defaultdict(lambda: [0.0, 0.0, 0])
    for t in trips:
        mo = t["exit_ts"][:7]
        months[mo][0] += t["pnl"]
        months[mo][1] += t["alpha"]
        months[mo][2] += 1
    print("\nby exit month:  " + "  ".join(f"{k} n={v[2]} ${v[0]:+.0f} (alpha ${v[1]:+.0f})"
                                         for k, v in sorted(months.items())))
    worst = sorted(trips, key=lambda x: x["pnl"])[:8]
    print("\nlargest losses: " + "; ".join(
        f"{x['symbol']} {x['family']} ${x['pnl']:+.0f} ({x['ret']:+.0%})" for x in worst))

    # ---- account level: daily returns ex cash flows, vs SPY ----
    ph = _get(f"{TRADE}/v2/account/portfolio/history",
              {"period": "1A", "timeframe": "1D", "intraday_reporting": "market_hours"})
    eq, pl, ts = ph["equity"], ph["profit_loss"], ph["timestamp"]
    days, rets = [], []
    for i in range(1, len(eq)):
        if eq[i - 1] and eq[i - 1] > 0 and eq[i] is not None and pl[i] is not None:
            days.append(datetime.fromtimestamp(ts[i], timezone.utc).date().isoformat())
            rets.append(pl[i] / eq[i - 1])
    j = _get(f"{DATA}/v2/stocks/SPY/bars", {"timeframe": "1Day", "start": days[0] if days else "2026-01-01",
                                            "adjustment": "all", "feed": "sip", "limit": 10000})
    spyc = {b["t"][:10]: float(b["c"]) for b in j.get("bars") or []}
    sd = sorted(spyc)
    spyr = {sd[i]: spyc[sd[i]] / spyc[sd[i - 1]] - 1 for i in range(1, len(sd))}
    pairs = [(r, spyr[d]) for d, r in zip(days, rets) if d in spyr and abs(r) < 0.5]
    if len(pairs) > 20:
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        mb, ma = st.mean(b), st.mean(a)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)
        beta = cov / st.variance(b)
        res = [x - beta * y for x, y in zip(a, b)]
        alpha_ann = st.mean(res) * 252
        te = st.stdev(res) * math.sqrt(252)
        sharpe = ma / st.stdev(a) * math.sqrt(252) if st.stdev(a) else float("nan")
        spy_sharpe = mb / st.stdev(b) * math.sqrt(252)
        cum, peak, mdd = 1.0, 1.0, 0.0
        for x in a:
            cum *= 1 + x
            peak = max(peak, cum)
            mdd = min(mdd, cum / peak - 1)
        scum, speak, smdd = 1.0, 1.0, 0.0
        for y in b:
            scum *= 1 + y
            speak = max(speak, scum)
            smdd = min(smdd, scum / speak - 1)
        print(f"\nACCOUNT ({len(pairs)} days {days[0]}..{days[-1]}, cash flows excluded): "
              f"return {cum - 1:+.1%} vs SPY {scum - 1:+.1%} | beta {beta:.2f} | "
              f"alpha {alpha_ann:+.0%}/yr (t {st.mean(res) / (st.stdev(res) / math.sqrt(len(res))):.1f}) | "
              f"Sharpe {sharpe:.2f} vs SPY {spy_sharpe:.2f} | tracking err {te:.0%} | "
              f"max DD {mdd:.1%} vs SPY {smdd:.1%} | corr {cov / (st.stdev(a) * st.stdev(b)):.2f}")

    if "--json" in sys.argv:
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump({"trips": trips, "open": open_lots}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
