"""Replay the live AI scorer over the labelled signal log.

    python tools/llm_scorer_eval.py [--cache FILE] [--limit N] [--workers 6]

Every row of dman_signal_features.json that has a label (what the signal
would have done: next-open entry, its own stop and target, 5 sessions) is
rebuilt into the prompt the scanner sends, for each prompt version in
dman_algo, and the model's answer is compared with the outcome.

What it reports, per prompt, on ALL rows and on the later half by date
(the half the v2 prompt's stated base rates were NOT taken from):
  - rank correlation of the score with the labelled P&L
  - the veto split: mean P&L and win rate of score >= 6 against score < 6
  - the same on one row per ticker (repeats of one name are not evidence)
and, as the bar to beat, the rank correlation of two plain numbers that need
no model at all: price, and ATR as a percent of price.

Limits, stated once: the feature log does not keep the rule's free-text
reason or the VIX / rates / dollar readings, so the v1 prompt is replayed
with those lines blank. Nothing here places, blocks or changes a trade.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_algo():
    os.environ.setdefault("TELEGRAM_TOKEN", "")
    with contextlib.redirect_stdout(io.StringIO()):
        import dman_algo
    return dman_algo


def _signal_from_row(r: dict) -> SimpleNamespace:
    entry, stop = float(r.get("entry") or 0), float(r.get("stop") or 0)
    risk = abs(entry - stop)
    long_ = str(r.get("bias", "LONG")).upper() != "SHORT"
    return SimpleNamespace(
        ticker=r.get("ticker", ""), bias=r.get("bias", "LONG"), setup=r.get("setup", ""),
        entry=entry, stop=stop, risk_per_share=risk,
        target1=r.get("target1", 0),
        target2=round(entry + (3 * risk if long_ else -3 * risk), 2),
        rr=r.get("rr", 0), rsi=r.get("rsi", 0), rvol=r.get("rvol", 0),
        confluence_score=r.get("score", 0), reason="(not logged)",
        atr=r.get("atr", 0), beta=r.get("beta", 0),
        catalyst_tier=r.get("catalyst_tier", ""), news_boost=r.get("news_boost", False),
        mtf_ok=r.get("mtf_ok", False), divergence_free=r.get("divergence_free", False),
    )


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 5:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _split_line(rows, key):
    hi = [r for r in rows if r[key] >= 6]
    lo = [r for r in rows if r[key] < 6]
    f = lambda g: (f"n={len(g):<4d} mean {_mean([x['pnl'] for x in g]):+6.2f}%  "
                   f"win {100 * _mean([1.0 if x['pnl'] > 0 else 0.0 for x in g]):4.0f}%") if g else "n=0"
    return f"score>=6: {f(hi)}   |   score<6: {f(lo)}"


def _first_per_ticker(rows):
    seen, out = set(), []
    for r in sorted(rows, key=lambda x: x["date"]):
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "llm_scorer_eval_cache.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    algo = _load_algo()
    with open(os.path.join(ROOT, "dman_signal_features.json"), encoding="utf-8") as f:
        raw = json.load(f)
    rows = [r for r in raw if r.get("label_outcome") and r.get("entry") and r.get("stop")]
    if args.limit:
        rows = rows[-args.limit:]
    try:
        with open(args.cache, encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    builders = {1: algo._ai_score_prompt_v1, 2: algo._ai_score_prompt_v2}

    def one(job):
        ver, r = job
        key = f"v{ver}|{algo.AI_SCORE_MODEL}|{r['date']}|{r['ticker']}|{r['setup']}"
        if key in cache:
            return key, cache[key]
        regime = {"regime": r.get("regime", "?"), "score": r.get("regime_score", 0), "details": {}}
        prompt = builders[ver](_signal_from_row(r), regime)
        for attempt in range(4):
            text = algo._ai_score_request(prompt, max_tokens=300 if ver >= 2 else 40)
            if text:
                score, p = algo._ai_parse_reply(text, ver)
                return key, {"score": score, "p": p}
            time.sleep(3 * (attempt + 1))
        return key, None

    jobs = [(v, r) for v in builders for r in rows]
    done = 0
    with contextlib.redirect_stdout(io.StringIO()):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for key, val in ex.map(one, jobs):
                if val is not None:
                    cache[key] = val
                done += 1
                if done % 100 == 0:
                    with open(args.cache, "w", encoding="utf-8") as f:
                        json.dump(cache, f)
    with open(args.cache, "w", encoding="utf-8") as f:
        json.dump(cache, f)

    table = []
    for r in rows:
        entry, stop = float(r["entry"]), float(r["stop"])
        row = {"date": r["date"], "ticker": r["ticker"], "pnl": float(r["label_pnl_pct"]),
               "price": entry, "atrp": float(r.get("atr") or 0) / entry * 100 if entry else 0.0}
        ok = True
        for v in builders:
            c = cache.get(f"v{v}|{algo.AI_SCORE_MODEL}|{r['date']}|{r['ticker']}|{r['setup']}")
            if not c or c.get("score") is None:
                ok = False
                break
            row[f"s{v}"] = c["score"]
            row[f"p{v}"] = c["p"] if c.get("p") is not None else c["score"] / 10.0
        if ok:
            table.append(row)

    dates = sorted(x["date"] for x in table)
    cut = dates[len(dates) // 2] if dates else ""
    views = [("ALL ROWS", table),
             (f"LATER HALF (from {cut})", [x for x in table if x["date"] >= cut]),
             ("ONE ROW PER TICKER", _first_per_ticker(table)),
             ("ONE PER TICKER, LATER HALF", _first_per_ticker([x for x in table if x["date"] >= cut]))]
    print(f"model {algo.AI_SCORE_MODEL} | labelled rows {len(rows)} | scored by both prompts {len(table)} "
          f"| tickers {len(set(x['ticker'] for x in table))}")
    for name, v in views:
        if len(v) < 10:
            continue
        pnl = [x["pnl"] for x in v]
        print(f"\n== {name}: n={len(v)}, mean {_mean(pnl):+.2f}%, "
              f"win {100 * _mean([1.0 if p > 0 else 0.0 for p in pnl]):.0f}%")
        print(f"   no-model bars   rank corr: price {spearman([x['price'] for x in v], pnl):+.3f}   "
              f"-ATR% {spearman([-x['atrp'] for x in v], pnl):+.3f}")
        for ver in builders:
            print(f"   prompt v{ver}       rank corr: score {spearman([x[f's{ver}'] for x in v], pnl):+.3f}   "
                  f"p {spearman([x[f'p{ver}'] for x in v], pnl):+.3f}")
            print(f"                   {_split_line(v, f's{ver}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
