"""Point-in-time test: does NEWS SENTIMENT (the emotional reaction) predict
returns over SPY? Uses Massive/Benzinga /v2/reference/news `insights[]`
(per-article, per-ticker positive/neutral/negative, labelled by an LLM at
ingestion; present from 2024-07). Same bar as tools/pit_harness.py.

Timing (no look-ahead): an article published after 16:00 ET on day t-1 and
up to 16:00 ET on day t belongs to day t; the signal is decided at t's close,
entered at t+1's open, exited at t+hold's close, 0.10% round trip, vs SPY.
Universe: pit_harness bars (delisted included), $10+, $20M/day prior ADV.

RESULT 2026-11-06 (2024-07-02..2026-10-30, 402,005 labels, ~1,740 liquid
names/day): 27 of 27 stock cells FAIL. Sentiment is already in the price:
  good news (net >= +2)          h1 -0.15% vs SPY  t -5.79  every year negative
  good news & stock up >3%       h1 -0.23%         t -3.18
  good news & stock down         h1 -0.12%         t -5.60
  bad news / overreaction / divergence / news surges: all |mean| < 0.8%, no t >= 2
  -> NO stock-level buy signal. Good news is, if anything, a short-term FADE
     (long-only: an avoid filter at most; untested as one).
  Market mood (share of negative labels, z >= 2 vs 60d, n=23 days): SPY 5d
  +0.93% vs +0.30% all days, 20d +2.39% vs +1.25% -- contrarian, same sign as
  VIX >= 30 (5d +1.84%, t 3.3, 2006+). Fear marks short-term bounces.

Usage:
  py -3 tools/news_sentiment_study.py download   # ~850 calls -> news_sentiment.pkl
  py -3 tools/news_sentiment_study.py            # run the tests
"""
import os, sys, time, pickle, datetime as dt
import numpy as np, pandas as pd, requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pit_harness as ph

CACHE = "news_sentiment.pkl"
START = dt.date(2024, 7, 1)
URL = "https://api.massive.com/v2/reference/news"


def download(end: dt.date = None):
    """Rows (published_utc, ticker, sentiment, n_tickers) for every insight."""
    key = os.environ["MASSIVE_API_KEY"]
    end = end or dt.date.today()
    rows = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else []
    done = {r[0][:10] for r in rows}
    d = START
    while d < end:
        nxt = d + dt.timedelta(days=1)
        if d.isoformat() in done:
            d = nxt
            continue
        url, p = URL, {"published_utc.gte": f"{d}T00:00:00Z", "published_utc.lt": f"{nxt}T00:00:00Z",
                       "limit": 1000, "order": "asc", "sort": "published_utc", "apiKey": key}
        while True:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code == 429:
                time.sleep(15)
                continue
            r.raise_for_status()
            j = r.json()
            for a in j.get("results", []):
                ins = a.get("insights") or []
                for i in ins:
                    if i.get("ticker") and i.get("sentiment"):
                        rows.append((a["published_utc"], i["ticker"].upper(), i["sentiment"], len(ins)))
            if not j.get("next_url") or not j.get("results"):
                break
            url, p = j["next_url"], {"apiKey": key}
        if d.day == 1:
            print(d, len(rows), flush=True)
            pickle.dump(rows, open(CACHE, "wb"))
        d = nxt
    pickle.dump(rows, open(CACHE, "wb"))
    return rows


def daily_counts(rows, index, columns):
    """(pos, neg, total) frames, date x ticker, by the 16:00 ET cutoff."""
    df = pd.DataFrame(rows, columns=["ts", "ticker", "sent", "nt"])
    t = pd.to_datetime(df.ts, utc=True).dt.tz_convert("America/New_York")
    # after the close -> next calendar day; then roll weekends/holidays forward
    day = (t + pd.to_timedelta((t.dt.hour >= 16).astype(int), unit="D")).dt.date
    sess = pd.Index(sorted(index))
    pos = np.searchsorted(np.array(sess, dtype="datetime64[D]"), np.array(day, dtype="datetime64[D]"))
    ok = pos < len(sess)
    df = df[ok].assign(date=np.array(sess)[pos[ok]])
    df = df[df.ticker.isin(set(columns))]
    g = df.groupby(["date", "ticker", "sent"]).size().unstack("sent", fill_value=0)
    out = {}
    for s in ("positive", "negative", "neutral"):
        out[s] = (g[s] if s in g else pd.Series(0, index=g.index)).unstack("ticker").reindex(
            index=index, columns=columns).fillna(0)
    return out["positive"], out["negative"], out["positive"] + out["negative"] + out["neutral"]


def run():
    o, c, dv = ph.load_bars()
    rows = pickle.load(open(CACHE, "rb"))
    pos, neg, tot = daily_counts(rows, c.index, c.columns)
    first = pd.to_datetime(min(r[0] for r in rows)).date()
    keep = c.index >= first
    o, c, dv, pos, neg, tot = (x[keep] for x in (o, c, dv, pos, neg, tot))
    liq = (ph._adv(dv) >= 20e6) & (c >= 10)
    r1 = c.pct_change(fill_method=None)
    avg_tot = tot.rolling(20, min_periods=10).mean().shift(1)
    net = pos - neg
    cells = {
        "net sentiment >= +2 (good news)":        (net >= 2) & (tot >= 2),
        "net sentiment <= -2 (bad news)":         (net <= -2) & (tot >= 2),
        "bad news, stock UP on the day":          (net <= -1) & (r1 > 0),
        "bad news, stock DOWN >3% (overreaction)": (net <= -1) & (r1 < -0.03),
        "good news, stock DOWN on the day":       (net >= 1) & (r1 < 0),
        "good news, stock UP >3%":                (net >= 1) & (r1 > 0.03),
        "news surge >= 3x 20d avg":               (tot >= 3 * avg_tot.clip(lower=1)) & (tot >= 3),
        "news surge & positive":                  (tot >= 3 * avg_tot.clip(lower=1)) & (tot >= 3) & (net >= 2),
        "news surge & negative":                  (tot >= 3 * avg_tot.clip(lower=1)) & (tot >= 3) & (net <= -2),
    }
    print(f"news {first}..{c.index[-1]}  labels {len(rows):,}  liquid names/day "
          f"{int(liq.sum(axis=1).median())}")
    for lab, m in cells.items():
        for hold in (1, 5, 20):
            s = ph.evaluate(m & liq, o, c, hold)
            ok, why = ph.verdict(s)
            if s.get("n"):
                print(f"{lab:42s} h{hold:2d} n={s['n']:6d} d={s['days']:3d} mean {s['mean']:+.2f} "
                      f"med {s['median']:+.2f} beat {s['beat_spy']:.0%} t {s['t_clustered']:+.2f} "
                      f"yrs {s['years']} {why[:4]}")
    # market mood: share of negative labels across ALL tickers -> SPY forward
    sp = c["SPY"]
    mood = (neg.sum(axis=1) / tot.sum(axis=1).replace(0, np.nan))
    z = (mood - mood.rolling(60, min_periods=30).mean()) / mood.rolling(60, min_periods=30).std()
    for h in (1, 5, 20):
        f = (sp.shift(-h) / sp - 1) * 100
        for lab, m in (("market mood very negative (z>=2)", z >= 2), ("very positive (z<=-2)", z <= -2)):
            x = f[m].dropna()
            print(f"{lab:42s} h{h:2d} n={len(x):3d} SPY mean {x.mean():+.2f} vs all {f.mean():+.2f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        download()
    else:
        run()
