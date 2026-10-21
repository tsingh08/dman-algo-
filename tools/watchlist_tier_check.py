"""Is anything institutionally liquid still filed as a junk smallcap?

The two universes do very different things. TICKER_SECTOR's keys become
WATCHLIST, which is auto-executable. DMAN_SMALLCAP_WATCHLIST is scan/alert
ONLY, and rightly so -- measured 2026-10-16 at -4.97% per five sessions, a
buy-hold median of -95.2%, and 82% of its names down more than half.

Nothing checked which list a name belonged in. On 2026-10-20 that cost a
trade: SPCX -- SpaceX, listed 2026-06-12 in the largest IPO ever, trading
$13.2 BILLION a day, more than AMD -- was sitting in the smallcap list, so the
algo could not act when it reported and moved +8.66% after hours. It had been
written down once and never re-examined.

    python tools/watchlist_tier_check.py            # audit
    python tools/watchlist_tier_check.py --verbose  # show every name checked

Flags two things, and only two, because a checker that reports taste gets
ignored:

  1. OVERLAP. A name in both lists would be handled twice under different
     rules. dman_algo.py now subtracts TICKER_SECTOR from the smallcap list at
     import, so this should be impossible -- this asserts the enforcement
     still exists rather than trusting it.
  2. MISFILED. A smallcap-listed name trading above MIN_LIQUID_DOLLAR_VOL. The
     threshold sits in the gap the data actually has: when this was written
     the four names above it ran $580M-$13,164M a day and the next one down
     was WFF at $133M. A fourfold gap is a real boundary, not a round number.

It deliberately does NOT flag the reverse (a quiet name in WATCHLIST). Low
volume is a reason to size small, not a misclassification, and the auto
universe already gates on liquidity at entry.
"""
import contextlib
import io
import os
import sys

MIN_LIQUID_DOLLAR_VOL = 500e6
LOOKBACK_SESSIONS = 20

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)
with contextlib.redirect_stdout(io.StringIO()):
    import dman_algo as a


def dollar_volume(ticker: str):
    """Average dollar volume over LOOKBACK_SESSIONS, or None if unknown."""
    try:
        df = a.fetch_df(ticker, 60)
        if df is None or len(df) < LOOKBACK_SESSIONS:
            return None
        c = [float(x) for x in df["Close"]][-LOOKBACK_SESSIONS:]
        v = [float(x) for x in df["Volume"]][-LOOKBACK_SESSIONS:]
        return sum(c[i] * v[i] for i in range(len(c))) / float(len(c))
    except Exception:
        return None


def audit(verbose: bool = False) -> int:
    auto = set(a.WATCHLIST)
    small = list(a.DMAN_SMALLCAP_WATCHLIST)
    print("  auto-exec WATCHLIST: %d    alert-only smallcap: %d"
          % (len(auto), len(small)))

    fails = []
    overlap = sorted(auto & set(small))
    if overlap:
        fails.append("%d name(s) in BOTH universes: %s -- one is auto-executable "
                     "and the other alert-only, so these are handled twice under "
                     "different rules" % (len(overlap), ", ".join(overlap)))
    else:
        print("  overlap: none (TICKER_SECTOR is subtracted at import)")

    misfiled, unknown = [], []
    for t in small:
        dv = dollar_volume(t)
        if dv is None:
            unknown.append(t)
            continue
        if verbose:
            print("     %-6s $%9.0fM/day" % (t, dv / 1e6))
        if dv >= MIN_LIQUID_DOLLAR_VOL:
            misfiled.append((dv, t))
    misfiled.sort(reverse=True)
    for dv, t in misfiled:
        fails.append("%s trades $%.0fM/day and is alert-only -- add it to "
                     "TICKER_SECTOR with its sector to make it executable"
                     % (t, dv / 1e6))

    print()
    if unknown:
        print("  note: no usable price history for %d name(s): %s"
              % (len(unknown), ", ".join(sorted(unknown)[:12])))
    if fails:
        print("  %d tier problem(s):" % len(fails))
        for f in fails:
            print("    - %s" % f)
        return 1
    print("  every alert-only name is genuinely illiquid, and the two "
          "universes do not overlap.")
    return 0


if __name__ == "__main__":
    print("Watchlist tier audit (auto-exec vs alert-only)")
    sys.exit(audit("--verbose" in sys.argv))
