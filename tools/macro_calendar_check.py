"""Is the hardcoded macro calendar still what the agencies actually publish?

_FOMC_DATES, _CPI_DATES and _PPI_DATES are hand-entered. check_macro_safe()
already refuses to trade when they RUN OUT -- _macro_calendar_state() nags for
a refill -- but nothing ever checked whether the dates it still has are RIGHT.
They were not.

Found 2026-10-16 by accident, while checking whether PPI landed on a Thursday:

    CPI   next release   BLS 2026-11-10   hardcoded 2026-11-12   2 days out
    PPI                  BLS 2026-12-15   hardcoded 2026-12-11   4 days out
    FOMC  December        Fed 2026-12-09   hardcoded 2026-12-16   7 days out

October was correct throughout, and every error is forward-dated -- the
signature of estimates entered past the confirmed window (the file even keeps
_FOMC_LAST_CONFIRMED_YEAR for this reason).

A wrong date fails in the worst direction available. On the real release the
gate does not blackout, which is precisely the binary event it exists to sit
out; on the fictional one it blocks a clean session for nothing. An expired
calendar is loud. A wrong calendar is silent.

    python tools/macro_calendar_check.py          # report drift
    python tools/macro_calendar_check.py --sets   # print corrected date sets

Sources are the agencies themselves:
  CPI/PPI  bls.gov/schedule/news_release/{cpi,ppi}.htm   (forward schedule)
  FOMC     federalreserve.gov/monetarypolicy/fomccalendars.htm
Only FORWARD dates are published, so past entries cannot be rechecked here --
which is fine, because only future ones can still gate a trade.
"""
import datetime as dt
import re
import sys
import urllib.request

UA = "DManAlgo research singh.tanveer2081@gmail.com"
MON = {m: i + 1 for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
FULL = {m: i + 1 for i, m in enumerate(
    "January February March April May June July August September "
    "October November December".split())}


def _text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", "replace")
    return body


def bls_schedule(which: str) -> set:
    """Release dates BLS currently publishes for cpi or ppi."""
    body = _text("https://www.bls.gov/schedule/news_release/%s.htm" % which)
    cells = [re.sub(r"<[^>]+>", "", c).strip()
             for c in re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)]
    out = set()
    for i, c in enumerate(cells):
        if re.match(r"^\d{2}:\d{2} [AP]M$", c) and i >= 1:
            m = re.search(r"([A-Z][a-z]{2})\.?\s+(\d{1,2}),\s+(20\d\d)", cells[i - 1])
            if m:
                out.add(dt.date(int(m.group(3)), MON[m.group(1)], int(m.group(2))))
    return out


def fomc_decision_days() -> set:
    """Second day of each scheduled FOMC meeting -- the decision day."""
    body = _text("https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm")
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))
    out = set()
    for ym in re.finditer(r"(20\d\d) FOMC Meetings", txt):
        year = int(ym.group(1))
        nxt = txt.find("FOMC Meetings", ym.end())
        seg = txt[ym.end():nxt if nxt > 0 else len(txt)]
        # The page carries a footnote -- "A two-day meeting is scheduled for
        # January 25-26, 2028" -- inside the last year's block. Without this
        # cut the checker reported a MISSING 2027-01-26, which is the 2028
        # note mis-stamped, and I would have written a wrong date while
        # fixing wrong dates.
        _note = seg.find("Note:")
        if _note > 0:
            seg = seg[:_note]
        for m in re.finditer(r"(%s)\s+(\d{1,2})-(\d{1,2})" % "|".join(FULL), seg):
            try:
                out.add(dt.date(year, FULL[m.group(1)], int(m.group(3))))
            except ValueError:
                pass
    return out


def bea_pce_schedule() -> set:
    """Release dates BEA currently publishes for Personal Income and Outlays.

    PCE is the one macro calendar the first version of this checker could not
    verify, and it is also the one closest to running out (2026-12-23, already
    nagging), so "unverifiable" was the wrong place to leave it.

    The schedule page lists a release line per row as, flattened:

        October 29 8:30 AM News Personal Income and Outlays, September 2026

    The RELEASE year is not on the row -- only the reference month carries one.
    PCE is published about a month after its reference month, so the release
    year equals the reference year except when the release month wraps round
    to January, which is the December-reference release.
    """
    body = _text("https://www.bea.gov/news/schedule")
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))
    out = set()
    pat = (r"([A-Z][a-z]+)\s+(\d{1,2})\s+\d{1,2}:\d{2}\s*[AP]M\s+N\s*ews\s+"
           r"Personal Income and Outlays,\s+([A-Z][a-z]+)\s+(20\d\d)")
    for m in re.finditer(pat, txt):
        rel_mon, day, ref_mon, ref_year = m.groups()
        if rel_mon not in FULL or ref_mon not in FULL:
            continue
        year = int(ref_year)
        if FULL[rel_mon] < FULL[ref_mon]:
            year += 1          # January release of a December reference month
        try:
            out.add(dt.date(year, FULL[rel_mon], int(day)))
        except ValueError:
            continue
    return out


def main() -> int:
    sys.path.insert(0, ".")
    import dman_algo as a

    today = a._et_today()
    checks = (
        ("CPI", a._CPI_DATES, lambda: bls_schedule("cpi")),
        ("PPI", a._PPI_DATES, lambda: bls_schedule("ppi")),
        ("FOMC", a._FOMC_DATES, fomc_decision_days),
        ("PCE", a._PCE_DATES, bea_pce_schedule),
    )
    bad = 0
    corrected = {}
    _drift_lines: list = []
    for name, hard, fetch in checks:
        try:
            live = fetch()
        except Exception as exc:
            print("  %-5s could not verify (%s) -- not treated as drift" % (name, exc))
            continue
        if not live:
            print("  %-5s source returned nothing -- not treated as drift" % name)
            continue
        horizon = max(live)
        mine = {d for d in hard if today <= d <= horizon}
        theirs = {d for d in live if d >= today}
        missing = sorted(theirs - mine)     # real releases the algo does not know
        extra = sorted(mine - theirs)       # algo dates the agency does not list
        ok = not missing and not extra
        print("  %-5s %s  (%d published through %s)"
              % (name, "ok" if ok else "DRIFT", len(theirs), horizon))
        for d in missing:
            print("        MISSING  %s  <- agency publishes this, gate will NOT fire" % d)
            _drift_lines.append("%s MISSING %s (gate will NOT fire)" % (name, d))
        for d in extra:
            print("        EXTRA    %s  <- not on the agency schedule" % d)
            _drift_lines.append("%s EXTRA %s (not on the agency schedule)" % (name, d))
        if not ok:
            bad += 1
        corrected[name] = sorted({d for d in hard if d < today} | theirs)

    if "--sets" in sys.argv:
        print()
        for name, dates in corrected.items():
            print("# %s -- agency-verified %s" % (name, dt.date.today()))
            print("_%s_DATES = {" % name)
            for d in dates:
                print("    date(%d, %2d, %2d)," % (d.year, d.month, d.day))
            print("}")
            print()

    if bad:
        print()
        print("  %d calendar(s) drifted. A wrong date fails OPEN on the real" % bad)
        print("  release and blocks a clean session on the fictional one.")
        print("  Re-run with --sets for corrected date sets.")
        # --alert lets the weekend cloud routine run this unattended. Without
        # it the check is only as current as the last time someone typed it,
        # which is how the dates drifted for months in the first place.
        if "--alert" in sys.argv:
            try:
                a.send_telegram(
                    "⚠️ <b>Macro calendar has drifted</b>" + chr(10)
                    + chr(10).join(_drift_lines[:12]) + chr(10) + chr(10)
                    + "A wrong date means NO blackout on the real release and a "
                    + "blackout on a clean session. Fix with "
                    + "<code>tools/macro_calendar_check.py --sets</code>.")
            except Exception as exc:
                print("  (could not send the alert: %s)" % exc)
        return 1
    print()
    print("  All verifiable calendars match their agency schedule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
