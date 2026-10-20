"""Do the UTC crons still cover their intended ET windows after DST ends?

Every cron in .github/workflows is UTC. The market runs on ET. The two drift
by an hour on the first Sunday in November, and the failure is quiet: the jobs
still fire and still self-gate on ET, they just get DELIVERED an hour earlier
in market terms. A range that ends at 20:xx UTC covers 16:xx ET in summer and
15:xx ET in winter, so the last hour of the session goes dark with nothing
erroring.

    python tools/dst_cron_check.py            # audit, non-zero exit on a gap
    python tools/dst_cron_check.py --verbose  # every cron's ET windows

TWO KINDS OF CRON, and conflating them is how the first version of this script
reported eight problems that were all its own:

  RANGE jobs (daemon, scanner, watchdog) fire repeatedly and self-gate on ET.
  Only the END of the window matters. Delivery starting an hour early is
  harmless -- dman_daemon.yml's own comment says the extra EDT run "lands
  after the close, where the job's own ET gate no-ops it" -- so flagging an
  early start is noise, and noise is how a checker gets ignored.

  POINT jobs (the premarket briefing, the open check) fire once for a moment
  that matters. These are handled by shipping BOTH crons, one per clock, with
  a runtime guard picking whichever lands in the window: dman_open_check.yml
  carries 13:50 and 14:50 UTC and says "the guard below runs whichever of
  these lands inside 9:45-10:30 ET and skips the other". So the test is
  whether SOME cron lands in the window under each clock, never whether every
  one does. Judging them individually flagged a correct pair as broken.

INTENDED_ET is the declaration this checks against. It has to be written down
somewhere, and a constant next to the checker beats a comment inside a YAML
file that nothing reads.
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WF = os.path.join(ROOT, ".github", "workflows")

# name -> (kind, first ET hour, last ET hour), or None for jobs with no
# session window to protect (nightly/weekend work an hour's drift cannot hurt).
INTENDED_ET = {
    # Three session starts (9:23 / 12:53 / 16:53 ET). The requirement is not
    # that the last one lands at 16:53 -- it is that an EVENING session starts
    # before the close, so post-market coverage to 20:05 ET exists at all. A
    # start that drifts an hour earlier just hands over earlier, and
    # cancel-in-progress makes the overlap a no-op.
    "dman_daemon.yml": ("range", 8, 15),
    # The ORB and dip-shadow steps need a run after 16:05 ET (the file says
    # so at its 0 22 cron). That is the binding end, not the 18:00 ET nightly
    # readiness slot, which is not session-bound at all.
    "dman_scanner.yml": ("range", 3, 16),
    # Self-gates to 9:30-16:00; what matters is that delivery still reaches
    # the close.
    "dman_watchdog.yml": ("range", 8, 16),
    # Must land BEFORE the open, every time.
    "dman_premarket.yml": ("point", 6, 9),
    # Must land just AFTER it.
    "dman_open_check.yml": ("point", 9, 10),
    "dman_stocktwits.yml": None,
    "dman_review.yml": None,
    "dman_weekend.yml": None,
    "ci.yml": None,
}

CRON = re.compile(r"^\s*-\s*cron:\s*['\"]([^'\"]+)['\"]", re.M)


def hours(field: str) -> list:
    """Expand a cron hour field into the hours it fires on."""
    out = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, _, s = part.partition("/")
            step = int(s or 1)
        if part == "*":
            lo, hi = 0, 23
        elif "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
        else:
            lo = hi = int(part)
        out.update(range(lo, hi + 1, step))
    return sorted(out)


def et_hours(utc_hours: list, offset: int) -> list:
    return sorted({(h - offset) % 24 for h in utc_hours})


def audit(verbose: bool = False) -> int:
    fails, notes = [], []
    for path in sorted(glob.glob(os.path.join(WF, "*.yml"))):
        name = os.path.basename(path)
        spec = INTENDED_ET.get(name, "UNDECLARED")
        crons = CRON.findall(io.open(path, encoding="utf-8").read())
        if not crons:
            continue
        if spec == "UNDECLARED":
            notes.append("%s has %d cron(s) and no INTENDED_ET entry -- add one "
                         "or this file is unaudited" % (name, len(crons)))
            continue
        if spec is None:
            if verbose:
                print("  %-26s %d cron(s), no session window to protect"
                      % (name, len(crons)))
            continue

        kind, lo, hi = spec
        bad = [c for c in crons if len(c.split()) != 5]
        for c in bad:
            fails.append("%s: cron %r is not 5 fields" % (name, c))
        good = [c for c in crons if len(c.split()) == 5]
        if not good:
            continue

        for label, off in (("EDT", 4), ("EST", 5)):
            if kind == "range":
                # Only the end matters; an early start is no-oped by the job.
                reach = max(max(et_hours(hours(c.split()[1]), off)) for c in good)
                ok = reach >= hi
                if verbose or not ok:
                    print("  %-26s %-5s reaches %02d:00 ET, needs %02d:00  %s"
                          % (name, label, reach, hi, "ok" if ok else "GAP"))
                if not ok:
                    fails.append(
                        "%s: under %s the schedule only reaches %02d:00 ET but the "
                        "window ends at %02d:00 -- widen the UTC hour range by one"
                        % (name, label, reach, hi))
            else:
                # Some cron has to land inside the window; that is what the
                # paired-cron-plus-guard pattern is for.
                landing = [h for c in good for h in et_hours(hours(c.split()[1]), off)
                           if lo <= h <= hi]
                ok = bool(landing)
                if verbose or not ok:
                    print("  %-26s %-5s lands at %s ET, window %02d-%02d  %s"
                          % (name, label,
                             ",".join("%02d" % h for h in sorted(set(landing))) or "nothing",
                             lo, hi, "ok" if ok else "GAP"))
                if not ok:
                    fails.append(
                        "%s: under %s no cron lands inside %02d:00-%02d:00 ET -- "
                        "add the other clock's cron alongside the existing one"
                        % (name, label, lo, hi))
    print()
    for n in notes:
        print("  note: %s" % n)
    if fails:
        print("  %d DST coverage problem(s):" % len(fails))
        for f in fails:
            print("    - %s" % f)
        return 1
    print("  every declared window is still covered under BOTH clocks.")
    return 0


if __name__ == "__main__":
    print("DST cron audit (DST ends the first Sunday in November)")
    sys.exit(audit("--verbose" in sys.argv))
