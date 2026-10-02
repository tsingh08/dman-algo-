"""Fail CI if an operationally critical file has vanished from the repo.

On 2026-09-29 `git status` in a working tree showed
scripts/register_fallback_task.ps1 and scripts/run_fallback_guard.ps1 as
DELETED. Nothing had touched them, no commit removed them, and both were intact
on origin/main -- the cause was never established. They were restored by name
and nothing was lost.

The cause is not the interesting part. A `git add -A` or `git commit -a` at that
moment would have silently deleted the GitHub-Actions-independent fallback guard,
and the only thing standing between that and a push was someone reading
`git status` carefully. That is not a control, it is a habit, and habits fail
exactly when a session is long and the diff is noisy.

This converts "a file quietly disappeared" into a red build. It says nothing
about WHY something went missing; it only guarantees you find out before the
loss reaches main. Every entry here is a file whose absence would remove a
safety net or a scheduled job without any other signal -- losing one of these is
invisible precisely because nothing references it at import time.

Adding a file here is cheap. Removing one deliberately means editing this list in
the same commit, which is the point: a deletion becomes a decision.
"""
import os
import sys

# path -> why its absence would matter
CRITICAL = {
    # Money paths and their tests
    "dman_algo.py": "the algorithm",
    "dman_daemon.py": "the always-on session loop",
    "test_dman_algo.py": "the regression suite",
    "requirements.txt": "pinned dependencies -- an unpinned install is a silent behaviour change",

    # Scheduling. The cloud IS the timer; if a workflow file goes, that job just
    # never runs again and nothing errors.
    ".github/workflows/ci.yml": "the only gate before main",
    ".github/workflows/dman_scanner.yml": "the scanner + briefing/daemon dispatch backstop",
    ".github/workflows/dman_daemon.yml": "the live session",
    ".github/workflows/dman_premarket.yml": "the 8:10 ET briefing",
    ".github/workflows/dman_watchdog.yml": "restarts a stuck daemon",
    ".github/workflows/dman_open_check.yml": "the open-time verification",

    # The local fallback, which is what actually went missing.
    "scripts/register_fallback_task.ps1": "registers the local fallback guard task",
    "scripts/run_fallback_guard.ps1": "the GitHub-Actions-independent fallback guard",
    # Added 2026-10-02 after the SAME disappearance happened a second time, to
    # two files the first manifest did not cover. Cause still unestablished; the
    # response is to widen what a vanished file cannot quietly do.
    "start_daemon.bat": "local daemon launcher",
    "dman_premarket_advisory.md": "the premarket advisory the briefing reads from",

    # Guards that only work if they are present to run.
    "tools/py311_check.py": "catches 3.12-only syntax before the 3.11 jobs hit it",
    "tools/hollow_assert_check.py": "catches tests that cannot fail",

    # Characterization harnesses: without these the goldens cannot be checked,
    # and a missing harness looks identical to a passing one.
    "golden/submit_signals_harness.py": "order-submission golden",
    "golden/monitor_option_harness.py": "option-exit golden",
    "golden/scanner_harness.py": "what trades at all",
    "golden/premarket_briefing_harness.py": "the briefing golden",
    "golden/callee_harness.py": "telegram/callee golden",
}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    missing = [(p, why) for p, why in sorted(CRITICAL.items())
               if not os.path.exists(os.path.join(root, p))]
    print(f"critical files: {len(CRITICAL) - len(missing)}/{len(CRITICAL)} present")
    if not missing:
        return 0
    print(f"\n{len(missing)} MISSING -- refusing the build:\n")
    for p, why in missing:
        print(f"  {p}")
        print(f"     {why}")
    print("\nIf one of these was removed on purpose, delete its line from")
    print("tools/critical_files_check.py in the SAME commit, so the removal is")
    print("a decision someone made rather than something that happened.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
