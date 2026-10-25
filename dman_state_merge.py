"""Three-way reconciliation of the state files two writers share.

The cloud daemon and the cron workflows (scanner, watchdog, stocktwits,
premarket) all rewrite the same dman_*.json files and push them to main.
Both used to settle a clash with git's line merge -- the daemon through
stash/pull/pop, the workflows through `rebase` + `checkout --theirs` -- and
each quietly dropped the other side's writes. This is the one rule both now
use: a side that did not change a file never overrides one that did, and
when both changed it, the file is merged by meaning where a rule exists.

Deliberately free of dman_algo imports: the persist-state action runs it
mid-rebase as a plain script.

    python dman_state_merge.py FILE...   # resolve conflicted FILEs mid-rebase
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

# Merged field by field by dman_algo.sync_all_state_with_remote(), which runs
# after every pull on both writers -- so the local copy is the one to keep.
SEMANTIC_MERGE_FILES = {
    "dman_positions.json", "dman_scan_log.json", "dman_win_rate.json",
    "dman_live_signals.json", "dman_alpaca_sync.json", "dman_news_log.json",
    "dman_daily_pnl.json", "dman_monthly_pnl.json", "dman_earnings_pending.json",
}
# {key: ISO timestamp} dedup stores. A lost key re-sends an alert (or, for
# __BZONE_ENTRY__, re-arms an entry), so both sides are kept, latest time wins.
TIMESTAMP_DICT_FILES = {"dman_last_alerts.json", "dman_alerts_dedup.json"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, timeout=90)


def merge_text(base: Optional[str], local: str, origin: str) -> Optional[str]:
    """3-way line merge (git merge-file), or None when the two sides overlap."""
    paths = []
    try:
        for txt in (local, base or "", origin):
            fd, path = tempfile.mkstemp(suffix=".merge")
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(txt)
            paths.append(path)
        r = _git("merge-file", "-p", *paths)
        return r.stdout if r.returncode == 0 else None
    finally:
        for path in paths:
            try:
                os.remove(path)
            except OSError:
                pass


def reconcile(p: str, base: Optional[str], local: Optional[str],
              origin: Optional[str]) -> tuple[Optional[str], str]:
    """(content to keep, how) for one state file; None content = file absent."""
    if local == origin:
        return local, "same"
    if local == base:
        return origin, "origin"
    if origin == base:
        return local, "local"
    if local is None or origin is None:
        return origin, "origin (deleted on one side, changed on the other)"
    name = os.path.basename(p)
    if name in SEMANTIC_MERGE_FILES:
        return local, "semantic"
    if name in TIMESTAMP_DICT_FILES:
        merged = json.loads(origin) if origin.strip() else {}
        for k, v in (json.loads(local) if local.strip() else {}).items():
            if k not in merged or str(v) > str(merged[k]):
                merged[k] = v
        return json.dumps(merged), "union"
    if p.endswith(".csv"):
        lines = origin.splitlines()
        seen = set(lines)
        lines += [ln for ln in local.splitlines() if ln not in seen]
        return "\n".join(lines) + "\n", "union"
    merged = merge_text(base, local, origin)
    if merged is not None:
        return merged, "merged"
    return origin, "origin (both sides changed the same lines)"


def _stage(n: int, p: str) -> Optional[str]:
    r = _git("show", f":{n}:{p}")
    return r.stdout if r.returncode == 0 else None


def resolve_rebase_conflicts(state_files: list[str]) -> list[str]:
    """Settle every conflicted path of an in-progress `git rebase origin/main`.

    Mid-rebase, stage 2 is the branch being rebased ONTO (origin) and stage 3
    is the commit being replayed (this run's own writes). A conflicted path
    that is not a state file keeps this run's copy, as the action always did.
    """
    out = _git("diff", "--name-only", "--diff-filter=U")
    report = []
    wanted = set(state_files)
    for p in [x for x in out.stdout.splitlines() if x]:
        if p not in wanted:
            _git("checkout", "--theirs", "--", p)
            _git("add", "--", p)
            report.append(f"{p} [ours, not a state file]")
            continue
        try:
            content, how = reconcile(p, _stage(1, p), _stage(3, p), _stage(2, p))
        except Exception as exc:
            _git("checkout", "--theirs", "--", p)
            _git("add", "--", p)
            report.append(f"{p} [ours, merge failed: {exc}]")
            continue
        if content is None:
            _git("rm", "-q", "--", p)
        else:
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(content)
            _git("add", "--", p)
        report.append(f"{p} [{how}]")
    return report


if __name__ == "__main__":
    for line in resolve_rebase_conflicts(sys.argv[1:]):
        print(f"  reconciled {line}")
