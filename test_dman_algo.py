"""
Regression suite targeting the specific bug CLASSES that have actually
broken production this week, not an exhaustive line-by-line test of every
function. Each test exists because something in its class already caused
a real incident:

  - test_argparse_*        : the StockTwits mode wasn't in argparse's
                              choices list for days, silently erroring on
                              every scheduled run.
  - test_bars_to_df_*,
    test_fetch_df_*,
    test_simulate_trade_*  : yfinance silently changed its default column
                              shape on a version bump; _simulate_trade_outcome
                              crashed for a full day before being traced.
  - test_merge_positions_* : the daemon and cron scanner can both update
                              the same tracked position from separate
                              processes; a naive merge could regress a
                              stop-raise back to a less-protective value.
  - test_two_processes_*,
    test_identical_local_*,
    test_max_entries_*,
    test_new_remote_*,
    test_plain_string_*    : merge_json_lists() covers the high-frequency
                              append-only files (scan_log, win_rate,
                              live_signals, alpaca_sync) where a whole-file
                              `git checkout --theirs` conflict resolution
                              silently dropped a full trading day of scan
                              entries. Locks in the fix after two earlier
                              designs each had a real, only-caught-by-
                              testing flaw (a full-history dedup that
                              collapsed unrelated pre-existing duplicates,
                              and a concat-before-diff bug that doubled the
                              file whenever local == remote).
  - test_check_macro_safe_*: blackout windows (FOMC, tariff-deadline-style
                              one-off events) gate real capital — an
                              off-by-one here means trading through an
                              event that should have been sat out, or
                              needlessly sitting out a clear day.
  - test_check_mtf_*       : the catalyst override that unblocked the
                              GOOGL/TSLA-style earnings gap play must only
                              fire for the two setups it's scoped to, and
                              only with a real news catalyst present.

Run locally:   py -3 -m unittest test_dman_algo -v
Run in CI:     python -m unittest test_dman_algo -v
No network calls are made — every yfinance/Alpaca/GitHub dependency is
mocked so this suite is fast and deterministic regardless of market hours.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock

# Credentials must be blank BEFORE import — several module-level constants
# read env vars at import time, and a stray real key in the environment
# (already bit us once this week during manual testing) must never let a
# test suite make a live network call or, worse, a live order attempt.
for _k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "APCA_API_KEY_ID",
           "APCA_API_SECRET_KEY", "ANTHROPIC_API_KEY", "BENZINGA_API_KEY",
           "BENZINGA_EARNING_API_KEY", "MASSIVE_API_KEY"):
    os.environ[_k] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dman_algo as a  # noqa: E402

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dman_algo.py")
with open(_SRC_PATH, encoding="utf-8") as _f:
    _SOURCE = _f.read()


class TestArgparseDispatchConsistency(unittest.TestCase):
    """Prevents the exact StockTwits incident from ever recurring: a mode
    dispatched in main() but missing from argparse's choices list errors
    out on every single invocation, silently, for as long as it takes
    someone to notice the workflow has been failing."""

    def _get_choices(self) -> set[str]:
        m = re.search(r'--mode",\s*default="scan",\s*choices=\[(.*?)\]',
                      _SOURCE, re.DOTALL)
        self.assertIsNotNone(m, "could not locate --mode choices list in source")
        return set(re.findall(r'"([\w-]+)"', m.group(1)))

    def _get_dispatched_modes(self) -> set[str]:
        return set(re.findall(r'args\.mode\s*==\s*"([\w-]+)"', _SOURCE))

    def test_every_dispatched_mode_is_a_valid_choice(self):
        choices = self._get_choices()
        dispatched = self._get_dispatched_modes()
        missing = dispatched - choices
        self.assertEqual(
            missing, set(),
            f"mode(s) {missing} are dispatched via 'elif args.mode == ...' "
            f"but missing from argparse's choices list — every invocation "
            f"with this mode will fail immediately with an argparse error, "
            f"exactly like the StockTwits workflow did for days."
        )

    def test_every_choice_has_a_dispatch_branch(self):
        # Softer check: a choice with no branch is dead code, not a crash
        # risk, but still worth knowing about.
        choices = self._get_choices()
        dispatched = self._get_dispatched_modes()
        orphaned = choices - dispatched
        self.assertEqual(
            orphaned, set(),
            f"mode(s) {orphaned} are valid --mode choices but have no "
            f"'elif args.mode == ...' branch — dead/unreachable option."
        )


class TestYfinanceMultiIndexDefense(unittest.TestCase):
    """Direct regression tests for the incident that cost hours: newer
    yfinance versions return MultiIndex columns even for single-ticker
    downloads. Every function that calls yf.download() directly must
    survive that shape without crashing, regardless of what version of
    yfinance happens to be installed at test time."""

    @staticmethod
    def _multiindex_df(rows: int = 30):
        import pandas as pd
        import numpy as np
        idx = pd.date_range("2026-06-01", periods=rows, freq="D")
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["TESTX"]])
        data = np.random.uniform(10, 20, size=(rows, len(cols)))
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_bars_to_df_flattens_multiindex_or_rejects_stale(self):
        # _bars_to_df takes a list of Bar-like objects, not a DataFrame —
        # verify it handles too-few and stale-date cases without raising,
        # which is the actual failure mode class (silent exception ->
        # everything treated as "corrupted").
        class FakeBar:
            def __init__(self, ts):
                self.open = self.high = self.low = self.close = 10.0
                self.volume = 1000
                self.timestamp = ts
        too_few = [FakeBar(datetime(2026, 7, 20))]
        self.assertIsNone(a._bars_to_df(too_few, min_bars=20))

    def test_simulate_trade_outcome_survives_multiindex_columns(self):
        """The actual incident. If this test ever fails again, the fix
        regressed — do not weaken this test to make it pass."""
        mock_df = self._multiindex_df(rows=30)
        with patch.object(a.yf, "download", return_value=mock_df):
            try:
                result = a._simulate_trade_outcome(
                    ticker="TESTX", entry=15.0, stop=12.0,
                    target1=20.0, target2=25.0, bias="LONG",
                    start_date="2026-06-01",
                )
            except TypeError as exc:
                self.fail(
                    f"_simulate_trade_outcome crashed on MultiIndex-column "
                    f"input — the exact 2026-07-24 production incident has "
                    f"regressed: {exc}"
                )
            # Function may legitimately return None (e.g. dates line up
            # such that the loop finds nothing conclusive) — the only
            # requirement is that it doesn't raise.
            self.assertTrue(result is None or isinstance(result, dict))

    def test_fetch_df_survives_multiindex_columns(self):
        mock_df = self._multiindex_df(rows=30)
        a._cache.clear()
        with patch.object(a.yf, "download", return_value=mock_df):
            with patch.object(a, "_fetch_alpaca_daily", return_value=None):
                try:
                    result = a.fetch_df("TESTX")
                except TypeError as exc:
                    self.fail(f"fetch_df crashed on MultiIndex-column input: {exc}")
        a._cache.clear()


class TestSetupMinConfluenceOverrides(unittest.TestCase):
    """Added 2026-08-14 after reviewing a month of live trades: 'Low Float
    Catalyst' had a 0% win rate on every non-BE trade (IOTR -8.7%,
    CLRO -28.1%, FGL -37.9%), avg loss -24.9% -- confirmed on FGL/CLRO
    specifically that plain-stop fills on these illiquid low-float names
    slip 8-18 points past the intended stop with no liquidity in between.
    SETUP_MIN_CONFLUENCE is the existing, already-battle-tested lever for
    this (same one Gap & Short / Vol Breakdown / MACD Bear already use) --
    this locks in that the override actually exists and is set high enough
    to matter (above VOLATILE_MIN_CONFLUENCE, not just above the default)."""

    def test_low_float_catalyst_has_a_stricter_override(self):
        self.assertIn("Low Float Catalyst", a.SETUP_MIN_CONFLUENCE)
        self.assertGreaterEqual(a.SETUP_MIN_CONFLUENCE["Low Float Catalyst"],
                                a.VOLATILE_MIN_CONFLUENCE)


class TestMergePositionsSnapshots(unittest.TestCase):
    """The daemon (60s cadence) and hourly cron scanner can both update the
    same tracked position from separate processes/checkouts. These tests
    lock in the rule that made that safe: never let a merge regress a
    stop-raise or resurrect a fully-sold position."""

    def test_more_protective_stop_wins(self):
        local  = [{"ticker": "SMCI", "stop": 2.0, "shares": 200}]
        remote = [{"ticker": "SMCI", "stop": 4.0, "shares": 200}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["stop"], 4.0,
                         "a stale, less-protective stop must never beat a "
                         "raised one, regardless of which side is 'local'")

    def test_fewer_shares_wins_tie_on_partial_sell(self):
        local  = [{"ticker": "SMCI", "stop": 4.0, "shares": 200}]
        remote = [{"ticker": "SMCI", "stop": 4.0, "shares": 100}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["shares"], 100,
                         "a partial sell that already executed must not "
                         "be reverted by a stale full-size snapshot")

    def test_ticker_on_only_one_side_is_kept_not_dropped(self):
        local  = [{"ticker": "AAPL", "stop": 200, "shares": 100}]
        remote: list[dict] = []
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(len(merged), 1,
                         "a position missing from one side (e.g. not yet "
                         "synced) must be kept, not silently dropped — "
                         "losing a stop-raise is worse than briefly "
                         "resurrecting an already-closed ticket")

    def test_identical_snapshots_produce_no_spurious_change(self):
        snap = [{"ticker": "AAPL", "stop": 200, "shares": 100}]
        merged = a.merge_positions_snapshots(snap, snap)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["stop"], 200)

    def test_two_options_legs_on_same_ticker_both_survive(self):
        # Confirmed live 2026-08-11: a real SMCI call+put earnings strangle
        # shares ticker "SMCI" for both legs. The old ticker-keyed merge
        # collapsed "4 local + 4 remote -> 3 merged", silently dropping the
        # call leg from tracking entirely (no stop, no milestone alerts) for
        # hours before it was caught by manual account inspection.
        call = {"ticker": "SMCI", "setup": "Options Call SMCI260814C00034000 ($34C exp 2026-08-14)",
                "stop": 0.66, "shares": 300}
        put  = {"ticker": "SMCI", "setup": "Options Put SMCI260814P00029500 ($29.5P exp 2026-08-14)",
                "stop": 0.565, "shares": 200}
        merged = a.merge_positions_snapshots([call, put], [call, put])
        setups = {m["setup"] for m in merged}
        self.assertEqual(len(merged), 2,
                          "both options legs must survive a merge even though they share a ticker")
        self.assertIn(call["setup"], setups)
        self.assertIn(put["setup"], setups)

    def test_closed_identity_absent_from_local_is_not_resurrected_from_remote(self):
        # Reproduces the real 2026-08-13 FGL incident: sync_alpaca_fills()
        # confirmed (via a real Alpaca fill) that FGL is closed and removed
        # it locally -- local_list no longer has it -- but origin/main
        # hasn't been pushed with that removal yet, so remote_list still
        # does. Without the closed_identities guard, the plain union rule
        # resurrected it on every single git_sync() cycle for 35+ minutes.
        local: list[dict] = []
        remote = [{"ticker": "FGL", "setup": "Low Float Catalyst", "stop": 0.64, "shares": 100}]
        merged = a.merge_positions_snapshots(local, remote, closed_identities={"FGL"})
        self.assertEqual(merged, [], "a confirmed-closed identity must never be "
                                      "re-added just because remote hasn't caught up yet")

    def test_closed_identity_present_in_local_is_not_blocked(self):
        # A same-day legitimate re-entry on the same ticker must never be
        # silently dropped by a stale tombstone -- local's own presence is
        # the authoritative signal here, not the tombstone.
        local  = [{"ticker": "FGL", "setup": "Low Float Catalyst", "stop": 1.1, "shares": 100}]
        remote: list[dict] = []
        merged = a.merge_positions_snapshots(local, remote, closed_identities={"FGL"})
        self.assertEqual(len(merged), 1, "a freshly-reopened position must survive "
                                          "even if its identity is still tombstoned")
        self.assertEqual(merged[0]["stop"], 1.1)

    def test_closed_identities_defaults_to_no_effect(self):
        # Omitting the new parameter entirely must reproduce the exact old
        # union-keep behavior -- every existing caller passes nothing.
        local: list[dict] = []
        remote = [{"ticker": "AAPL", "stop": 200, "shares": 100}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(len(merged), 1)


class TestClosedIdentityTombstone(unittest.TestCase):
    """Added 2026-08-13 alongside the FGL-resurrection fix: _mark_identity_closed()
    writes into the sync-state file's closed_identities dict (mutating in
    place, same pattern as recorded_ids), and _recent_closed_identities()
    reads it back with a TTL. These are the two primitives
    merge_positions_snapshots()'s new guard depends on for correctness."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.write("{}")
        self._tmp.close()
        self._patch = patch.object(a, "ALPACA_SYNC_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_mark_then_recent_round_trips(self):
        state = a._load_sync_state()
        a._mark_identity_closed(state, "FGL", "Low Float Catalyst")
        a._save_sync_state(state)
        self.assertIn("FGL", a._recent_closed_identities())

    def test_expired_entry_is_not_returned(self):
        state = a._load_sync_state()
        a._mark_identity_closed(state, "FGL", "Low Float Catalyst")
        # Backdate past the TTL directly, bypassing the real clock.
        ident = a._position_identity("FGL", "Low Float Catalyst")
        state["closed_identities"][ident] = time.time() - a._CLOSED_IDENTITY_TOMBSTONE_S - 1
        a._save_sync_state(state)
        self.assertNotIn("FGL", a._recent_closed_identities())

    def test_options_leg_uses_occ_identity_not_bare_ticker(self):
        state = a._load_sync_state()
        a._mark_identity_closed(state, "SMCI", "Options Call SMCI260814C00034000 ($34C exp 2026-08-14)")
        a._save_sync_state(state)
        closed = a._recent_closed_identities()
        self.assertIn("SMCI260814C00034000", closed)
        self.assertNotIn("SMCI", closed)

    def test_missing_state_file_returns_empty_set_not_a_crash(self):
        os.unlink(self._tmp.name)
        self.assertEqual(a._recent_closed_identities(), set())


class TestSyncPositionsWithRemoteTombstone(unittest.TestCase):
    """End-to-end reproduction of the real 2026-08-13 incident: FGL closed
    (stopped out, -37.89%) and sync_alpaca_fills() correctly removed it
    from the local dman_positions.json, but origin/main still had it (not
    yet pushed) -- and sync_positions_with_remote()'s merge kept adding it
    right back on every single cycle, before the commit that would have
    caught origin up ever got a chance to land. Reproduces the exact
    local-empty / remote-has-it / tombstone-present shape via a mocked
    `git show origin/main:...` call, same pattern as TestSyncScanLogWithRemote."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._pos_tmp.close()
        self._sync_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._sync_tmp.write("{}")
        self._sync_tmp.close()
        self._patches = [
            patch.object(a, "POSITIONS_FILE", self._pos_tmp.name),
            patch.object(a, "ALPACA_SYNC_FILE", self._sync_tmp.name),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.unlink(self._pos_tmp.name)
        os.unlink(self._sync_tmp.name)

    def test_tombstoned_close_survives_a_stale_remote_copy(self):
        with open(self._pos_tmp.name, "w") as f:
            json.dump([], f)   # FGL already closed locally
        state = a._load_sync_state()
        a._mark_identity_closed(state, "FGL", "Low Float Catalyst")
        a._save_sync_state(state)

        remote = [{"ticker": "FGL", "setup": "Low Float Catalyst", "stop": 0.64, "shares": 100}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_positions_with_remote()

        with open(self._pos_tmp.name) as f:
            result = json.load(f)
        self.assertEqual(result, [], "the confirmed-closed FGL must not "
                                      "reappear just because remote is stale")

    def test_without_a_tombstone_the_old_union_behavior_still_applies(self):
        # Sanity check that the fix is additive, not a behavior change for
        # the ordinary "not yet synced" case this rule exists to protect.
        with open(self._pos_tmp.name, "w") as f:
            json.dump([], f)
        remote = [{"ticker": "AAPL", "setup": "Gap & Hold", "stop": 200, "shares": 100}]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_positions_with_remote()
        with open(self._pos_tmp.name) as f:
            result = json.load(f)
        self.assertEqual(len(result), 1, "a position missing locally with no "
                                          "tombstone must still be kept (union)")


class TestPositionTrackerMultiLegOptions(unittest.TestCase):
    """PositionTracker.open()/close() dedup/remove by ticker alone, which
    collides two options legs (call + put) sharing the same underlying
    ticker. _position_identity() (OCC symbol for options, ticker otherwise)
    fixes this — these tests lock in that a second leg's open/close can't
    silently wipe out the first."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.write("[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _call_pos(self):
        return a.OpenPosition(
            ticker="SMCI", bias="LONG", setup="Options Call SMCI260814C00034000 ($34C exp 2026-08-14)",
            entry=1.32, stop=0.66, target1=1.98, target2=3.3, shares=300, entry_date="2026-08-10",
        )

    def _put_pos(self):
        return a.OpenPosition(
            ticker="SMCI", bias="SHORT", setup="Options Put SMCI260814P00029500 ($29.5P exp 2026-08-14)",
            entry=1.13, stop=0.565, target1=1.695, target2=2.825, shares=200, entry_date="2026-08-10",
        )

    def test_opening_second_leg_does_not_evict_the_first(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._call_pos())
        pt.open(self._put_pos())
        self.assertEqual(len(pt.positions), 2)

    def test_closing_one_leg_by_occ_symbol_leaves_the_other_open(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._call_pos())
        pt.open(self._put_pos())
        pt.close("SMCI", occ_symbol="SMCI260814C00034000")
        self.assertEqual(len(pt.positions), 1)
        self.assertTrue(pt.positions[0].setup.startswith("Options Put"))

    def test_bare_ticker_close_does_not_match_either_options_leg(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._call_pos())
        pt.open(self._put_pos())
        found = pt.close("SMCI")
        self.assertIsNone(found, "a bare-ticker close must not ambiguously match an options leg")
        self.assertEqual(len(pt.positions), 2)


class TestMergeJsonLists(unittest.TestCase):
    """merge_json_lists() covers the high-frequency append-only files
    (scan_log, win_rate, live_signals pending, alpaca_sync recorded_ids)
    that merge_positions_snapshots() doesn't handle — a whole-file
    `git checkout --theirs` conflict resolution silently drops whichever
    side's new entry loses, even though both were genuinely new. Two
    earlier designs of this function were wrong in ways only caught by
    testing against the real, 6000+ entry dman_win_rate.json before
    shipping: a full-history content dedup that collapsed pre-existing
    (unrelated) duplicates as an undiscussed side effect, and a
    concat-before-diff bug that doubled the file's size whenever local
    and remote were already identical. These tests lock in the fix."""

    def test_two_processes_each_add_a_different_new_entry(self):
        base = ["BASE"]
        local = base + ["JOB_A"]
        remote = base + ["JOB_B"]
        merged = a.merge_json_lists(local, remote, key_fn=lambda x: x)
        self.assertEqual(set(merged), {"BASE", "JOB_A", "JOB_B"},
                         "both sides' genuinely new entries must survive — "
                         "this is the exact race that caused a full trading "
                         "day of dman_scan_log.json entries to vanish")

    def test_identical_local_and_remote_do_not_double(self):
        snap = [{"ticker": "AAPL", "date": "2024-01-01", "result": "win"}] * 3
        merged = a.merge_json_lists(
            snap, snap, key_fn=lambda x: json.dumps(x, sort_keys=True))
        self.assertEqual(len(merged), 3,
                         "when local == remote, remote must contribute "
                         "nothing — an earlier design doubled the file's "
                         "size on every sync because it concatenated before "
                         "checking for already-shared history")

    def test_max_entries_caps_positionally_not_by_date_sort(self):
        # Real dman_win_rate.json data is NOT chronologically ordered by
        # its own "date" field (backtest runs over arbitrary historical
        # windows get appended in whatever order they were processed).
        # Sorting by "date" before capping to max_entries would silently
        # keep whichever entries have the highest date value instead of
        # the N most recently appended ones — corrupting what
        # rolling_stats()'s "last N trades" means for Kelly sizing.
        records = (
            [{"ticker": "OLD", "date": "2025-12-31"}] +
            [{"ticker": f"KEEP{i}", "date": "2020-01-01"} for i in range(5)]
        )
        merged = a.merge_json_lists(
            records, records,
            key_fn=lambda x: json.dumps(x, sort_keys=True),
            max_entries=5,
        )
        self.assertEqual(merged, records[-5:],
                         "truncation must be positional (last N as appended), "
                         "matching WinRateTracker._save()'s own records[-500:] "
                         "— not re-sorted by an unreliable date field")

    def test_new_remote_entry_survives_the_cap(self):
        local = [{"ticker": f"T{i}", "date": "2020-01-01"} for i in range(5)]
        new_entry = {"ticker": "NEWEST", "date": "2026-07-27"}
        remote = local + [new_entry]
        merged = a.merge_json_lists(
            local, remote,
            key_fn=lambda x: json.dumps(x, sort_keys=True),
            max_entries=5,
        )
        self.assertIn(new_entry, merged,
                     "the actual race being fixed: remote's genuinely new "
                     "entry must not be discarded just because the combined "
                     "list now exceeds max_entries")
        self.assertEqual(len(merged), 5)

    def test_plain_string_entries_and_no_key_fn(self):
        # dman_alpaca_sync.json's recorded_ids is a flat list of plain
        # strings, not dicts — must not assume dict shape.
        merged = a.merge_json_lists(["id1", "id2"], ["id2", "id3"],
                                    key_fn=lambda x: x)
        self.assertEqual(merged, ["id1", "id2", "id3"])
        # No key_fn at all: no identity to dedup on, fall back to
        # whichever list is at least as complete rather than blindly
        # concatenating (which double-counts the common already-synced case).
        self.assertEqual(a.merge_json_lists(["x"], ["y"], key_fn=None), ["x"])


class TestAppendScanLog(unittest.TestCase):
    """Confirmed live 2026-08-12: _append_scan_log() used a positional
    log[-max_entries:] slice, unfixed by the 2026-08-11 sync_scan_log_
    with_remote() sort-by-ts fix -- that fix only touched the cross-
    process MERGE step, not this function, which runs FIRST on every
    single scan, before any merge happens. Real trading day evidence:
    entries from 6 days earlier were still present alongside only-
    through-11:47am entries from the CURRENT day, with every scan from
    11:54am to market close silently missing -- this function's own
    truncation dropped them the moment the on-disk file wasn't already
    perfectly sorted, because it trusted list position instead of the ts
    field every entry already carries."""

    def _entry(self, ts):
        return {"ts": ts, "signals": 0, "universe": "curated"}

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "SCAN_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_new_entry_survives_when_file_starts_out_of_order(self):
        # Reproduces the real incident: the on-disk file already has old
        # entries (6 days back) interleaved with recent ones, NOT sorted
        # by ts -- exactly what a stale git-level conflict resolution can
        # leave behind. A positional [-20:] slice on this unsorted input
        # can drop the brand-new append; sorting first cannot.
        scrambled = ([self._entry("2026-08-06T10:00:00-04:00")] * 19 +
                     [self._entry("2026-08-12T11:47:00-04:00")])
        with open(a.SCAN_LOG_FILE, "w") as f:
            json.dump(scrambled, f)
        a._append_scan_log(self._entry("2026-08-12T11:56:00-04:00"), max_entries=20)
        with open(a.SCAN_LOG_FILE) as f:
            result = json.load(f)
        self.assertIn("2026-08-12T11:56:00-04:00", [e["ts"] for e in result],
                      "the entry just appended must survive regardless of on-disk order")
        self.assertEqual(result[-1]["ts"], "2026-08-12T11:56:00-04:00",
                          "newest entry must be last (ascending order), matching "
                          "print_scan_log()'s reversed() convention")

    def test_oldest_entries_evicted_first_when_over_cap(self):
        entries = [self._entry(f"2026-08-{d:02d}T10:00:00-04:00") for d in range(1, 21)]
        with open(a.SCAN_LOG_FILE, "w") as f:
            json.dump(entries, f)
        a._append_scan_log(self._entry("2026-08-21T10:00:00-04:00"), max_entries=20)
        with open(a.SCAN_LOG_FILE) as f:
            result = json.load(f)
        self.assertEqual(len(result), 20)
        self.assertNotIn("2026-08-01T10:00:00-04:00", [e["ts"] for e in result])
        self.assertIn("2026-08-21T10:00:00-04:00", [e["ts"] for e in result])

    def test_missing_file_starts_fresh(self):
        a._append_scan_log(self._entry("2026-08-12T10:00:00-04:00"))
        with open(a.SCAN_LOG_FILE) as f:
            result = json.load(f)
        self.assertEqual(len(result), 1)


class TestLogScanHalt(unittest.TestCase):
    """Added 2026-08-13: run_pro_scanner()'s three circuit breakers
    (consecutive-loss / monthly-loss / daily-loss) all `return []` before
    ever reaching the function's own _append_scan_log() call ~350 lines
    later -- meaning the scan log went completely dark for the rest of
    2026-08-13 the moment the daily loss limit tripped (~10:40 AM ET),
    with zero visible difference from the scanner silently being broken.
    Confirmed live: 9+ real scanner runs that day, zero scan_log entries.
    _log_scan_halt() closes that gap without needing a full regime lookup
    for a run that never got that far."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "SCAN_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_writes_a_minimal_halted_entry(self):
        a._log_scan_halt("daily_loss_limit", ["AAPL", "MSFT"], 85)
        with open(a.SCAN_LOG_FILE) as f:
            result = json.load(f)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertTrue(entry["halted"])
        self.assertEqual(entry["halt_reason"], "daily_loss_limit")
        self.assertEqual(entry["tickers_total"], 2)
        self.assertEqual(entry["signals"], 0)

    def test_failure_is_silent_never_raises(self):
        with patch.object(a, "_append_scan_log", side_effect=Exception("disk full")):
            a._log_scan_halt("daily_loss_limit", [], 85)   # must not raise

    def test_print_scan_log_renders_a_halted_entry(self):
        a._log_scan_halt("consecutive_losses", ["AAPL"], 75)
        with patch("builtins.print") as mock_print:
            a.print_scan_log()
        rendered = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("HALTED", rendered)
        self.assertIn("consecutive_losses", rendered)


class TestRunProScannerHaltLogging(unittest.TestCase):
    """Confirms run_pro_scanner() actually calls _log_scan_halt() from
    each of the three circuit-breaker early returns -- the point of
    _log_scan_halt existing is defeated if a guard forgets to call it."""

    def setUp(self):
        self._patches = [
            patch.object(a, "send_telegram", return_value=True),
            patch.object(a, "_is_duplicate_alert", return_value=False),
            patch.object(a, "_save_last_alert", return_value=None),
            patch.object(a, "resolve_live_outcomes", return_value=0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_consecutive_loss_guard_logs_a_halt(self):
        with patch.object(a.WinRateTracker, "rolling_stats",
                          return_value={"consec_losses": a.MAX_CONSEC_LOSSES, "consec_wins": 0}), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "consecutive_losses")

    def test_monthly_loss_guard_logs_a_halt(self):
        with patch.object(a.WinRateTracker, "rolling_stats",
                          return_value={"consec_losses": 0, "consec_wins": 0}), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "monthly_loss_limit")

    def test_daily_loss_guard_logs_a_halt(self):
        with patch.object(a.WinRateTracker, "rolling_stats",
                          return_value={"consec_losses": 0, "consec_wins": 0}), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "get_todays_loss", return_value=-(a.DAILY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "daily_loss_limit")


class TestSyncScanLogWithRemote(unittest.TestCase):
    """Confirmed live 2026-08-11: sync_scan_log_with_remote() used to cap
    the merged list positionally (merged[-20:]), matching
    merge_json_lists()'s own local+new-from-remote concatenation order.
    Whenever local and remote had each independently been appended-and-
    capped-at-20 by separate runs (the normal case — hourly scanner + 60s
    daemon) and diverged in content, every remote entry could look "new"
    to local's byte-for-byte key_fn, landing all of remote after local in
    the concatenation — so merged[-20:] kept ONLY remote's tail, silently
    discarding 100% of local's fresh entries including the one just
    appended this run. The rewritten local file then matched origin
    exactly, so the next git diff found nothing to commit — repeating
    forever with no error. dman_scan_log.json sat frozen at a full day-old
    snapshot this way. The fix sorts by the always-reliable wall-clock
    `ts` field before capping, so genuinely newest entries always survive
    regardless of which side of the concatenation produced them."""

    def _entry(self, ts):
        return {"ts": ts, "signals": 0, "universe": "curated"}

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "SCAN_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_fresh_local_entry_survives_a_diverged_same_size_remote(self):
        # local: 19 old entries + 1 brand-new one just appended this run.
        local = [self._entry(f"2026-08-01T{h:02d}:00:00-04:00") for h in range(19)]
        local.append(self._entry("2026-08-11T21:00:00-04:00"))   # the newest entry
        # remote: a fully-diverged, equally-sized, older snapshot (the
        # "frozen a day ago" scenario) — none of it byte-matches local.
        remote = [self._entry(f"2026-08-10T{h:02d}:30:00-04:00") for h in range(20)]
        with open(self._tmp.name, "w") as f:
            json.dump(local, f)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(remote))
            a.sync_scan_log_with_remote()
        with open(self._tmp.name) as f:
            result = json.load(f)
        self.assertEqual(len(result), 20)
        self.assertIn("2026-08-11T21:00:00-04:00", [e["ts"] for e in result],
                       "the newest local entry must survive the merge, not be "
                       "evicted by an equally-sized but older remote snapshot")
        self.assertEqual(result[-1]["ts"], "2026-08-11T21:00:00-04:00",
                          "newest entry must remain last (ascending order), "
                          "matching print_scan_log()'s reversed() convention")


class TestSubmitAlpacaTradeErrorSurfacing(unittest.TestCase):
    """A stretch of live orders silently failed (stale Alpaca key → 401) with
    zero visibility: submit_alpaca_trade() only printed the exception to the
    GitHub Actions log, and the Telegram alert just said "check GitHub Actions
    logs immediately" with no actual reason. The user had to manually check
    the Alpaca dashboard to even notice nothing was executing. Locks in that
    the real exception text now comes back to the caller instead of being
    swallowed after a print()."""

    def _signal(self):
        return a.ProSignal(
            ticker="TESTX", setup="Gap & Hold", bias="LONG",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            rr=2.0, rsi=60, rvol=2.0, reason="test", shares=10,
        )

    def test_auth_failure_surfaces_the_real_exception_text(self):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("401 Client Error: Unauthorized")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            oid, err = a.submit_alpaca_trade(self._signal())
        self.assertIsNone(oid)
        self.assertIn("Unauthorized", err,
                       "the caller must get the actual API error back, not a "
                       "generic message pointing at logs nobody checks")

    def test_success_returns_order_id_and_no_error(self):
        mock_order = MagicMock()
        mock_order.id = "abc123"
        mock_client = MagicMock()
        mock_client.submit_order.return_value = mock_order
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            oid, err = a.submit_alpaca_trade(self._signal())
        self.assertEqual(oid, "abc123")
        self.assertIsNone(err)

    def test_missing_client_returns_a_descriptive_error(self):
        with patch.object(a, "get_alpaca_client", return_value=None):
            oid, err = a.submit_alpaca_trade(self._signal())
        self.assertIsNone(oid)
        self.assertIsNotNone(err)

    def test_stop_loss_leg_has_no_limit_price(self):
        # Confirmed live 2026-08-06: a bracket's STOP_LIMIT child leg
        # reliably gets stuck in OrderStatus.HELD on this account after the
        # entry fills, while the take-profit LIMIT sibling activates fine —
        # reproduced 3/3 times (W, CLRO, CELZ), each one left completely
        # unprotected. The fix is a plain STOP (no limit_price) for the
        # stop-loss leg, which activated correctly all 3/3 times it was used
        # as a manual repair. A regression here (limit_price reappearing)
        # means going back to a stop that may never actually activate.
        mock_order = MagicMock(); mock_order.id = "abc123"
        mock_client = MagicMock(); mock_client.submit_order.return_value = mock_order
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            a.submit_alpaca_trade(self._signal())
        req = mock_client.submit_order.call_args[0][0]
        self.assertIsNone(req.stop_loss.limit_price)
        self.assertEqual(req.stop_loss.stop_price, 9.0)

    def test_swing_mode_stop_loss_leg_also_has_no_limit_price(self):
        # Same fix applies to the OTO (swing-mode) path — identical
        # StopLossRequest construction, same account-level HELD risk.
        sig = self._signal()
        sig.swing_mode = True
        mock_order = MagicMock(); mock_order.id = "abc123"
        mock_client = MagicMock(); mock_client.submit_order.return_value = mock_order
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            a.submit_alpaca_trade(sig)
        req = mock_client.submit_order.call_args[0][0]
        self.assertIsNone(req.stop_loss.limit_price)


class TestMacroBlackoutWindows(unittest.TestCase):
    """Blackout logic gates real capital. An off-by-one here means either
    trading through an event that should have been sat out, or needlessly
    sitting out a day that was actually clear."""

    def test_fomc_day_is_blocked(self):
        fomc_day = sorted(a._FOMC_DATES)[0]
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = fomc_day
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            safe, _ = a.check_macro_safe()
        self.assertFalse(safe, "FOMC day itself must be blocked")

    def test_day_after_blackout_window_is_clear(self):
        fomc_day = sorted(a._FOMC_DATES)[0]
        clear_day = date(fomc_day.year, fomc_day.month, fomc_day.day) \
            .fromordinal(fomc_day.toordinal() + 3)
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = clear_day
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            safe, _ = a.check_macro_safe()
        self.assertTrue(safe, "3 days after an FOMC date is well outside "
                              "the ±1 day blackout and must be clear")

    def test_major_macro_event_date_is_blocked(self):
        if not a._MAJOR_MACRO_EVENT_DATES:
            self.skipTest("no _MAJOR_MACRO_EVENT_DATES currently configured")
        event_day = sorted(a._MAJOR_MACRO_EVENT_DATES)[0]
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = event_day
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            safe, _ = a.check_macro_safe()
        self.assertFalse(safe, "a configured major macro event day must "
                               "be blocked the same way FOMC is")


class TestMtfCatalystOverride(unittest.TestCase):
    """The override that unblocks a news-confirmed earnings-gap play (the
    GOOGL/TSLA scenario) must be scoped tightly: only Gap & Hold / Bear Gap
    Hold, and only with news_boost actually True. It must never silently
    loosen the bar for an unrelated setup or an unconfirmed gap.

    score_signal() calls several other network-backed sub-checks
    (relative strength, sector, earnings safety, sector-ETF momentum) that
    have nothing to do with what's under test here — all patched to fast,
    deterministic stand-ins so this suite stays network-free as designed
    (a fake ticker otherwise spams real 404s at yfinance on every run).
    """

    def _make_signal(self, setup: str, news_boost: bool):
        sig = a.ProSignal(
            ticker="TESTX", bias="SHORT", setup=setup,
            entry=10.0, stop=11.0, target1=8.0, target2=6.0,
            shares=100, rr=2.0, rsi=30.0, rvol=2.0,
            reason="test", confluence_score=0,
        )
        sig.news_boost = news_boost
        return sig

    def _score(self, sig):
        with patch.object(a, "check_mtf", return_value=(False, 5)), \
             patch.object(a, "check_relative_strength", return_value=(True, 8)), \
             patch.object(a, "check_sector", return_value=(True, 5)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 5)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", return_value=None):
            return a.score_signal(sig, _fake_df(), _fake_regime(), a.WinRateTracker())

    def test_override_fires_for_bear_gap_hold_with_news(self):
        scored = self._score(self._make_signal("Bear Gap Hold", news_boost=True))
        self.assertTrue(scored.mtf_ok,
                        "a confirmed catalyst must override a failing MTF "
                        "check for Bear Gap Hold — this is the fix that "
                        "would have let the GOOGL/TSLA gap plays through")

    def test_override_does_not_fire_without_news(self):
        scored = self._score(self._make_signal("Bear Gap Hold", news_boost=False))
        self.assertFalse(scored.mtf_ok,
                         "without a real catalyst, the MTF bar must stay "
                         "exactly as strict as it always was")

    def test_override_does_not_fire_for_unrelated_setup(self):
        scored = self._score(self._make_signal("Morning Runner", news_boost=True))
        self.assertFalse(scored.mtf_ok,
                         "the override is scoped to Gap & Hold / Bear Gap "
                         "Hold only — it must not loosen anything else")


class TestEarningsCalendarDictParsing(unittest.TestCase):
    """Regression test for a confirmed production bug: yfinance's .calendar
    returns a plain dict — {'Earnings Date': [date(...), ...], ...} — not a
    DataFrame. The old code checked cal.empty/cal.columns, which raised
    AttributeError on every real call (dict has neither attribute), silently
    caught by a bare except — meaning check_earnings_safe() always returned
    "safe" and get_upcoming_earnings() always returned [], so
    EARNINGS_BLACKOUT never actually blocked a signal in production."""

    def test_dict_shaped_calendar_blocks_earnings_today(self):
        fake_cal = {"Earnings Date": [date.today()], "Earnings High": 5.0}
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = fake_cal
            safe, score = a.check_earnings_safe("TESTX")
        self.assertFalse(safe, "earnings today must be blocked, not silently passed")
        self.assertEqual(score, 0)

    def test_dict_shaped_calendar_allows_earnings_far_out(self):
        far_date = date.today() + timedelta(days=a.EARNINGS_BLACKOUT + 10)
        fake_cal = {"Earnings Date": [far_date]}
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = fake_cal
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe, "earnings well outside the blackout window must pass")
        self.assertEqual(score, 5)

    def test_none_calendar_is_treated_as_safe(self):
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = None
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe)
        self.assertEqual(score, 5)

    def test_empty_dict_calendar_is_treated_as_safe(self):
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = {}
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe)

    def test_get_upcoming_earnings_returns_dict_shaped_dates(self):
        near_date = date.today() + timedelta(days=2)
        fake_cal = {"Earnings Date": [near_date]}
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = fake_cal
            result = a.get_upcoming_earnings(["TESTX"], days_ahead=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "TESTX")
        self.assertEqual(result[0]["earn_date"], near_date)
        self.assertEqual(result[0]["days_away"], 2)

    def test_get_upcoming_earnings_excludes_dates_beyond_window(self):
        far_date = date.today() + timedelta(days=30)
        fake_cal = {"Earnings Date": [far_date]}
        with patch.object(a.yf, "Ticker") as mock_tk:
            mock_tk.return_value.calendar = fake_cal
            result = a.get_upcoming_earnings(["TESTX"], days_ahead=5)
        self.assertEqual(result, [])


class TestBarSetContainsBugFix(unittest.TestCase):
    """Regression test for a confirmed production bug: alpaca-py's BarSet does
    not support `in` the way a dict does — `ticker in resp` was always False
    even when resp.data[ticker] had real bars, confirmed live 2026-07-29 by
    directly calling the SDK. _fetch_alpaca_daily(), prewarm_alpaca_bars(),
    and _fetch_intraday_bars() all used this pattern and always silently
    returned [] / fell back to a slower path, in production, until fixed to
    use resp.data.get(ticker, []) instead."""

    def test_fetch_alpaca_daily_uses_data_get_not_in(self):
        class FakeBarSet:
            def __init__(self, data):
                self.data = data
            def __contains__(self, key):
                return False   # reproduces the real BarSet's actual behavior

        fake_bar = MagicMock(open=10.0, high=11.0, low=9.5, close=10.5, volume=1000,
                             timestamp=datetime.now() - timedelta(days=1))
        fake_resp = FakeBarSet({"TESTX": [fake_bar] * 25})
        mock_dc = MagicMock()
        mock_dc.get_stock_bars.return_value = fake_resp
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            df = a._fetch_alpaca_daily("TESTX", 30)
        self.assertIsNotNone(df, "must recover real bars via resp.data.get(), "
                                 "not silently return None via `ticker in resp`")
        self.assertEqual(len(df), 25)


class TestOptionContractStrikeStringConversion(unittest.TestCase):
    """Regression test for a confirmed production bug: GetOptionContractsRequest's
    strike_price_gte/lte fields require STRING values (pydantic Optional[str]),
    but the code passed floats — every real call raised a ValidationError,
    silently caught, meaning _find_best_call_contract/_find_best_put_contract
    ALWAYS returned None and every options signal always fell back to an
    equity order. Confirmed live 2026-07-29 against the real installed library."""

    def test_find_best_call_contract_passes_string_strikes(self):
        from alpaca.trading.requests import GetOptionContractsRequest
        captured_requests = []

        class FakeContract:
            symbol = "TESTX260814C00100000"
            strike_price = 100.0

        def fake_get_option_contracts(req):
            captured_requests.append(req)
            return MagicMock(option_contracts=[FakeContract()])

        mock_client = MagicMock()
        mock_client.get_option_contracts.side_effect = fake_get_option_contracts

        fake_snap = {"bid": 5.0, "ask": 5.2, "mid": 5.1, "spread_pct": 0.04,
                    "delta": 0.65, "gamma": 0.01, "theta": -0.05, "vega": 0.1,
                    "iv": 0.3, "oi": 100}
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=fake_snap):
                a._find_best_call_contract(mock_client, "TESTX", 100.0)

        self.assertTrue(captured_requests, "expected at least one get_option_contracts call")
        for req in captured_requests:
            self.assertIsInstance(req.strike_price_gte, str,
                                  "strike_price_gte must be a string — GetOptionContractsRequest "
                                  "rejects floats with a pydantic ValidationError")
            self.assertIsInstance(req.strike_price_lte, str)


class TestBlackScholesDeltaEstimate(unittest.TestCase):
    """Added 2026-08-08: confirmed live that this account has options
    TRADING approved (level 3, real buying power) but NOT the OPRA
    market-DATA subscription -- a separate Alpaca product. The
    "indicative" fallback feed sometimes returns a snapshot with NO
    greeks at all (bare bid/ask only), which _get_option_snapshot()
    defaults to delta=0.0, and the ITM hard filter in
    _find_best_call/put_contract (`delta < 0.40 -> reject`) then rejected
    every such contract unconditionally -- live-verified as one real
    contributing cause of zero options fills despite full trading
    approval. This is a pure-math regression check on the BS estimate
    itself; the wiring is covered by TestMissingGreeksDeltaEstimateFallback
    below."""

    def test_deep_itm_call_has_high_delta(self):
        # $100 stock, $70 strike call, 14 DTE -- deeply in the money.
        delta = a._estimate_bs_delta(100.0, 70.0, 14, True, sigma=0.40)
        self.assertGreater(delta, 0.90)

    def test_deep_otm_call_has_low_delta(self):
        delta = a._estimate_bs_delta(100.0, 160.0, 14, True, sigma=0.40)
        self.assertLess(delta, 0.10)

    def test_atm_call_delta_is_roughly_half(self):
        delta = a._estimate_bs_delta(100.0, 100.0, 14, True, sigma=0.40)
        self.assertTrue(0.40 <= delta <= 0.65)

    def test_deep_itm_put_delta_is_strongly_negative(self):
        # $100 stock, $130 strike put -- deeply in the money for a put.
        delta = a._estimate_bs_delta(100.0, 130.0, 14, False, sigma=0.40)
        self.assertLess(delta, -0.90)

    def test_deep_otm_put_delta_is_near_zero(self):
        delta = a._estimate_bs_delta(100.0, 60.0, 14, False, sigma=0.40)
        self.assertGreater(delta, -0.10)

    def test_bad_input_fails_open_to_zero_not_a_crash(self):
        self.assertEqual(a._estimate_bs_delta(100.0, 0.0, 14, True, sigma=0.40), 0.0)
        self.assertEqual(a._estimate_bs_delta(100.0, 100.0, 14, True, sigma=0.0), 0.0)


class TestRealizedVolEstimate(unittest.TestCase):
    """The sigma input feeding _estimate_bs_delta() -- a bad estimate here
    (e.g. a real crash instead of the documented 0.40 fallback) would
    propagate into every contract selection made while OPRA is
    unentitled."""

    def setUp(self):
        a._REALIZED_VOL_CACHE.clear()

    def tearDown(self):
        a._REALIZED_VOL_CACHE.clear()

    def _hist_df(self, closes):
        import pandas as pd
        return pd.DataFrame({"Close": closes})

    def test_computes_a_positive_annualized_vol_from_real_closes(self):
        import numpy as np
        rng = np.random.default_rng(42)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 40))
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.history.return_value = self._hist_df(closes)
            vol = a._realized_vol_estimate("TESTX")
        self.assertTrue(0.15 <= vol <= 2.0)

    def test_data_failure_falls_back_to_default(self):
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("network down")
            vol = a._realized_vol_estimate("TESTX")
        self.assertEqual(vol, 0.40)

    def test_result_is_cached(self):
        import numpy as np
        rng = np.random.default_rng(1)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.02, 40))
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.history.return_value = self._hist_df(closes)
            first = a._realized_vol_estimate("TESTX")
            mock_yf.Ticker.side_effect = Exception("should not be called again")
            second = a._realized_vol_estimate("TESTX")
        self.assertEqual(first, second)


class TestMissingGreeksDeltaEstimateFallback(unittest.TestCase):
    """Wiring test: when the broker snapshot has delta==0.0 (no real
    Greeks -- the indicative-feed placeholder), _find_best_call_contract
    and _find_best_put_contract must estimate one via Black-Scholes
    instead of letting the hard ITM filter reject the contract outright.
    A regression here silently reverts to the confirmed-live zero-fills
    bug: every missing-Greeks contract rejected, no exceptions."""

    class _FakeContract:
        def __init__(self, symbol, strike):
            self.symbol = symbol
            self.strike_price = strike

    def _client_returning(self, symbol, strike):
        client = MagicMock()
        client.get_option_contracts.return_value = MagicMock(
            option_contracts=[self._FakeContract(symbol, strike)]
        )
        return client

    def test_call_with_zero_delta_snapshot_gets_estimated_and_selected(self):
        # Deep ITM by construction (current price way above strike) --
        # the estimate should land comfortably above the 0.40 floor.
        zero_greeks_snap = {"bid": 30.0, "ask": 30.5, "mid": 30.25,
                             "spread_pct": 0.02, "delta": 0.0, "gamma": 0.0,
                             "theta": 0.0, "vega": 0.0, "iv": 0.0, "oi": 0}
        client = self._client_returning("TESTX260821C00070000", 70.0)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=zero_greeks_snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s), \
                 patch.object(a, "_realized_vol_estimate", return_value=0.40):
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        self.assertIsNotNone(contract, "a deep-ITM contract with only missing Greeks must "
                                        "not be rejected outright")
        self.assertTrue(contract["delta_estimated"])
        self.assertGreaterEqual(contract["delta"], 0.40)

    def test_call_with_real_nonzero_delta_is_not_overridden(self):
        # Broker-supplied delta already present (e.g. real OPRA data) --
        # the estimate must never override a genuine value.
        real_snap = {"bid": 5.0, "ask": 5.2, "mid": 5.1, "spread_pct": 0.04,
                     "delta": 0.62, "gamma": 0.01, "theta": -0.05, "vega": 0.1,
                     "iv": 0.3, "oi": 100}
        client = self._client_returning("TESTX260821C00095000", 95.0)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=real_snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s), \
                 patch.object(a, "_realized_vol_estimate") as mock_vol:
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        mock_vol.assert_not_called()
        self.assertIsNotNone(contract)
        self.assertNotIn("delta_estimated", contract)
        self.assertEqual(contract["delta"], 0.62)

    def test_put_with_zero_delta_snapshot_gets_estimated_and_selected(self):
        # Deep ITM put by construction (strike way above current price).
        zero_greeks_snap = {"bid": 30.0, "ask": 30.5, "mid": 30.25,
                             "spread_pct": 0.02, "delta": 0.0, "gamma": 0.0,
                             "theta": 0.0, "vega": 0.0, "iv": 0.0, "oi": 0}
        client = self._client_returning("TESTX260821P00130000", 130.0)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=zero_greeks_snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s), \
                 patch.object(a, "_realized_vol_estimate", return_value=0.40):
                contract = a._find_best_put_contract(client, "TESTX", 100.0)
        self.assertIsNotNone(contract)
        self.assertTrue(contract["delta_estimated"])
        self.assertLess(contract["delta"], 0, "put delta must be negative")
        self.assertGreaterEqual(abs(contract["delta"]), 0.40)


class TestEarningsSpreadMlegOrderConstruction(unittest.TestCase):
    """Locks in the first-ever use of OrderClass.MLEG in this codebase — every
    prior options function submits single-leg orders only. A malformed legs
    list here means partial fills / naked short legs in a LIVE brokerage
    account, which is exactly the risk this feature exists to eliminate."""

    def _plan(self, both_sides=True):
        p = {"ticker": "META", "sets": 1, "net_debit": 8.34,
             "put": {"long_occ": "META260807P00540000", "short_occ": "META260807P00510000"}}
        if both_sides:
            p["call"] = {"long_occ": "META260807C00650000", "short_occ": "META260807C00680000"}
        return p

    def test_double_spread_submits_exactly_four_legs_correct_sides(self):
        mock_order = MagicMock(); mock_order.id = "xyz"
        mock_client = MagicMock(); mock_client.submit_order.return_value = mock_order
        with patch.object(a, "get_available_cash", return_value=1_000_000.0):
            oid, err = a._submit_earnings_spread(mock_client, self._plan())
        self.assertIsNone(err)
        self.assertEqual(oid, "xyz")
        req = mock_client.submit_order.call_args[0][0]
        self.assertEqual(req.order_class, a.OrderClass.MLEG)
        self.assertEqual(len(req.legs), 4)
        sides = [(l.symbol, l.side) for l in req.legs]
        self.assertIn(("META260807C00650000", a.OrderSide.BUY),  sides)
        self.assertIn(("META260807C00680000", a.OrderSide.SELL), sides)
        self.assertIn(("META260807P00540000", a.OrderSide.BUY),  sides)
        self.assertIn(("META260807P00510000", a.OrderSide.SELL), sides)

    def test_single_side_spread_submits_exactly_two_legs(self):
        mock_client = MagicMock(); mock_client.submit_order.return_value = MagicMock(id="abc")
        with patch.object(a, "get_available_cash", return_value=1_000_000.0):
            _, err = a._submit_earnings_spread(mock_client, self._plan(both_sides=False))
        self.assertIsNone(err)
        req = mock_client.submit_order.call_args[0][0]
        self.assertEqual(len(req.legs), 2)

    def test_limit_price_is_positive_for_a_debit(self):
        mock_client = MagicMock(); mock_client.submit_order.return_value = MagicMock(id="1")
        with patch.object(a, "get_available_cash", return_value=1_000_000.0):
            a._submit_earnings_spread(mock_client, self._plan())
        req = mock_client.submit_order.call_args[0][0]
        self.assertGreater(req.limit_price, 0)

    def test_submit_failure_returns_error_text_not_swallowed(self):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("insufficient buying power")
        with patch.object(a, "get_available_cash", return_value=1_000_000.0):
            oid, err = a._submit_earnings_spread(mock_client, self._plan())
        self.assertIsNone(oid)
        self.assertIn("insufficient buying power", err)

    def test_cash_guard_blocks_when_insufficient_cash(self):
        mock_client = MagicMock(); mock_client.submit_order.return_value = MagicMock(id="1")
        with patch.object(a, "get_available_cash", return_value=1.0):
            oid, err = a._submit_earnings_spread(mock_client, self._plan())
        self.assertIsNone(oid)
        self.assertIn("cash", err)
        mock_client.submit_order.assert_not_called()

    def test_no_legs_is_rejected_before_hitting_the_api(self):
        mock_client = MagicMock()
        oid, err = a._submit_earnings_spread(mock_client, {"ticker": "X", "sets": 1, "net_debit": 1})
        self.assertIsNone(oid)
        self.assertIsNotNone(err)
        mock_client.submit_order.assert_not_called()


class TestCloseEarningsSpread(unittest.TestCase):
    """_close_earnings_spread() must invert each leg's side/intent from how it
    was opened (bought-to-open -> sell-to-close, sold-to-open -> buy-to-close).
    Getting this backwards would submit the WRONG side on a live account —
    e.g. buying MORE of a short leg instead of closing it."""

    def _pos(self):
        return {"ticker": "META", "spread_qty": 1,
               "legs": ["META260807C00650000", "META260807C00680000",
                        "META260807P00540000", "META260807P00510000"]}

    def test_sides_and_intents_are_inverted_from_opening(self):
        mock_held = MagicMock(qty="1")
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = mock_held
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="close1")
        fake_snap = {"bid": 5.0, "ask": 5.5}
        with patch.object(a, "_get_option_snapshot", return_value=fake_snap):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                st, oid = a._close_earnings_spread(self._pos(), "test")
        self.assertEqual(st, "submitted")
        req = mock_client.submit_order.call_args[0][0]
        by_symbol = {l.symbol: l.side for l in req.legs}
        self.assertEqual(by_symbol["META260807C00650000"], a.OrderSide.SELL)   # was long -> sell to close
        self.assertEqual(by_symbol["META260807C00680000"], a.OrderSide.BUY)    # was short -> buy to close
        self.assertEqual(by_symbol["META260807P00540000"], a.OrderSide.SELL)
        self.assertEqual(by_symbol["META260807P00510000"], a.OrderSide.BUY)

    def test_already_closed_when_no_leg_is_held(self):
        mock_client = MagicMock()
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            st, oid = a._close_earnings_spread(self._pos(), "test")
        self.assertEqual(st, "already_closed")
        mock_client.submit_order.assert_not_called()


class TestRecentEarningsSurprise(unittest.TestCase):
    """Real actual-vs-estimate beat/miss data (Massive's /benzinga/v1/earnings
    response) was already being fetched for consensus-estimate fields but
    the actual_eps/eps_surprise_percent fields were never surfaced anywhere
    — this is the one piece of "market sentiment" data actually entitled on
    the current Massive plan (analyst-ratings/bulls-bears-say return 403).
    A regression here means a Gap & Hold alert on a just-reported ticker
    goes out with no explanation of why it gapped."""

    def _item(self, ticker="UBER", actual_eps=0.81, eps_pct=-0.0241,
              actual_rev=14.191e9, rev_pct=-0.0031, when="2026-08-05"):
        return {"ticker": ticker, "date": when, "actual_eps": actual_eps,
                "eps_surprise_percent": eps_pct, "actual_revenue": actual_rev,
                "revenue_surprise_percent": rev_pct}

    def test_reported_ticker_returns_the_surprise_data(self):
        with patch.object(a, "_fetch_massive_earnings", return_value=[self._item()]):
            result = a._recent_earnings_surprise("UBER")
        self.assertIsNotNone(result)
        self.assertEqual(result["actual_eps"], 0.81)

    def test_estimate_only_entry_is_not_treated_as_reported(self):
        # actual_eps is None until the company has actually reported —
        # an estimate-only row must not be mistaken for a beat/miss.
        estimate_only = self._item()
        estimate_only["actual_eps"] = None
        with patch.object(a, "_fetch_massive_earnings", return_value=[estimate_only]):
            result = a._recent_earnings_surprise("UBER")
        self.assertIsNone(result)

    def test_no_earnings_in_window_returns_none(self):
        with patch.object(a, "_fetch_massive_earnings", return_value=[]):
            result = a._recent_earnings_surprise("UBER")
        self.assertIsNone(result)

    def test_format_note_reports_beat_and_miss_correctly(self):
        beat = self._item(eps_pct=0.05, rev_pct=0.02)
        note = a._format_earnings_surprise_note(beat)
        self.assertIn("beat", note)
        self.assertNotIn("missed", note)

        miss = self._item(eps_pct=-0.05, rev_pct=-0.02)
        note = a._format_earnings_surprise_note(miss)
        self.assertIn("missed", note)
        self.assertNotIn("beat", note)


class TestSharesFallbackPolicy(unittest.TestCase):
    """Policy set 2026-08-05: grow the account on options, shares only for
    DMan's own curated small-cap watchlist — not as a silent fallback for
    expensive large-cap signals when an options fill isn't available. A
    regression here means the algo goes back to buying full-price shares
    of high-value stocks instead of skipping the trade."""

    def test_dman_watchlist_ticker_allowed(self):
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertTrue(a._shares_fallback_allowed("APVO"))

    def test_large_cap_ticker_not_allowed(self):
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertFalse(a._shares_fallback_allowed("AMZN"))


class TestSharesFallbackBudgetShares(unittest.TestCase):
    """User decision 2026-08-08: an options-eligible signal (WATCHLIST/
    OPTIONS_SETUPS) that genuinely can't get an options fill and isn't on
    the small-cap watchlist now falls back to a BUDGET-CAPPED shares
    position instead of skipping entirely -- capped to the same
    options-equivalent budget, not the signal's normal full-size equity
    sizing. Confirmed live: AMBA/AMZN/RBLX-style signals were previously
    alerted and then silently produced zero trade. A regression here
    either buys far more shares than the risk budget intends (the
    max(1, ...) floor this replaced would buy 1 share of even a $5,000
    stock against a $1,250 budget) or refuses to buy anything affordable."""

    def test_computes_whole_shares_within_budget(self):
        self.assertEqual(a._shares_fallback_budget_shares(current_price=10.0, budget=1250.0), 125)

    def test_rounds_down_not_up(self):
        # $87 AMBA-style price against the $1,250 budget -> 14 shares
        # ($1,218), not 15 ($1,305, over budget).
        self.assertEqual(a._shares_fallback_budget_shares(current_price=87.35, budget=1250.0), 14)

    def test_price_exceeding_budget_returns_zero_not_one(self):
        # Must NOT force a floor of 1 share when even a single share blows
        # past the budget (e.g. a $2,000+ stock against a $1,250 budget) --
        # that would silently exceed the intended risk cap.
        self.assertEqual(a._shares_fallback_budget_shares(current_price=2093.0, budget=1250.0), 0)

    def test_zero_or_negative_price_fails_closed(self):
        self.assertEqual(a._shares_fallback_budget_shares(current_price=0.0, budget=1250.0), 0)
        self.assertEqual(a._shares_fallback_budget_shares(current_price=-5.0, budget=1250.0), 0)

    def test_zero_budget_fails_closed(self):
        self.assertEqual(a._shares_fallback_budget_shares(current_price=10.0, budget=0.0), 0)


class TestWinRateLiveOnlyFiltering(unittest.TestCase):
    """Added 2026-08-07: dman_win_rate.json's 500 records span back to
    2024 -- almost entirely backtest simulation, with real trades so
    sparse they barely register in the rolling-50 window. rolling_stats
    (live_only=True) is what lets the win rate actually mean "what this
    account has really done" instead of quietly blending in two years of
    backtest data. A regression here means that blend becomes invisible
    again."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _record(self, tracker, outcome, is_live, pnl_pct=1.0):
        tracker.record(a.TradeRecord(
            ticker="TESTX", date="2026-08-07", bias="LONG", setup="Gap & Hold",
            entry=10.0, exit=11.0, outcome=outcome, pnl_pct=pnl_pct,
            score=100, is_live=is_live,
        ))

    def test_live_only_excludes_backtest_records(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(10):
            self._record(tracker, "WIN", is_live=False)
        self._record(tracker, "LOSS", is_live=True)
        stats = tracker.rolling_stats(live_only=True)
        self.assertEqual(stats["total"], 1)
        self.assertEqual(stats["losses"], 1)

    def test_default_rolling_stats_still_blends_everything(self):
        # Confirms this is additive, not a behavior change to the existing
        # default call sites (e.g. adaptive_min_score()) that intentionally
        # still read the full pool.
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(3):
            self._record(tracker, "WIN", is_live=False)
        self._record(tracker, "LOSS", is_live=True)
        stats = tracker.rolling_stats()
        self.assertEqual(stats["total"], 4)

    def test_old_records_without_is_live_field_default_to_backtest(self):
        with open(self._tmp.name, "w") as f:
            json.dump([{"ticker": "OLD", "date": "2024-01-01", "bias": "LONG",
                        "setup": "Gap & Hold", "entry": 1.0, "exit": 1.1,
                        "outcome": "WIN", "pnl_pct": 10.0, "score": 90}], f)
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        self.assertFalse(tracker.records[0].is_live)
        self.assertEqual(tracker.rolling_stats(live_only=True)["total"], 0)


class TestProgressEquityStopToTrailing(unittest.TestCase):
    """Confirmed live 2026-08-08: CELZ sat at +24.58% with its original
    entry-time stop completely untouched -- the T1 alert only ever told a
    human to "move stop to breakeven," nothing executed it. This is the
    highest-stakes new logic shipped tonight (directly cancels/replaces/
    creates real orders on live positions) and the market is closed, so it
    cannot be live-tested before Monday -- these tests carry the full
    weight of verifying correctness. Every path must leave the position
    with SOME live protective order; the market is closed, so it cannot be
    live-tested before Monday -- these tests carry the full weight of
    verifying correctness."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(json.dumps([{
            "ticker": "CELZ", "bias": "LONG", "setup": "Low Float Catalyst",
            "entry": 1.1398, "stop": 0.96, "target1": 1.68, "target2": 1.99,
            "shares": 284, "entry_date": "2026-08-06", "stop_stage": "initial",
        }]).encode())
        self._pos_tmp.close()
        self._patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._pos_tmp.name)

    def _pos(self, **overrides):
        base = dict(ticker="CELZ", bias="LONG", setup="Low Float Catalyst",
                    entry=1.1398, stop=0.96, target1=1.68, target2=1.99,
                    shares=284, entry_date="2026-08-06", stop_stage="initial")
        base.update(overrides)
        return a.OpenPosition(**base)

    def _stop_order(self, order_id="stop-1"):
        from alpaca.trading.enums import OrderType
        o = MagicMock()
        o.id = order_id
        o.order_type = OrderType.STOP
        return o

    def test_below_t1_does_nothing(self):
        mock_client = MagicMock()
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.20)
        self.assertIsNone(result)
        mock_client.get_orders.assert_not_called()

    def test_already_trailing_does_nothing(self):
        mock_client = MagicMock()
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(
                self._pos(stop_stage="trailing"), cur_price=2.00)
        self.assertIsNone(result)
        mock_client.get_orders.assert_not_called()

    def test_custom_trigger_price_below_it_does_nothing(self):
        # Added 2026-08-10 for the early-profit-lock feature: a caller-
        # supplied trigger_price replaces the T1 gate entirely -- below it,
        # even a price that's already well past the position's real T1
        # target must still do nothing if it hasn't reached the custom gate.
        mock_client = MagicMock()
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(
                self._pos(), cur_price=1.30, trigger_price=1.40)
        self.assertIsNone(result)
        mock_client.get_orders.assert_not_called()

    def test_custom_trigger_price_below_t1_still_progresses(self):
        # The core of the fix: a trigger_price BELOW pos.target1 must let
        # this fire even though cur_price never reached the real T1.
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.submit_order.return_value = MagicMock(id="early-trail-1")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(
                self._pos(target1=14.73), cur_price=11.56, trigger_price=1.31)
        self.assertIsNotNone(result, "a trigger_price below the real T1 must still "
                              "allow progression -- this is what protects a position "
                              "that never reaches its full target")
        mock_client.replace_order_by_id.assert_called_once()

    def test_successful_transition_updates_stage_and_uses_capped_trail(self):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.submit_order.return_value = MagicMock(id="trail-order-1")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.70)
        self.assertIsNotNone(result)
        self.assertIn("breakeven", result)
        self.assertIn("trailing", result)
        # replace called with breakeven stop_price
        replace_call = mock_client.replace_order_by_id.call_args
        self.assertEqual(replace_call[0][0], "stop-1")
        self.assertEqual(replace_call[0][1].stop_price, round(1.1398, 2))
        # cancel called on the same order before the new trailing submission
        mock_client.cancel_order_by_id.assert_called_once_with("stop-1")
        # trail% must be capped so initial level can't sit below breakeven:
        # original_stop_pct = (1.1398-0.96)/1.1398*100 = 15.8%
        # current_gain_pct  = (1.70-1.1398)/1.70*100    = 32.9%
        # capped trail = min(15.8, 32.9) = 15.8
        submit_call = mock_client.submit_order.call_args[0][0]
        self.assertAlmostEqual(submit_call.trail_percent, 15.79, places=1)
        # tracker updated to reflect the new stage
        tracker = a.PositionTracker(filepath=self._pos_tmp.name)
        self.assertEqual(tracker.positions[0].stop_stage, "trailing")

    def test_trail_percent_never_exceeds_current_gain(self):
        # A position with a very wide original stop but only just past T1 --
        # the gain-from-entry distance must win the cap, not the original
        # stop%, or the initial trailing level could land below breakeven.
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.submit_order.return_value = MagicMock(id="trail-order-2")
        wide_stop_pos = self._pos(entry=10.0, stop=8.2, target1=10.5)  # 18% original stop
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._progress_equity_stop_to_trailing(wide_stop_pos, cur_price=10.6)
        # current_gain_pct = (10.6-10.0)/10.6*100 = 5.66%, well under 18%
        submit_call = mock_client.submit_order.call_args[0][0]
        self.assertLess(submit_call.trail_percent, 18.0)
        self.assertAlmostEqual(submit_call.trail_percent, 5.66, places=1)

    def test_trailing_submission_failure_falls_back_to_plain_breakeven_stop(self):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.submit_order.side_effect = [
            Exception("trailing stop rejected"),   # first call: trailing stop fails
            MagicMock(id="fallback-stop-1"),        # second call: plain stop succeeds
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.70)
        self.assertIsNotNone(result)
        self.assertIn("fallback", result)
        self.assertEqual(mock_client.submit_order.call_count, 2)
        # must NOT claim "trailing" if it actually fell back to a plain stop
        tracker = a.PositionTracker(filepath=self._pos_tmp.name)
        self.assertEqual(tracker.positions[0].stop_stage, "initial")

    def test_both_trailing_and_fallback_fail_sends_emergency_alert(self):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.submit_order.side_effect = Exception("Alpaca is down")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.70)
        self.assertIsNone(result)
        mock_tg.assert_called_once()
        self.assertIn("stop management failed", mock_tg.call_args[0][0])

    def test_no_live_stop_order_found_does_not_crash(self):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.70)
        self.assertIsNone(result)
        mock_client.replace_order_by_id.assert_not_called()

    def test_breakeven_replace_failure_stops_before_any_cancel(self):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [self._stop_order()]
        mock_client.replace_order_by_id.side_effect = Exception("replace rejected")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            result = a._progress_equity_stop_to_trailing(self._pos(), cur_price=1.70)
        self.assertIsNone(result)
        # must not cancel the only live protective order if the replace
        # (which would have made it safe) never actually succeeded
        mock_client.cancel_order_by_id.assert_not_called()


class TestCheckEquityPositionTarget(unittest.TestCase):
    """Added 2026-08-09: extracted from run_momentum_watch() so the
    daemon's new continuous equity guard loop (run_equity_guard) can reuse
    the exact same T1/T2/stop-progression logic instead of a parallel copy
    that could drift out of sync. A regression here means either a missed
    T1/T2 alert on a real position, or a double-alert from the two call
    sites (hourly cron + daemon) racing each other."""

    def setUp(self):
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.write(b"{}")
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

    def tearDown(self):
        self._dedup_patch.stop()
        os.unlink(self._dedup_tmp.name)

    def _pos(self, **overrides):
        base = dict(ticker="CELZ", entry=1.1398, target1=1.68, target2=1.99,
                    stop=0.96, shares=284, bias="LONG", setup="Low Float Catalyst",
                    entry_date="2026-08-06", stop_stage="initial")
        base.update(overrides)
        return base

    def test_price_below_both_targets_does_nothing(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(self._pos(), cur_price=1.30)
        mock_tg.assert_not_called()

    def test_t1_hit_alerts_and_progresses_stop(self):
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value="stop raised") as mock_prog:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(self._pos(), cur_price=1.70)
        mock_tg.assert_called_once()
        self.assertIn("T1 HIT", mock_tg.call_args[0][0])
        mock_prog.assert_called_once()

    def test_t2_hit_takes_precedence_over_t1(self):
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value=None):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(self._pos(), cur_price=2.50)
        mock_tg.assert_called_once()
        self.assertIn("T2 HIT", mock_tg.call_args[0][0])

    def test_already_alerted_today_does_not_refire(self):
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value=None):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(self._pos(), cur_price=1.70)
                a._check_equity_position_target(self._pos(), cur_price=1.71)
        self.assertEqual(mock_tg.call_count, 1, "the dedup file must prevent a "
                          "second alert for the same target on the same day, "
                          "regardless of how many times this is called")

    def test_explicit_cur_price_skips_the_rest_call(self):
        with patch.object(a, "get_live_price") as mock_price:
            with patch.object(a, "send_telegram", return_value=True):
                a._check_equity_position_target(self._pos(), cur_price=1.30)
        mock_price.assert_not_called()

    def test_no_cur_price_falls_back_to_get_live_price(self):
        with patch.object(a, "get_live_price", return_value=1.30) as mock_price:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(self._pos())
        mock_price.assert_called_once_with("CELZ")
        mock_tg.assert_not_called()

    def test_unavailable_price_does_not_crash_or_alert(self):
        with patch.object(a, "get_live_price", return_value=None):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(self._pos())
        mock_tg.assert_not_called()

    def test_zero_entry_returns_immediately(self):
        with patch.object(a, "get_live_price") as mock_price:
            a._check_equity_position_target(self._pos(entry=0), cur_price=100.0)
        mock_price.assert_not_called()

    def test_early_lock_engages_before_t1(self):
        # Real CLRO incident, 2026-08-10: entry $9.80, T1 $14.73, peaked at
        # $14.08 (+43.7%) and reversed hard -- T1 was never reached, so the
        # T1 branch never engaged and this position got zero protection.
        # +18% here is well past the 15% trigger but nowhere near T1.
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value="stop raised to breakeven, now trailing") as mock_prog:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(pos, cur_price=11.56)   # +18%
        mock_tg.assert_called_once()
        self.assertIn("Early profit lock", mock_tg.call_args[0][0])
        # must pass a trigger_price BELOW T1, not fall through to the T1 gate
        self.assertEqual(mock_prog.call_args.kwargs.get("trigger_price"), round(9.80 * 1.15, 4))

    def test_below_early_lock_threshold_does_nothing(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)   # +12.2%, below the 15% trigger
        mock_tg.assert_not_called()

    def test_early_lock_does_not_refire_same_day(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value="locked"):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(pos, cur_price=11.56)
                a._check_equity_position_target(pos, cur_price=12.00)
        self.assertEqual(mock_tg.call_count, 1)

    def test_early_lock_silent_when_progression_returns_none(self):
        # Already trailing (from an earlier check) -- no new alert, no dedup mark.
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83, stop_stage="trailing")
        with patch.object(a, "_progress_equity_stop_to_trailing", return_value=None):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_equity_position_target(pos, cur_price=11.56)
        mock_tg.assert_not_called()


class TestRunEquityGuard(unittest.TestCase):
    """Added 2026-08-09: the always-on daemon's continuous counterpart to
    run_options_guard(), which only ever covered options/earnings-spread
    positions -- plain equity positions (everything actually held,
    confirmed live: CELZ, CLRO) had NO continuous daemon-side monitoring
    at all before this, only the hourly cron. A regression here means
    either options positions get double-checked here too (harmless but
    wrong) or equity positions get silently skipped."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.close()
        self._patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._pos_tmp.name):
            os.unlink(self._pos_tmp.name)

    def _write_positions(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def test_only_plain_equity_positions_are_checked(self):
        self._write_positions([
            {"ticker": "CELZ", "entry": 1.14, "setup": "Low Float Catalyst"},
            {"ticker": "AAPL260821C00310000", "entry": 5.0, "setup": "Options Call AAPL260821C00310000 ($310C exp 2026-08-21)"},
            {"ticker": "META", "entry": 500.0, "setup": "Earnings META spread"},
        ])
        with patch.object(a, "_check_equity_position_target") as mock_check:
            a.run_equity_guard()
        checked_tickers = [c.args[0]["ticker"] for c in mock_check.call_args_list]
        self.assertEqual(checked_tickers, ["CELZ"])

    def test_get_price_fn_is_used_per_ticker(self):
        self._write_positions([{"ticker": "CELZ", "entry": 1.14, "setup": "Low Float Catalyst"}])
        prices = {"CELZ": 1.75}
        with patch.object(a, "_check_equity_position_target") as mock_check:
            a.run_equity_guard(get_price_fn=lambda t: prices.get(t))
        self.assertEqual(mock_check.call_args.kwargs.get("cur_price"), 1.75)

    def test_missing_positions_file_does_not_crash(self):
        os.unlink(self._pos_tmp.name)
        with patch.object(a, "_check_equity_position_target") as mock_check:
            a.run_equity_guard()
        mock_check.assert_not_called()


class TestOptionsPnlMilestone(unittest.TestCase):
    """Added 2026-08-10: tiered P&L notifications for options legs the user
    wants live visibility into (e.g. an earnings hold) independent of the
    stop/trail/T1 alerts. A regression here means either silence on a real
    swing or repeat-alert spam on every check once past the threshold."""

    def setUp(self):
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.write(b"{}")
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._pos_patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._pos_patch.start()

    def tearDown(self):
        self._dedup_patch.stop(); os.unlink(self._dedup_tmp.name)
        self._pos_patch.stop();   os.unlink(self._pos_tmp.name)

    def _pos(self, **overrides):
        base = {"ticker": "SMCI", "milestone_gain_alerted": 0.0, "milestone_loss_alerted": 0.0}
        base.update(overrides)
        return base

    def test_below_start_threshold_does_nothing(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.10, entry_prem=1.00, underlying_px=33.0)   # +10%, below the 15% start
        mock_tg.assert_not_called()

    def test_threshold_tightened_to_15_fires_where_old_30_would_not(self):
        # Locks in the 2026-08-15 tightening (direct instruction to catch
        # any meaningful move on the live UMAC play, not wait for a 30%
        # swing) -- +22% is below the OLD 30% start but above the new 15%.
        # (Deliberately not exactly +20% -- (1.20-1.00)/1.00*100 lands at
        # 19.999999999999996 in float, one bucket below what it looks like
        # on paper; +22% sits comfortably inside the 20% bucket instead.)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.22, entry_prem=1.00, underlying_px=33.0)   # +22%
        mock_tg.assert_called_once()
        self.assertIn("+20%", mock_tg.call_args[0][0])

    def test_gain_milestone_fires_at_30(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.35, entry_prem=1.00, underlying_px=34.0)   # +35%
        mock_tg.assert_called_once()
        self.assertIn("+30%", mock_tg.call_args[0][0])

    def test_gain_milestone_does_not_refire_same_bucket(self):
        pos = self._pos(milestone_gain_alerted=30.0)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(pos, "CALL", "SMCI260814C00034000",
                                           cur_prem=1.38, entry_prem=1.00, underlying_px=34.0)   # +38%, still bucket 30
        mock_tg.assert_not_called()

    def test_gain_milestone_fires_at_new_higher_bucket(self):
        pos = self._pos(milestone_gain_alerted=30.0)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(pos, "CALL", "SMCI260814C00034000",
                                           cur_prem=1.45, entry_prem=1.00, underlying_px=35.0)   # +45% -> bucket 40
        mock_tg.assert_called_once()
        self.assertIn("+40%", mock_tg.call_args[0][0])

    def test_loss_milestone_fires_independently_of_gain_side(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "PUT", "SMCI260814P00029500",
                                           cur_prem=0.65, entry_prem=1.00, underlying_px=33.0)   # -35%
        mock_tg.assert_called_once()
        self.assertIn("-30%", mock_tg.call_args[0][0])


class TestOptionsTrailGivebackPct(unittest.TestCase):
    """_options_trail_giveback_pct() replaced a flat 30% giveback with one
    that widens as peak profit deepens -- confirmed live 2026-08-12: SMCI's
    call peaked +160%, a flat 30% giveback exited at +80% (option $2.38),
    and premium then ran to a $4.25 high / $3.75 close the same day. Pure
    math tests for the interpolation, independent of the monitor plumbing."""

    def test_at_or_below_activation_returns_min(self):
        self.assertEqual(a._options_trail_giveback_pct(0.0), a.OPTIONS_TRAIL_GIVEBACK_MIN_PCT)
        self.assertEqual(a._options_trail_giveback_pct(a.OPTIONS_TRAIL_ACTIVATE_GAIN_PCT),
                          a.OPTIONS_TRAIL_GIVEBACK_MIN_PCT)

    def test_at_max_gain_threshold_returns_max(self):
        self.assertAlmostEqual(a._options_trail_giveback_pct(a.OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT),
                               a.OPTIONS_TRAIL_GIVEBACK_MAX_PCT)

    def test_beyond_max_gain_threshold_stays_capped_at_max(self):
        self.assertAlmostEqual(a._options_trail_giveback_pct(a.OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT + 500),
                               a.OPTIONS_TRAIL_GIVEBACK_MAX_PCT)

    def test_midpoint_interpolates_linearly(self):
        mid_gain = (a.OPTIONS_TRAIL_ACTIVATE_GAIN_PCT + a.OPTIONS_TRAIL_GIVEBACK_MAX_AT_PCT) / 2
        expected = (a.OPTIONS_TRAIL_GIVEBACK_MIN_PCT + a.OPTIONS_TRAIL_GIVEBACK_MAX_PCT) / 2
        self.assertAlmostEqual(a._options_trail_giveback_pct(mid_gain), expected)

    def test_smci_incident_peak_gain_gives_capped_tolerance(self):
        # Real numbers: entry 1.32, peak 3.44 -> peak_gain ~160.6%, past
        # the 150% ceiling, so tolerance is capped at MAX_PCT (50%) --
        # verified this alone is enough to have kept the position open
        # through the real $2.38 exit (only a 30.8% giveback).
        peak_gain = (3.44 - 1.32) / 1.32 * 100
        self.assertAlmostEqual(a._options_trail_giveback_pct(peak_gain), a.OPTIONS_TRAIL_GIVEBACK_MAX_PCT)


class TestMonitorOptionPositionTrailingExit(unittest.TestCase):
    """Added 2026-08-10: replaced the fixed T2 (+150%) auto-close with a
    trailing exit (activate after OPTIONS_TRAIL_ACTIVATE_GAIN_PCT gain,
    close on OPTIONS_TRAIL_GIVEBACK_PCT giveback from the peak) after the
    CLRO incident showed a fixed target either gets missed on a fast
    reversal or fires too rigidly. This is real-money-managing, previously
    completely untested code -- every path must leave the position either
    correctly closed or correctly still-open, never in between."""

    def setUp(self):
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.write(b"{}")
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.close()
        self._pos_patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._pos_patch.start()

        self._milestone_patch = patch.object(a, "_check_options_pnl_milestone", return_value=None)
        self._milestone_patch.start()
        self._price_patch = patch.object(a, "get_live_price", return_value=33.0)
        self._price_patch.start()

    def tearDown(self):
        self._dedup_patch.stop();    os.unlink(self._dedup_tmp.name)
        self._pos_patch.stop();      os.unlink(self._pos_tmp.name)
        self._milestone_patch.stop()
        self._price_patch.stop()

    def _write_positions(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def _pos(self, **overrides):
        base = {"ticker": "SMCI", "setup": "Options Call SMCI260814C00034000 ($34C exp 2026-08-14)",
                "entry": 1.00, "stop": 0.50, "target1": 1.50, "target2": 2.50,
                "atr": 0.45, "shares": 100, "peak_premium": 0.0,
                "milestone_gain_alerted": 0.0, "milestone_loss_alerted": 0.0}
        base.update(overrides)
        return base

    def _snap(self, mid, delta=0.6, theta=-0.05, iv=0.3):
        return {"bid": mid - 0.02, "ask": mid + 0.02, "mid": mid, "delta": delta,
                "gamma": 0.01, "theta": theta, "vega": 0.1, "iv": iv, "oi": 500}

    def test_pre_trail_stop_loss_still_closes(self):
        # Never got meaningfully profitable -- baseline floor must still work.
        self._write_positions([self._pos()])
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(0.48)):   # -52%
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord1")) as mock_close:
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._monitor_option_position(self._pos(), "CALL")
        mock_close.assert_called_once()
        self.assertIn("OPTIONS STOP", mock_tg.call_args[0][0])

    def test_trail_not_active_before_activation_gain_even_with_a_drop(self):
        # Peak only ever reached +10% (below the 25% activation) -- a drop
        # from there must NOT trigger a trail exit (trail was never armed).
        pos = self._pos(peak_premium=1.10)   # +10% peak, stop=0.50 keeps this above baseline stop
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(0.90)):   # -10% from entry, -18% off peak
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(pos, "CALL")
        mock_close.assert_not_called()
        self.assertIn("not yet active", result)

    def test_trail_exit_fires_after_activation_and_giveback(self):
        # Peak hit +30% (past the 25% activation), now given back to +5%
        # from entry -- a ~19% drop off peak... need past 30% giveback.
        # Peak 1.35 (+35%), current 0.90 -> giveback (1.35-0.90)/1.35=33.3% > 30%.
        pos = self._pos(peak_premium=1.35)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(0.90)):
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord2")) as mock_close:
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._monitor_option_position(pos, "CALL")
        mock_close.assert_called_once()
        self.assertIn("TRAIL EXIT", mock_tg.call_args[0][0])

    def test_trail_active_but_giveback_not_enough_does_not_exit(self):
        # Peak +35%, current only given back ~11% off peak (1.35 -> 1.20) --
        # under the 30% giveback threshold, must stay open.
        pos = self._pos(peak_premium=1.35, target1=999.0)   # target1 sky-high so T1 branch can't interfere
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.20)):
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(pos, "CALL")
        mock_close.assert_not_called()
        self.assertIn("trailing active", result)

    def test_smci_style_deep_winner_survives_a_giveback_that_used_to_exit(self):
        # Reproduces the real 2026-08-12 incident: SMCI's call peaked at
        # +160% (option $3.44 off a $1.32 entry), the OLD flat 30% giveback
        # closed the remaining position at $2.38 (a 30.8% pullback from
        # peak), and premium then ran on to a $4.25 high / $3.75 close the
        # SAME DAY -- verified missed continuation. Same shape here: peak
        # +160% (2.60 off 1.00), current giveback ~32.7% off peak -- past
        # the OLD flat 30% (would have exited) but under the new widened
        # tolerance at this depth of profit (interpolated toward 50%).
        pos = self._pos(peak_premium=2.60, target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.75)):
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(pos, "CALL")
        mock_close.assert_not_called()
        self.assertIn("trailing active", result)

    def test_get_snapshot_fn_used_instead_of_rest_when_it_returns_data(self):
        # Added 2026-08-12 alongside the options WebSocket stream: a
        # provided get_snapshot_fn (the daemon's real-time quote cache)
        # must win over the REST snapshot, and the REST path must not be
        # called at all when the injected one already has data -- same
        # contract as run_equity_guard's get_price_fn.
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot") as mock_rest:
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(
                        pos, "CALL", get_snapshot_fn=lambda occ: self._snap(1.05))
        mock_rest.assert_not_called()
        mock_close.assert_not_called()
        self.assertIn("+5%", result)

    def test_get_snapshot_fn_falls_back_to_rest_when_it_returns_none(self):
        # A cold/stale/disconnected stream returns None -- must fall
        # through to the REST snapshot exactly as if no fn were given at
        # all, never leaving the position unchecked.
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)) as mock_rest:
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(
                        pos, "CALL", get_snapshot_fn=lambda occ: None)
        mock_rest.assert_called_once()
        mock_close.assert_not_called()
        self.assertIn("+5%", result)

    def test_greeks_default_gracefully_when_stream_snapshot_omits_them(self):
        # The real-time feed only carries bid/ask/mid/sizes, no Greeks --
        # this must not crash and must fall back to the entry-delta /
        # zeroed Greeks the function already defaults to via .get().
        pos = self._pos(target1=999.0, atr=0.62)
        stream_snap = {"bid": 1.03, "ask": 1.07, "mid": 1.05}
        with patch.object(a, "_get_option_snapshot") as mock_rest:
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(
                        pos, "CALL", get_snapshot_fn=lambda occ: stream_snap)
        mock_rest.assert_not_called()
        mock_close.assert_not_called()
        self.assertIn("Δ 0.62", result)   # entry delta default
        self.assertIn("θ 0.000", result)  # theta defaults to 0

    def test_get_price_fn_used_for_milestone_underlying_display_price(self):
        # Added 2026-08-15 alongside GUARD_EVERY_S dropping to 10s:
        # get_live_price() is an uncached REST call, so blindly calling it
        # every cycle for the milestone alert's cosmetic underlying-price
        # line would 6x that call volume for zero decision-making benefit.
        # get_price_fn (the daemon's real-time equity quote cache) must be
        # tried first, with get_live_price() only as a fallback.
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)):
            with patch.object(a, "get_live_price") as mock_rest_price:
                with patch.object(a, "_submit_options_close"):
                    with patch.object(a, "send_telegram", return_value=True):
                        a._monitor_option_position(
                            pos, "CALL", get_price_fn=lambda t: 42.5)
        mock_rest_price.assert_not_called()
        self.assertEqual(a._check_options_pnl_milestone.call_args.args[5], 42.5)

    def test_get_price_fn_falls_back_to_rest_when_it_returns_none(self):
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)):
            with patch.object(a, "get_live_price", return_value=33.0) as mock_rest_price:
                with patch.object(a, "_submit_options_close"):
                    with patch.object(a, "send_telegram", return_value=True):
                        a._monitor_option_position(
                            pos, "CALL", get_price_fn=lambda t: None)
        mock_rest_price.assert_called_once()
        self.assertEqual(a._check_options_pnl_milestone.call_args.args[5], 33.0)

    def test_extreme_giveback_still_exits_even_at_deep_profit(self):
        # The widened tolerance is not unlimited -- a giveback past even
        # the MAX_PCT ceiling (interpolated to 50% at this +160% peak gain)
        # must still trigger the exit.
        pos = self._pos(peak_premium=2.60)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.20)):   # 53.8% off peak
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord3")) as mock_close:
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._monitor_option_position(pos, "CALL")
        mock_close.assert_called_once()
        self.assertIn("TRAIL EXIT", mock_tg.call_args[0][0])

    def test_peak_premium_updates_on_new_high(self):
        self._write_positions([self._pos(peak_premium=1.00)])
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.10)):
            with patch.object(a, "_submit_options_close"):
                with patch.object(a, "send_telegram", return_value=True):
                    with patch.object(a, "_update_option_position_field") as mock_update:
                        a._monitor_option_position(self._pos(peak_premium=1.00), "CALL")
        mock_update.assert_any_call("SMCI260814C00034000", peak_premium=1.10)

    def test_peak_premium_does_not_regress_on_a_lower_price(self):
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(0.80)):
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord4")):
                with patch.object(a, "send_telegram", return_value=True):
                    with patch.object(a, "_update_option_position_field") as mock_update:
                        a._monitor_option_position(self._pos(peak_premium=1.50), "CALL")
        for c in mock_update.call_args_list:
            self.assertNotIn("peak_premium", c.kwargs)

    def test_t1_half_sell_unaffected_by_trailing_change(self):
        # Real regression guard: T1 logic must still work exactly as before.
        pos = self._pos(target1=1.20, shares=200)   # 2 contracts
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.25)):   # past T1, below trail-activate
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord3")) as mock_close:
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._monitor_option_position(pos, "CALL")
        mock_close.assert_called_once_with("SMCI260814C00034000", 1, "SMCI CALL T1 half")
        self.assertIn("T1 HIT", mock_tg.call_args[0][0])

    def test_milestone_check_is_invoked(self):
        # setUp already patches _check_options_pnl_milestone to a MagicMock --
        # just inspect that mock directly rather than re-patching mid-test.
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)):
            with patch.object(a, "send_telegram", return_value=True):
                a._monitor_option_position(self._pos(), "CALL")
        a._check_options_pnl_milestone.assert_called_once()


class TestRunOptionsGuardSnapshotFnPassthrough(unittest.TestCase):
    """Added 2026-08-12: run_options_guard() must forward get_snapshot_fn
    through to _monitor_option_position() for every naked call/put leg --
    the actual point of the injection point is defeated if it's silently
    dropped at the dispatch layer."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.close()
        self._patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._pos_tmp.name):
            os.unlink(self._pos_tmp.name)

    def _write_positions(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def test_snapshot_fn_forwarded_to_calls_and_puts(self):
        self._write_positions([
            {"ticker": "AAPL", "setup": "Options Call AAPL260821C00310000 ($310C exp 2026-08-21)"},
            {"ticker": "TSLA", "setup": "Options Put TSLA260821P00250000 ($250P exp 2026-08-21)"},
        ])
        sentinel_fn = lambda occ: None
        with patch.object(a, "_monitor_option_position", return_value=None) as mock_mon:
            a.run_options_guard(verbose=False, get_snapshot_fn=sentinel_fn)
        self.assertEqual(mock_mon.call_count, 2)
        for c in mock_mon.call_args_list:
            self.assertIs(c.kwargs.get("get_snapshot_fn"), sentinel_fn)


class TestSyncAlpacaFillsStatusMatching(unittest.TestCase):
    """Confirmed live 2026-08-07: sync_alpaca_fills() compared
    str(order.status) against the bare string "filled", but a real Alpaca
    order's str() is "OrderStatus.FILLED" -- meaning this check was true
    for EVERY order regardless of actual status, and the function has
    never once auto-recorded a real trade close since it was written.
    IOTR, ARTL, AMZN, FERG, and W all needed manual --mode record
    intervention because of this single line. A regression here means
    going back to a supposedly-automatic sync that silently does nothing."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._sync_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._sync_tmp.write(b"{}")
        self._sync_tmp.close()
        self._wr_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._wr_tmp.write(b"[]")
        self._wr_tmp.close()
        self._pnl_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pnl_tmp.write(b"{}")
        self._pnl_tmp.close()
        # PositionTracker's filepath is an early-bound default parameter
        # (filepath: str = POSITIONS_FILE, evaluated once at class-definition
        # time) -- patch.object(a, "POSITIONS_FILE", ...) does NOT affect
        # calls that don't pass filepath explicitly. Confirmed the hard way:
        # an earlier version of this test that relied on the patch alone
        # silently operated on the REAL production dman_positions.json,
        # adding a fake IOTR entry and deleting the real CLRO position via
        # pt.close() inside sync_alpaca_fills(). Every PositionTracker(...)
        # call in this test passes filepath explicitly for that reason.
        # sync_alpaca_fills() itself instantiates PositionTracker() with no
        # explicit filepath too, so patching POSITIONS_FILE alone can't
        # reach it either -- patch the class reference itself so ANY
        # PositionTracker() call anywhere in this test's call graph is
        # forced onto the isolated temp file, regardless of how it's
        # constructed.
        import functools
        _isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patches = [
            patch.object(a, "PositionTracker", _isolated_pt),
            patch.object(a, "ALPACA_SYNC_FILE", self._sync_tmp.name),
            patch.object(a, "DAILY_PNL_FILE", self._pnl_tmp.name),
            patch.object(a, "send_telegram", return_value=True),
        ]
        for p in self._patches:
            p.start()
        _isolated_pt().open(a.OpenPosition(
            ticker="IOTR", bias="LONG", setup="Low Float Catalyst",
            entry=3.5168, stop=3.0, target1=4.0, target2=4.5,
            shares=7, entry_date="2026-07-08",
        ))

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._pos_tmp, self._sync_tmp, self._wr_tmp, self._pnl_tmp):
            os.unlink(f.name)

    def _order(self, side, status, filled_avg_price=None, filled_qty="7"):
        o = MagicMock()
        o.id = "order-1"
        o.side = side
        o.status = status
        o.filled_avg_price = filled_avg_price
        o.filled_qty = filled_qty
        o.qty = filled_qty
        o.filled_at = None
        return o

    def test_real_filled_sell_order_is_recorded(self):
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # IOTR no longer held
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 1)
        self.assertEqual(len(tracker.records), 1)
        self.assertEqual(tracker.records[0].ticker, "IOTR")
        self.assertEqual(tracker.records[0].exit, 3.21)
        self.assertTrue(tracker.records[0].is_live)

    def test_real_close_writes_a_tombstone_for_the_merge_guard(self):
        # See merge_positions_snapshots()'s docstring for the full incident
        # this closes: without a tombstone recorded here, the NEXT
        # git_sync() cycle's merge_positions_snapshots() call had no way to
        # know this exact close was already confirmed against real Alpaca
        # ground truth, and kept resurrecting it from a stale remote copy.
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            a.sync_alpaca_fills(tracker)
        self.assertIn("IOTR", a._recent_closed_identities())

    def test_non_filled_order_is_not_recorded(self):
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.PENDING_NEW),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0)

    def test_already_recorded_close_still_clears_a_resurrected_entry(self):
        # Confirmed live 2026-08-11: CLRO's closing stop order (filled
        # 2026-08-06) was already in recorded_ids from when it first closed
        # correctly, but merge_positions_snapshots() later resurrected the
        # tracked entry from a stale snapshot. Every sync_alpaca_fills()
        # cycle since then just `continue`-d past the already-recorded order
        # id and never reached a pt.close() call -- removal used to live
        # ONLY inside the "found a NEW closing fill" branch -- leaving CLRO
        # stuck as a phantom "open" position for 5 real days. Once
        # _alp_sym is confirmed absent from Alpaca, the stale entry must be
        # cleared regardless of whether its closing order was recorded before.
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # not held at Alpaca
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21),
        ]
        with open(self._sync_tmp.name, "w") as f:
            json.dump({"last_sync": None, "recorded_ids": ["order-1"]}, f)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0, "an already-recorded close must not be double-recorded")
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0,
                          "the stale entry must still be cleared even though nothing new was recorded")

    def test_matching_win_rate_record_blocks_a_second_recording_even_with_a_new_order_id(self):
        # Confirmed live 2026-08-11: _restore_corrupted_json()'s git-checkout
        # fallback (dman_daemon.py) can revert dman_alpaca_sync.json to a
        # PAST commit that predates a real order id being safely pushed,
        # un-recording it from recorded_ids -- so the SAME real close (CLRO)
        # got detected as "new" and re-recorded 5 times in one day, each one
        # double-counting P&L into dman_win_rate.json and dman_daily_pnl.json
        # (which accumulated to a phantom -24.76%, well past the 3% daily
        # loss circuit breaker). This is the independent ledger-level guard:
        # even with a genuinely different (never-seen) order id, a matching
        # ticker+setup+exit already in win-rate history must block a second
        # recording.
        from alpaca.trading.enums import OrderStatus, OrderSide
        tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
        tracker.record(a.TradeRecord(
            ticker="IOTR", date="2026-08-06", bias="LONG", setup="Low Float Catalyst",
            entry=3.52, exit=3.21, outcome="LOSS", pnl_pct=-8.7, score=0, is_live=True,
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        order = self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21)
        order.id = "a-different-never-seen-order-id"
        mock_client.get_orders.return_value = [order]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0, "a ledger-matching close must not be recorded again")
        self.assertEqual(len([r for r in tracker.records if r.ticker == "IOTR"]), 1,
                          "win-rate history must still have exactly one IOTR record")


class TestAiThemeMomentumBonus(unittest.TestCase):
    """Added 2026-08-07, direct instruction: AI isn't a GICS/SPDR sector, so
    AI-heavy names (NVDA, PLTR, SMCI, ...) only ever got generic Technology/
    XLK momentum scoring even when the AI theme specifically was moving much
    sharper than broad tech. This is an ADDITIVE bonus on top of the
    existing sector score, not a replacement -- a regression here means
    either the bonus stops firing for real AI names, or it starts leaking
    onto non-AI tickers / non-AI-driven sector moves."""

    def _etf_df(self, chg_pct):
        import pandas as pd
        base = 100.0
        return pd.DataFrame({"Close": [base, base * (1 + chg_pct / 100)]})

    def _signal(self, ticker, bias="LONG"):
        return a.ProSignal(
            ticker=ticker, bias=bias, setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            shares=100, rr=2.0, rsi=50.0, rvol=2.0,
            reason="test", confluence_score=0,
        )

    def _score(self, sig, ai_etf_chg_pct):
        def _fetch_df_side_effect(symbol, *args, **kwargs):
            if symbol == a.AI_THEME_ETF:
                return self._etf_df(ai_etf_chg_pct)
            return None   # sector ETF / everything else — no-op cleanly
        # Mocked components kept deliberately low, leaving headroom below
        # the confluence_score's min(100, total) cap -- otherwise both the
        # flat and boosted runs saturate at 100 and the +3 bonus becomes
        # invisible to a score-difference assertion.
        with patch.object(a, "check_mtf", return_value=(True, 2)), \
             patch.object(a, "check_relative_strength", return_value=(True, 2)), \
             patch.object(a, "check_sector", return_value=(True, 2)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 1)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", side_effect=_fetch_df_side_effect):
            return a.score_signal(sig, _fake_df(), _fake_regime(), a.WinRateTracker())

    def test_ai_ticker_with_strong_ai_momentum_gets_the_bonus(self):
        self.assertIn("NVDA", a.AI_THEME_TICKERS)
        scored = self._score(self._signal("NVDA"), ai_etf_chg_pct=2.0)
        self.assertGreaterEqual(scored.confluence_score, 3)

    def test_ai_ticker_with_flat_ai_momentum_gets_no_bonus(self):
        base = self._score(self._signal("NVDA"), ai_etf_chg_pct=0.0)
        boosted = self._score(self._signal("NVDA"), ai_etf_chg_pct=2.0)
        self.assertLess(base.confluence_score, boosted.confluence_score)

    def test_non_ai_ticker_never_gets_the_bonus_even_on_a_hot_ai_tape(self):
        self.assertNotIn("KO", a.AI_THEME_TICKERS)
        with_hot_ai = self._score(self._signal("KO"), ai_etf_chg_pct=5.0)
        flat_ai = self._score(self._signal("KO"), ai_etf_chg_pct=0.0)
        self.assertEqual(with_hot_ai.confluence_score, flat_ai.confluence_score)


class TestDay2ContinuationExtensionCap(unittest.TestCase):
    """Confirmed live 2026-08-06: AMZN passed every one of Day 2
    Continuation's existing gates by a wide margin (12.5% d1_gap vs 4% min,
    80% d1_held vs 60% min) and still bought the exact top of its whole
    holding period, already +20.8% from the pre-gap baseline by entry. None
    of the existing gates check total cumulative extension, only the shape
    of day 1's gap. A regression here means going back to chasing an
    already-exhausted multi-day move as if it were a fresh continuation."""

    def test_amzn_style_overextension_is_blocked(self):
        # Real numbers from the live incident: pre-gap close $235.50,
        # entry $284.43 -- +20.8% cumulative, above the 15% cap.
        self.assertFalse(a._day2_continuation_not_overextended(284.43, 235.50))

    def test_modest_continuation_is_allowed(self):
        # A healthier, more typical Day 2 Continuation shape -- gapped and
        # continued, but nowhere near AMZN's already-exhausted move.
        self.assertTrue(a._day2_continuation_not_overextended(255.0, 235.50))

    def test_exactly_at_cap_is_allowed(self):
        pre_gap = 100.0
        entry = pre_gap * (1 + a.DAY2_MAX_CUMULATIVE_MOVE_PCT / 100)
        self.assertTrue(a._day2_continuation_not_overextended(entry, pre_gap))

    def test_just_above_cap_is_blocked(self):
        pre_gap = 100.0
        entry = pre_gap * (1 + (a.DAY2_MAX_CUMULATIVE_MOVE_PCT + 0.1) / 100)
        self.assertFalse(a._day2_continuation_not_overextended(entry, pre_gap))

    def test_bad_baseline_data_fails_open(self):
        # Can't compute a ratio against a zero/negative baseline -- must not
        # block a legitimate signal just because of a bad data point.
        self.assertTrue(a._day2_continuation_not_overextended(284.43, 0.0))


class TestSectorEtfLookupConsistency(unittest.TestCase):
    """Confirmed live 2026-08-07: a duplicate SECTOR_ETF (singular) dict used
    GICS full names ("Health Care", "Communication Services", "Consumer
    Discretionary") that didn't match the abbreviated labels TICKER_SECTOR
    actually assigns ("Healthcare", "Comm Services", "Consumer Disc") --
    silently zeroing the 8-pt sector-ETF-momentum confluence score for 28 of
    ~84 curated tickers. A regression here (e.g. a new ticker tagged with a
    sector name that doesn't exist in SECTOR_ETFS) means the same silent
    scoring gap for whatever's newly mismatched."""

    def test_every_sector_used_in_ticker_map_resolves_to_a_real_etf(self):
        used_sectors = set(a.TICKER_SECTOR.values()) - {""}
        for sector in used_sectors:
            etf = a.SECTOR_ETFS.get(sector, "")
            self.assertTrue(etf, f"sector {sector!r} (used in TICKER_SECTOR) has no ETF in SECTOR_ETFS")

    def test_known_previously_broken_tickers_now_resolve(self):
        # These are real tickers from the live incident -- GOOGL/NFLX
        # (Comm Services), AMZN/TSLA (Consumer Disc), LLY (Healthcare).
        cases = {"GOOGL": "XLC", "AMZN": "XLY", "LLY": "XLV"}
        for ticker, expected_etf in cases.items():
            sector = a.TICKER_SECTOR.get(ticker, "")
            etf = a.SECTOR_ETFS.get(sector, "")
            self.assertEqual(etf, expected_etf)


class TestDynamicTickerPriceFloor(unittest.TestCase):
    """Widened 2026-08-07: the $2.00 price floor excluded every sub-$2 penny
    stock from organic smallcap discovery entirely, even though DMan's own
    real style explicitly includes sub-$1 plays. A regression here means
    going back to silently dropping exactly the kind of ticker this net is
    supposed to catch."""

    def _quote(self, symbol, price, chg_pct=5.0, vol=1_000_000, avg_vol=200_000):
        return {"symbol": symbol, "regularMarketPrice": price,
                "regularMarketChangePercent": chg_pct,
                "regularMarketVolume": vol, "averageDailyVolume10Day": avg_vol}

    def _mock_response(self, quotes):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"finance": {"result": [{"quotes": quotes}]}}
        return resp

    def test_sub_two_dollar_ticker_now_passes(self):
        quotes = [self._quote("CISS", 0.85)]
        with patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = self._mock_response(quotes)
            result = a.fetch_dman_dynamic_tickers()
        self.assertIn("CISS", result)

    def test_below_new_floor_still_excluded(self):
        # 0.20 is the new floor -- true sub-penny/halted-risk names below
        # that are still deliberately excluded, this isn't "no floor at all".
        quotes = [self._quote("SUBPENNY", 0.05)]
        with patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = self._mock_response(quotes)
            result = a.fetch_dman_dynamic_tickers()
        self.assertNotIn("SUBPENNY", result)

    def test_requests_50_per_screener_not_25(self):
        with patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = self._mock_response([])
            a.fetch_dman_dynamic_tickers()
        for call in mock_requests.get.call_args_list:
            self.assertIn("count=50", call[0][0])


class TestNewsSentimentVerdict(unittest.TestCase):
    """_news_sentiment_verdict() is a majority vote over
    _fetch_massive_reference_news()'s per-article insights[] entries.
    Confirmed live 2026-08-13: Massive's /v2/reference/news endpoint is
    accessible under the current MASSIVE_API_KEY (real multi-publisher
    results with sentiment) even though the separate analyst-ratings tier
    is 403'd -- these tests lock in the vote-counting logic in isolation
    from the catalyst-gate integration."""

    def _article(self, ticker, sentiment):
        return {"title": "test", "insights": [{"ticker": ticker, "sentiment": sentiment}]}

    def test_no_articles_returns_none(self):
        with patch.object(a, "_fetch_massive_reference_news", return_value=[]):
            self.assertIsNone(a._news_sentiment_verdict("FGL"))

    def test_majority_positive_wins(self):
        articles = [self._article("FGL", "positive"), self._article("FGL", "positive"),
                    self._article("FGL", "negative")]
        with patch.object(a, "_fetch_massive_reference_news", return_value=articles):
            self.assertEqual(a._news_sentiment_verdict("FGL"), "positive")

    def test_majority_negative_wins(self):
        articles = [self._article("FGL", "negative"), self._article("FGL", "negative"),
                    self._article("FGL", "neutral")]
        with patch.object(a, "_fetch_massive_reference_news", return_value=articles):
            self.assertEqual(a._news_sentiment_verdict("FGL"), "negative")

    def test_only_matching_ticker_insights_counted(self):
        # An article can tag multiple tickers -- only THIS ticker's own
        # insight entry should count, not an unrelated co-mentioned symbol.
        articles = [self._article("OTHERCO", "negative"), self._article("FGL", "positive")]
        with patch.object(a, "_fetch_massive_reference_news", return_value=articles):
            self.assertEqual(a._news_sentiment_verdict("FGL"), "positive")

    def test_articles_with_no_matching_insights_return_none(self):
        articles = [self._article("OTHERCO", "negative")]
        with patch.object(a, "_fetch_massive_reference_news", return_value=articles):
            self.assertIsNone(a._news_sentiment_verdict("FGL"))


class TestNewsBoostAfterSentimentVeto(unittest.TestCase):
    """Added 2026-08-14, direct instruction to make sure news actually
    helps get into plays: run_pro_scanner()'s news_boost (+5 confluence
    points, and an MTF override for Gap & Hold / Bear Gap Hold — see
    score_signal) used to fire off a bare "does a headline exist" check,
    with zero regard for whether that headline was good or bad news.
    These lock in the sentiment veto in isolation from the full scan."""

    def test_no_headline_returns_false_without_a_sentiment_call(self):
        with patch.object(a, "_news_sentiment_verdict") as mock_verdict:
            result = a._news_boost_after_sentiment_veto(False, "FGL")
        self.assertFalse(result)
        mock_verdict.assert_not_called()

    def test_headline_with_negative_sentiment_is_vetoed(self):
        with patch.object(a, "_news_sentiment_verdict", return_value="negative"):
            self.assertFalse(a._news_boost_after_sentiment_veto(True, "FGL"))

    def test_headline_with_positive_sentiment_passes(self):
        with patch.object(a, "_news_sentiment_verdict", return_value="positive"):
            self.assertTrue(a._news_boost_after_sentiment_veto(True, "FGL"))

    def test_headline_with_unknown_sentiment_passes(self):
        # None (no sentiment data found) must NOT veto -- same "don't let a
        # missing refinement override an already-confirmed catalyst" rule
        # detect_low_float_catalyst()'s gate already uses.
        with patch.object(a, "_news_sentiment_verdict", return_value=None):
            self.assertTrue(a._news_boost_after_sentiment_veto(True, "FGL"))

    def test_headline_with_neutral_sentiment_passes(self):
        with patch.object(a, "_news_sentiment_verdict", return_value="neutral"):
            self.assertTrue(a._news_boost_after_sentiment_veto(True, "FGL"))


class TestPrewarmAlpacaBarsChunking(unittest.TestCase):
    """Added 2026-08-13 alongside the chunk_size 50->100 / inter-chunk
    sleep 0.2->0.1 retuning (API-usage audit found the original values
    were an untested guess from before this account ever ran a single
    scan under real Algo Trader Plus credentials -- since proven safe
    across every live session with zero recorded 429s). Locks in the new
    default and confirms chunking still respects whatever chunk_size is
    actually passed, so a future retune can't silently break batching."""

    def setUp(self):
        self._cache_backup = dict(a._cache)
        a._cache.clear()

    def tearDown(self):
        a._cache.clear()
        a._cache.update(self._cache_backup)

    def _fake_bar(self):
        return MagicMock(open=10.0, high=11.0, low=9.5, close=10.5, volume=1000,
                         timestamp=datetime.now() - timedelta(days=1))

    def test_default_chunk_size_is_100(self):
        import inspect
        sig = inspect.signature(a.prewarm_alpaca_bars)
        self.assertEqual(sig.parameters["chunk_size"].default, 100)

    def test_batches_respect_explicit_chunk_size(self):
        tickers = [f"TEST{i}" for i in range(5)]
        calls: list[list[str]] = []

        def fake_get_stock_bars(req):
            calls.append(list(req.symbol_or_symbols))
            resp = MagicMock()
            # _bars_to_df requires >= 20 bars (min_bars default) or it
            # returns None and this ticker silently doesn't get warmed.
            resp.data = {t: [self._fake_bar()] * 25 for t in req.symbol_or_symbols}
            return resp

        mock_dc = MagicMock()
        mock_dc.get_stock_bars.side_effect = fake_get_stock_bars
        with patch.object(a, "ALPACA_AVAILABLE", True), \
             patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            warmed = a.prewarm_alpaca_bars(tickers, chunk_size=2)
        self.assertEqual([len(c) for c in calls], [2, 2, 1])
        self.assertEqual(warmed, 5)

    def test_no_alpaca_client_returns_zero(self):
        with patch.object(a, "ALPACA_AVAILABLE", True), \
             patch.object(a, "get_alpaca_data_client", return_value=None):
            self.assertEqual(a.prewarm_alpaca_bars(["AAPL"]), 0)

    def test_alpaca_unavailable_returns_zero_without_a_call(self):
        with patch.object(a, "ALPACA_AVAILABLE", False), \
             patch.object(a, "get_alpaca_data_client") as mock_get_client:
            self.assertEqual(a.prewarm_alpaca_bars(["AAPL"]), 0)
        mock_get_client.assert_not_called()


class TestAlertMassiveApiFailure(unittest.TestCase):
    """Added 2026-08-13 after an API audit found fetch_earnings_mover_tickers()
    and _fetch_massive_reference_news() both failed open with zero logging
    or alerting on a non-200/exception -- the same silent-failure shape
    that cost a full week of missed options signals during the 2026-08-06
    OPRA entitlement 403 before anyone noticed. This is real production
    Telegram/dedup-file plumbing, not a throwaway helper -- must never hit
    the live dman_alerts_dedup.json or send a real Telegram message from
    a test run."""

    def setUp(self):
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.write(b"{}")
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

    def tearDown(self):
        self._dedup_patch.stop()
        os.unlink(self._dedup_tmp.name)

    def test_sends_one_telegram_alert_on_first_failure(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._alert_massive_api_failure("earnings-mover", "HTTP 500")
        mock_tg.assert_called_once()
        self.assertIn("earnings-mover", mock_tg.call_args[0][0])
        self.assertIn("HTTP 500", mock_tg.call_args[0][0])

    def test_second_failure_same_day_same_source_is_deduped(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._alert_massive_api_failure("earnings-mover", "HTTP 500")
            a._alert_massive_api_failure("earnings-mover", "HTTP 500 again")
        mock_tg.assert_called_once()

    def test_different_sources_each_get_their_own_alert(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._alert_massive_api_failure("earnings-mover", "HTTP 500")
            a._alert_massive_api_failure("reference-news", "HTTP 403")
        self.assertEqual(mock_tg.call_count, 2)

    def test_telegram_send_failure_does_not_raise(self):
        with patch.object(a, "send_telegram", side_effect=Exception("telegram down")):
            a._alert_massive_api_failure("earnings-mover", "HTTP 500")   # must not raise


class TestFetchMassiveReferenceNews(unittest.TestCase):
    """Isolated tests for the raw HTTP wrapper — fail-open behavior and
    the real endpoint/params shape confirmed live 2026-08-13."""

    def setUp(self):
        a._MASSIVE_SENTIMENT_CACHE.clear()

    def test_no_api_key_returns_empty_without_a_call(self):
        with patch.object(a, "MASSIVE_API_KEY", ""), \
             patch.object(a, "requests") as mock_requests:
            result = a._fetch_massive_reference_news("FGL")
        self.assertEqual(result, [])
        mock_requests.get.assert_not_called()

    def test_non_200_returns_empty(self):
        with patch.object(a, "MASSIVE_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_alert_massive_api_failure") as mock_alert:
            mock_requests.get.return_value = MagicMock(status_code=403)
            result = a._fetch_massive_reference_news("FGL")
        self.assertEqual(result, [])
        mock_alert.assert_called_once()
        self.assertEqual(mock_alert.call_args[0][0], "reference-news")

    def test_exception_fails_open_to_empty_list(self):
        with patch.object(a, "MASSIVE_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_alert_massive_api_failure") as mock_alert:
            mock_requests.get.side_effect = Exception("network error")
            result = a._fetch_massive_reference_news("FGL")
        self.assertEqual(result, [])
        mock_alert.assert_called_once()

    def test_successful_call_hits_the_confirmed_endpoint(self):
        with patch.object(a, "MASSIVE_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = MagicMock(
                status_code=200, json=lambda: {"results": [{"title": "x"}]})
            result = a._fetch_massive_reference_news("FGL")
        self.assertEqual(result, [{"title": "x"}])
        url = mock_requests.get.call_args[0][0]
        self.assertEqual(url, "https://api.massive.com/v2/reference/news")
        params = mock_requests.get.call_args[1]["params"]
        self.assertEqual(params["ticker"], "FGL")


class TestFetchEarningsMoverTickers(unittest.TestCase):
    """Confirmed live 2026-08-12: CRWV beat EPS by +30.9% and gapped +15.7%
    the next morning on 24M avg daily volume, and the algo never once
    considered it because every other earnings lookup in this file only
    checks tickers already on the fixed WATCHLIST. fetch_earnings_mover_tickers()
    closes that blind spot via Massive's earnings endpoint, which (confirmed
    live the same day) supports a plain date-range query with no ticker
    filter. These tests lock in the filtering: a real beat AND a real live
    gap AND enough liquidity, restricted to names not already covered by
    the normal watchlist path."""

    def _hist(self, prev_close, avg_vol):
        import pandas as pd
        return pd.DataFrame({
            "Close":  [prev_close] * 10,
            "Volume": [avg_vol] * 10,
        })

    def _earnings_response(self, records):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"results": records}
        return resp

    def _record(self, ticker, importance=4, eps_pct=0.30, rev_pct=0.01):
        return {"ticker": ticker, "importance": importance,
                "eps_surprise_percent": eps_pct, "revenue_surprise_percent": rev_pct}

    def setUp(self):
        self._patches = [
            patch.object(a, "ENABLE_EARNINGS_MOVER_SCAN", True),
            patch.object(a, "MASSIVE_API_KEY", "test-key"),
            patch.object(a, "WATCHLIST", ["AAPL", "MSFT"]),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_real_beat_and_gap_is_found(self):
        # The actual CRWV shape: importance 4, +30.9% EPS beat, live price
        # gapped well above prior close on real volume.
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=104.51), \
             patch.object(a, "fetch_df", return_value=self._hist(90.32, 24_000_000)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("CRWV")])
            result = a.fetch_earnings_mover_tickers()
        self.assertIn("CRWV", result)

    def test_ticker_already_on_watchlist_is_excluded(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=104.51), \
             patch.object(a, "fetch_df", return_value=self._hist(90.32, 24_000_000)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("AAPL")])
            result = a.fetch_earnings_mover_tickers()
        self.assertNotIn("AAPL", result,
                          "the normal watchlist path already covers this ticker")

    def test_below_importance_threshold_excluded(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=104.51), \
             patch.object(a, "fetch_df", return_value=self._hist(90.32, 24_000_000)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("CRWV", importance=2)])
            result = a.fetch_earnings_mover_tickers()
        self.assertNotIn("CRWV", result)

    def test_beat_on_paper_but_no_real_gap_excluded(self):
        # SLAB's actual shape: +22.4% EPS beat but the stock didn't move at
        # all -- market already priced it in, not a signal.
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=218.905), \
             patch.object(a, "fetch_df", return_value=self._hist(218.94, 178_940)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("SLAB", eps_pct=0.224, rev_pct=-0.0009)])
            result = a.fetch_earnings_mover_tickers()
        self.assertNotIn("SLAB", result,
                          "a beat with no real price reaction must not be treated as a signal")

    def test_miss_is_never_included_shorts_disabled(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=60.0), \
             patch.object(a, "fetch_df", return_value=self._hist(80.0, 5_000_000)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("MISS", eps_pct=-0.30, rev_pct=-0.10)])
            result = a.fetch_earnings_mover_tickers()
        self.assertNotIn("MISS", result,
                          "ALLOW_SHORTS is False -- a miss, however dramatic the gap down, isn't tradeable")

    def test_illiquid_name_excluded(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=12.0), \
             patch.object(a, "fetch_df", return_value=self._hist(10.0, 50_000)):
            mock_requests.get.return_value = self._earnings_response(
                [self._record("THIN")])
            result = a.fetch_earnings_mover_tickers()
        self.assertNotIn("THIN", result)

    def test_flag_disabled_returns_empty(self):
        with patch.object(a, "ENABLE_EARNINGS_MOVER_SCAN", False):
            with patch.object(a, "requests") as mock_requests:
                result = a.fetch_earnings_mover_tickers()
        mock_requests.get.assert_not_called()
        self.assertEqual(result, [])

    def test_no_api_key_returns_empty(self):
        with patch.object(a, "MASSIVE_API_KEY", ""):
            with patch.object(a, "requests") as mock_requests:
                result = a.fetch_earnings_mover_tickers()
        mock_requests.get.assert_not_called()
        self.assertEqual(result, [])

    def test_http_error_fails_open_to_empty_list(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_alert_massive_api_failure") as mock_alert:
            mock_requests.get.side_effect = Exception("network error")
            result = a.fetch_earnings_mover_tickers()
        self.assertEqual(result, [])
        mock_alert.assert_called_once()
        self.assertEqual(mock_alert.call_args[0][0], "earnings-mover")

    def test_non_200_fails_open_and_alerts(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_alert_massive_api_failure") as mock_alert:
            mock_requests.get.return_value = MagicMock(status_code=500)
            result = a.fetch_earnings_mover_tickers()
        self.assertEqual(result, [])
        mock_alert.assert_called_once()
        self.assertEqual(mock_alert.call_args[0][0], "earnings-mover")


class TestAlertsIncludeOptionsHint(unittest.TestCase):
    """Direct feedback: when the algo sends a play, the user wants to act
    on it immediately via the /options -> /buy Telegram flow rather than
    typing the ticker in by hand. Every alert that represents an actual
    play (a real signal, or a StockTwits call worth a look) must include a
    ready-to-use "/options TICKER" line."""

    def _signal(self, ticker="ABCD", entry=50.0):
        return a.ProSignal(
            ticker=ticker, bias="LONG", setup="Gap & Hold",
            entry=entry, stop=48.0, target1=54.0, target2=58.0,
            shares=10, rr=2.0, rsi=55.0, rvol=2.0,
            reason="test signal", confluence_score=85,
        )

    def test_format_signal_telegram_includes_options_hint(self):
        msg = a.format_signal_telegram(self._signal(), {"regime": "BULL"})
        self.assertIn("/options ABCD", msg)

    def test_format_smallcap_telegram_includes_options_hint(self):
        msg = a.format_smallcap_telegram(self._signal(ticker="MICRO", entry=0.50), fl_m=5.0, sh_pct=0.0)
        self.assertIn("/options MICRO", msg)


class TestStocktwitsQuickTake(unittest.TestCase):
    """_stocktwits_quick_take() attaches a live technical snapshot to a
    freshly-detected StockTwits call so the Telegram alert itself answers
    'is this worth looking at right now' instead of just 'ticker added,
    check back later.' Deliberately a lightweight heuristic, not a
    duplicate of the real scanner's scoring -- these tests lock in that
    it degrades safely (never blocks the alert) and that the strong/weak
    labels point the right direction."""

    def _uptrend_df(self, n=40, spike_vol=True):
        import pandas as pd
        closes = [10.0 + i * 0.15 for i in range(n)]   # steady uptrend
        highs  = [c + 0.1 for c in closes]
        lows   = [c - 0.1 for c in closes]
        vols   = [500_000] * (n - 1) + ([2_000_000] if spike_vol else [500_000])
        return pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                              "Close": closes, "Volume": vols})

    def test_strong_uptrend_gets_positive_label(self):
        df = self._uptrend_df()
        last_close = float(df["Close"].iloc[-1])
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "get_live_price", return_value=last_close * 1.08):
            result = a._stocktwits_quick_take("XYZ")
        self.assertIn("Strong setup", result)

    def test_insufficient_data_fails_open(self):
        import pandas as pd
        with patch.object(a, "fetch_df", return_value=pd.DataFrame({"Close": [1, 2]})):
            result = a._stocktwits_quick_take("XYZ")
        self.assertIn("unavailable", result)

    def test_none_data_fails_open(self):
        with patch.object(a, "fetch_df", return_value=None):
            result = a._stocktwits_quick_take("XYZ")
        self.assertIn("unavailable", result)

    def test_exception_fails_open_never_raises(self):
        with patch.object(a, "fetch_df", side_effect=Exception("network error")):
            result = a._stocktwits_quick_take("XYZ")
        self.assertIn("unavailable", result)

    def test_declining_low_volume_does_not_get_strong_label(self):
        import pandas as pd
        n = 40
        closes = [20.0 - i * 0.15 for i in range(n)]   # steady downtrend
        highs  = [c + 0.1 for c in closes]
        lows   = [c - 0.1 for c in closes]
        vols   = [500_000] * n
        df = pd.DataFrame({"Open": closes, "High": highs, "Low": lows,
                            "Close": closes, "Volume": vols})
        last_close = float(df["Close"].iloc[-1])
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "get_live_price", return_value=last_close * 0.94):
            result = a._stocktwits_quick_take("XYZ")
        self.assertNotIn("Strong setup", result)


class TestFetchAvailableExpiries(unittest.TestCase):
    """Confirmed live 2026-08-12: GetOptionContractsRequest WITHOUT explicit
    expiration_date_gte/lte bounds silently narrows to just the single
    nearest expiry instead of the real listed ladder -- for SMCI that was
    1 expiry (68 contracts) vs. the real 7 weekly expiries (344 contracts)
    once explicit bounds were passed. This is the exact call shape /options
    TICKER needs to let a real "browse other expiries" list exist at all."""

    def test_returns_sorted_distinct_expiries(self):
        class C:
            def __init__(self, exp):
                self.expiration_date = exp
        import datetime as _dt
        raw = MagicMock(option_contracts=[
            C(_dt.date(2026, 8, 21)), C(_dt.date(2026, 8, 14)),
            C(_dt.date(2026, 8, 14)),  # duplicate strike, same expiry
            C(_dt.date(2026, 8, 28)),
        ])
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = raw
        result = a._fetch_available_expiries(mock_client, "SMCI")
        self.assertEqual(result, [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)])
        # Must have passed explicit bounds -- confirmed live this is required
        # for Alpaca to return more than just the nearest expiry.
        _, kwargs = mock_client.get_option_contracts.call_args
        req = mock_client.get_option_contracts.call_args[0][0]
        self.assertIsNotNone(getattr(req, "expiration_date_gte", None))
        self.assertIsNotNone(getattr(req, "expiration_date_lte", None))

    def test_exception_fails_open_to_empty_list(self):
        mock_client = MagicMock()
        mock_client.get_option_contracts.side_effect = Exception("network error")
        result = a._fetch_available_expiries(mock_client, "SMCI")
        self.assertEqual(result, [])


class TestFetchOptionChainForDisplay(unittest.TestCase):
    """_fetch_option_chain_for_display() is the browse-side counterpart to
    _find_best_call_contract/_find_best_put_contract — instead of picking
    ONE ITM-biased 'best' contract, it returns a symmetric band of BOTH
    calls and puts around ATM for a human to choose from via /options.
    These tests lock in the liquidity gate and that contracts with no
    usable quote are dropped rather than shown with a garbage price."""

    def _snap(self, bid, ask, delta=0.5):
        return {"bid": bid, "ask": ask, "delta": delta, "spread_pct": 0.03}

    def test_illiquid_underlying_returns_none(self):
        mock_client = MagicMock()
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 100_000
            result = a._fetch_option_chain_for_display(mock_client, "THIN", 50.0)
        self.assertIsNone(result)
        mock_client.get_option_contracts.assert_not_called()

    def test_returns_both_calls_and_puts(self):
        class FakeContract:
            symbol = "TESTX260814C00100000"
            strike_price = 100.0

        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = MagicMock(option_contracts=[FakeContract()])
        with patch.object(a, "yf") as mock_yf, \
             patch.object(a, "_get_option_snapshot", return_value=self._snap(2.0, 2.1)):
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            result = a._fetch_option_chain_for_display(mock_client, "TESTX", 100.0, num_strikes=4)
        self.assertIsNotNone(result)
        types = {i["type"] for i in result["items"]}
        self.assertEqual(types, {"CALL", "PUT"})

    def test_contracts_with_no_quote_are_skipped(self):
        class FakeContract:
            symbol = "TESTX260814C00100000"
            strike_price = 100.0

        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = MagicMock(option_contracts=[FakeContract()])
        with patch.object(a, "yf") as mock_yf, \
             patch.object(a, "_get_option_snapshot", return_value=self._snap(0, 0)):
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            result = a._fetch_option_chain_for_display(mock_client, "TESTX", 100.0, num_strikes=4)
        self.assertIsNone(result, "a chain with zero usable quotes must return None, not an empty-ish menu")


class TestTelegramOptionsBrowseAndBuy(unittest.TestCase):
    """The /options -> /buy -> YES/NO Telegram flow places REAL orders, so
    these tests are deliberately thorough: staging a confirmation must
    never itself place an order, an expired/invalid menu or confirmation
    must be rejected rather than guessed at, and halt/macro-blackout must
    be enforced at the final YES step regardless of what was staged
    earlier."""

    def _menu(self, ticker="SMCI", expires_in_min=10, expiry="2026-08-14"):
        now = datetime.now(a.ET) if hasattr(a, "ET") else __import__("datetime").datetime.now()
        return {
            "ticker": ticker,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=expires_in_min)).isoformat(),
            "expiry": expiry,
            "items": [
                {"idx": 1, "type": "CALL", "occ_symbol": "SMCI260814C00034000",
                 "strike": 34.0, "bid": 1.50, "ask": 1.55, "delta": 0.51, "delta_estimated": False},
                {"idx": 2, "type": "PUT", "occ_symbol": "SMCI260814P00029500",
                 "strike": 29.5, "bid": 1.05, "ask": 1.10, "delta": -0.42, "delta_estimated": False},
            ],
        }

    def setUp(self):
        self._menu_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._menu_tmp.close()
        self._buy_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._buy_tmp.close()
        self._pos_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._pos_tmp.write("[]")
        self._pos_tmp.close()
        import functools
        _isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patches = [
            patch.object(a, "TELEGRAM_OPTIONS_MENU_FILE", self._menu_tmp.name),
            patch.object(a, "TELEGRAM_MANUAL_BUY_FILE", self._buy_tmp.name),
            patch.object(a, "PositionTracker", _isolated_pt),
            patch.object(a, "send_telegram", return_value=True),
            patch.object(a, "is_halted", return_value=False),
            patch.object(a, "check_macro_safe", return_value=(True, 5)),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._menu_tmp, self._buy_tmp, self._pos_tmp):
            try:
                os.unlink(f.name)
            except FileNotFoundError:
                pass

    # ── /options ──────────────────────────────────────────────────────
    def test_options_command_no_ticker_shows_usage(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options"])
        self.assertIn("Usage", mock_tg.call_args[0][0])

    def test_options_command_no_chain_sends_error(self):
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=[]), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=None), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI"])
        self.assertIn("No liquid options chain", mock_tg.call_args[0][0])

    def test_options_command_saves_menu_and_numbers_items(self):
        chain = {"ticker": "SMCI", "expiry": "2026-08-14", "dte": 2, "underlying_price": 34.0,
                  "items": [{"type": "CALL", "occ_symbol": "X", "strike": 34.0,
                              "bid": 1.5, "ask": 1.55, "delta": 0.51, "delta_estimated": False}]}
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=[]), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=chain), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "smci"])
        with open(self._menu_tmp.name) as f:
            saved = json.load(f)
        self.assertEqual(saved["ticker"], "SMCI")
        self.assertEqual(saved["items"][0]["idx"], 1)
        self.assertIn("CALLS", mock_tg.call_args[0][0])

    def test_options_command_shows_other_expiries_footer(self):
        import datetime as _dt
        chain = {"ticker": "SMCI", "expiry": "2026-08-14", "dte": 2, "underlying_price": 34.0,
                  "items": [{"type": "CALL", "occ_symbol": "X", "strike": 34.0,
                              "bid": 1.5, "ask": 1.55, "delta": 0.51, "delta_estimated": False}]}
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=chain), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI"])
        msg = mock_tg.call_args[0][0]
        self.assertIn("Other expiries", msg)
        self.assertIn("Aug 21", msg)

    def test_options_command_with_valid_expiry_index_uses_that_expiry(self):
        import datetime as _dt
        chain = {"ticker": "SMCI", "expiry": "2026-08-28", "dte": 16, "underlying_price": 34.0,
                  "items": [{"type": "CALL", "occ_symbol": "X", "strike": 34.0,
                              "bid": 1.5, "ask": 1.55, "delta": 0.51, "delta_estimated": False}]}
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=chain) as mock_fetch, \
             patch.object(a, "send_telegram", return_value=True):
            a._handle_options_command(["/options", "SMCI", "3"])
        _, kwargs = mock_fetch.call_args
        self.assertEqual(kwargs.get("expiry"), _dt.date(2026, 8, 28))

    def test_options_command_with_invalid_expiry_index_shows_error(self):
        import datetime as _dt
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21)]
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI", "9"])
        self.assertIn("No expiry #9", mock_tg.call_args[0][0])

    # ── /buy ──────────────────────────────────────────────────────────
    def test_buy_command_no_menu_file(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1"])
        self.assertIn("No active /options menu", mock_tg.call_args[0][0])

    def test_buy_command_expired_menu(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(expires_in_min=-5), f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1"])
        self.assertIn("expired", mock_tg.call_args[0][0])

    def test_buy_command_invalid_index(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "99"])
        self.assertIn("No item #99", mock_tg.call_args[0][0])

    def test_buy_command_defaults_to_one_contract_at_ask(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1"])
        mock_gac.assert_not_called()   # staging a confirmation must never touch the broker
        with open(self._buy_tmp.name) as f:
            pending = json.load(f)
        self.assertEqual(pending["occ_symbol"], "SMCI260814C00034000")
        self.assertEqual(pending["contracts"], 1)
        self.assertEqual(pending["limit_price"], 1.55)   # item 1's ask in _menu()
        self.assertIn("Confirm", mock_tg.call_args[0][0])
        self.assertIn("asking price", mock_tg.call_args[0][0])

    def test_buy_command_with_explicit_qty_and_price(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1", "3", "1.50"])
        with open(self._buy_tmp.name) as f:
            pending = json.load(f)
        self.assertEqual(pending["contracts"], 3)
        self.assertEqual(pending["limit_price"], 1.50)
        self.assertAlmostEqual(pending["total_cost"], 3 * 1.50 * 100)
        msg = mock_tg.call_args[0][0]
        self.assertIn("Buy 3 calls", msg)
        self.assertIn("your limit price of $1.50", msg)

    def test_buy_command_qty_capped_at_max_contracts(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "send_telegram", return_value=True):
            a._handle_buy_command(["/buy", "1", "999", "0.10"])
        with open(self._buy_tmp.name) as f:
            pending = json.load(f)
        self.assertEqual(pending["contracts"], a.MANUAL_BUY_MAX_CONTRACTS)

    def test_buy_command_fat_finger_price_rejected(self):
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            # item 1's ask is 1.55 -- "15.00" is a plausible "meant 1.50, typed
            # 15.00" fat-finger, well past the 2x-ask sanity ceiling.
            a._handle_buy_command(["/buy", "1", "1", "15.00"])
        self.assertIn("looks like a typo", mock_tg.call_args[0][0])
        self.assertEqual(os.path.getsize(self._buy_tmp.name), 0,
                          "a rejected fat-finger price must not stage a pending confirmation")

    def test_buy_command_low_price_below_ask_is_allowed(self):
        # A passive limit below the ask is a completely normal, legitimate
        # choice (may or may not fill) -- must never be blocked the way an
        # absurdly HIGH price is.
        with open(self._menu_tmp.name, "w") as f:
            json.dump(self._menu(), f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1", "1", "0.90"])
        with open(self._buy_tmp.name) as f:
            pending = json.load(f)
        self.assertEqual(pending["limit_price"], 0.90)
        self.assertIn("Confirm", mock_tg.call_args[0][0])

    def test_buy_command_over_per_trade_cap_rejected(self):
        menu = self._menu()
        menu["items"][0]["bid"] = 200.0
        menu["items"][0]["ask"] = 201.0
        with open(self._menu_tmp.name, "w") as f:
            json.dump(menu, f)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1", "3"])   # 3 contracts @ $201 ask = $60,300
        self.assertIn("per-trade cap", mock_tg.call_args[0][0])
        self.assertEqual(os.path.getsize(self._buy_tmp.name), 0,
                          "a rejected /buy must not stage a pending confirmation")

    # ── YES/NO reply ─────────────────────────────────────────────────
    def _pending(self, expires_in_min=5):
        now = datetime.now(a.ET)
        return {
            "ticker": "SMCI", "occ_symbol": "SMCI260814C00034000", "option_type": "CALL",
            "strike": 34.0, "expiry": "2026-08-14", "contracts": 3, "ask_at_confirm": 1.55,
            "limit_price": 1.55, "total_cost": 465.0,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=expires_in_min)).isoformat(),
        }

    def test_non_yes_no_text_returns_false(self):
        self.assertFalse(a._handle_manual_options_buy_reply("hello there"))

    def test_yes_with_nothing_pending_returns_false(self):
        self.assertFalse(a._handle_manual_options_buy_reply("YES"))

    def test_no_cancels_without_ordering(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        with patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            handled = a._handle_manual_options_buy_reply("NO")
        self.assertTrue(handled)
        mock_gac.assert_not_called()
        self.assertIn("cancelled", mock_tg.call_args[0][0])

    def test_expired_confirmation_declines_without_ordering(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(expires_in_min=-5), f)
        with patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("expired", mock_tg.call_args[0][0])

    def test_halted_blocks_order(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        with patch.object(a, "is_halted", return_value=True), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("halted", mock_tg.call_args[0][0])

    def test_macro_blackout_blocks_order(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        with patch.object(a, "check_macro_safe", return_value=(False, 0)), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("blackout", mock_tg.call_args[0][0])

    def test_yes_submits_and_registers_position(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        snap = {"bid": 1.52, "ask": 1.56, "delta": 0.51}
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="order-abc-123")
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "_get_option_snapshot", return_value=snap), \
             patch.object(a, "_cash_available_for", return_value=(True, "")), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_client.submit_order.assert_called_once()
        positions = a.PositionTracker().positions
        self.assertEqual(len(positions), 1)
        self.assertTrue(positions[0].setup.startswith("Options Call SMCI260814C00034000"))
        self.assertIn("Submitted", mock_tg.call_args[0][0])


class TestSubmitManualOptionsBuy(unittest.TestCase):
    """_submit_manual_options_buy() is the actual order-placement half of
    the /options -> /buy -> YES flow. Isolated from the Telegram-handler
    tests above so the staleness/cash/MAX_POSITIONS guards can be tested
    directly against its (order_id, error) return contract."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._pos_tmp.write("[]")
        self._pos_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patch = patch.object(a, "PositionTracker", self._isolated_pt)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._pos_tmp.name)

    def _pending(self):
        return {
            "ticker": "SMCI", "occ_symbol": "SMCI260814C00034000", "option_type": "CALL",
            "strike": 34.0, "expiry": "2026-08-14", "contracts": 3, "ask_at_confirm": 1.55,
            "limit_price": 1.55,
        }

    def test_price_drift_past_limit_aborts(self):
        mock_client = MagicMock()
        with patch.object(a, "_get_option_snapshot", return_value={"bid": 2.0, "ask": 2.05, "delta": 0.5}):
            order_id, err = a._submit_manual_options_buy(mock_client, self._pending())
        self.assertIsNone(order_id)
        self.assertIn("moved", err)
        mock_client.submit_order.assert_not_called()

    def test_no_live_quote_aborts(self):
        mock_client = MagicMock()
        with patch.object(a, "_get_option_snapshot", return_value=None):
            order_id, err = a._submit_manual_options_buy(mock_client, self._pending())
        self.assertIsNone(order_id)
        mock_client.submit_order.assert_not_called()

    def test_insufficient_cash_aborts(self):
        mock_client = MagicMock()
        with patch.object(a, "_get_option_snapshot", return_value={"bid": 1.52, "ask": 1.56, "delta": 0.5}), \
             patch.object(a, "_cash_available_for", return_value=(False, "insufficient cash")):
            order_id, err = a._submit_manual_options_buy(mock_client, self._pending())
        self.assertIsNone(order_id)
        self.assertIn("insufficient cash", err)
        mock_client.submit_order.assert_not_called()

    def test_max_positions_cancels_the_just_placed_order(self):
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="order-xyz")
        # Fill the tracker to MAX_POSITIONS before submitting.
        pt = a.PositionTracker(filepath=self._pos_tmp.name)
        for i in range(a.MAX_POSITIONS):
            pt.open(a.OpenPosition(
                ticker=f"T{i}", bias="LONG", setup="Low Float Catalyst",
                entry=1.0, stop=0.9, target1=1.1, target2=1.2,
                shares=10, entry_date="2026-08-01",
            ))
        with patch.object(a, "_get_option_snapshot", return_value={"bid": 1.52, "ask": 1.56, "delta": 0.5}), \
             patch.object(a, "_cash_available_for", return_value=(True, "")):
            order_id, err = a._submit_manual_options_buy(mock_client, self._pending())
        self.assertIsNone(order_id)
        self.assertIn("MAX_POSITIONS", err)
        mock_client.cancel_order_by_id.assert_called_once_with("order-xyz")

    def test_successful_submission_uses_the_users_chosen_price_not_live_ask(self):
        # Confirmed live 2026-08-12 (direct user feedback): the order used
        # to always submit at live_ask*1.03 regardless of what price the
        # user confirmed in /buy -- defeating the whole point of letting
        # them specify their own limit. The live ask here (1.56) is
        # deliberately different from the pending confirmation's chosen
        # price (1.55) to prove the submitted order uses the latter.
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="order-xyz")
        with patch.object(a, "_get_option_snapshot", return_value={"bid": 1.52, "ask": 1.56, "delta": 0.5}), \
             patch.object(a, "_cash_available_for", return_value=(True, "")):
            order_id, err = a._submit_manual_options_buy(mock_client, self._pending())
        self.assertEqual(order_id, "order-xyz")
        self.assertIsNone(err)
        _, kwargs = mock_client.submit_order.call_args
        submitted_order = mock_client.submit_order.call_args[0][0]
        self.assertEqual(submitted_order.limit_price, 1.55)
        pos = a.PositionTracker(filepath=self._pos_tmp.name).positions[0]
        self.assertAlmostEqual(pos.entry, 1.55)
        self.assertAlmostEqual(pos.target1, round(1.55 * 1.5, 2))
        self.assertAlmostEqual(pos.target2, round(1.55 * 2.5, 2))
        self.assertAlmostEqual(pos.stop, round(1.55 * 0.5, 2))
        self.assertEqual(pos.shares, 300)   # 3 contracts * 100


class TestGithubHealthDiagnosis(unittest.TestCase):
    """_diagnose_github_health() drives run_fallback_guard() -- the whole
    point is running the scan locally the moment GitHub Actions can't, so
    it needs to reliably tell a genuine GitHub platform outage apart from
    a normal healthy state (must not trigger constantly) and from a real
    code bug (which fails fast, not at GitHub's ~15-min runner-queue
    timeout). Confirmed live 2026-08-06: 3 different workflows with 3
    different configured timeout-minutes all failed at the same ~15 min
    mark during the actual outage -- that's the platform-level signal
    this locks in."""

    def _run(self, minutes_ago, status="completed", conclusion="success",
              duration_min=None):
        now = datetime.now(a.ET)
        created = now - timedelta(minutes=minutes_ago)
        updated = created + timedelta(minutes=duration_min if duration_min is not None else 1)
        return {
            "created_at": created.isoformat().replace("+00:00", "Z") if created.tzinfo else created.isoformat() + "Z",
            "updated_at": updated.isoformat().replace("+00:00", "Z") if updated.tzinfo else updated.isoformat() + "Z",
            "status": status,
            "conclusion": conclusion,
        }

    def test_no_runs_at_all_is_unhealthy(self):
        unhealthy, reason = a._diagnose_github_health([], datetime.now(a.ET))
        self.assertTrue(unhealthy)

    def test_recent_successful_run_is_healthy(self):
        run = self._run(minutes_ago=10, status="completed", conclusion="success")
        unhealthy, _ = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertFalse(unhealthy)

    def test_15min_failure_matches_runner_queue_timeout_signature(self):
        run = self._run(minutes_ago=15, status="completed", conclusion="failure", duration_min=15)
        unhealthy, reason = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertTrue(unhealthy)
        self.assertIn("runner-queue timeout", reason)

    def test_fast_failure_is_not_treated_as_platform_outage(self):
        # A genuine code bug fails in well under a minute -- must not be
        # mistaken for GitHub's runner-queue timeout, or the fallback
        # would just re-run the same broken code locally forever.
        run = self._run(minutes_ago=2, status="completed", conclusion="failure", duration_min=0.5)
        unhealthy, _ = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertFalse(unhealthy)

    def test_stuck_queued_past_20min_is_unhealthy(self):
        run = self._run(minutes_ago=25, status="queued", conclusion=None)
        unhealthy, reason = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertTrue(unhealthy)
        self.assertIn("queued", reason)

    def test_recently_queued_is_not_yet_unhealthy(self):
        run = self._run(minutes_ago=5, status="queued", conclusion=None)
        unhealthy, _ = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertFalse(unhealthy)

    def test_no_run_in_90_min_window_is_unhealthy(self):
        run = self._run(minutes_ago=120, status="completed", conclusion="success")
        unhealthy, reason = a._diagnose_github_health([run], datetime.now(a.ET))
        self.assertTrue(unhealthy)
        self.assertIn("no scan run started", reason)


class TestFallbackGuardWiring(unittest.TestCase):
    """run_fallback_guard() is the actual entry point Windows Task Scheduler
    calls -- these lock in the outer plumbing (market-hours gate, and that
    an unhealthy diagnosis actually triggers a local scan + alert) since
    _diagnose_github_health() being correct is worthless if nothing calls it
    at the right time or acts on its answer."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self._patch = patch.object(a, "LAST_ALERTS_FILE", self._tmp.name)
        self._patch.start()
        self._token_patch = patch.object(a, "GITHUB_TOKEN", "fake-token")
        self._token_patch.start()

    def tearDown(self):
        self._patch.stop()
        self._token_patch.stop()
        os.unlink(self._tmp.name)

    def _weekday_market_hours(self):
        # A fixed Tuesday at 11:00 AM ET -- inside market hours, a weekday.
        return datetime(2026, 8, 4, 11, 0, tzinfo=a.ET)

    def test_skips_outside_market_hours(self):
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 4, 20, 0, tzinfo=a.ET)  # 8 PM ET
            with patch.object(a, "requests") as mock_requests:
                a.run_fallback_guard()
        mock_requests.get.assert_not_called()

    def test_unhealthy_github_triggers_local_scan_and_alert(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"workflow_runs": []}   # no runs = unhealthy
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = self._weekday_market_hours()
            mock_dt.fromisoformat = datetime.fromisoformat
            with patch.object(a, "requests") as mock_requests:
                mock_requests.get.return_value = mock_resp
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0)
                        a.run_fallback_guard()
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        self.assertIn("--submit", called_args)
        mock_tg.assert_called_once()
        self.assertIn("Local fallback activated", mock_tg.call_args[0][0])

    def test_healthy_github_does_not_trigger_local_scan(self):
        mock_resp = MagicMock()
        recent = self._weekday_market_hours() - timedelta(minutes=5)
        mock_resp.json.return_value = {"workflow_runs": [{
            "created_at": recent.isoformat(), "updated_at": recent.isoformat(),
            "status": "completed", "conclusion": "success",
        }]}
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = self._weekday_market_hours()
            mock_dt.fromisoformat = datetime.fromisoformat
            with patch.object(a, "requests") as mock_requests:
                mock_requests.get.return_value = mock_resp
                with patch("subprocess.run") as mock_run:
                    a.run_fallback_guard()
        mock_run.assert_not_called()


class TestSmallcapPullbackToleranceScaling(unittest.TestCase):
    """Confirmed live 2026-08-06: CLRO gapped +217% intraday, CELZ +70%, both
    far beyond the ~15-50% range the flat 12% pullback tolerance was
    originally calibrated against. A regression here means going back to
    the same cushion for a 20% gap and a 200%+ gap, when the latter gives
    back ground far faster and harder."""

    def test_normal_gap_uses_baseline_tolerance(self):
        self.assertEqual(a._smallcap_pullback_tolerance_pct(25.0),
                          a.SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT)

    def test_extreme_gap_gets_tightest_tolerance(self):
        self.assertEqual(a._smallcap_pullback_tolerance_pct(217.0), 6.0)

    def test_moderate_extreme_gap_gets_middle_tolerance(self):
        self.assertEqual(a._smallcap_pullback_tolerance_pct(100.0), 9.0)

    def test_celz_style_70pct_gap_still_uses_baseline(self):
        # CELZ's real gap (70%) is below the 75% moderate-extreme cutoff --
        # its actual 2.6% pullback stayed well clear either way, but this
        # locks in that a 70% gap doesn't get needlessly tightened.
        self.assertEqual(a._smallcap_pullback_tolerance_pct(70.0),
                          a.SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT)


class TestIntradayHighPullbackGuard(unittest.TestCase):
    """Confirmed live 2026-08-06: CLRO scored a qualifying Low Float Catalyst
    setup on RVOL/float/MACD/RSI while actually 14.4% off its own intraday
    high of 22 minutes earlier and still falling -- every other gate is
    computed from data that's blind to the shape of TODAY specifically, so
    nothing caught buying into an active knife-catch. A regression here
    means going back to chasing spikes on the way down."""

    def _df(self, close, high, n=6):
        import pandas as pd
        # n rows: enough for the volume-averaging / 52wk-low lookback logic
        # to run without special-casing a too-short frame.
        rows = []
        for i in range(n - 1):
            rows.append({"Close": 8.0, "High": 8.2, "Low": 7.8, "Open": 7.9,
                         "RVOL": 1.0, "MACD": 0.1, "MACD_sig": 0.2,
                         "MACD_hist": -0.05, "RSI": 45.0, "Volume": 500_000})
        rows.append({"Close": close, "High": high, "Low": min(close, high) * 0.95,
                     "Open": high * 0.98, "RVOL": 6.0, "MACD": 1.0, "MACD_sig": 0.5,
                     "MACD_hist": 0.3, "RSI": 40.0, "Volume": 3_000_000})
        return pd.DataFrame(rows)

    def setUp(self):
        self._patches = [
            patch.object(a, "ENABLE_SMALLCAP", True),
            patch.object(a, "_get_short_float_data", return_value=(1.0, 5.0, 5.0, 0.1)),
            patch.object(a, "_is_recent_reverse_split", return_value=False),
            patch.object(a, "get_effective_account", return_value=5000.0),
            # Catalyst-confirmation gate added 2026-08-12 makes a real network
            # call otherwise -- these tests exercise the pullback guard, not
            # the news gate, and must not depend on live API state (a real
            # ticker like CELZ happening to have real news right now vs. a
            # fake one like TESTX not is exactly the kind of nondeterminism
            # a unit test must never have).
            patch.object(a, "_fetch_massive_benzinga_news",
                        side_effect=lambda tickers, **kw: {t: ["headline"] for t in tickers}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_clro_style_knife_catch_is_blocked(self):
        # High set 22 minutes before entry at 11.45, current price 9.80 —
        # the exact real numbers from the live incident: 14.4% off high.
        df = self._df(close=9.80, high=11.45)
        sig = a.detect_low_float_catalyst(df, "CLRO")
        self.assertIsNone(sig)

    def test_celz_style_small_pullback_still_passes(self):
        # Real numbers from the same day's CELZ entry: 2.6% off high,
        # comfortably under the threshold — must not be blocked.
        df = self._df(close=1.1398, high=1.17)
        sig = a.detect_low_float_catalyst(df, "CELZ")
        self.assertIsNotNone(sig)

    def test_pullback_exactly_at_threshold_is_not_blocked(self):
        # 12.0% pullback exactly — the guard uses a strict ">" so the
        # boundary itself is allowed, only pullbacks WORSE than the
        # threshold are skipped.
        high = 10.0
        close = high * (1 - a.SMALLCAP_MAX_PULLBACK_FROM_HIGH_PCT / 100)
        df = self._df(close=close, high=high)
        sig = a.detect_low_float_catalyst(df, "TESTX")
        self.assertIsNotNone(sig)


class TestLowFloatCatalystNewsGate(unittest.TestCase):
    """Confirmed live 2026-08-12: FGL cleared every technical gate in
    detect_low_float_catalyst() (float, RVOL -- a 28x float rotation that
    day --, MACD, RSI, 52wk-low proximity) with no findable news or filing
    behind it. Direct instruction after that trade: require a real,
    confirmed catalyst before committing capital to a low-float spike,
    not just trust volume as a proxy for one. These tests lock in that
    a non-watchlist ticker with no news is blocked, one with news passes,
    and DMan's own curated watchlist is exempt (his call already IS the
    catalyst -- same reasoning as its separately-lower RVOL bar)."""

    def _df(self, close=8.5, high=8.6, n=6):
        import pandas as pd
        rows = []
        for i in range(n - 1):
            rows.append({"Close": 8.0, "High": 8.2, "Low": 7.8, "Open": 7.9,
                         "RVOL": 1.0, "MACD": 0.1, "MACD_sig": 0.2,
                         "MACD_hist": -0.05, "RSI": 45.0, "Volume": 500_000})
        rows.append({"Close": close, "High": high, "Low": min(close, high) * 0.95,
                     "Open": high * 0.98, "RVOL": 6.0, "MACD": 1.0, "MACD_sig": 0.5,
                     "MACD_hist": 0.3, "RSI": 40.0, "Volume": 3_000_000})
        return pd.DataFrame(rows)

    def setUp(self):
        self._patches = [
            patch.object(a, "ENABLE_SMALLCAP", True),
            patch.object(a, "_get_short_float_data", return_value=(1.0, 5.0, 5.0, 0.1)),
            patch.object(a, "_is_recent_reverse_split", return_value=False),
            patch.object(a, "get_effective_account", return_value=5000.0),
            patch.object(a, "DMAN_SMALLCAP_WATCHLIST", {"DMANPICK"}),
            # Default to a neutral sentiment verdict so existing tests that
            # only care about the news-existence gate aren't incidentally
            # blocked by the separate sentiment check added 2026-08-13.
            patch.object(a, "_news_sentiment_verdict", return_value="neutral"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_no_news_blocks_non_watchlist_ticker(self):
        with patch.object(a, "_fetch_massive_benzinga_news", return_value={"FGL": []}):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNone(sig)

    def test_confirmed_news_allows_non_watchlist_ticker(self):
        with patch.object(a, "_fetch_massive_benzinga_news",
                          return_value={"FGL": ["Company announces XYZ"]}):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNotNone(sig)

    def test_dman_watchlist_ticker_exempt_from_news_check(self):
        with patch.object(a, "_fetch_massive_benzinga_news") as mock_news, \
             patch.object(a, "_news_sentiment_verdict") as mock_sentiment:
            sig = a.detect_low_float_catalyst(self._df(), "DMANPICK")
        mock_news.assert_not_called()
        mock_sentiment.assert_not_called()
        self.assertIsNotNone(sig)

    def test_negative_sentiment_blocks_despite_confirmed_news(self):
        # Confirmed live 2026-08-13: Massive's /v2/reference/news provides
        # real per-article sentiment even though the analyst-ratings tier
        # is 403'd. "There is news" alone doesn't mean it supports a LONG
        # entry -- a volume spike whose only real news is negative is a
        # sell-off, not a catalyst, on a long-only system.
        with patch.object(a, "_fetch_massive_benzinga_news",
                          return_value={"FGL": ["Company misses guidance"]}), \
             patch.object(a, "_news_sentiment_verdict", return_value="negative"):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNone(sig)

    def test_positive_sentiment_allows_entry(self):
        with patch.object(a, "_fetch_massive_benzinga_news",
                          return_value={"FGL": ["Company announces contract win"]}), \
             patch.object(a, "_news_sentiment_verdict", return_value="positive"):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNotNone(sig)

    def test_unknown_sentiment_does_not_override_confirmed_news(self):
        # No sentiment data found (None) must NOT block -- the primary
        # news-existence check already confirmed a real catalyst; this is
        # an additional refinement on top, not a replacement gate.
        with patch.object(a, "_fetch_massive_benzinga_news",
                          return_value={"FGL": ["Company announces XYZ"]}), \
             patch.object(a, "_news_sentiment_verdict", return_value=None):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNotNone(sig)

    def test_news_api_failure_fails_closed(self):
        # Fail-closed: an API error (empty dict from the existing
        # _fetch_massive_benzinga_news failure handling) must block entry,
        # not silently let an unverified spike through.
        with patch.object(a, "_fetch_massive_benzinga_news", return_value={}):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNone(sig)


class TestSendSignalAlertBatch(unittest.TestCase):
    """Confirmed live 2026-08-13 API audit: send_telegram() had zero
    batching logic -- a scan producing several signals fired that many
    separate Telegram notifications. _send_signal_alert_batch() combines
    them into one digest message, splitting only if the combined length
    would exceed Telegram's practical limit."""

    def test_empty_list_sends_nothing(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._send_signal_alert_batch([])
        mock_tg.assert_not_called()

    def test_single_message_sent_unchanged_no_header(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._send_signal_alert_batch(["one signal's full alert text"])
        mock_tg.assert_called_once_with("one signal's full alert text")

    def test_multiple_messages_combined_into_one_send(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._send_signal_alert_batch(["alert A", "alert B", "alert C"])
        mock_tg.assert_called_once()
        combined = mock_tg.call_args[0][0]
        for piece in ("alert A", "alert B", "alert C"):
            self.assertIn(piece, combined)
        self.assertIn("3", combined)   # "3 new plays found" header

    def test_oversized_batch_splits_into_multiple_sends(self):
        huge_messages = ["x" * 3000 for _ in range(3)]   # 3 * 3000 > one safe chunk
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._send_signal_alert_batch(huge_messages)
        self.assertGreater(mock_tg.call_count, 1)
        # every original message must still appear somewhere across the sends
        all_sent = "".join(c[0][0] for c in mock_tg.call_args_list)
        self.assertEqual(all_sent.count("x" * 3000), 3)


class TestRegisterTelegramCommands(unittest.TestCase):
    """Confirmed live 2026-08-13 API audit: only sendMessage and getUpdates
    were ever used anywhere in this file -- setMyCommands was never called,
    so /options, /buy, etc. never showed up in Telegram's command
    autocomplete. TELEGRAM_COMMANDS is the single source of truth both this
    function and the unknown-command help text read from, so they can't
    silently drift apart."""

    def test_sends_every_command_with_its_description(self):
        with patch.object(a, "TELEGRAM_TOKEN", "test-token"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=200,
                                                         json=lambda: {"ok": True})
            result = a._register_telegram_commands()
        self.assertTrue(result)
        sent = mock_requests.post.call_args[1]["json"]["commands"]
        sent_names = {c["command"] for c in sent}
        expected_names = {c for c, _ in a.TELEGRAM_COMMANDS}
        self.assertEqual(sent_names, expected_names)

    def test_no_token_returns_false_without_a_call(self):
        with patch.object(a, "TELEGRAM_TOKEN", ""), \
             patch.object(a, "requests") as mock_requests:
            result = a._register_telegram_commands()
        self.assertFalse(result)
        mock_requests.post.assert_not_called()

    def test_http_failure_fails_open_returns_false_never_raises(self):
        with patch.object(a, "TELEGRAM_TOKEN", "test-token"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.side_effect = Exception("network error")
            result = a._register_telegram_commands()
        self.assertFalse(result)

    def test_help_text_lists_every_registered_command(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/unknowncommand")
        msg = mock_tg.call_args[0][0]
        for cmd, _ in a.TELEGRAM_COMMANDS:
            self.assertIn(f"/{cmd}", msg)


class TestIsMarketOpenHolidayAware(unittest.TestCase):
    """Confirmed live 2026-08-13 audit: is_market_open() was a pure
    weekday+time check with zero holiday awareness, despite _MARKET_HOLIDAYS
    already existing in this file for other purposes. The caller that
    matters most is _submit_signals_to_alpaca()'s belt-and-suspenders gate
    right before real order submission -- specifically for a manual/off-
    schedule invocation (--mode alpaca, workflow_dispatch) that bypasses
    whatever normally keeps the cron schedule from running on a holiday."""

    def test_thanksgiving_during_normal_hours_is_closed(self):
        # Thanksgiving 2026-11-26 is a Thursday -- a weekday+time-only
        # check would incorrectly report the market open.
        fake_now = datetime(2026, 11, 26, 11, 0, tzinfo=a.ET)
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            self.assertFalse(a.is_market_open())

    def test_normal_weekday_during_hours_is_still_open(self):
        # A regular non-holiday Wednesday must be unaffected by the fix.
        fake_now = datetime(2026, 8, 12, 11, 0, tzinfo=a.ET)
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            self.assertTrue(a.is_market_open())


class TestDayStartEquityBaseline(unittest.TestCase):
    """Alpaca's last_equity is the prior TRADING DAY's close -- it doesn't
    account for a same-day deposit, so (equity - last_equity) treats new
    capital as trading profit. Confirmed live 2026-08-06: a $2,000 deposit
    inflated the EOD Telegram alert to "+66.26%" for what was actually a
    real loss day. A regression here means that false-profit alert again."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "_DAY_START_EQUITY_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_first_call_of_the_day_seeds_current_equity_as_baseline(self):
        baseline = a._get_day_start_equity(5000.0)
        self.assertEqual(baseline, 5000.0)

    def test_deposit_after_baseline_is_seeded_does_not_inflate_pnl(self):
        # First read of the day (e.g. 2 AM, deposit already landed) seeds
        # the baseline. A later read the same day, after real trading moved
        # equity, must compare against that SAME baseline, not re-seed.
        a._get_day_start_equity(4943.73)
        baseline = a._get_day_start_equity(4895.85)
        self.assertEqual(baseline, 4943.73)
        day_pl_pct = (4895.85 - baseline) / baseline * 100
        self.assertAlmostEqual(day_pl_pct, -0.97, places=1)

    def test_new_calendar_day_reseeds_the_baseline(self):
        with open(self._tmp.name, "w") as f:
            json.dump({"date": "2020-01-01", "equity": 1000.0}, f)
        baseline = a._get_day_start_equity(5000.0)
        self.assertEqual(baseline, 5000.0)


class TestOptionsFeedResolution(unittest.TestCase):
    """OPRA entitlement on this account has flipped on/off before with zero
    code change on our end -- confirmed working 2026-07-30, confirmed 403
    again 2026-08-06, and the hardcoded OPTIONS_DATA_FEED constant meant a
    full week where every single options signal silently failed
    (_get_option_snapshot returning None for every contract) and fell
    through to skip/equity, discovered only by manually testing the raw
    endpoint. A regression here means going back to trusting a feed that
    may not actually be entitled, with no visibility when it isn't."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)   # start absent -- matches a fresh checkout with no prior state
        self._patch = patch.object(a, "_OPTIONS_FEED_STATE_FILE", self._tmp.name)
        self._patch.start()
        a._options_feed_state = {"feed": None, "checked_at": 0.0}

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)
        a._options_feed_state = {"feed": None, "checked_at": 0.0}

    def _mock_contract(self):
        contracts = MagicMock()
        contracts.option_contracts = [MagicMock(symbol="AAPL260821C00310000")]
        return contracts

    def test_403_falls_back_to_indicative_and_alerts(self):
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    feed = a._resolve_options_feed()
        self.assertEqual(feed, "indicative")
        mock_tg.assert_called_once()
        self.assertIn("not entitled", mock_tg.call_args[0][0])

    def test_200_keeps_preferred_feed_no_alert(self):
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=200)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    feed = a._resolve_options_feed()
        self.assertEqual(feed, a.OPTIONS_DATA_FEED)
        mock_tg.assert_not_called()

    def test_repeat_resolution_within_window_does_not_reprobe(self):
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp) as mock_get:
                with patch.object(a, "send_telegram", return_value=True):
                    a._resolve_options_feed()
                    a._resolve_options_feed()
        mock_get.assert_called_once()

    def test_same_state_on_recheck_does_not_realert(self):
        # First resolution alerts (state change None -> indicative). Age the
        # PERSISTED checked_at past the recheck window (simulating real
        # elapsed time, not just an in-memory reset) and force a reload --
        # the resulting real re-probe must not alert twice for the same
        # known-broken entitlement.
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._resolve_options_feed()
                    with open(self._tmp.name) as f:
                        state = json.load(f)
                    state["checked_at"] -= (a._OPTIONS_FEED_RECHECK_S + 1)
                    with open(self._tmp.name, "w") as f:
                        json.dump(state, f)
                    a._options_feed_state["checked_at"] = 0.0   # allow a reload from disk
                    a._resolve_options_feed()
        self.assertEqual(mock_tg.call_count, 1)

    def test_fresh_process_after_alert_does_not_realert(self):
        # The actual bug reported live 2026-08-08: GitHub Actions starts a
        # brand-new process for every scan/daemon-session run, so the
        # in-memory _options_feed_state used to reset to defaults every
        # time -- making every single run's first probe look like a fresh
        # None -> indicative transition and re-send the same Telegram
        # alert, repeating every ~10 min across scan/momentum-watch runs.
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._resolve_options_feed()
                    # Simulate a brand-new process: reset in-memory state
                    # exactly like a fresh module import would, WITHOUT
                    # touching the persisted file on disk.
                    a._options_feed_state = {"feed": None, "checked_at": 0.0}
                    a._resolve_options_feed()
        self.assertEqual(mock_tg.call_count, 1, "a fresh process must not re-alert "
                          "for an already-known, unchanged entitlement state")

    def test_fresh_process_within_recheck_window_does_not_reprobe(self):
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp) as mock_get:
                with patch.object(a, "send_telegram", return_value=True):
                    a._resolve_options_feed()
                    a._options_feed_state = {"feed": None, "checked_at": 0.0}   # simulate fresh process
                    a._resolve_options_feed()
        mock_get.assert_called_once()


class TestStockFeedResolution(unittest.TestCase):
    """Mirrors TestOptionsFeedResolution for the stock-side SIP entitlement.
    Algo Trader Plus billing voided twice before finally activating
    2026-08-09 -- if it ever lapses again, _fetch_alpaca_daily /
    prewarm_alpaca_bars / _fetch_intraday_bars would otherwise silently
    swallow the resulting 403 and fall through to yfinance with zero
    visibility, the same gap that let OPRA's entitlement flip go unnoticed
    for up to a week before _resolve_options_feed() existed. A regression
    here means going back to trusting SIP is entitled with no way to know
    if it stops being true."""

    def setUp(self):
        self._key_patch    = patch.object(a, "ALPACA_API_KEY", "test-key")
        self._secret_patch = patch.object(a, "ALPACA_SECRET_KEY", "test-secret")
        self._key_patch.start()
        self._secret_patch.start()

        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)   # start absent -- matches a fresh checkout with no prior state
        self._patch = patch.object(a, "_STOCK_FEED_STATE_FILE", self._tmp.name)
        self._patch.start()
        a._stock_feed_state = {"feed": None, "checked_at": 0.0}

    def tearDown(self):
        self._patch.stop()
        self._key_patch.stop()
        self._secret_patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)
        a._stock_feed_state = {"feed": None, "checked_at": 0.0}

    def test_403_falls_back_to_iex_and_alerts(self):
        mock_resp = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_resp):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                feed = a._resolve_stock_feed()
        self.assertEqual(feed, "iex")
        mock_tg.assert_called_once()
        self.assertIn("not entitled", mock_tg.call_args[0][0])

    def test_200_keeps_preferred_feed_no_alert(self):
        mock_resp = MagicMock(status_code=200)
        with patch.object(a.requests, "get", return_value=mock_resp):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                feed = a._resolve_stock_feed()
        self.assertEqual(feed, a.STOCK_DATA_FEED)
        mock_tg.assert_not_called()

    def test_repeat_resolution_within_window_does_not_reprobe(self):
        mock_resp = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_resp) as mock_get:
            with patch.object(a, "send_telegram", return_value=True):
                a._resolve_stock_feed()
                a._resolve_stock_feed()
        mock_get.assert_called_once()

    def test_same_state_on_recheck_does_not_realert(self):
        mock_resp = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_resp):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._resolve_stock_feed()
                with open(self._tmp.name) as f:
                    state = json.load(f)
                state["checked_at"] -= (a._STOCK_FEED_RECHECK_S + 1)
                with open(self._tmp.name, "w") as f:
                    json.dump(state, f)
                a._stock_feed_state["checked_at"] = 0.0   # allow a reload from disk
                a._resolve_stock_feed()
        self.assertEqual(mock_tg.call_count, 1)

    def test_fresh_process_after_alert_does_not_realert(self):
        mock_resp = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_resp):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._resolve_stock_feed()
                # Simulate a brand-new process: reset in-memory state exactly
                # like a fresh module import would, WITHOUT touching the
                # persisted file on disk.
                a._stock_feed_state = {"feed": None, "checked_at": 0.0}
                a._resolve_stock_feed()
        self.assertEqual(mock_tg.call_count, 1, "a fresh process must not re-alert "
                          "for an already-known, unchanged entitlement state")

    def test_fresh_process_within_recheck_window_does_not_reprobe(self):
        mock_resp = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_resp) as mock_get:
            with patch.object(a, "send_telegram", return_value=True):
                a._resolve_stock_feed()
                a._stock_feed_state = {"feed": None, "checked_at": 0.0}
                a._resolve_stock_feed()
        mock_get.assert_called_once()

    def test_restoration_after_fallback_alerts_success(self):
        mock_403 = MagicMock(status_code=403)
        mock_200 = MagicMock(status_code=200)
        with patch.object(a.requests, "get", return_value=mock_403):
            with patch.object(a, "send_telegram", return_value=True):
                a._resolve_stock_feed()
        with open(self._tmp.name) as f:
            state = json.load(f)
        state["checked_at"] -= (a._STOCK_FEED_RECHECK_S + 1)
        with open(self._tmp.name, "w") as f:
            json.dump(state, f)
        a._stock_feed_state["checked_at"] = 0.0
        with patch.object(a.requests, "get", return_value=mock_200):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                feed = a._resolve_stock_feed()
        self.assertEqual(feed, a.STOCK_DATA_FEED)
        mock_tg.assert_called_once()
        self.assertIn("restored", mock_tg.call_args[0][0])

    def test_missing_credentials_returns_preferred_feed_without_a_call(self):
        with patch.object(a, "ALPACA_API_KEY", ""):
            with patch.object(a.requests, "get") as mock_get:
                feed = a._resolve_stock_feed()
        self.assertEqual(feed, a.STOCK_DATA_FEED)
        mock_get.assert_not_called()


class TestMacroCalendarProximity(unittest.TestCase):
    """_days_to_next_macro_print() feeds a real, current-condition sizing
    adjustment in _fetch_global_context() — added 2026-08-06 because
    check_macro_safe()'s hard gate only blocks entries before 10 AM ET on
    the actual NFP/CPI/PPI release day, leaving the day-before (real
    pre-print positioning risk) and the release-day afternoon with zero
    extra caution. A regression here means the account sizes up normally
    heading into a known high-volatility print instead of derating."""

    def setUp(self):
        self._nfp_patch = patch.object(a, "_nfp_dates", return_value={date(2026, 8, 7)})
        self._nfp_patch.start()
        self._cpi_patch = patch.object(a, "_CPI_DATES", {date(2026, 8, 12)})
        self._cpi_patch.start()
        self._ppi_patch = patch.object(a, "_PPI_DATES", set())
        self._ppi_patch.start()

    def tearDown(self):
        self._nfp_patch.stop()
        self._cpi_patch.stop()
        self._ppi_patch.stop()

    def test_day_before_nfp_returns_one(self):
        self.assertEqual(a._days_to_next_macro_print(date(2026, 8, 6)), 1)

    def test_nfp_day_itself_returns_zero(self):
        self.assertEqual(a._days_to_next_macro_print(date(2026, 8, 7)), 0)

    def test_two_days_out_returns_none(self):
        self.assertIsNone(a._days_to_next_macro_print(date(2026, 8, 5)))

    def test_unrelated_date_returns_none(self):
        self.assertIsNone(a._days_to_next_macro_print(date(2026, 8, 10)))


class TestEarningsSpreadSizing(unittest.TestCase):
    """5% of a $2,997.77 account is ~$150 — far below a realistic large-cap
    debit spread's cost. Locks in that a too-expensive-even-at-minimum-width
    spread is skipped explicitly (with a reason), never silently forced into
    a degenerate 0-contract/0-cost plan."""

    def test_skips_when_even_min_width_exceeds_budget_slack(self):
        with patch.object(a, "get_effective_account", return_value=1000.0):
            with patch.object(a, "_last_n_earnings_moves", return_value=[]):
                with patch.object(a, "_find_spread_legs", return_value={
                        "long_occ": "X1", "short_occ": "X2", "long_strike": 100,
                        "short_strike": 110, "net_debit": 50.0,  # $5000/contract — absurdly expensive
                        "expiry": "2026-08-07", "dte": 9, "long_oi": 0, "short_oi": 0}):
                    plan = a.build_earnings_spread_plan(
                        MagicMock(), "TESTX", 500.0, a.date.today(), "AMC")
        self.assertIsNone(plan)


class TestBrokerSideStopCoverageCheck(unittest.TestCase):
    """_check_open_position_risk()'s orphan check only asks "do WE know
    about this position" — it never asked whether Alpaca actually has a
    LIVE protective stop working. Confirmed live 2026-08-04: a position's
    stop-limit leg was stuck HELD (its take-profit sibling had claimed all
    shares via Alpaca's held_for_orders accounting, breaking the OCO link)
    and nothing caught it until a manual API query. A regression here means
    a real position can sit with zero downside protection with no alert."""

    def setUp(self):
        from alpaca.trading.enums import AssetClass, OrderType, OrderStatus, PositionIntent
        self.AssetClass  = AssetClass
        self.OrderType   = OrderType
        self.OrderStatus = OrderStatus

        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._pos_patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._pos_patch.start()
        # PositionTracker's filepath is an early-bound default parameter --
        # patching POSITIONS_FILE alone doesn't reach a bare PositionTracker()
        # call (used by both the orphan check and _auto_restore_missing_stop).
        # Patching the class reference forces every construction in the call
        # graph onto the isolated temp file regardless of how it's called.
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._pt_patch = patch.object(a, "PositionTracker", self._isolated_pt)
        self._pt_patch.start()

        self._sig_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._sig_tmp.write(b'{"pending": []}')
        self._sig_tmp.close()
        self._sig_patch = patch.object(a, "LIVE_SIGNALS_FILE", self._sig_tmp.name)
        self._sig_patch.start()

        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

    def tearDown(self):
        self._pos_patch.stop();    os.unlink(self._pos_tmp.name)
        self._pt_patch.stop()
        self._sig_patch.stop();    os.unlink(self._sig_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)

    def _equity_position(self, symbol="W", qty="3"):
        p = MagicMock()
        p.symbol = symbol
        p.asset_class = self.AssetClass.US_EQUITY
        p.qty = qty
        p.avg_entry_price = "116.25"
        p.unrealized_pl = "-2.40"
        p.unrealized_plpc = "-0.02"
        return p

    def _order(self, symbol, order_type, status):
        o = MagicMock()
        o.symbol = symbol
        o.order_type = order_type
        o.status = status
        return o

    def _messages(self, mock_tg):
        return [c[0][0] for c in mock_tg.call_args_list]

    def test_position_with_no_stop_order_at_all_triggers_alert(self):
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("W")]
        mock_client.get_orders.return_value = []   # no orders whatsoever
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})
        msgs = self._messages(mock_tg)
        stop_msgs = [m for m in msgs if "NO LIVE STOP" in m]
        self.assertEqual(len(stop_msgs), 1)
        self.assertIn("W", stop_msgs[0])

    def test_successful_auto_restore_reports_as_restored_not_manual_action(self):
        # End-to-end: W is tracked with a real stop price, so
        # _auto_restore_missing_stop should succeed, and the alert text
        # must reflect that rather than always reading as "manual action
        # needed" even when nothing manual is actually required anymore.
        with open(self._pos_tmp.name, "w") as f:
            json.dump([{"ticker": "W", "bias": "LONG", "setup": "Gap & Hold",
                       "entry": 118.0, "stop": 110.0, "target1": 130.0, "target2": 140.0,
                       "shares": 3, "entry_date": "2026-08-01"}], f)
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("W")]
        # get_orders is called twice: once for the broad stop-coverage scan
        # (empty — that's WHY this is unprotected), once inside
        # _auto_restore_missing_stop for W specifically (also empty — nothing to cancel).
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="restored-order-id")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})
        msgs = self._messages(mock_tg)
        restore_msgs = [m for m in msgs if "STOP AUTO-RESTORED" in m]
        self.assertEqual(len(restore_msgs), 1)
        self.assertIn("✅", restore_msgs[0])
        mock_client.submit_order.assert_called_once()

    def test_held_stop_does_not_count_as_coverage(self):
        # This is the exact failure mode from the live incident: an order
        # exists and is the right type, but its status means it isn't
        # actually working on the exchange.
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("W")]
        mock_client.get_orders.return_value = [
            self._order("W", self.OrderType.STOP_LIMIT, self.OrderStatus.HELD)
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})
        stop_msgs = [m for m in self._messages(mock_tg) if "NO LIVE STOP" in m]
        self.assertEqual(len(stop_msgs), 1)

    def test_live_stop_order_suppresses_the_alert(self):
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("W")]
        mock_client.get_orders.return_value = [
            self._order("W", self.OrderType.STOP_LIMIT, self.OrderStatus.NEW)
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})
        for _call in mock_tg.call_args_list:
            self.assertNotIn("NO LIVE STOP", _call[0][0])

    def test_options_positions_are_excluded(self):
        mock_client = MagicMock()
        opt_pos = self._equity_position("META260807C00650000")
        opt_pos.asset_class = self.AssetClass.US_OPTION
        mock_client.get_all_positions.return_value = [opt_pos]
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})
        for _call in mock_tg.call_args_list:
            self.assertNotIn("NO LIVE STOP", _call[0][0])


class TestAutoRestoreMissingStop(unittest.TestCase):
    """Added 2026-08-15 after the no-live-stop gap hit twice in 48 hours
    (a real position's stop stuck HELD, another CANCELED by a duplicate-
    order race) and both needed manual intervention despite the alert
    firing correctly. These lock in _auto_restore_missing_stop() in
    isolation: it must never guess a stop price for an untracked ticker,
    must clear conflicting sell orders before resubmitting, and must
    always use a plain STOP.

    PositionTracker's filepath is an early-bound default parameter
    (filepath: str = POSITIONS_FILE, evaluated once at class-definition
    time) -- patch.object(a, "POSITIONS_FILE", ...) does NOT affect
    PositionTracker() calls that don't pass filepath explicitly, and
    _auto_restore_missing_stop() is exactly such a call. Patching the
    PositionTracker class reference itself (forcing every construction
    anywhere in the call graph onto an isolated temp file) is the only
    reliable isolation -- confirmed the hard way: an earlier version of
    these tests used a bare ticker name and silently read this machine's
    REAL production dman_positions.json instead of the test fixture.
    ZTEST9x is used as the ticker specifically to never collide with a
    real held position even if isolation were somehow to fail again."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._pt_patch = patch.object(a, "PositionTracker", self._isolated_pt)
        self._pt_patch.start()

    def tearDown(self):
        self._pt_patch.stop()
        os.unlink(self._pos_tmp.name)

    def _write_positions(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def _tracked_litx(self, stop=28.01):
        return {"ticker": "ZTEST9x", "bias": "LONG", "setup": "Gap & Hold",
                "entry": 35.85, "stop": stop, "target1": 55.45, "target2": 67.21,
                "shares": 47, "entry_date": "2026-08-12"}

    def test_untracked_ticker_refuses_to_guess_a_stop(self):
        self._write_positions([])   # nothing tracked
        mock_client = MagicMock()
        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok)
        self.assertIn("not in PositionTracker", detail)
        mock_client.submit_order.assert_not_called()

    def test_zero_stop_on_record_refuses_to_guess(self):
        self._write_positions([self._tracked_litx(stop=0)])
        mock_client = MagicMock()
        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok)
        mock_client.submit_order.assert_not_called()

    def test_options_leg_is_not_matched_by_bare_ticker(self):
        # A tracked options position on the same underlying must not be
        # mistaken for an equity stop level -- setups are filtered out
        # the same way _check_open_position_risk's orphan check does.
        self._write_positions([{
            "ticker": "ZTEST9x", "bias": "LONG",
            "setup": "Options Call ZTEST9x260828C00040000 ($40C exp 2026-08-28)",
            "entry": 3.0, "stop": 1.5, "target1": 4.5, "target2": 6.0,
            "shares": 100, "entry_date": "2026-08-12",
        }])
        mock_client = MagicMock()
        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok)
        self.assertIn("not in PositionTracker", detail)

    def test_successful_restore_submits_a_plain_stop_at_the_tracked_price(self):
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []   # nothing to cancel
        mock_order = MagicMock(id="new-stop-order-id-12345678")
        mock_client.submit_order.return_value = mock_order

        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        self.assertIn("28.01", detail)
        mock_client.submit_order.assert_called_once()
        req = mock_client.submit_order.call_args[0][0]
        from alpaca.trading.requests import StopOrderRequest
        self.assertIsInstance(req, StopOrderRequest)
        self.assertEqual(req.stop_price, 28.01)
        self.assertEqual(req.qty, 47)

    def test_conflicting_sell_orders_are_cancelled_before_resubmitting(self):
        # Reproduces the actual root cause on both real incidents: a
        # take-profit (or duplicate entry) leg claims all shares via
        # Alpaca's held_for_orders accounting, so a fresh stop can't get
        # share allocation until that sibling is cleared first.
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        stale_tp = MagicMock(id="stale-take-profit-id")
        from alpaca.trading.enums import OrderSide
        stale_tp.side = OrderSide.SELL
        mock_client.get_orders.return_value = [stale_tp]
        mock_client.submit_order.return_value = MagicMock(id="new-id")

        with patch.object(a.time, "sleep"):   # don't actually block the test suite
            ok, _ = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        mock_client.cancel_order_by_id.assert_called_once_with("stale-take-profit-id")

    def test_submission_failure_is_reported_not_raised(self):
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.side_effect = Exception("insufficient buying power")
        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok)
        self.assertIn("insufficient buying power", detail)


class TestPendingSignalsFilteredToRealPositions(unittest.TestCase):
    """Confirmed live 2026-08-08: dman_live_signals.json's "pending" list
    tracks every signal that was ever ALERTED (_log_live_signal fires at
    alert time, independent of --submit or whether the order actually
    filled) -- not signals that became real positions. The risk check used
    to treat every pending entry as if it were an open position: FGL and
    AMZN both showed up with live stop-distance numbers despite neither
    ever having filled a real order, reading as real exposure that didn't
    exist. A regression here means a stale alerted-but-never-filled signal
    can trigger a false "OPEN POSITION RISK ALERT" for something you don't
    actually hold, or (the fail-open direction) go silent about a position
    you genuinely do hold because Alpaca was briefly unreachable."""

    def setUp(self):
        from alpaca.trading.enums import AssetClass
        self.AssetClass = AssetClass

        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._pos_patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._pos_patch.start()

        self._sig_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._sig_tmp.close()
        self._sig_patch = patch.object(a, "LIVE_SIGNALS_FILE", self._sig_tmp.name)
        self._sig_patch.start()

        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

    def tearDown(self):
        self._pos_patch.stop();    os.unlink(self._pos_tmp.name)
        self._sig_patch.stop();    os.unlink(self._sig_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)

    def _write_pending(self, tickers):
        with open(self._sig_tmp.name, "w") as f:
            json.dump({"pending": [
                {"ticker": t, "bias": "LONG", "entry": 10.0, "stop": 9.9,
                 "target1": 12.0, "score": 100}
                for t in tickers
            ]}, f)

    def _equity_position(self, symbol):
        p = MagicMock()
        p.symbol = symbol
        p.asset_class = self.AssetClass.US_EQUITY
        p.qty = "10"
        p.avg_entry_price = "10.0"
        p.unrealized_pl = "0.0"
        p.unrealized_plpc = "0.0"
        return p

    def test_alerted_but_never_filled_ticker_does_not_trigger_a_risk_alert(self):
        # FGL/AMZN-style case: alerted once, never a real position. Price is
        # set BELOW stop (would definitely alert if not filtered out).
        self._write_pending(["FGL"])
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # no real positions at all
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        for _call in mock_tg.call_args_list:
            self.assertNotIn("FGL", _call[0][0])

    def test_real_held_position_still_gets_checked_and_alerted(self):
        self._write_pending(["CLRO"])
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("CLRO")]
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        risk_msgs = [c[0][0] for c in mock_tg.call_args_list if "OPEN POSITION RISK" in c[0][0]]
        self.assertEqual(len(risk_msgs), 1)
        self.assertIn("CLRO", risk_msgs[0])

    def test_mixed_real_and_phantom_only_real_one_alerts(self):
        self._write_pending(["CLRO", "FGL"])
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("CLRO")]
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        risk_msgs = [c[0][0] for c in mock_tg.call_args_list if "OPEN POSITION RISK" in c[0][0]]
        self.assertEqual(len(risk_msgs), 1)
        self.assertIn("CLRO", risk_msgs[0])
        self.assertNotIn("FGL", risk_msgs[0])

    def test_alpaca_unreachable_fails_open_still_shows_pending(self):
        # Can't verify real positions this cycle -- must NOT silently hide a
        # genuine at-risk position just because Alpaca was unreachable.
        self._write_pending(["CLRO"])
        with patch.object(a, "get_alpaca_client", return_value=None), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        risk_msgs = [c[0][0] for c in mock_tg.call_args_list if "OPEN POSITION RISK" in c[0][0]]
        self.assertEqual(len(risk_msgs), 1)
        self.assertIn("CLRO", risk_msgs[0])

    def test_zero_real_positions_filters_pending_to_nothing_not_fail_open(self):
        # Distinguishes "couldn't verify" (fail open, above) from "verified
        # you hold nothing" (must filter to empty, not show stale signals).
        self._write_pending(["FGL"])
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        for _call in mock_tg.call_args_list:
            self.assertNotIn("OPEN POSITION RISK", _call[0][0])


class TestEarningsReactionScoring(unittest.TestCase):
    """Added 2026-08-11: _recent_earnings_surprise() already existed and
    fed alert TEXT, but never the actual confluence score -- a signal
    looked identical whether or not real numbers backed it up. A
    regression here means either a fundamentally-backed setup doesn't get
    credited, or (worse) an averaged EPS+revenue score wrongly penalizes
    a case like RIOT (missed EPS -106%, beat revenue +14.6%, rallied
    +22.9% overnight, confirmed live 2026-08-10)."""

    def _surprise(self, rev_pct=None, eps_pct=None, when="2026-08-10"):
        return {"date": when, "revenue_surprise_percent": rev_pct, "eps_surprise_percent": eps_pct}

    def test_no_recent_earnings_returns_zero(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=None):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 0)

    def test_strong_revenue_beat_scores_max_tier(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.12)):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertEqual(score, 5)

    def test_moderate_revenue_beat_scores_mid_tier(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.07)):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertEqual(score, 3)

    def test_small_revenue_beat_below_threshold_scores_zero(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.02)):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertEqual(score, 0)

    def test_eps_beat_adds_a_secondary_bonus(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.12, eps_pct=0.10)):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertEqual(score, 7)   # 5 (revenue) + 2 (EPS)

    def test_riot_style_eps_miss_with_revenue_beat_is_not_penalized(self):
        # Real numbers, confirmed live 2026-08-10: EPS missed by -106%,
        # revenue beat by +14.6%, price rallied +22.9% overnight.
        with patch.object(a, "_recent_earnings_surprise",
                          return_value=self._surprise(rev_pct=0.1457, eps_pct=-1.0606)):
            ok, score = a.check_earnings_reaction("RIOT", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 5, "the EPS miss must not drag down the genuine revenue-beat bonus")

    def test_short_bias_inverts_the_direction(self):
        # A miss confirms a SHORT/bearish setup, not a beat.
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=-0.12)):
            ok, score = a.check_earnings_reaction("TESTX", "SHORT")
        self.assertEqual(score, 5)

    def test_beat_does_not_confirm_a_short_bias(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.12)):
            ok, score = a.check_earnings_reaction("TESTX", "SHORT")
        self.assertEqual(score, 0)

    def test_score_is_capped_at_seven(self):
        with patch.object(a, "_recent_earnings_surprise", return_value=self._surprise(rev_pct=0.50, eps_pct=0.50)):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertEqual(score, 7)

    def test_exception_fails_open(self):
        with patch.object(a, "_recent_earnings_surprise", side_effect=Exception("boom")):
            ok, score = a.check_earnings_reaction("TESTX", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 0)


class TestEarningsAlreadyReportedCheck(unittest.TestCase):
    """_check_earnings_already_reported() must match real earnings-release
    headline phrasing and fail closed (False, not a crash) when Benzinga
    returns nothing — confirmed live that Benzinga's single-ticker filter is
    unreliable (META/TSLA returned zero articles even unfiltered by time),
    so "no headlines" must mean "can't confirm," not "definitely not reported.\""""

    def test_matching_headline_returns_true(self):
        with patch.object(a, "_fetch_benzinga_ticker_news",
                          return_value={"TESTX": ["TESTX Reports Q2 Earnings, Beats Estimates"]}):
            self.assertTrue(a._check_earnings_already_reported("TESTX"))

    def test_unrelated_headline_returns_false(self):
        with patch.object(a, "_fetch_benzinga_ticker_news",
                          return_value={"TESTX": ["TESTX Announces New Product Line"]}):
            self.assertFalse(a._check_earnings_already_reported("TESTX"))

    def test_no_headlines_returns_false_not_a_crash(self):
        with patch.object(a, "_fetch_benzinga_ticker_news", return_value={}):
            self.assertFalse(a._check_earnings_already_reported("TESTX"))

    def test_fetch_exception_fails_closed(self):
        with patch.object(a, "_fetch_benzinga_ticker_news", side_effect=Exception("network error")):
            self.assertFalse(a._check_earnings_already_reported("TESTX"))


class TestEarningsSpreadScanSkipsUnresolvedTiming(unittest.TestCase):
    """_resolve_earnings_timing()'s own docstring says UNKNOWN-TODAY means
    "caller should skip, not guess" — but run_earnings_spread_scan() never
    actually implemented that skip until now; it just displayed the timing
    string and built a plan regardless. Same requirement for the new
    ALREADY-REPORTED state. A regression here means offering a spread for a
    ticker that already reported or whose timing couldn't be confirmed."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._patch.start()
        a._save_earnings_pending([])

        # _is_alerted_today/_mark_alerted read/write the REAL dman_alerts_dedup.json
        # (repo-relative, not test-isolated like EARNINGS_SPREAD_PENDING_FILE above).
        # Confirmed live 2026-07-30: test_amc_still_builds_a_plan writes a real
        # "TESTX_EARNSPREAD_OFFER_<today>" dedup entry to disk on a pass, so a
        # second same-day run of this suite short-circuits before ever calling
        # build_earnings_spread_plan() — a false failure with no code regression.
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)
        self._dedup_patch.stop()
        os.unlink(self._dedup_tmp.name)

    def _run_with_candidate(self, timing):
        candidate = {"ticker": "TESTX", "earn_date": date.today(), "days_away": 0,
                    "timing": timing, "current_price": 100.0}
        with patch.object(a, "is_market_open", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=MagicMock()):
                with patch.object(a, "get_earnings_spread_candidates", return_value=[candidate]):
                    with patch.object(a, "build_earnings_spread_plan") as mock_build:
                        with patch.object(a, "send_telegram", return_value=True):
                            a.run_earnings_spread_scan()
        return mock_build

    def test_unknown_today_never_builds_a_plan(self):
        mock_build = self._run_with_candidate("UNKNOWN-TODAY")
        mock_build.assert_not_called()

    def test_already_reported_never_builds_a_plan(self):
        mock_build = self._run_with_candidate("ALREADY-REPORTED")
        mock_build.assert_not_called()

    def test_amc_still_builds_a_plan(self):
        mock_build = self._run_with_candidate("AMC")
        mock_build.assert_called_once()


class TestEarningsApprovalTelegramFlow(unittest.TestCase):
    """The approve-gate is the entire safety rationale for this feature —
    every earnings spread requires an explicit human YES (permanent gate, no
    auto-promotion). A reply-parsing bug here either submits an unapproved
    live order or permanently ignores a legitimate approval. Uses an isolated
    temp file for pending state — never the real dman_earnings_pending.json."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._patch.start()
        a._save_earnings_pending([])

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _plan(self):
        return {"ticker": "HOOD", "earn_date": "2026-07-29", "sets": 1,
               "net_debit": 1.91, "total_cost": 191.0, "max_loss": 191.0,
               "directional": None,
               "call": {"long_occ": "HOOD260807C00096000", "short_occ": "HOOD260807C00099000",
                        "max_gain": 193.0}}

    def _add_pending(self, ticker, minutes_until_expiry=30):
        entry = {"ticker": ticker, "earn_date": "2026-07-29",
                 "created_at": datetime.now(a.ET).isoformat(),
                 "expires_at": (datetime.now(a.ET) + timedelta(minutes=minutes_until_expiry)).isoformat(),
                 "status": "awaiting_approval", "plan": self._plan()}
        pending = a._load_earnings_pending()
        pending.append(entry)
        a._save_earnings_pending(pending)

    def test_bare_yes_applies_to_the_only_pending_offer(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                    with patch.object(a, "PositionTracker") as MockPT:
                        consumed = a._handle_earnings_approval_reply("yes")
        self.assertTrue(consumed)
        mock_client.submit_order.assert_called_once()
        self.assertEqual(a._load_earnings_pending(), [])

    def test_yes_with_wrong_ticker_does_not_match(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                consumed = a._handle_earnings_approval_reply("yes META")
        mock_client.submit_order.assert_not_called()
        # the HOOD offer must still be there — a mismatched ticker isn't a rejection
        self.assertEqual(len(a._load_earnings_pending()), 1)

    def test_ambiguous_bare_yes_with_two_pending_is_not_silently_guessed(self):
        self._add_pending("HOOD")
        self._add_pending("RIVN")
        mock_client = MagicMock()
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertEqual(len(a._load_earnings_pending()), 2, "both offers must remain pending")

    def test_no_rejects_and_does_not_submit_an_order(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                consumed = a._handle_earnings_approval_reply("no HOOD")
        self.assertTrue(consumed)
        mock_client.submit_order.assert_not_called()
        self.assertEqual(a._load_earnings_pending(), [])

    def test_expired_offer_is_not_approvable(self):
        self._add_pending("HOOD", minutes_until_expiry=-5)   # already expired
        mock_client = MagicMock()
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()

    def test_non_yes_no_text_is_not_consumed(self):
        self._add_pending("HOOD")
        consumed = a._handle_earnings_approval_reply("what's the weather")
        self.assertFalse(consumed)
        self.assertEqual(len(a._load_earnings_pending()), 1)

    def _plan_with_snapshot(self, current_price):
        p = self._plan()
        p["current_price"] = current_price
        return p

    def _add_pending_with_plan(self, ticker, plan, minutes_until_expiry=240):
        entry = {"ticker": ticker, "earn_date": "2026-07-29",
                 "created_at": datetime.now(a.ET).isoformat(),
                 "expires_at": (datetime.now(a.ET) + timedelta(minutes=minutes_until_expiry)).isoformat(),
                 "status": "awaiting_approval", "plan": plan}
        pending = a._load_earnings_pending()
        pending.append(entry)
        a._save_earnings_pending(pending)

    def test_small_price_drift_still_submits(self):
        # Added 2026-08-10 alongside widening the approval window to 4hrs:
        # a small move since the offer was built must not block a real approval.
        self._add_pending_with_plan("HOOD", self._plan_with_snapshot(100.0))
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                    with patch.object(a, "get_live_price", return_value=103.0):   # +3%, under the 8% limit
                        with patch.object(a, "PositionTracker"):
                            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_called_once()

    def test_large_price_drift_aborts_without_submitting(self):
        # The core of the fix: a widened approval window means the snapshot
        # pricing/strikes can go stale enough to no longer reflect reality --
        # must abort rather than submit a spread built on outdated numbers.
        self._add_pending_with_plan("HOOD", self._plan_with_snapshot(100.0))
        mock_client = MagicMock()
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                with patch.object(a, "get_live_price", return_value=112.0):   # +12%, over the 8% limit
                    consumed = a._handle_earnings_approval_reply("yes")
        self.assertTrue(consumed)
        mock_client.submit_order.assert_not_called()
        self.assertIn("NOT submitted", mock_tg.call_args[0][0])
        self.assertEqual(a._load_earnings_pending(), [], "an aborted-on-drift offer must not "
                          "stay pending forever -- it's consumed, not re-approvable")

    def test_drift_check_skipped_when_no_snapshot_price_stored(self):
        # Older/malformed plans without a current_price snapshot must fail
        # open to the pre-existing submit path, not silently block forever.
        self._add_pending("HOOD")   # no current_price key
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        with patch.object(a, "send_telegram", return_value=True):
            with patch.object(a, "get_alpaca_client", return_value=mock_client):
                with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                    with patch.object(a, "get_live_price", return_value=99999.0):
                        with patch.object(a, "PositionTracker"):
                            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_called_once()


class TestWorkflowRestartFeature(unittest.TestCase):
    """The whole point of /restart and the watchdog auto-heal is that they
    work even when the daemon is frozen — a bug here means the "last resort"
    has no resort. Never let these tests make a real network call: GitHub's
    dispatch API would actually restart the live daemon."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "LAST_ALERTS_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_trigger_restart_no_token_fails_closed(self):
        with patch.object(a, "GITHUB_TOKEN", ""):
            ok, msg = a._trigger_workflow_restart("dman_daemon.yml")
        self.assertFalse(ok)
        self.assertIn("GITHUB_TOKEN", msg)

    def test_trigger_restart_success_on_204(self):
        mock_resp = MagicMock(status_code=204)
        with patch.object(a, "GITHUB_TOKEN", "fake-token"):
            with patch.object(a.requests, "post", return_value=mock_resp) as mock_post:
                ok, msg = a._trigger_workflow_restart("dman_daemon.yml")
        self.assertTrue(ok)
        # Confirm it hit the real dispatch endpoint shape, not just "some URL"
        call_url = mock_post.call_args[0][0]
        self.assertIn("actions/workflows/dman_daemon.yml/dispatches", call_url)
        self.assertEqual(mock_post.call_args[1]["json"], {"ref": "main"})

    def test_trigger_restart_surfaces_non_204_as_failure(self):
        mock_resp = MagicMock(status_code=403, text="Resource not accessible")
        with patch.object(a, "GITHUB_TOKEN", "fake-token"):
            with patch.object(a.requests, "post", return_value=mock_resp):
                ok, msg = a._trigger_workflow_restart("dman_daemon.yml")
        self.assertFalse(ok)
        self.assertIn("403", msg)

    def test_restart_command_dispatches_and_confirms(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(True, "dispatched")) as mock_trigger:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/restart")
        mock_trigger.assert_called_once_with("dman_daemon.yml")
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("Restart requested" in t for t in sent_texts))
        self.assertTrue(any("dispatched" in t.lower() for t in sent_texts))

    def test_restart_command_reports_failure_with_fallback_instructions(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(False, "HTTP 403: nope")):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/reboot")
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("failed" in t.lower() and "Run workflow" in t for t in sent_texts))

    def test_watchdog_auto_restarts_when_daemon_stale_past_45_min(self):
        stale_sync = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        stale_sync.close()
        stale_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        with open(stale_sync.name, "w") as _f:
            json.dump({"last_sync": stale_time}, _f)
        with patch.object(a, "ALPACA_SYNC_FILE", stale_sync.name):
            with patch.object(a, "SCAN_LOG_FILE", "/nonexistent/path.json"):
                with patch.object(a, "_RATE_LIMIT_EVENTS_FILE", "/nonexistent/path2.json"):
                    with patch.object(a, "datetime") as mock_dt:
                        # Fix "now" mid-session so the 930-1600 ET window check passes
                        fixed_now = datetime(2026, 8, 3, 11, 0, tzinfo=a.ET)
                        mock_dt.now.side_effect = lambda tz=None: (fixed_now if tz else datetime.now())
                        mock_dt.fromisoformat = datetime.fromisoformat
                        with patch.object(a, "_trigger_workflow_restart",
                                          return_value=(True, "dispatched")) as mock_trigger:
                            with patch.object(a, "send_telegram", return_value=True):
                                with patch.object(a, "requests") as mock_requests:
                                    mock_requests.get.return_value = MagicMock(
                                        json=lambda: {"workflow_runs": []})
                                    a.run_watchdog()
        os.unlink(stale_sync.name)
        mock_trigger.assert_called_once_with("dman_daemon.yml")

    def test_watchdog_does_not_restart_for_mild_staleness(self):
        # 35 min stale — past the 30-min "flag it" threshold but well under
        # the 45-min "auto-restart" threshold. Should notify, not restart.
        stale_sync = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        stale_sync.close()
        stale_time = (datetime.now() - timedelta(minutes=35)).isoformat()
        with open(stale_sync.name, "w") as _f:
            json.dump({"last_sync": stale_time}, _f)
        with patch.object(a, "ALPACA_SYNC_FILE", stale_sync.name):
            with patch.object(a, "SCAN_LOG_FILE", "/nonexistent/path.json"):
                with patch.object(a, "_RATE_LIMIT_EVENTS_FILE", "/nonexistent/path2.json"):
                    with patch.object(a, "datetime") as mock_dt:
                        fixed_now = datetime(2026, 8, 3, 11, 0, tzinfo=a.ET)
                        mock_dt.now.side_effect = lambda tz=None: (fixed_now if tz else datetime.now())
                        mock_dt.fromisoformat = datetime.fromisoformat
                        with patch.object(a, "_trigger_workflow_restart") as mock_trigger:
                            with patch.object(a, "send_telegram", return_value=True):
                                with patch.object(a, "requests") as mock_requests:
                                    mock_requests.get.return_value = MagicMock(
                                        json=lambda: {"workflow_runs": []})
                                    a.run_watchdog()
        os.unlink(stale_sync.name)
        mock_trigger.assert_not_called()

    def test_watchdog_restart_respects_cooldown_no_double_dispatch(self):
        # A restart already recorded within the cooldown window must not
        # trigger a second dispatch on the very next watchdog tick — that
        # would restart-storm a daemon that's simply still coming back up.
        stale_sync = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        stale_sync.close()
        stale_time = (datetime.now() - timedelta(minutes=60)).isoformat()
        with open(stale_sync.name, "w") as _f:
            json.dump({"last_sync": stale_time}, _f)
        with patch.object(a, "ALPACA_SYNC_FILE", stale_sync.name):
            with patch.object(a, "SCAN_LOG_FILE", "/nonexistent/path.json"):
                with patch.object(a, "_RATE_LIMIT_EVENTS_FILE", "/nonexistent/path2.json"):
                    with patch.object(a, "datetime") as mock_dt:
                        fixed_now = datetime(2026, 8, 3, 11, 0, tzinfo=a.ET)
                        mock_dt.now.side_effect = lambda tz=None: (fixed_now if tz else datetime.now())
                        mock_dt.fromisoformat = datetime.fromisoformat
                        # Record the "already restarted" marker using the SAME
                        # mocked clock the cooldown check will compare against —
                        # using the real wall-clock here would make the elapsed-
                        # time math meaningless (could be a stale-looking gap of
                        # hours purely from test/real-time drift, not the
                        # 2-minutes-ago scenario this test actually means).
                        a._save_last_alert("__WATCHDOG_AUTO_RESTART__")
                        with patch.object(a, "_trigger_workflow_restart") as mock_trigger:
                            with patch.object(a, "send_telegram", return_value=True):
                                with patch.object(a, "requests") as mock_requests:
                                    mock_requests.get.return_value = MagicMock(
                                        json=lambda: {"workflow_runs": []})
                                    a.run_watchdog()
        os.unlink(stale_sync.name)
        mock_trigger.assert_not_called()


class TestInsiderBuyingSignal(unittest.TestCase):
    """Added 2026-08-08 ("free upgrades... completely" -- explicitly in
    place of any further paid Massive/Benzinga tiers). SEC EDGAR Form 4 is
    free and public but slow (live-tested: ~70s for a single ticker across
    a few filings), so the network budget/caching here is load-bearing,
    not incidental. A regression could either burn the per-process network
    budget on nothing (unknown-ticker / old-filing / non-Form-4 paths not
    short-circuiting before a fetch) or count routine option-exercise/
    tax-withholding transactions (codes M/F) as if they were genuine
    open-market insider conviction (codes P/S only)."""

    def setUp(self):
        a._INSIDER_TXN_CACHE.clear()
        a._insider_network_calls_used = 0
        self._sec_cik_map_backup = a._sec_cik_map
        a._sec_cik_map = {"TEST": "0000320193"}

    def tearDown(self):
        a._INSIDER_TXN_CACHE.clear()
        a._insider_network_calls_used = 0
        a._sec_cik_map = self._sec_cik_map_backup

    def _submissions_response(self, forms, dates, accessions=None):
        accessions = accessions or [f"0001-26-{i:06d}" for i in range(len(forms))]
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "filings": {"recent": {
                "form": forms, "filingDate": dates, "accessionNumber": accessions,
            }}
        }
        return resp

    def _form4_xml(self, codes):
        txns = "".join(
            f"""<nonDerivativeTransaction>
                    <transactionCoding><transactionCode>{c}</transactionCode></transactionCoding>
                    <transactionAmounts>
                        <transactionShares><value>1000</value></transactionShares>
                        <transactionPricePerShare><value>10.0</value></transactionPricePerShare>
                    </transactionAmounts>
                    <transactionDate><value>2026-08-01</value></transactionDate>
                </nonDerivativeTransaction>"""
            for c in codes
        )
        xml = (f'<?xml version="1.0"?><ownershipDocument>'
               f'<reportingOwner><reportingOwnerId><rptOwnerName>Test Insider'
               f'</rptOwnerName></reportingOwnerId></reportingOwner>{txns}</ownershipDocument>')
        resp = MagicMock(status_code=200)
        resp.content = xml.encode()
        return resp

    def test_filters_to_p_and_s_codes_only(self):
        today = a.date.today().isoformat()
        sub_resp = self._submissions_response(["4"], [today])
        xml_resp = self._form4_xml(["M", "F", "P"])
        with patch.object(a.requests, "get", side_effect=[sub_resp, xml_resp]):
            txns = a._fetch_recent_insider_transactions("TEST")
        self.assertEqual([t["code"] for t in txns], ["P"])

    def test_filing_older_than_days_back_is_excluded_without_a_fetch(self):
        old_date = (a.date.today() - a.timedelta(days=30)).isoformat()
        sub_resp = self._submissions_response(["4"], [old_date])
        with patch.object(a.requests, "get", return_value=sub_resp) as mock_get:
            txns = a._fetch_recent_insider_transactions("TEST", days_back=14)
        self.assertEqual(txns, [])
        mock_get.assert_called_once()   # only the submissions call -- no wasted XML fetch

    def test_non_form4_filings_are_skipped_without_a_fetch(self):
        today = a.date.today().isoformat()
        sub_resp = self._submissions_response(["10-K", "8-K"], [today, today])
        with patch.object(a.requests, "get", return_value=sub_resp) as mock_get:
            txns = a._fetch_recent_insider_transactions("TEST")
        self.assertEqual(txns, [])
        mock_get.assert_called_once()

    def test_unknown_ticker_returns_empty_without_any_network_call(self):
        with patch.object(a.requests, "get") as mock_get:
            txns = a._fetch_recent_insider_transactions("NOTATICKER")
        self.assertEqual(txns, [])
        mock_get.assert_not_called()

    def test_result_is_cached_second_call_makes_no_extra_network_request(self):
        today = a.date.today().isoformat()
        sub_resp = self._submissions_response(["4"], [today])
        xml_resp = self._form4_xml(["P"])
        with patch.object(a.requests, "get", side_effect=[sub_resp, xml_resp]) as mock_get:
            first = a._fetch_recent_insider_transactions("TEST")
            second = a._fetch_recent_insider_transactions("TEST")
        self.assertEqual(first, second)
        self.assertEqual(mock_get.call_count, 2)   # not 4 -- second call hit the cache

    def test_network_budget_exhaustion_returns_empty_without_a_call(self):
        a._insider_network_calls_used = a._INSIDER_NETWORK_BUDGET
        with patch.object(a.requests, "get") as mock_get:
            txns = a._fetch_recent_insider_transactions("TEST")
        self.assertEqual(txns, [])
        mock_get.assert_not_called()

    def test_check_insider_activity_scores_open_market_purchase_for_long(self):
        with patch.object(a, "_fetch_recent_insider_transactions",
                           return_value=[{"code": "P", "value": 50000.0}]):
            ok, score = a.check_insider_activity("TEST", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 4)

    def test_check_insider_activity_scores_open_market_sale_for_short(self):
        with patch.object(a, "_fetch_recent_insider_transactions",
                           return_value=[{"code": "S", "value": 50000.0}]):
            ok, score = a.check_insider_activity("TEST", "SHORT")
        self.assertTrue(ok)
        self.assertEqual(score, 4)

    def test_purchase_does_not_count_toward_a_short_signal(self):
        with patch.object(a, "_fetch_recent_insider_transactions",
                           return_value=[{"code": "P", "value": 50000.0}]):
            ok, score = a.check_insider_activity("TEST", "SHORT")
        self.assertTrue(ok)
        self.assertEqual(score, 0)

    def test_small_dollar_transaction_gets_the_lower_bonus(self):
        with patch.object(a, "_fetch_recent_insider_transactions",
                           return_value=[{"code": "P", "value": 5000.0}]):
            ok, score = a.check_insider_activity("TEST", "LONG")
        self.assertEqual(score, 2)

    def test_no_matching_transactions_never_blocks_the_signal(self):
        with patch.object(a, "_fetch_recent_insider_transactions", return_value=[]):
            ok, score = a.check_insider_activity("TEST", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 0)

    def test_fetch_failure_fails_open_never_blocks(self):
        with patch.object(a, "_fetch_recent_insider_transactions", side_effect=Exception("boom")):
            ok, score = a.check_insider_activity("TEST", "LONG")
        self.assertTrue(ok)
        self.assertEqual(score, 0)


class TestInsiderScoreSignalGating(unittest.TestCase):
    """The insider check only runs inside score_signal() for candidates
    whose partial score-so-far (MTF+RS+Sector+SectorETF+AI) already
    clears a bar -- this is what protects the slow SEC lookup from firing
    on every single scanned ticker every cycle. A regression here either
    burns the network budget on weak candidates or silently stops the
    bonus from ever reaching genuinely strong ones."""

    def _signal(self, ticker="TEST", bias="LONG"):
        return a.ProSignal(
            ticker=ticker, bias=bias, setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            shares=100, rr=2.0, rsi=50.0, rvol=2.0,
            reason="test", confluence_score=0,
        )

    def _score(self, sig, mtf, rs, sec, check_insider_return=(True, 0)):
        with patch.object(a, "check_mtf", return_value=(True, mtf)), \
             patch.object(a, "check_relative_strength", return_value=(True, rs)), \
             patch.object(a, "check_sector", return_value=(True, sec)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 1)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", return_value=None), \
             patch.object(a, "check_insider_activity", return_value=check_insider_return) as mock_insider:
            scored = a.score_signal(sig, _fake_df(), _fake_regime(), a.WinRateTracker())
        return scored, mock_insider

    def test_weak_candidate_never_triggers_the_insider_check(self):
        # mtf+rs+sec = 2+2+2 = 6, well below the 25 gate (sector-ETF/AI both
        # 0 since fetch_df is mocked to None).
        _, mock_insider = self._score(self._signal(), mtf=2, rs=2, sec=2)
        mock_insider.assert_not_called()

    def test_strong_candidate_triggers_the_insider_check(self):
        # mtf+rs+sec = 15+10+8 = 33 clears the 25 gate on its own.
        _, mock_insider = self._score(self._signal(), mtf=15, rs=10, sec=8)
        mock_insider.assert_called_once_with("TEST", "LONG")

    def test_insider_bonus_is_added_to_the_total_score(self):
        base, _ = self._score(self._signal(), mtf=15, rs=10, sec=8, check_insider_return=(True, 0))
        boosted, _ = self._score(self._signal(), mtf=15, rs=10, sec=8, check_insider_return=(True, 4))
        self.assertEqual(boosted.confluence_score, base.confluence_score + 4)


def _fake_df():
    import pandas as pd
    import numpy as np
    n = 60
    idx = pd.date_range("2026-05-01", periods=n, freq="D")
    close = np.linspace(10, 12, n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": [1_000_000] * n,
        "RSI": [40.0] * n, "MACD": [-0.1] * n, "MACD_sig": [0.1] * n,
        "MACD_hist": [-0.1] * n, "EMA9": close, "EMA20": close,
        "EMA50": close, "ATR": [0.5] * n, "AvgVol20": [5_000_000] * n,
        "RVOL": [2.0] * n,
    }, index=idx)


def _fake_regime():
    return {"regime": "CHOP", "score": 10, "vix_ok": True,
           "details": {}}


if __name__ == "__main__":
    unittest.main(verbosity=2)
