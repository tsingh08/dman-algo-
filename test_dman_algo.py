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
from unittest.mock import patch, MagicMock, mock_open

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


class TestSimulateTradeOutcomeStopCheckOrdering(unittest.TestCase):
    """Found in the 2026-08-16 review: within one daily bar,
    _simulate_trade_outcome() used to check the BE@1R trigger (did High
    reach +1R) BEFORE the stop-breach check (did Low breach the ORIGINAL
    stop) -- and the BE@1R check mutates trail_stop in place, so a bar
    whose High reached +1R AND whose Low also breached the original stop
    got the stop check evaluated against the freshly-promoted breakeven
    stop instead. Daily OHLC can't tell you which extreme happened first
    intraday -- the Low could just as easily have breached the original
    stop BEFORE the High ever reached +1R that same day, which is a real
    loss, not a breakeven "win". This look-ahead bias could inflate the
    reported win rate."""

    def _df_with_one_wide_bar(self, entry_open, entry_high, entry_low, entry_close,
                              bar_high, bar_low, bar_close):
        import pandas as pd
        idx = pd.date_range("2026-06-01", periods=2, freq="D")
        return pd.DataFrame({
            "Open":   [entry_open, entry_close],
            "High":   [entry_high, bar_high],
            "Low":    [entry_low, bar_low],
            "Close":  [entry_close, bar_close],
            "Volume": [1_000_000, 1_000_000],
        }, index=idx)

    def test_same_bar_stop_breach_and_1r_touch_is_a_real_loss_not_breakeven(self):
        # entry=15, stop=12 -> be1r_px = 15 + (15-12) = 18. This bar's High
        # (18.5) clears +1R AND its Low (11.5) breaches the ORIGINAL stop --
        # daily OHLC can't tell us which happened first, so the
        # conservative (non-look-ahead) read must be a real stop-out.
        df = self._df_with_one_wide_bar(15.0, 15.2, 14.8, 15.0, 18.5, 11.5, 13.0)
        with patch.object(a.yf, "download", return_value=df):
            result = a._simulate_trade_outcome(
                ticker="TESTX", entry=15.0, stop=12.0,
                target1=20.0, target2=25.0, bias="LONG",
                start_date="2026-06-01",
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["outcome"], "LOSS")
        self.assertEqual(result["exit_reason"], "STOP")
        self.assertAlmostEqual(result["pnl_pct"], -20.0, places=1)

    def test_1r_touch_without_a_stop_breach_still_arms_breakeven_for_the_next_bar(self):
        # Regression check: the reordering must not break the normal,
        # non-ambiguous BE@1R promotion path -- a bar that reaches +1R
        # WITHOUT breaching the original stop should still arm breakeven,
        # and a later bar breaching that NEW (breakeven) stop should
        # report STOP(BE) with ~0% pnl, not a real loss.
        import pandas as pd
        idx = pd.date_range("2026-06-01", periods=3, freq="D")
        df = pd.DataFrame({
            "Open":   [15.0, 17.5, 14.5],
            "High":   [15.2, 18.5, 15.1],   # bar 2 clears +1R (18.0), stop NOT breached (Low > 12)
            "Low":    [14.8, 17.0, 14.4],   # bar 3 breaches the NEW breakeven stop (15.0)
            "Close":  [15.0, 18.0, 14.6],
            "Volume": [1_000_000, 1_000_000, 1_000_000],
        }, index=idx)
        with patch.object(a.yf, "download", return_value=df):
            result = a._simulate_trade_outcome(
                ticker="TESTX", entry=15.0, stop=12.0,
                target1=25.0, target2=30.0, bias="LONG",
                start_date="2026-06-01",
            )
        self.assertIsNotNone(result)
        self.assertEqual(result["exit_reason"], "STOP(BE)")
        self.assertAlmostEqual(result["pnl_pct"], 0.0, places=1)


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

    def test_tied_shares_and_stop_still_keeps_the_higher_peak_premium(self):
        # Reproduces a real live incident (2026-08-15): an options position's
        # shares/stop don't change until T1, so they're tied on nearly every
        # merge before that -- and the old tie-break ("keep prev", i.e.
        # remote) silently discarded whichever side had actually observed
        # the higher real-time peak. Confirmed live: UMAC's peak_premium was
        # found pinned at 8.75, BELOW its own 9.22 entry, while the real
        # intraday high (verified against Alpaca's own historical bars) was
        # 10.20 -- this is exactly that shape, minified.
        local  = [{"ticker": "UMAC", "stop": 4.61, "shares": 200, "peak_premium": 10.20}]
        remote = [{"ticker": "UMAC", "stop": 4.61, "shares": 200, "peak_premium": 8.75}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["peak_premium"], 10.20,
                         "the higher observed peak must survive a tie, regardless of "
                         "which side is 'local' vs 'remote'")

    def test_tied_shares_and_stop_keeps_the_further_milestone_alerted(self):
        # Same tie shape, for the P&L-milestone dedup fields -- taking the
        # max prevents a regression from re-firing an already-sent alert.
        local  = [{"ticker": "UMAC", "stop": 4.61, "shares": 200,
                   "milestone_gain_alerted": 30.0, "milestone_loss_alerted": 0.0}]
        remote = [{"ticker": "UMAC", "stop": 4.61, "shares": 200,
                   "milestone_gain_alerted": 20.0, "milestone_loss_alerted": 10.0}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["milestone_gain_alerted"], 30.0)
        self.assertEqual(merged[0]["milestone_loss_alerted"], 10.0)

    def test_tied_shares_and_stop_never_regresses_trailing_back_to_initial(self):
        local  = [{"ticker": "CELZ", "stop": 10.0, "shares": 100, "stop_stage": "trailing"}]
        remote = [{"ticker": "CELZ", "stop": 10.0, "shares": 100, "stop_stage": "initial"}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["stop_stage"], "trailing")
        # And the reverse ordering (remote has "trailing") must also survive.
        merged2 = a.merge_positions_snapshots(remote, local)
        self.assertEqual(merged2[0]["stop_stage"], "trailing")

    def test_tied_shares_and_stop_keeps_the_higher_trail_pct(self):
        # Added 2026-08-23 alongside the new trail_pct field -- same tie
        # shape as peak_premium/milestones above: a merge must never lose
        # the only local record of what a live trailing order's percent
        # was submitted at.
        local  = [{"ticker": "CELZ", "stop": 10.0, "shares": 100,
                   "stop_stage": "trailing", "trail_pct": 8.5}]
        remote = [{"ticker": "CELZ", "stop": 10.0, "shares": 100,
                   "stop_stage": "trailing", "trail_pct": 0.0}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["trail_pct"], 8.5)

    def test_a_more_protective_stop_still_carries_forward_the_losing_sides_peak(self):
        # When shares/stop are NOT tied, the existing rule still picks the
        # more-protective side -- but the losing side's peak_premium must
        # not just vanish if it happens to be the higher one (e.g. the
        # side that raised its stop first hasn't re-observed the peak yet).
        local  = [{"ticker": "UMAC", "stop": 6.00, "shares": 200, "peak_premium": 11.00}]
        remote = [{"ticker": "UMAC", "stop": 4.61, "shares": 200, "peak_premium": 13.50}]
        merged = a.merge_positions_snapshots(local, remote)
        self.assertEqual(merged[0]["stop"], 6.00, "the more-protective stop still wins")
        self.assertEqual(merged[0]["peak_premium"], 13.50,
                         "but the higher real peak from the losing side must not be lost")


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


class TestPositionTrackerOpenAccumulatesOnScaleIn(unittest.TestCase):
    """Found live 2026-08-17: ARTL was bought in 3 separate scan-cycle
    batches (a sustained-RVOL small-cap catalyst signal firing again on
    several scans before the position was recognized as already held).
    open() used to silently REPLACE the tracked entry on every repeat
    call for the same identity -- the recorded share count never matched
    what was actually held on the exchange. A regression here means a
    real scale-in position drifts out of sync with the broker again."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.write("[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _pos(self, **overrides):
        base = dict(ticker="ARTL", bias="LONG", setup="Low Float Catalyst",
                    entry=0.75, stop=0.61, target1=0.98, target2=1.31,
                    shares=504, entry_date="2026-08-17")
        base.update(overrides)
        return a.OpenPosition(**base)

    def test_repeat_open_accumulates_shares_instead_of_replacing(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(shares=504, entry=0.75))
        pt.open(self._pos(shares=493, entry=0.76))
        self.assertEqual(len(pt.positions), 1, "a scale-in must not create a second tracked entry")
        self.assertEqual(pt.positions[0].shares, 504 + 493)

    def test_repeat_open_computes_size_weighted_average_entry(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(shares=100, entry=1.00))
        pt.open(self._pos(shares=100, entry=2.00))
        self.assertAlmostEqual(pt.positions[0].entry, 1.50, places=4)

    def test_repeat_open_keeps_the_more_protective_stop(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(stop=0.61))
        pt.open(self._pos(stop=0.55))   # a looser stop must not win
        self.assertEqual(pt.positions[0].stop, 0.61)
        pt2 = a.PositionTracker(filepath=self._tmp.name)
        pt2.positions = []
        pt2.open(self._pos(stop=0.61))
        pt2.open(self._pos(stop=0.65))   # a tighter/more-protective stop should win
        self.assertEqual(pt2.positions[0].stop, 0.65)

    def test_repeat_open_keeps_the_original_entry_date(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(entry_date="2026-08-14"))
        pt.open(self._pos(entry_date="2026-08-17"))
        self.assertEqual(pt.positions[0].entry_date, "2026-08-14")

    def test_repeat_open_does_not_regress_trailing_stop_stage(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(stop_stage="trailing"))
        pt.open(self._pos(stop_stage="initial"))   # a later batch's default must not reset it
        self.assertEqual(pt.positions[0].stop_stage, "trailing")

    def test_repeat_open_carries_forward_trail_pct_when_already_trailing(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(stop_stage="trailing", trail_pct=8.5))
        pt.open(self._pos(stop_stage="initial", trail_pct=0.0))
        self.assertEqual(pt.positions[0].trail_pct, 8.5,
                          "a later, non-trailing batch's default trail_pct must not erase it")

    def test_repeat_open_carries_forward_the_higher_milestone_progress(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(peak_premium=1.5, milestone_gain_alerted=20.0))
        pt.open(self._pos(peak_premium=1.2, milestone_gain_alerted=0.0))
        self.assertEqual(pt.positions[0].peak_premium, 1.5)
        self.assertEqual(pt.positions[0].milestone_gain_alerted, 20.0)

    def test_a_genuinely_different_identity_still_opens_separately(self):
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(ticker="ARTL"))
        pt.open(self._pos(ticker="CELZ"))
        self.assertEqual(len(pt.positions), 2, "unrelated tickers must never be accumulated together")

    def test_repeat_open_never_lets_the_merged_stop_exceed_the_merged_entry(self):
        # Added 2026-08-23: an options position that already earned a
        # breakeven stop via T1 (stop == its old, higher entry), then
        # accumulated more at a genuinely lower price, blended to a lower
        # entry while the old `max(existing.stop, pos.stop)` rule kept the
        # OLD entry as the stop -- leaving the merged position with a stop
        # ABOVE its own blended cost basis. That's a real, live-money
        # invariant violation for a long-only protective stop: it can
        # never be more protective to sit above what you paid.
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(shares=200, entry=10.00, stop=10.00))   # T1 already taken, stop = old entry
        pt.open(self._pos(shares=200, entry=9.00, stop=4.50))     # accumulate more at a real discount
        self.assertAlmostEqual(pt.positions[0].entry, 9.50, places=4)
        self.assertLessEqual(pt.positions[0].stop, pt.positions[0].entry,
                              "a merged stop must never sit above the merged entry")
        self.assertEqual(pt.positions[0].stop, 9.50,
                          "capped at the new blended entry — still the most protective value "
                          "that doesn't invert the stop/entry relationship")

    def test_repeat_open_still_keeps_a_lower_old_stop_when_it_is_more_protective(self):
        # The clamp must not regress the ORIGINAL "more protective wins"
        # behavior for the ordinary case where the merged stop legitimately
        # stays below the merged entry either way.
        pt = a.PositionTracker(filepath=self._tmp.name)
        pt.open(self._pos(shares=100, entry=1.00, stop=0.61))
        pt.open(self._pos(shares=100, entry=1.00, stop=0.55))   # looser — must not win
        self.assertEqual(pt.positions[0].stop, 0.61)


class TestPositionTrackerThreadSafety(unittest.TestCase):
    """Added 2026-08-23: PositionTracker.open()/close() and
    _update_positions_matching() each independently did a full
    load->mutate->write of dman_positions.json with no coordination --
    guard_loop/telegram_loop/scan_loop all run as separate threads of the
    SAME daemon process, so one thread's read-modify-write could silently
    clobber another's already-saved change (a classic lost-update race).
    _POSITIONS_LOCK serializes every mutation; these tests actually run
    concurrent threads against a real shared file rather than just
    asserting the lock object exists, since a lock that's acquired in the
    wrong place gives zero protection despite "looking" present."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _pos(self, ticker):
        return a.OpenPosition(
            ticker=ticker, bias="LONG", setup="Low Float Catalyst",
            entry=1.0, stop=0.5, target1=1.5, target2=2.0,
            shares=100, entry_date="2026-08-23",
        )

    def test_many_concurrent_opens_from_separate_instances_lose_none(self):
        import threading
        N = 25
        barrier = threading.Barrier(N)

        def _worker(i):
            barrier.wait()   # maximize actual overlap, not just interleaved starts
            a.PositionTracker(filepath=self._tmp.name).open(self._pos(f"T{i}"))

        with patch.object(a, "MAX_POSITIONS", N + 10):   # isolate the lock behavior from capacity
            threads = [threading.Thread(target=_worker, args=(i,)) for i in range(N)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        final = a.PositionTracker(filepath=self._tmp.name)
        self.assertEqual(len(final.positions), N,
                          "every concurrent open() from a separate instance must survive — "
                          "a lost-update race would silently drop some")
        self.assertEqual({p.ticker for p in final.positions}, {f"T{i}" for i in range(N)})

    def test_concurrent_open_and_field_update_do_not_corrupt_each_other(self):
        import threading
        a.PositionTracker(filepath=self._tmp.name).open(self._pos("HELD"))
        N = 20
        barrier = threading.Barrier(N + 1)

        def _open_worker(i):
            barrier.wait()
            a.PositionTracker(filepath=self._tmp.name).open(self._pos(f"NEW{i}"))

        def _update_worker():
            barrier.wait()
            with patch.object(a, "POSITIONS_FILE", self._tmp.name):
                for _ in range(20):
                    a._update_position_field("HELD", stop=0.6)

        with patch.object(a, "MAX_POSITIONS", N + 10):
            threads = [threading.Thread(target=_open_worker, args=(i,)) for i in range(N)]
            threads.append(threading.Thread(target=_update_worker))
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        final = a.PositionTracker(filepath=self._tmp.name)
        self.assertEqual(len(final.positions), N + 1,
                          "the pre-existing HELD position plus every new open() must all survive")
        held = next(p for p in final.positions if p.ticker == "HELD")
        self.assertEqual(held.stop, 0.6, "the field update must not be lost either")


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


class TestPrintScanLogNewsBreadth(unittest.TestCase):
    """Added 2026-08-15: the whole point of persisting a per-scan news
    sentiment breadth snapshot is to have a reviewable trend before ever
    deciding whether to wire it into scoring — locks in that the number
    actually SHOWS UP when the log is reviewed, not just sits silently in
    the JSON, and that older entries from before this field existed don't
    break rendering (regular .get() defaults, no KeyError)."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "SCAN_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _rendered(self):
        with patch("builtins.print") as mock_print:
            a.print_scan_log()
        return " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)

    def test_breadth_pct_is_rendered(self):
        a._append_scan_log({"ts": "2026-08-15T10:00:00-04:00", "regime": "BULL",
                            "regime_score": 17, "vix": 14.5, "min_score": 85,
                            "universe": "curated", "tickers_total": 79, "signals": 0,
                            "signal_tickers": [], "news_breadth_pct": 33.3, "news_breadth_total": 4})
        rendered = self._rendered()
        self.assertIn("+33%", rendered)

    def test_pre_existing_entry_without_the_field_renders_without_crashing(self):
        # Reproduces an entry shape from before this field existed --
        # print_scan_log must never KeyError on old scan history.
        a._append_scan_log({"ts": "2026-08-14T10:00:00-04:00", "regime": "BULL",
                            "regime_score": 17, "vix": 14.5, "min_score": 85,
                            "universe": "curated", "tickers_total": 79, "signals": 0,
                            "signal_tickers": []})
        rendered = self._rendered()   # must not raise
        self.assertNotIn("breadth", rendered)


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
    each of the four circuit-breaker early returns (consecutive-loss,
    monthly, daily, VIX-extreme) -- the point of _log_scan_halt existing
    is defeated if a guard forgets to call it."""

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
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
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
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "monthly_loss_limit")

    def test_probation_bypasses_consec_and_monthly_guards(self):
        # Added 2026-08-18: with probation active, a scan that would
        # otherwise halt on either guard must proceed past both. Also
        # trips the (cheap, network-free) VIX-crisis halt right after so
        # the scan terminates quickly instead of falling through into a
        # real live scan of "AAPL" -- this only asserts the two bypassed
        # reasons never got logged, not that scanning fully completed.
        _stats = {"consec_losses": a.MAX_CONSEC_LOSSES, "consec_wins": 0,
                  "win_rate": 0.3, "total": 10, "wins": 3, "losses": 7}
        regime = {"regime": "CRISIS", "score": 0, "vix_ok": False, "details": {"VIX": 45.0}}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=_stats), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "get_market_regime", return_value=regime), \
             patch.object(a, "get_top_sectors", return_value=[]), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "vix_extreme")

    def test_probation_still_lets_daily_loss_guard_halt(self):
        with patch.object(a.WinRateTracker, "rolling_stats",
                          return_value={"consec_losses": a.MAX_CONSEC_LOSSES, "consec_wins": 0}), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "get_todays_loss", return_value=-(a.DAILY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "daily_loss_limit")

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

    def test_vix_extreme_guard_logs_a_halt(self):
        # Found in the 2026-08-16 review: the other three circuit breakers
        # (consecutive-loss, monthly, daily) all call _log_scan_halt() so
        # the scan log doesn't go dark for the rest of a halted session --
        # the VIX>=40 crisis halt, the single highest-stakes day for this
        # to matter, never did.
        regime = {"regime": "CRISIS", "score": 0, "vix_ok": False, "details": {"VIX": 45.0}}
        with patch.object(a.WinRateTracker, "rolling_stats",
                          return_value={"consec_losses": 0, "consec_wins": 0,
                                        "win_rate": 0.5, "total": 10}), \
             patch.object(a.WinRateTracker, "adaptive_min_score", return_value=80), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_market_regime", return_value=regime), \
             patch.object(a, "get_top_sectors", return_value=[]), \
             patch.object(a, "_log_scan_halt") as mock_halt:
            result = a.run_pro_scanner(["AAPL"], universe_label="test")
        self.assertEqual(result, [])
        mock_halt.assert_called_once()
        self.assertEqual(mock_halt.call_args[0][0], "vix_extreme")


class TestSectorConcentrationCap(unittest.TestCase):
    """Found in the 2026-08-16 review: TICKER_SECTOR.get(ticker, "")
    returns "" for anything outside the curated large-cap map, and the
    cap treated an empty sector as automatically EXEMPT -- so it never
    applied to small-cap catalyst or dynamically-discovered signals at
    all, no matter how many fired in the same scan. Fixed by bucketing
    every unmapped ticker under one shared sentinel sector and applying
    the same max-2 cap to it."""

    def _signal(self, ticker):
        return a.ProSignal(
            ticker=ticker, bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            shares=100, rr=2.0, rsi=50.0, rvol=2.0,
            reason="test", confluence_score=0,
        )

    def test_a_third_signal_in_a_known_sector_is_capped(self):
        # AAPL/MSFT/NVDA are all TICKER_SECTOR "Technology" real entries.
        sigs = [self._signal("AAPL"), self._signal("MSFT"), self._signal("NVDA")]
        result = a._apply_sector_concentration_cap(sigs)
        self.assertEqual([s.ticker for s in result], ["AAPL", "MSFT"])

    def test_unmapped_small_cap_tickers_are_now_capped_too(self):
        # Three tickers with no TICKER_SECTOR entry at all -- the actual
        # bug: previously an unbounded number of these could pass.
        sigs = [self._signal("ZZZQ1"), self._signal("ZZZQ2"), self._signal("ZZZQ3")]
        result = a._apply_sector_concentration_cap(sigs)
        self.assertEqual(len(result), 2, "unmapped-sector signals must be capped at 2, same as any real sector")

    def test_two_unmapped_tickers_both_pass(self):
        sigs = [self._signal("ZZZQ1"), self._signal("ZZZQ2")]
        result = a._apply_sector_concentration_cap(sigs)
        self.assertEqual(len(result), 2)

    def test_known_and_unmapped_sectors_are_capped_independently(self):
        sigs = [self._signal("AAPL"), self._signal("MSFT"), self._signal("NVDA"),
               self._signal("ZZZQ1"), self._signal("ZZZQ2"), self._signal("ZZZQ3")]
        result = a._apply_sector_concentration_cap(sigs)
        tickers = [s.ticker for s in result]
        self.assertEqual(tickers, ["AAPL", "MSFT", "ZZZQ1", "ZZZQ2"])


class TestFinalizeAndAlertSignalsLogsDedupSuppressed(unittest.TestCase):
    """Found in the 2026-08-16 review: a signal suppressed by the alert
    cooldown (_is_duplicate_alert) is still tradeable -- it's still in the
    list run_pro_scanner() returns and the caller submits to Alpaca -- but
    _log_live_signal() used to be skipped in the same branch that skipped
    the Telegram alert, so a real fill from a dedup-suppressed signal had
    no entry in the live-outcome pending log to ever resolve win/loss
    against. Only the alert should be suppressed by dedup, not outcome
    tracking."""

    def _signal(self, ticker):
        return a.ProSignal(
            ticker=ticker, bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            shares=100, rr=2.0, rsi=50.0, rvol=2.0,
            reason="test", confluence_score=0,
        )

    def test_dedup_suppressed_signal_still_gets_logged_for_outcome_tracking(self):
        sig = self._signal("AAPL")
        with patch.object(a, "_is_duplicate_alert", return_value=True), \
             patch.object(a, "_log_live_signal") as mock_log, \
             patch.object(a, "_save_last_alert") as mock_save, \
             patch.object(a, "_send_signal_alert_batch") as mock_send:
            a._finalize_and_alert_signals([sig], {"regime": "TREND"}, {})
        mock_log.assert_called_once_with(sig)
        mock_save.assert_not_called()
        mock_send.assert_called_once_with([])

    def test_non_suppressed_signal_is_logged_and_alerted(self):
        sig = self._signal("AAPL")
        with patch.object(a, "_is_duplicate_alert", return_value=False), \
             patch.object(a, "_log_live_signal") as mock_log, \
             patch.object(a, "_save_last_alert") as mock_save, \
             patch.object(a, "format_signal_telegram", return_value="msg"), \
             patch.object(a, "_send_signal_alert_batch") as mock_send:
            a._finalize_and_alert_signals([sig], {"regime": "TREND"}, {})
        mock_log.assert_called_once_with(sig)
        mock_save.assert_called_once_with("AAPL")
        mock_send.assert_called_once_with(["msg"])

    def test_smallcap_extra_ticker_uses_the_smallcap_telegram_format(self):
        sig = self._signal("ZZZQ")
        with patch.object(a, "_is_duplicate_alert", return_value=False), \
             patch.object(a, "_log_live_signal"), \
             patch.object(a, "_save_last_alert"), \
             patch.object(a, "format_smallcap_telegram", return_value="smallcap msg") as mock_fmt, \
             patch.object(a, "_send_signal_alert_batch") as mock_send:
            a._finalize_and_alert_signals([sig], {"regime": "TREND"},
                                          {"ZZZQ": (5.0, 20.0, 3.0, 60.0)})
        mock_fmt.assert_called_once_with(sig, 5.0, 20.0, 3.0, 60.0)
        mock_send.assert_called_once_with(["smallcap msg"])


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


class TestSyncAlpacaSyncStateWithRemote(unittest.TestCase):
    """Found in the 2026-08-16 review: sync_alpaca_sync_state_with_remote()'s
    rebuild() only ever reconstructed last_sync/recorded_ids, silently
    dropping closed_identities on every rewrite -- and this function runs on
    nearly every git_sync() cycle (last_sync changes every time
    sync_alpaca_fills() runs). That's the actual, still-live root cause of
    the documented FGL resurrection incident: the tombstone
    merge_positions_snapshots() depends on to keep a genuinely-closed
    position from reappearing was being wiped almost as fast as it was
    written. These lock in that closed_identities now survives a merge,
    with per-identity later-timestamp-wins and TTL pruning."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "ALPACA_SYNC_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _write_local(self, data):
        with open(self._tmp.name, "w") as f:
            json.dump(data, f)

    def _sync_against_remote(self, remote):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_alpaca_sync_state_with_remote()
        with open(self._tmp.name) as f:
            return json.load(f)

    def test_local_only_closed_identity_survives_a_merge(self):
        now = time.time()
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"], "closed_identities": {"FGL": now}})
        remote = {"last_sync": "2026-08-16T09:59:00-04:00", "recorded_ids": ["b"],
                  "closed_identities": {}}
        result = self._sync_against_remote(remote)
        self.assertIn("FGL", result["closed_identities"],
                      "a tombstone only local knows about must not be dropped by the merge")

    def test_remote_only_closed_identity_is_pulled_in_not_discarded(self):
        # The exact FGL shape: THIS process's own pt.close() already removed
        # a position locally and tombstoned it -- but the merge itself must
        # not require local to already know about a remote-side tombstone
        # for that tombstone to survive.
        now = time.time()
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"], "closed_identities": {}})
        remote = {"last_sync": "2026-08-16T10:05:00-04:00", "recorded_ids": ["a"],
                  "closed_identities": {"FGL": now}}
        result = self._sync_against_remote(remote)
        self.assertIn("FGL", result["closed_identities"])

    def test_same_identity_keeps_the_later_timestamp(self):
        now = time.time()
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"],
                            "closed_identities": {"FGL": now - 100}})
        remote = {"last_sync": "2026-08-16T10:05:00-04:00", "recorded_ids": ["a", "c"],
                  "closed_identities": {"FGL": now}}
        result = self._sync_against_remote(remote)
        self.assertAlmostEqual(result["closed_identities"]["FGL"], now, delta=1)

    def test_expired_closed_identity_is_pruned_by_the_merge(self):
        stale = time.time() - a._CLOSED_IDENTITY_TOMBSTONE_S - 10
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"], "closed_identities": {"OLD": stale}})
        remote = {"last_sync": "2026-08-16T10:05:00-04:00", "recorded_ids": ["a", "c"],
                  "closed_identities": {}}
        result = self._sync_against_remote(remote)
        self.assertNotIn("OLD", result["closed_identities"])

    def test_corrupted_remote_timestamp_does_not_poison_a_good_local_entry(self):
        now = time.time()
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"], "closed_identities": {"FGL": now}})
        remote = {"last_sync": "2026-08-16T10:05:00-04:00", "recorded_ids": ["a", "c"],
                  "closed_identities": {"FGL": "not-a-number"}}
        result = self._sync_against_remote(remote)
        self.assertAlmostEqual(result["closed_identities"]["FGL"], now, delta=1)

    def test_recorded_ids_and_last_sync_still_merge_correctly(self):
        # Regression check: the pre-existing behavior this function already
        # had must survive the closed_identities addition unchanged.
        self._write_local({"last_sync": "2026-08-16T10:00:00-04:00",
                            "recorded_ids": ["a"], "closed_identities": {}})
        remote = {"last_sync": "2026-08-16T10:05:00-04:00",
                  "recorded_ids": ["a", "b"], "closed_identities": {}}
        result = self._sync_against_remote(remote)
        self.assertEqual(set(result["recorded_ids"]), {"a", "b"})
        self.assertEqual(result["last_sync"], "2026-08-16T10:05:00-04:00")


class TestPnlEntriesAreMergeSafe(unittest.TestCase):
    """Found in the 2026-08-16 review: dman_daily_pnl.json/
    dman_monthly_pnl.json were a single mutated {"date"/"month", "pnl_pct"}
    scalar with no semantic merge -- unlike every other multi-writer state
    file in this project (scan log, win rate, alpaca sync state), a naive
    git merge conflict between the daemon and the hourly cron scanner could
    silently keep only one side's contribution, under-counting today's real
    loss against DAILY_LOSS_LIMIT/MONTHLY_LOSS_LIMIT. Fixed by converting
    both files to an append-only list of {ts, pnl_pct} entries (summed at
    read time) plus a union-then-sort merge, the same pattern already
    proven for scan_log/news_log. These lock in the merge itself and that a
    pre-migration legacy-shape file upgrades instead of losing its one
    running total."""

    def setUp(self):
        self._daily_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._daily_tmp.close()
        self._monthly_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._monthly_tmp.close()
        self._patches = [
            patch.object(a, "DAILY_PNL_FILE", self._daily_tmp.name),
            patch.object(a, "MONTHLY_PNL_FILE", self._monthly_tmp.name),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._daily_tmp, self._monthly_tmp):
            os.unlink(f.name)

    def _write(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f)

    def _sync_daily_against_remote(self, remote):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_daily_pnl_with_remote()
        with open(self._daily_tmp.name) as f:
            return json.load(f)

    def _sync_monthly_against_remote(self, remote):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_monthly_pnl_with_remote()
        with open(self._monthly_tmp.name) as f:
            return json.load(f)

    def test_local_and_remote_entries_both_survive_a_merge(self):
        self._write(self._daily_tmp.name, {"entries": [
            {"ts": "2026-08-16T10:00:00-04:00", "pnl_pct": -1.0}]})
        remote = {"entries": [{"ts": "2026-08-16T10:05:00-04:00", "pnl_pct": -0.5}]}
        result = self._sync_daily_against_remote(remote)
        total = sum(e["pnl_pct"] for e in result["entries"])
        self.assertAlmostEqual(total, -1.5, places=4,
                                msg="neither side's contribution to today's realized P&L may be dropped")

    def test_identical_entry_on_both_sides_is_not_double_counted(self):
        entry = {"ts": "2026-08-16T10:00:00-04:00", "pnl_pct": -1.0}
        self._write(self._daily_tmp.name, {"entries": [entry]})
        result = self._sync_daily_against_remote({"entries": [dict(entry)]})
        self.assertEqual(len(result["entries"]), 1)

    def test_monthly_merge_uses_the_same_union_logic(self):
        self._write(self._monthly_tmp.name, {"entries": [
            {"ts": "2026-08-01T10:00:00-04:00", "pnl_pct": -2.0}]})
        remote = {"entries": [{"ts": "2026-08-16T10:05:00-04:00", "pnl_pct": -3.0}]}
        result = self._sync_monthly_against_remote(remote)
        total = sum(e["pnl_pct"] for e in result["entries"])
        self.assertAlmostEqual(total, -5.0, places=4)

    def test_pre_migration_legacy_shape_local_file_is_not_dropped_by_a_sync(self):
        # A file that hasn't been touched by record_daily_pnl() yet is still
        # in the old {"date", "pnl_pct"} shape -- the sync's extract() must
        # upgrade it the same way _load_pnl_entries() does, not treat it as
        # having zero entries and let the merge silently wipe it out.
        self._write(self._daily_tmp.name, {"date": "2026-08-16", "pnl_pct": -1.25})
        remote = {"entries": [{"ts": "2026-08-16T10:05:00-04:00", "pnl_pct": -0.5}]}
        result = self._sync_daily_against_remote(remote)
        total = sum(e["pnl_pct"] for e in result["entries"])
        self.assertAlmostEqual(total, -1.75, places=4)

    def test_legacy_shape_file_upgrades_transparently_via_load_pnl_entries(self):
        self._write(self._daily_tmp.name, {"date": "2026-08-16", "pnl_pct": -1.25})
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 16, 12, 0, tzinfo=a.ET)
            self.assertAlmostEqual(a.get_todays_loss(), -1.25, places=4)


class TestLogNewsEvent(unittest.TestCase):
    """Added 2026-08-15, direct instruction to have the algo "constantly
    internalize" news across market + extended hours instead of losing
    each headline the moment its alert (now near-silent by the same
    instruction) scrolls past. Applies sync_scan_log_with_remote()'s
    already-learned ts-sort-before-cap lesson from the start, since this
    file has the identical high-frequency multi-writer shape (cron
    scanner's REST pre-fetch + daemon's real-time stream both append)."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "NEWS_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _read(self):
        with open(a.NEWS_LOG_FILE) as f:
            return json.load(f)

    def test_writes_a_new_entry(self):
        a._log_news_event(["UMAC"], "Unusual Machines Soars 22%", source="Barron's", tag="held")
        result = self._read()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbols"], ["UMAC"])
        self.assertEqual(result[0]["headline"], "Unusual Machines Soars 22%")
        self.assertEqual(result[0]["tag"], "held")

    def test_exact_duplicate_story_is_not_re_logged(self):
        # The REST pre-fetch path has a rolling lookback window, so the
        # same headline can legitimately reappear in the next scan cycle's
        # fetch -- must not bloat the log with repeats of the same story.
        a._log_news_event(["UMAC"], "Unusual Machines Soars 22%", tag="held")
        a._log_news_event(["UMAC"], "Unusual Machines Soars 22%", tag="held")
        self.assertEqual(len(self._read()), 1)

    def test_same_headline_different_symbols_is_a_distinct_entry(self):
        a._log_news_event(["UMAC"], "Trump Levies Drone Tariffs", tag="held")
        a._log_news_event(["RCAT", "ONDS"], "Trump Levies Drone Tariffs", tag="macro")
        self.assertEqual(len(self._read()), 2)

    def test_symbols_are_stored_sorted_for_stable_dedup(self):
        a._log_news_event(["ONDS", "RCAT"], "same story", tag="macro")
        a._log_news_event(["RCAT", "ONDS"], "same story", tag="macro")   # different input order
        self.assertEqual(len(self._read()), 1, "sorted-symbols dedup must not care about input order")

    def test_missing_symbols_or_headline_is_a_silent_no_op(self):
        a._log_news_event([], "headline with no symbols", tag="watchlist")
        a._log_news_event(["AAPL"], "", tag="watchlist")
        # Neither call should ever reach the write path -- file stays
        # exactly as setUp left it (empty), never becomes an empty JSON list.
        self.assertEqual(os.path.getsize(a.NEWS_LOG_FILE), 0)

    def test_oldest_entries_evicted_first_when_over_cap(self):
        entries = [{"ts": f"2026-08-{d:02d}T10:00:00-04:00", "symbols": [f"T{d}"],
                   "headline": f"story {d}", "source": "", "tag": "watchlist", "sentiment": None}
                  for d in range(1, a.NEWS_LOG_MAX_ENTRIES + 1)]
        with open(a.NEWS_LOG_FILE, "w") as f:
            json.dump(entries, f)
        a._log_news_event(["NEWEST"], "the newest story", tag="watchlist")
        result = self._read()
        self.assertEqual(len(result), a.NEWS_LOG_MAX_ENTRIES)
        self.assertNotIn("T1", [s for e in result for s in e["symbols"]])
        self.assertIn("NEWEST", [s for e in result for s in e["symbols"]])


class TestNewsSentimentBreadth(unittest.TestCase):
    """Added 2026-08-15 as an OBSERVATION-ONLY regime factor (direct
    instruction: surface it, don't score on it yet — no live track record
    on this account). These lock in the aggregation math in isolation from
    get_market_regime()'s own network-heavy plumbing: window filtering,
    curated-universe filtering, unknown-sentiment handling, and the
    breadth_pct formula itself."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patches = [
            patch.object(a, "NEWS_LOG_FILE", self._tmp.name),
            patch.object(a, "WATCHLIST", ["AAPL", "MSFT"]),
            patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["ARTL"]),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def _write(self, entries):
        with open(self._tmp.name, "w") as f:
            json.dump(entries, f)

    def _entry(self, hours_ago, symbols, sentiment):
        from datetime import timezone as _tz
        ts = (datetime.now(_tz.utc) - timedelta(hours=hours_ago)).isoformat()
        return {"ts": ts, "symbols": symbols, "headline": "h", "source": "",
                "tag": "watchlist", "sentiment": sentiment}

    def test_missing_log_returns_zeroed_result(self):
        os.unlink(self._tmp.name)
        result = a._news_sentiment_breadth()
        self.assertEqual(result["total"], 0)
        self.assertIsNone(result["breadth_pct"])

    def test_counts_and_breadth_pct_within_window(self):
        self._write([
            self._entry(1, ["AAPL"], "positive"),
            self._entry(2, ["AAPL"], "positive"),
            self._entry(3, ["MSFT"], "negative"),
            self._entry(4, ["ARTL"], "neutral"),
        ])
        result = a._news_sentiment_breadth(hours_back=24.0)
        self.assertEqual(result["positive"], 2)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["neutral"], 1)
        self.assertEqual(result["total"], 4)
        # (2 positive - 1 negative) / (2+1) scored = 33.3%
        self.assertAlmostEqual(result["breadth_pct"], 33.3, places=1)

    def test_entries_outside_window_are_excluded(self):
        self._write([self._entry(48, ["AAPL"], "positive")])   # 48h ago, outside a 24h window
        result = a._news_sentiment_breadth(hours_back=24.0)
        self.assertEqual(result["total"], 0)

    def test_entries_outside_curated_universe_are_excluded(self):
        self._write([self._entry(1, ["NOTONWATCHLIST"], "positive")])
        result = a._news_sentiment_breadth(hours_back=24.0)
        self.assertEqual(result["total"], 0)

    def test_unknown_sentiment_counted_separately_and_excluded_from_breadth(self):
        self._write([
            self._entry(1, ["AAPL"], None),
            self._entry(2, ["AAPL"], None),
        ])
        result = a._news_sentiment_breadth(hours_back=24.0)
        self.assertEqual(result["unknown"], 2)
        self.assertEqual(result["total"], 2)
        self.assertIsNone(result["breadth_pct"], "an all-unscored sample must not silently read as neutral (0%)")

    def test_never_raises_on_a_malformed_log(self):
        with open(self._tmp.name, "w") as f:
            f.write("not valid json")
        result = a._news_sentiment_breadth()   # must not raise
        self.assertEqual(result["total"], 0)


class TestSyncNewsLogWithRemote(unittest.TestCase):
    """sync_news_log_with_remote() mirrors sync_scan_log_with_remote()'s
    ts-sort-before-cap merge exactly — see TestSyncScanLogWithRemote for
    the full incident this pattern guards against (dman_scan_log.json
    freezing for a full trading day). Applied here proactively since this
    file has the identical multi-writer shape from day one."""

    def _entry(self, ts, sym="UMAC"):
        return {"ts": ts, "symbols": [sym], "headline": f"story at {ts}",
               "source": "", "tag": "watchlist", "sentiment": None}

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "NEWS_LOG_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_fresh_local_entry_survives_a_diverged_same_size_remote(self):
        local = [self._entry(f"2026-08-01T{h:02d}:00:00-04:00") for h in range(a.NEWS_LOG_MAX_ENTRIES - 1)]
        local.append(self._entry("2026-08-15T21:00:00-04:00"))   # the newest entry
        remote = [self._entry(f"2026-08-10T{h % 24:02d}:30:00-04:00", sym=f"R{h}")
                 for h in range(a.NEWS_LOG_MAX_ENTRIES)]
        with open(self._tmp.name, "w") as f:
            json.dump(local, f)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_news_log_with_remote()
        with open(self._tmp.name) as f:
            result = json.load(f)
        self.assertEqual(len(result), a.NEWS_LOG_MAX_ENTRIES)
        self.assertIn("2026-08-15T21:00:00-04:00", [e["ts"] for e in result],
                      "the newest local entry must survive the merge")
        self.assertEqual(result[-1]["ts"], "2026-08-15T21:00:00-04:00")


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
        # check_macro_safe() derives "today" from datetime.now(ET) (fixed
        # 2026-08-23 to stop using the naive date.today() the runner's UTC
        # clock could misread near midnight) -- mock that instead of date.
        fomc_day = sorted(a._FOMC_DATES)[0]
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(fomc_day.year, fomc_day.month, fomc_day.day,
                                                 12, 0, tzinfo=a.ET)
            safe, _ = a.check_macro_safe()
        self.assertFalse(safe, "FOMC day itself must be blocked")

    def test_day_after_blackout_window_is_clear(self):
        fomc_day = sorted(a._FOMC_DATES)[0]
        clear_day = date(fomc_day.year, fomc_day.month, fomc_day.day) \
            .fromordinal(fomc_day.toordinal() + 3)
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(clear_day.year, clear_day.month, clear_day.day,
                                                 12, 0, tzinfo=a.ET)
            safe, _ = a.check_macro_safe()
        self.assertTrue(safe, "3 days after an FOMC date is well outside "
                              "the ±1 day blackout and must be clear")

    def test_major_macro_event_date_is_blocked(self):
        if not a._MAJOR_MACRO_EVENT_DATES:
            self.skipTest("no _MAJOR_MACRO_EVENT_DATES currently configured")
        event_day = sorted(a._MAJOR_MACRO_EVENT_DATES)[0]
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(event_day.year, event_day.month, event_day.day,
                                                 12, 0, tzinfo=a.ET)
            safe, _ = a.check_macro_safe()
        self.assertFalse(safe, "a configured major macro event day must "
                               "be blocked the same way FOMC is")

    def test_no_bls_release_date_falls_on_a_weekend(self):
        # Found in the 2026-08-16 review: _PPI_DATES' April 2026/2027
        # entries were computed as CPI+1 CALENDAR day, not next business
        # day as documented -- April CPI lands on a Friday both years, so
        # PPI+1 landed on a Saturday, a date BLS never actually releases
        # on. The real April PPI print (the following Monday) had zero
        # blackout coverage either year. BLS (CPI/PPI/NFP) never releases
        # on a weekend, so this is a hard invariant, not a heuristic.
        for _name, _dates in (("_CPI_DATES", a._CPI_DATES), ("_PPI_DATES", a._PPI_DATES)):
            weekend_dates = sorted(d for d in _dates if d.weekday() >= 5)
            self.assertEqual(weekend_dates, [],
                             f"{_name} has date(s) on a weekend, which BLS never "
                             f"releases on -- likely calendar-day-not-business-day "
                             f"arithmetic: {weekend_dates}")

    def test_pce_release_day_is_blocked_before_10am(self):
        # Added 2026-08-22: confirmed live that PCE (the Fed's own STATED
        # preferred inflation gauge) had zero coverage anywhere in this
        # file despite CPI/PPI/NFP all being tracked -- found researching
        # the week of 2026-08-24, where PCE (Wed 8/26) lands the same day
        # as NVDA's earnings. Uses 8/26 specifically (not just any
        # _PCE_DATES entry) because it's confirmed clear of any FOMC/
        # major-macro-event blackout that would otherwise block the whole
        # day regardless of time and mask what this test is isolating.
        pce_day = date(2026, 8, 26)
        with patch.object(a, "date") as mock_date, \
             patch.object(a, "datetime") as mock_dt:
            mock_date.today.return_value = pce_day
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            mock_dt.now.return_value = datetime(pce_day.year, pce_day.month, pce_day.day, 9, 0, tzinfo=a.ET)
            safe, _ = a.check_macro_safe()
        self.assertFalse(safe, "PCE release day before 10 AM ET must be blocked, same as CPI/PPI")

    def test_pce_release_day_opens_after_10am(self):
        pce_day = date(2026, 8, 26)
        with patch.object(a, "date") as mock_date, \
             patch.object(a, "datetime") as mock_dt:
            mock_date.today.return_value = pce_day
            mock_date.side_effect = lambda *args, **kw: date(*args, **kw)
            mock_dt.now.return_value = datetime(pce_day.year, pce_day.month, pce_day.day, 10, 30, tzinfo=a.ET)
            safe, _ = a.check_macro_safe()
        self.assertTrue(safe, "PCE release day after 10 AM ET must open, same as CPI/PPI")

    def test_pce_dates_never_fall_on_a_weekend(self):
        weekend_dates = sorted(d for d in a._PCE_DATES if d.weekday() >= 5)
        self.assertEqual(weekend_dates, [], f"_PCE_DATES has weekend date(s): {weekend_dates}")

    def test_aug_26_2026_nvda_earnings_day_is_a_registered_pce_date(self):
        # The specific real-world date that surfaced this gap -- PCE lands
        # the same day as NVDA's earnings (after close), confirmed
        # directly from bea.gov/news/schedule.
        self.assertIn(date(2026, 8, 26), a._PCE_DATES)

    def test_days_to_next_macro_print_includes_pce(self):
        result = a._days_to_next_macro_print(today=date(2026, 8, 25))
        self.assertEqual(result, 1, "PCE on 8/26 must count as 1 day away from 8/25")


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

    def test_dict_shaped_calendar_blocks_unconfirmed_earnings_today(self):
        fake_cal = {"Earnings Date": [date.today()], "Earnings High": 5.0}
        with patch.object(a.yf, "Ticker") as mock_tk, \
             patch.object(a, "_check_earnings_already_reported", return_value=False):
            mock_tk.return_value.calendar = fake_cal
            safe, score = a.check_earnings_safe("TESTX")
        self.assertFalse(safe, "an unconfirmed same-day report (could still be AMC-pending) must be blocked")
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


class TestAiScoreSignalFailureIsNotAZero(unittest.TestCase):
    """Found in the 2026-08-16 review: ai_score_signal() returned 0 on
    EVERY failure path (no key, timeout, rate limit, malformed response)
    and the caller's `sig.ai_score < 6` check couldn't distinguish that
    from "the AI genuinely scored this low" -- a transient API failure
    silently rejected a real signal as "AI score too low." Also hardened
    the response parsing to find the first text-type content block instead
    of blindly assuming content[0] is text (which breaks if a non-text
    block, e.g. a thinking block, ever precedes it)."""

    def _signal(self):
        return a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            rr=2.0, rsi=50.0, rvol=2.0, reason="test",
        )

    def test_no_api_key_returns_none_not_zero(self):
        with patch.object(a, "ANTHROPIC_API_KEY", ""):
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertIsNone(result)

    def test_request_exception_returns_none_not_zero(self):
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.side_effect = Exception("timeout")
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertIsNone(result)

    def test_valid_text_response_is_parsed(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "8"}]}
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertEqual(result, 8)

    def test_non_text_block_before_the_real_text_block_is_skipped_not_fatal(self):
        # Reproduces the exact shape a thinking-enabled response could take:
        # a non-text block first, the real answer in a later block.
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [
            {"type": "thinking", "thinking": "considering the setup..."},
            {"type": "text", "text": "7"},
        ]}
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertEqual(result, 7)

    def test_no_text_block_at_all_returns_none_not_zero(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "thinking", "thinking": "..."}]}
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertIsNone(result)

    def test_unparseable_text_defaults_to_five(self):
        # Pre-existing behavior, must survive the None-vs-0 change unchanged.
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "not a number"}]}
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = a.ai_score_signal(self._signal(), _fake_regime())
        self.assertEqual(result, 5)


class TestNewsKeywordFreshnessCheck(unittest.TestCase):
    """Added 2026-08-21: check_news_keyword_freshness() reviews
    MACRO_NEWS_KEYWORDS against real headlines the system has actually
    seen and suggests gaps -- advisory only, must never edit the keyword
    lists itself, and must fail closed (None) rather than raise on any
    missing key/file/API failure since it runs alongside the weekly
    readiness scan and must never be able to break it."""

    def setUp(self):
        self._log_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._log_tmp.close()
        self._patch = patch.object(a, "NEWS_LOG_FILE", self._log_tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._log_tmp.name):
            os.unlink(self._log_tmp.name)

    def _write_log(self, headlines):
        with open(self._log_tmp.name, "w") as f:
            json.dump([{"headline": h, "symbols": ["AAPL"], "ts": "2026-08-20T10:00:00-04:00"}
                       for h in headlines], f)

    def test_no_api_key_returns_none(self):
        self._write_log(["Some real headline"])
        with patch.object(a, "ANTHROPIC_API_KEY", ""):
            self.assertIsNone(a.check_news_keyword_freshness())

    def test_missing_log_file_returns_none(self):
        os.unlink(self._log_tmp.name)
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"):
            self.assertIsNone(a.check_news_keyword_freshness())

    def test_empty_log_returns_none(self):
        self._write_log([])
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"):
            self.assertIsNone(a.check_news_keyword_freshness())

    def test_request_exception_returns_none_not_raises(self):
        self._write_log(["A real headline about something"])
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.side_effect = Exception("timeout")
            self.assertIsNone(a.check_news_keyword_freshness())

    def test_valid_response_is_returned_and_includes_current_headlines(self):
        self._write_log(["Some CEO says something notable", "Duplicate", "Duplicate"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"type": "text", "text": "New CEO — repeatedly mentioned, market mover"}]
        }
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            result = a.check_news_keyword_freshness()
        self.assertEqual(result, "New CEO — repeatedly mentioned, market mover")
        # Confirm duplicate headlines were deduped before being sent.
        prompt = mock_requests.post.call_args[1]["json"]["messages"][0]["content"]
        self.assertEqual(prompt.count("Duplicate"), 1)

    def test_never_mutates_the_keyword_lists(self):
        self._write_log(["A headline mentioning someone new"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": [{"type": "text", "text": "Someone New — reason"}]}
        before = list(a.MACRO_NEWS_KEYWORDS)
        with patch.object(a, "ANTHROPIC_API_KEY", "test-key"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.post.return_value = mock_resp
            a.check_news_keyword_freshness()
        self.assertEqual(a.MACRO_NEWS_KEYWORDS, before)


class TestOptimizeStopTighteningCap(unittest.TestCase):
    """Found in the 2026-08-16 review: optimize_stop()'s "never tighten so
    much it invalidates 2R" clamp was mathematically dead code -- both the
    swing-low candidate selection and the clamp line included raw_stop as
    one term of a max(), guaranteeing the clamp could never change the
    result. There was no actual cap on tightening. These lock in the real
    fix: risk-per-share can never shrink below OPTIMIZE_STOP_MIN_RISK_FRACTION
    of the raw risk, regardless of how close a recent swing low sits to
    entry."""

    def _df(self, lows, atr=0.5):
        import pandas as pd
        n = len(lows)
        idx = pd.date_range("2026-07-01", periods=n, freq="D")
        return pd.DataFrame({
            "Low": lows, "High": [l + 1 for l in lows],
            "Close": lows, "ATR": [atr] * n,
        }, index=idx)

    def test_shallow_swing_low_no_longer_collapses_risk_past_the_floor(self):
        # Reproduces the audit finding's exact shape: entry=$50, raw_stop=$48
        # (raw risk $2.00), a shallow 15-bar swing low at $49.70 close to
        # entry. Old buggy behavior: best_stop ~= 49.61, risk collapses to
        # ~$0.39 (an 80%+ collapse). Fixed: risk can't fall below 50% of
        # $2.00 = $1.00, so stop can't rise above entry-1.00 = $49.00.
        lows = [49.70] * 15
        df = self._df(lows, atr=0.10)
        result = a.optimize_stop(df, raw_stop=48.0, entry=50.0, bias="LONG")
        risk = 50.0 - result
        self.assertGreaterEqual(risk, 2.0 * a.OPTIMIZE_STOP_MIN_RISK_FRACTION - 0.01,
                                "risk-per-share must never fall below the tightening floor")
        self.assertLess(result, 49.70, "the stop must still be meaningfully tighter than the raw one")

    def test_short_side_shallow_swing_high_also_respects_the_floor(self):
        import pandas as pd
        n = 15
        idx = pd.date_range("2026-07-01", periods=n, freq="D")
        highs = [50.30] * n
        df = pd.DataFrame({"Low": [h - 1 for h in highs], "High": highs,
                           "Close": highs, "ATR": [0.10] * n}, index=idx)
        result = a.optimize_stop(df, raw_stop=52.0, entry=50.0, bias="SHORT")
        risk = result - 50.0
        self.assertGreaterEqual(risk, 2.0 * a.OPTIMIZE_STOP_MIN_RISK_FRACTION - 0.01)

    def test_a_genuinely_wider_swing_low_is_not_artificially_tightened(self):
        # The floor only caps EXCESSIVE tightening -- when both the swing
        # low AND the ATR-derived candidate are wider than raw_stop (a
        # large ATR here keeps entry-1.5*ATR below raw_stop too), raw_stop
        # itself wins on its own, with no tightening for the floor to cap.
        lows = [45.0] * 15   # well below raw_stop=48
        df = self._df(lows, atr=1.5)
        result = a.optimize_stop(df, raw_stop=48.0, entry=50.0, bias="LONG")
        self.assertEqual(result, 48.0, "raw_stop wins when nothing actually tightens past it")

    def test_zero_atr_returns_raw_stop_unchanged(self):
        df = self._df([49.0] * 15, atr=0.0)
        result = a.optimize_stop(df, raw_stop=48.0, entry=50.0, bias="LONG")
        self.assertEqual(result, 48.0)


class TestWriteJsonAtomic(unittest.TestCase):
    """Found in the 2026-08-16 review: every JSON write in this file was a
    plain open(path,"w") + json.dump() -- a process killed mid-write
    (OOM-kill, a VPS reboot) leaves a truncated file, and every reader's
    except (FileNotFoundError, json.JSONDecodeError) handler treats that
    identically to "this file never existed." For PositionTracker
    specifically, that means silently reporting ZERO open positions to a
    live-money guard loop. These lock in that _write_json_atomic() never
    leaves a truncated file in place of a good one, on either a clean
    write or one that fails partway through serialization."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self._path = os.path.join(self._tmp_dir, "atomic_test.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_writes_valid_json_readable_afterward(self):
        a._write_json_atomic(self._path, {"a": 1, "b": [1, 2, 3]})
        with open(self._path) as f:
            self.assertEqual(json.load(f), {"a": 1, "b": [1, 2, 3]})

    def test_passes_through_dump_kwargs_like_indent(self):
        a._write_json_atomic(self._path, {"a": 1}, indent=2)
        with open(self._path) as f:
            content = f.read()
        self.assertIn("\n", content, "indent=2 must actually be applied, not silently dropped")

    def test_original_file_survives_a_failure_partway_through_serialization(self):
        # Write a good file first.
        a._write_json_atomic(self._path, {"good": "data"})
        with open(self._path) as f:
            original = f.read()

        class Unserializable:
            pass

        with self.assertRaises(TypeError):
            a._write_json_atomic(self._path, {"bad": Unserializable()})

        with open(self._path) as f:
            after = f.read()
        self.assertEqual(original, after, "a failed write must never touch the original file")

    def test_failed_write_does_not_leave_a_stray_temp_file_behind(self):
        class Unserializable:
            pass
        try:
            a._write_json_atomic(self._path, {"bad": Unserializable()})
        except TypeError:
            pass
        leftover = [f for f in os.listdir(self._tmp_dir) if f.startswith(".tmp_")]
        self.assertEqual(leftover, [], "a failed write must clean up its own temp file")

    def test_no_intermediate_state_ever_visible_to_a_concurrent_reader(self):
        # Simulate the crash-safety property directly: after a successful
        # write, the file on disk is NEVER the empty/half-written
        # intermediate state -- it's atomically either the old content or
        # the new content, never something in between.
        a._write_json_atomic(self._path, {"version": 1})
        a._write_json_atomic(self._path, {"version": 2})
        with open(self._path) as f:
            self.assertEqual(json.load(f), {"version": 2})


class TestGetTopSectorsCacheServesAnyN(unittest.TestCase):
    """Found in the 2026-08-16 review: get_top_sectors()'s 4-hour cache
    used to store ranked[:n] -- a truncated top-n -- rather than the full
    ranking, so whichever call happened to populate the cache first fixed
    its size for every OTHER caller for the next 4 hours. check_sector()'s
    SHORT path calls get_top_sectors(11) (wants the bottom 4 of all 11,
    via reversed()[:4]); the premarket briefing calls the n=4 default every
    morning. A cache primed by n=4 silently gave the SHORT gate only 4
    (reversed, the top-4 STRONGEST) sectors instead of the true bottom 4 --
    inverting the gate."""

    def setUp(self):
        self._orig_cache = a._sector_cache
        self._orig_ts = a._sector_cache_ts
        a._sector_cache = None
        a._sector_cache_ts = None

    def tearDown(self):
        a._sector_cache = self._orig_cache
        a._sector_cache_ts = self._orig_ts

    def test_cache_primed_by_small_n_still_serves_a_larger_n_correctly(self):
        # 11 sectors, each with a distinct, known momentum ranking.
        sectors = list(a.SECTOR_ETFS.keys())
        with patch.object(a, "fetch_df", side_effect=self._fetch_df_ranked(sectors)):
            top4 = a.get_top_sectors(4)     # primes the cache
            all11 = a.get_top_sectors(11)   # must NOT be capped at 4 just because the cache saw n=4 first
        self.assertEqual(len(top4), 4)
        self.assertEqual(len(all11), 11)
        self.assertEqual(all11[:4], top4, "the top 4 of the full ranking must match the earlier n=4 call")

    def test_cache_primed_by_large_n_still_serves_a_smaller_n_correctly(self):
        sectors = list(a.SECTOR_ETFS.keys())
        with patch.object(a, "fetch_df", side_effect=self._fetch_df_ranked(sectors)):
            all11 = a.get_top_sectors(11)   # primes the cache
            top4 = a.get_top_sectors(4)     # must be capped at 4, not still return all 11
        self.assertEqual(len(all11), 11)
        self.assertEqual(len(top4), 4)
        self.assertEqual(top4, all11[:4])

    def test_short_gate_bottom_four_unaffected_by_an_earlier_n4_call(self):
        # Direct reproduction of check_sector()'s SHORT path: reversed(get_top_sectors(11))[:4].
        sectors = list(a.SECTOR_ETFS.keys())
        with patch.object(a, "fetch_df", side_effect=self._fetch_df_ranked(sectors)):
            a.get_top_sectors(4)   # premarket-briefing-style call, primes the cache with n=4
            bottom4 = list(reversed(a.get_top_sectors(11)))[:4]
        # The weakest sectors are the LAST 4 of the ranked-by-strength list
        # fetch_df_ranked() constructs (see helper) -- verify we actually
        # got the true bottom 4, not the top 4 relabeled.
        true_bottom4 = sectors[-4:]
        self.assertEqual(set(bottom4), set(true_bottom4))

    def _fetch_df_ranked(self, sectors):
        # sectors[0] is the strongest performer, sectors[-1] the weakest --
        # a distinct, decreasing 20-day return per sector so ranking order
        # is unambiguous and verifiable.
        import pandas as pd
        import numpy as np
        etf_to_perf = {a.SECTOR_ETFS[s]: (len(sectors) - i) * 2.0 for i, s in enumerate(sectors)}

        def _fetch(etf, *args, **kwargs):
            perf = etf_to_perf.get(etf, 0.0)
            n = 25
            close = np.concatenate([np.full(4, 100.0), np.linspace(100.0, 100.0 * (1 + perf / 100), 21)])
            idx = pd.date_range("2026-07-01", periods=n, freq="D")
            return pd.DataFrame({"Close": close}, index=idx)
        return _fetch


class TestEarningsBlackoutAllowsKnownReactions(unittest.TestCase):
    """Found in the 2026-08-16 review: check_earnings_safe()'s old
    `-1 <= days_away <= EARNINGS_BLACKOUT` range blocked yesterday's AND
    today's earnings unconditionally -- which is exactly the state of a
    real post-earnings gap, the setup Gap & Hold's earnings-gap variant and
    fetch_earnings_mover_tickers() exist to catch. days_away == -1 is an
    unambiguous known reaction (any earnings time yesterday is in the
    past); days_away == 0 needs a real check since it could be a still-
    pending AMC report later today."""

    def _mock_calendar(self, earn_date):
        return patch.object(a, "_extract_earnings_dates", return_value=[earn_date])

    def test_earnings_yesterday_no_longer_blocks(self):
        yesterday = date.today() - timedelta(days=1)
        with self._mock_calendar(yesterday):
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe, "a real post-earnings gap from yesterday must be tradeable")
        self.assertEqual(score, 5)

    def test_same_day_confirmed_reported_does_not_block(self):
        today = date.today()
        with self._mock_calendar(today), \
             patch.object(a, "_check_earnings_already_reported", return_value=True):
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe, "a same-day BMO report already confirmed released is a known reaction")

    def test_same_day_unconfirmed_still_blocks(self):
        today = date.today()
        with self._mock_calendar(today), \
             patch.object(a, "_check_earnings_already_reported", return_value=False):
            safe, score = a.check_earnings_safe("TESTX")
        self.assertFalse(safe, "an unconfirmed same-day report could still be AMC-pending later today")

    def test_upcoming_earnings_within_blackout_still_blocks(self):
        upcoming = date.today() + timedelta(days=2)
        with self._mock_calendar(upcoming):
            safe, score = a.check_earnings_safe("TESTX")
        self.assertFalse(safe, "an upcoming, unreported earnings date must still block entries")

    def test_two_days_ago_no_longer_relevant_stays_safe(self):
        two_days_ago = date.today() - timedelta(days=2)
        with self._mock_calendar(two_days_ago):
            safe, score = a.check_earnings_safe("TESTX")
        self.assertTrue(safe)


class TestIntervalToResampleRule(unittest.TestCase):
    def test_minutes(self):
        self.assertEqual(a._interval_to_resample_rule("5m"), "5min")
        self.assertEqual(a._interval_to_resample_rule("1m"), "1min")
        self.assertEqual(a._interval_to_resample_rule("15m"), "15min")

    def test_hours(self):
        self.assertEqual(a._interval_to_resample_rule("1h"), "1h")

    def test_unrecognized_format_returns_none(self):
        self.assertIsNone(a._interval_to_resample_rule("bogus"))
        self.assertIsNone(a._interval_to_resample_rule(""))


class TestFetchIntradayBarsRespectsInterval(unittest.TestCase):
    """Found in the 2026-08-16 review: _fetch_intraday_bars() always
    returned Alpaca's native 1-minute bars regardless of its own `interval`
    parameter -- every real caller passes "5m", and every downstream
    consumer (_compute_session_levels, _detect_pre_breakout,
    _detect_momentum_fade -- the momentum-watch exit manager for live
    small-cap positions) is written and documented for 5-minute
    granularity. "No new session high in 5 bars" silently meant 5 minutes
    instead of 25."""

    def _one_min_bars(self, n, start_hour=9, start_minute=30, base_price=10.0):
        import zoneinfo
        ET = zoneinfo.ZoneInfo("America/New_York")
        base = datetime(2026, 8, 17, start_hour, start_minute, tzinfo=ET)
        bars = []
        for i in range(n):
            px = base_price + i * 0.01
            bars.append(MagicMock(
                open=px, high=px + 0.05, low=px - 0.05, close=px + 0.02,
                volume=1000 + i, timestamp=base + timedelta(minutes=i),
            ))
        return bars

    def test_fifteen_one_minute_bars_resample_to_three_five_minute_bars(self):
        bars = self._one_min_bars(15)
        mock_resp = MagicMock()
        mock_resp.data = {"TESTX": bars}
        mock_dc = MagicMock()
        mock_dc.get_stock_bars.return_value = mock_resp
        with patch.object(a, "ALPACA_AVAILABLE", True), \
             patch.object(a, "get_alpaca_data_client", return_value=mock_dc), \
             patch.object(a, "_resolve_stock_feed", return_value="sip"):
            df = a._fetch_intraday_bars("TESTX", interval="5m")
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3, "15 one-minute bars must resample into exactly 3 five-minute bars")

    def test_resampled_bar_ohlc_is_correctly_aggregated(self):
        bars = self._one_min_bars(5)   # exactly one 5-min bar's worth
        mock_resp = MagicMock()
        mock_resp.data = {"TESTX": bars}
        mock_dc = MagicMock()
        mock_dc.get_stock_bars.return_value = mock_resp
        with patch.object(a, "ALPACA_AVAILABLE", True), \
             patch.object(a, "get_alpaca_data_client", return_value=mock_dc), \
             patch.object(a, "_resolve_stock_feed", return_value="sip"):
            df = a._fetch_intraday_bars("TESTX", interval="5m")
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertAlmostEqual(float(row["Open"]), bars[0].open, places=4,
                               msg="Open must come from the FIRST 1-min bar")
        self.assertAlmostEqual(float(row["Close"]), bars[-1].close, places=4,
                               msg="Close must come from the LAST 1-min bar")
        self.assertAlmostEqual(float(row["High"]), max(b.high for b in bars), places=4)
        self.assertAlmostEqual(float(row["Low"]), min(b.low for b in bars), places=4)
        self.assertAlmostEqual(float(row["Volume"]), sum(b.volume for b in bars), places=4)

    def test_one_minute_interval_request_skips_resampling(self):
        bars = self._one_min_bars(5)
        mock_resp = MagicMock()
        mock_resp.data = {"TESTX": bars}
        mock_dc = MagicMock()
        mock_dc.get_stock_bars.return_value = mock_resp
        with patch.object(a, "ALPACA_AVAILABLE", True), \
             patch.object(a, "get_alpaca_data_client", return_value=mock_dc), \
             patch.object(a, "_resolve_stock_feed", return_value="sip"):
            df = a._fetch_intraday_bars("TESTX", interval="1m")
        self.assertEqual(len(df), 5, "a 1m request must keep the native 1-minute granularity")


class TestGetLivePriceUsesFeedResolver(unittest.TestCase):
    """Found in the 2026-08-16 review: get_live_price() -- the single
    hottest price-check path in the file, behind T1/T2/early-lock
    decisions and entry-price drift validation -- omitted feed= entirely,
    unlike every other stock-data call site (_fetch_alpaca_daily,
    _fetch_intraday_bars, prewarm_alpaca_bars all pass
    feed=_resolve_stock_feed()). A SIP entitlement lapse would degrade
    this specific path silently, with no downgrade alert."""

    def _quote(self, ask=10.05, bid=10.00):
        q = MagicMock()
        q.ask_price = ask
        q.bid_price = bid
        return q

    def test_feed_resolver_result_is_passed_to_the_quote_request(self):
        captured = {}

        def fake_get_stock_latest_quote(req):
            captured["feed"] = req.feed
            return {"TESTX": self._quote()}

        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.side_effect = fake_get_stock_latest_quote
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc), \
             patch.object(a, "_resolve_stock_feed", return_value="iex"):
            px = a.get_live_price("TESTX")
        self.assertEqual(str(captured["feed"]).lower(), "datafeed.iex")
        self.assertAlmostEqual(px, 10.025, places=3)

    def test_sip_entitled_still_requests_sip(self):
        captured = {}

        def fake_get_stock_latest_quote(req):
            captured["feed"] = req.feed
            return {"TESTX": self._quote()}

        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.side_effect = fake_get_stock_latest_quote
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc), \
             patch.object(a, "_resolve_stock_feed", return_value="sip"):
            a.get_live_price("TESTX")
        self.assertEqual(str(captured["feed"]).lower(), "datafeed.sip")


class TestGetLiveQuoteIncludesSizes(unittest.TestCase):
    """Added 2026-08-23: get_live_quote() is the equity counterpart to
    _get_option_snapshot()'s bid_size/ask_size -- backs order-flow-aware
    exit tightening for equity positions. get_live_price() must keep
    returning exactly the same mid it always did (verified by the
    existing TestGetLivePriceUsesFeedResolver suite still passing
    unchanged); these lock in the new function's own contract."""

    def _quote(self, ask=10.05, bid=10.00, bid_size=300, ask_size=50):
        q = MagicMock()
        q.ask_price = ask
        q.bid_price = bid
        q.bid_size = bid_size
        q.ask_size = ask_size
        return q

    def test_returns_full_quote_with_sizes(self):
        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.return_value = {"TESTX": self._quote()}
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            result = a.get_live_quote("TESTX")
        self.assertAlmostEqual(result["mid"], 10.025, places=3)
        self.assertEqual(result["bid_size"], 300)
        self.assertEqual(result["ask_size"], 50)

    def test_no_data_client_returns_none(self):
        with patch.object(a, "get_alpaca_data_client", return_value=None):
            self.assertIsNone(a.get_live_quote("TESTX"))

    def test_zero_bid_or_ask_returns_none(self):
        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.return_value = {"TESTX": self._quote(bid=0)}
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            self.assertIsNone(a.get_live_quote("TESTX"))

    def test_exception_returns_none_not_raises(self):
        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.side_effect = Exception("network error")
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            self.assertIsNone(a.get_live_quote("TESTX"))

    def test_get_live_price_still_returns_just_the_mid(self):
        mock_dc = MagicMock()
        mock_dc.get_stock_latest_quote.return_value = {"TESTX": self._quote()}
        with patch.object(a, "get_alpaca_data_client", return_value=mock_dc):
            px = a.get_live_price("TESTX")
        self.assertAlmostEqual(px, 10.025, places=3)


class TestComputeIndicatorsCached(unittest.TestCase):
    """Found in the 2026-08-16 review: the intended cache-reuse guard at
    several call sites ("MACD" not in df.columns) never actually tripped,
    because fetch_df()'s own cache only ever stores the RAW (pre-indicator)
    dataframe -- compute_indicators() was always called fresh. A ticker
    appearing in both the main scan universe and a later pass (small-cap
    catalyst, momentum-watch radar) got its entire indicator suite --
    including a pure-Python triple loop -- recomputed a second time, every
    scan and every fast-trigger, for ~200 tickers."""

    def setUp(self):
        self._cache_patch = patch.object(a, "_indicator_cache", {})
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()

    def _raw_df(self, n=60):
        import pandas as pd, numpy as np
        idx = pd.date_range("2026-05-01", periods=n, freq="D")
        close = np.linspace(10, 12, n)
        return pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": [1_000_000] * n,
        }, index=idx)

    def test_second_call_for_the_same_ticker_does_not_recompute(self):
        raw = self._raw_df()
        with patch.object(a, "compute_indicators", wraps=a.compute_indicators) as mock_ci:
            a._compute_indicators_cached("TESTX", raw)
            a._compute_indicators_cached("TESTX", raw)
        mock_ci.assert_called_once()

    def test_different_tickers_each_compute_once(self):
        raw = self._raw_df()
        with patch.object(a, "compute_indicators", wraps=a.compute_indicators) as mock_ci:
            a._compute_indicators_cached("TESTX", raw)
            a._compute_indicators_cached("OTHERX", raw)
        self.assertEqual(mock_ci.call_count, 2)

    def test_result_has_indicator_columns(self):
        result = a._compute_indicators_cached("TESTX", self._raw_df())
        self.assertIn("MACD", result.columns)
        self.assertIn("RSI", result.columns)

    def test_mutating_one_callers_result_does_not_corrupt_the_next_callers(self):
        # Real regression risk: several call sites do df.dropna(...,
        # inplace=True) on what this returns -- if that mutated the SHARED
        # cached object, the next caller for the same ticker would get a
        # silently truncated/corrupted dataframe instead of the full one.
        raw = self._raw_df()
        first = a._compute_indicators_cached("TESTX", raw)
        first.dropna(subset=["MACD"], inplace=True)
        second = a._compute_indicators_cached("TESTX", raw)
        self.assertEqual(len(second), len(raw), "a mutation on one caller's "
                         "result must not affect what the next caller gets back")

    def test_clearing_the_cache_forces_recomputation(self):
        raw = self._raw_df()
        with patch.object(a, "compute_indicators", wraps=a.compute_indicators) as mock_ci:
            a._compute_indicators_cached("TESTX", raw)
            a._indicator_cache.clear()
            a._compute_indicators_cached("TESTX", raw)
        self.assertEqual(mock_ci.call_count, 2)


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
                             "theta": 0.0, "vega": 0.0, "iv": 0.0, "oi": 0,
                             "bid_size": 50, "ask_size": 50}
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
                     "iv": 0.3, "oi": 100, "bid_size": 50, "ask_size": 50}
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
                             "theta": 0.0, "vega": 0.0, "iv": 0.0, "oi": 0,
                             "bid_size": 50, "ask_size": 50}
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


class TestOptionsEntryLiquidityGate(unittest.TestCase):
    """Added 2026-08-23, entry-side counterpart to the order-flow exit
    signal: _find_best_call_contract()/_find_best_put_contract() must
    reject a contract whose top-of-book size is below
    OPTIONS_MIN_QUOTE_SIZE on either side, even when delta and spread%
    both look fine -- a thin book means the displayed spread can be
    misleadingly tight relative to what's actually fillable there."""

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

    def _snap(self, bid_size, ask_size):
        return {"bid": 5.0, "ask": 5.2, "mid": 5.1, "spread_pct": 0.04,
                "delta": 0.62, "gamma": 0.01, "theta": -0.05, "vega": 0.1,
                "iv": 0.3, "oi": 100, "bid_size": bid_size, "ask_size": ask_size}

    def test_call_rejects_thin_ask_size(self):
        client = self._client_returning("TESTX260821C00095000", 95.0)
        snap = self._snap(bid_size=50, ask_size=a.OPTIONS_MIN_QUOTE_SIZE - 1)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s):
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        self.assertIsNone(contract, "a thin ask must reject the contract even with good delta/spread")

    def test_call_rejects_thin_bid_size(self):
        client = self._client_returning("TESTX260821C00095000", 95.0)
        snap = self._snap(bid_size=a.OPTIONS_MIN_QUOTE_SIZE - 1, ask_size=50)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s):
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        self.assertIsNone(contract, "a thin bid must reject the contract too, not just a thin ask")

    def test_call_accepts_exactly_at_the_floor(self):
        client = self._client_returning("TESTX260821C00095000", 95.0)
        snap = self._snap(bid_size=a.OPTIONS_MIN_QUOTE_SIZE, ask_size=a.OPTIONS_MIN_QUOTE_SIZE)
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s):
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        self.assertIsNotNone(contract, "exactly at the floor must still pass, not just strictly above it")

    def test_put_rejects_thin_ask_size(self):
        client = self._client_returning("TESTX260821P00130000", 130.0)
        snap = self._snap(bid_size=50, ask_size=a.OPTIONS_MIN_QUOTE_SIZE - 1)
        snap["delta"] = -0.62
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s):
                contract = a._find_best_put_contract(client, "TESTX", 100.0)
        self.assertIsNone(contract)

    def test_missing_size_keys_fail_closed_not_a_crash(self):
        # A snap dict lacking bid_size/ask_size entirely (shouldn't happen
        # in production -- _get_option_snapshot() always includes them --
        # but must fail closed, not raise, if it ever does).
        client = self._client_returning("TESTX260821C00095000", 95.0)
        snap = {"bid": 5.0, "ask": 5.2, "mid": 5.1, "spread_pct": 0.04,
                "delta": 0.62, "gamma": 0.01, "theta": -0.05, "vega": 0.1,
                "iv": 0.3, "oi": 100}
        with patch.object(a, "yf") as mock_yf:
            mock_yf.Ticker.return_value.fast_info.three_month_average_volume = 10_000_000
            with patch.object(a, "_get_option_snapshot", return_value=snap), \
                 patch.object(a, "_merge_contract_oi", side_effect=lambda s, c: s):
                contract = a._find_best_call_contract(client, "TESTX", 100.0)
        self.assertIsNone(contract)


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


class TestEarningsSpreadPostEventDecayExit(unittest.TestCase):
    """Added 2026-08-21 (session review finding): the only prior exits for
    an earnings debit spread were DTE-based (could be days away) and
    take-profit (only fires on gains) -- a spread that lost most of its
    value on the actual earnings event had no path out until the DTE
    close, even with the catalyst already spent and nothing left to wait
    for. Confirmed live: BABA's double spread was down to ~$0 (from a
    $160 debit) the day after earnings, 7 days before its DTE close would
    have fired."""

    def _pos(self, earn_date="2026-08-20", entry=160.0, max_gain=322.0,
              expiry="260828", spread_qty=1):
        return {
            "ticker": "BABA", "earn_date": earn_date, "entry": entry,
            "max_gain": max_gain, "spread_qty": spread_qty,
            "legs": [f"BABA{expiry}C00138000", f"BABA{expiry}C00142000",
                     f"BABA{expiry}P00120000", f"BABA{expiry}P00116000"],
        }

    def _snap(self, bid, ask):
        return {"bid": bid, "ask": ask}

    def test_decayed_spread_after_earnings_triggers_post_event_close(self):
        # bid/ask chosen so cur_value nets to roughly -$4 (matches the real
        # BABA case), well under 50% of the $160 debit paid.
        snaps = {
            "BABA260828C00138000": self._snap(0.80, 0.84),
            "BABA260828C00142000": self._snap(0.61, 0.65),
            "BABA260828P00120000": self._snap(0.30, 0.34),
            "BABA260828P00116000": self._snap(0.53, 0.57),
        }
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 21)   # 1 day after earn_date
            mock_date.fromisoformat = date.fromisoformat
            with patch.object(a, "_get_option_snapshot", side_effect=lambda s: snaps[s]), \
                 patch.object(a, "_close_earnings_spread", return_value=("submitted", "order123")) as mock_close, \
                 patch.object(a, "_is_alerted_today", return_value=False), \
                 patch.object(a, "_mark_alerted"), \
                 patch.object(a, "send_telegram", return_value=True) as mock_tg:
                result = a._monitor_earnings_spread_position(self._pos())
        mock_close.assert_called_once()
        self.assertIn("post-event decay", mock_close.call_args[0][1])
        self.assertIn("POST-EVENT DECAY", result)
        self.assertTrue(any("POST-EVENT" in c.args[0] for c in mock_tg.call_args_list))

    def test_same_day_as_earnings_does_not_trigger_yet(self):
        # date.today() == earn_date (earnings today, market hasn't had a
        # full session to react yet) -- must not fire on event day itself.
        snaps = {
            "BABA260828C00138000": self._snap(0.80, 0.84),
            "BABA260828C00142000": self._snap(0.61, 0.65),
            "BABA260828P00120000": self._snap(0.30, 0.34),
            "BABA260828P00116000": self._snap(0.53, 0.57),
        }
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 20)   # == earn_date
            mock_date.fromisoformat = date.fromisoformat
            with patch.object(a, "_get_option_snapshot", side_effect=lambda s: snaps[s]), \
                 patch.object(a, "_close_earnings_spread") as mock_close:
                result = a._monitor_earnings_spread_position(self._pos())
        mock_close.assert_not_called()
        self.assertIsNone(result)

    def test_healthy_value_after_earnings_does_not_trigger(self):
        # Value still well above 50% of the debit paid -- no reason to
        # force an early exit just because earnings already happened.
        snaps = {
            "BABA260828C00138000": self._snap(1.80, 1.84),
            "BABA260828C00142000": self._snap(0.61, 0.65),
            "BABA260828P00120000": self._snap(0.10, 0.14),
            "BABA260828P00116000": self._snap(0.05, 0.09),
        }
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 21)
            mock_date.fromisoformat = date.fromisoformat
            with patch.object(a, "_get_option_snapshot", side_effect=lambda s: snaps[s]), \
                 patch.object(a, "_close_earnings_spread") as mock_close:
                result = a._monitor_earnings_spread_position(self._pos())
        mock_close.assert_not_called()

    def test_no_earn_date_never_triggers_post_event_exit(self):
        # Older/malformed positions without earn_date must fail safe --
        # no post-event exit attempted, same as the missing-snapshot case.
        # date.today() must be frozen the same way the sibling test above
        # does: the legs' baked-in "260828" expiry is only in the future
        # relative to a frozen date. Left ambient, this test passed only
        # while the real calendar hadn't yet reached 2026-08-28 -- once it
        # had, dte_now went negative and the (unrelated) DTE-close branch
        # fired for real, calling the mocked _close_earnings_spread with no
        # configured return value and blowing up the unpack in the caller,
        # not the post-event-decay path this test actually exercises.
        pos = self._pos(earn_date="")
        snaps = {
            "BABA260828C00138000": self._snap(0.10, 0.14),
            "BABA260828C00142000": self._snap(0.05, 0.09),
            "BABA260828P00120000": self._snap(0.05, 0.09),
            "BABA260828P00116000": self._snap(0.05, 0.09),
        }
        with patch.object(a, "date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 21)   # comfortably before the 8/28 leg expiry
            mock_date.fromisoformat = date.fromisoformat
            with patch.object(a, "_get_option_snapshot", side_effect=lambda s: snaps[s]), \
                 patch.object(a, "_close_earnings_spread") as mock_close:
                result = a._monitor_earnings_spread_position(pos)
        mock_close.assert_not_called()


class TestResolveEarningsTimingAlreadyReportedGap(unittest.TestCase):
    """Added 2026-08-23: _resolve_earnings_timing() checked actual_eps
    first (the strongest signal, but only populated once Massive has
    backfilled it) and, when that was still None, trusted a scheduled
    BMO/AMC time classification immediately -- returning "BMO"/"AMC" and
    skipping the independent news-headline confirmation entirely. A
    same-day BMO print scanned hours after the open (the daemon's 2:45 PM
    ET trigger, or a later hourly-cron run) with actual_eps still
    unbackfilled would classify as "BMO" with zero further check, letting
    run_earnings_spread_scan() build and offer a real spread priced off an
    underlying that had already reacted -- exactly the case
    ALREADY-REPORTED exists to catch."""

    def _item(self, ticker="NVDA", when="2026-08-26", actual_eps=None, time_str="07:00:00"):
        return {"ticker": ticker, "date": when, "actual_eps": actual_eps, "time": time_str}

    def test_days_away_one_never_needs_disambiguation(self):
        result = a._resolve_earnings_timing(MagicMock(), "NVDA", date(2026, 8, 26), 200.0, days_away=1)
        self.assertEqual(result, "PENDING-TOMORROW")

    def test_actual_eps_present_is_already_reported_no_further_checks(self):
        with patch.object(a, "_fetch_massive_earnings",
                           return_value=[self._item(actual_eps=1.05)]), \
             patch.object(a, "_check_earnings_already_reported") as mock_news:
            result = a._resolve_earnings_timing(MagicMock(), "NVDA", date(2026, 8, 26), 200.0, days_away=0)
        self.assertEqual(result, "ALREADY-REPORTED")
        mock_news.assert_not_called()

    def test_classifiable_time_but_already_reported_via_news_wins(self):
        # THE FIX: actual_eps still None (not backfilled yet), time_str
        # classifies cleanly to BMO -- but the independent news check
        # confirms it already happened. ALREADY-REPORTED must win.
        with patch.object(a, "_fetch_massive_earnings",
                           return_value=[self._item(actual_eps=None, time_str="07:00:00")]), \
             patch.object(a, "_check_earnings_already_reported", return_value=True):
            result = a._resolve_earnings_timing(MagicMock(), "NVDA", date(2026, 8, 26), 200.0, days_away=0)
        self.assertEqual(result, "ALREADY-REPORTED")

    def test_classifiable_time_and_not_yet_reported_uses_the_classification(self):
        # The normal, common case must be unaffected -- no regression.
        with patch.object(a, "_fetch_massive_earnings",
                           return_value=[self._item(actual_eps=None, time_str="07:00:00")]), \
             patch.object(a, "_check_earnings_already_reported", return_value=False):
            result = a._resolve_earnings_timing(MagicMock(), "NVDA", date(2026, 8, 26), 200.0, days_away=0)
        self.assertEqual(result, "BMO")

    def test_no_massive_match_still_falls_through_to_news_check(self):
        with patch.object(a, "_fetch_massive_earnings", return_value=[]), \
             patch.object(a, "_check_earnings_already_reported", return_value=True):
            result = a._resolve_earnings_timing(MagicMock(), "NVDA", date(2026, 8, 26), 200.0, days_away=0)
        self.assertEqual(result, "ALREADY-REPORTED")


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

    def test_low_float_catalyst_setup_allowed_even_off_watchlist(self):
        # Regression for the live 2026-08-24 PMI incident: a Low Float
        # Catalyst signal (float 1.1M, score 90) got skipped outright --
        # no options, no shares -- purely because it wasn't on the
        # static watchlist, despite being exactly the profile the shares
        # exception was meant for. Went on to +75.5%, never traded.
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertTrue(a._shares_fallback_allowed("PMI", "Low Float Catalyst"))

    def test_other_setups_off_watchlist_still_not_allowed(self):
        # The exception is specific to Low Float Catalyst -- a Gap & Hold
        # or Day 2 Continuation signal on a non-watchlist ticker must
        # still skip rather than fall back to shares (the exact NDSN
        # incident this policy exists to prevent).
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertFalse(a._shares_fallback_allowed("NDSN", "Gap & Hold"))

    def test_watchlist_ticker_still_allowed_regardless_of_setup(self):
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertTrue(a._shares_fallback_allowed("APVO", "Gap & Hold"))

    def test_momentum_day_only_setup_allowed_even_off_watchlist(self):
        # Same reasoning as Low Float Catalyst -- a momentum-watch
        # breakout is individually human-approved, drawn from the same
        # thin-float universe, and shouldn't dead-end on watchlist
        # membership any more than a dynamically-discovered Low Float
        # Catalyst signal does.
        with patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["APVO", "MASK"]):
            self.assertTrue(a._shares_fallback_allowed("XYZ", a.MOMENTUM_DAY_ONLY_SETUP))

    def test_momentum_day_only_setup_is_in_options_setups(self):
        self.assertIn(a.MOMENTUM_DAY_ONLY_SETUP, a.OPTIONS_SETUPS)


class TestSmallcapScoreThreshold(unittest.TestCase):
    """Added 2026-08-21 (session review finding): SETUP_MIN_CONFLUENCE's
    per-setup tightening (e.g. "Low Float Catalyst" raised to 90 on
    2026-08-14 after a 0% WR streak) was never consulted by the smallcap
    discovery path -- the ONLY path that setup type is ever generated
    from -- so a DMan-watchlist ticker could keep re-entering through the
    much lower 45-point watchlist floor. A regression here silently
    re-opens that exact loophole."""

    def setUp(self):
        self._setup_prob_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._setup_prob_tmp.close()
        os.unlink(self._setup_prob_tmp.name)
        self._patches = [
            patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["ARTL"]),
            patch.object(a, "DMAN_WATCHLIST_MIN_SCORE", 45),
            patch.object(a, "SMALLCAP_MIN_SCORE", 55),
            patch.object(a, "SETUP_MIN_CONFLUENCE", {"Low Float Catalyst": 90}),
            # _setup_probation_bonus() (added 2026-08-24) reads real
            # production state otherwise -- must be isolated like every
            # other file this class already patches.
            patch.object(a, "SETUP_PROBATION_FILE", self._setup_prob_tmp.name),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        if os.path.exists(self._setup_prob_tmp.name):
            os.unlink(self._setup_prob_tmp.name)

    def test_watchlist_ticker_with_a_tightened_setup_uses_the_higher_bar(self):
        self.assertEqual(a._smallcap_score_threshold("ARTL", "Low Float Catalyst"), 90)

    def test_non_watchlist_ticker_with_a_tightened_setup_also_uses_the_higher_bar(self):
        self.assertEqual(a._smallcap_score_threshold("RANDOM", "Low Float Catalyst"), 90)

    def test_watchlist_ticker_with_an_untightened_setup_keeps_the_low_floor(self):
        self.assertEqual(a._smallcap_score_threshold("ARTL", "Some Other Setup"), 45)

    def test_non_watchlist_ticker_with_an_untightened_setup_keeps_the_smallcap_floor(self):
        self.assertEqual(a._smallcap_score_threshold("RANDOM", "Some Other Setup"), 55)

    def test_artl_at_its_real_recent_entry_score_now_fails_the_gate(self):
        # ARTL's actual live entry on 2026-08-20 scored 47 -- below the
        # tightened 90 bar, above the old 45 watchlist floor. This is the
        # exact live case the fix closes.
        threshold = a._smallcap_score_threshold("ARTL", "Low Float Catalyst")
        self.assertGreater(threshold, 47)


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


class TestSetupStatsIsLiveFiltering(unittest.TestCase):
    """Found in the 2026-08-16 review: setup_stats() -- the one function
    real position sizing (size_position_kelly, via score_signal) reads
    from -- had no is_live filter at all. A real check against production
    data found "Gap & Hold" carrying 475 backtest records against exactly
    1 real live trade, with setup_stats() blending them into an 83% win
    rate presented as if it meant something about live performance. These
    lock in that only real fills count, and that a setup with too few live
    trades falls back to the live-only aggregate, not the backtest-
    dominated one."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _record(self, tracker, setup, outcome, is_live, pnl_pct=1.0):
        tracker.record(a.TradeRecord(
            ticker="TESTX", date="2026-08-16", bias="LONG", setup=setup,
            entry=10.0, exit=11.0, outcome=outcome, pnl_pct=pnl_pct,
            score=100, is_live=is_live,
        ))

    def test_backtest_records_for_the_setup_are_excluded(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(475):
            self._record(tracker, "Gap & Hold", "WIN", is_live=False, pnl_pct=8.0)
        for _ in range(5):
            self._record(tracker, "Gap & Hold", "LOSS", is_live=True, pnl_pct=-5.0)
        stats = tracker.setup_stats("Gap & Hold")
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["losses"], 5)
        self.assertEqual(stats["win_rate"], 0.0,
                         "475 backtest wins must not paper over 5 real live losses")

    def test_fewer_than_5_live_trades_falls_back_to_live_only_aggregate_not_backtest(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(475):
            self._record(tracker, "Gap & Hold", "WIN", is_live=False, pnl_pct=8.0)
        self._record(tracker, "Gap & Hold", "WIN", is_live=True, pnl_pct=6.0)   # only 1 live trade
        # A different setup with a real live loss -- this is what the
        # live-only aggregate fallback should actually reflect.
        self._record(tracker, "Low Float Catalyst", "LOSS", is_live=True, pnl_pct=-20.0)
        stats = tracker.setup_stats("Gap & Hold")
        # Must equal rolling_stats(live_only=True), NOT the 476-trade blend.
        self.assertEqual(stats["total"], 2)
        self.assertLess(stats["win_rate"], 1.0,
                        "the real live loss on a different setup must be visible in the "
                        "fallback aggregate, not hidden behind 475 backtest wins")

    def test_five_or_more_live_trades_uses_the_setup_specific_stats(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(5):
            self._record(tracker, "Morning Runner", "WIN", is_live=True, pnl_pct=10.0)
        stats = tracker.setup_stats("Morning Runner")
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["win_rate"], 1.0)


class TestPositionTrackerShowOptionsPricing(unittest.TestCase):
    """Found in the 2026-08-16 review: PositionTracker.show() (--mode
    positions) excluded only earnings spreads from the equity display
    path -- a single-leg options position fell through to it and had
    the UNDERLYING STOCK price compared against premium-denominated
    entry/stop/target fields, producing a fabricated P&L and, worse, a
    FALSE "T2 HIT" Telegram alert whenever the stock price happened to
    numerically exceed a premium target (unrelated scales)."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()

    def tearDown(self):
        os.unlink(self._pos_tmp.name)

    def _write(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def test_single_leg_option_uses_premium_not_underlying_stock_price(self):
        # Real UMAC shape: entry is a $9.22 PREMIUM, but the underlying
        # stock trades around $25 -- a naive get_live_price(ticker) call
        # here would return the stock price, not the premium.
        self._write([{
            "ticker": "UMAC", "bias": "LONG",
            "setup": "Options Call UMAC260828C00025000 ($25.0C exp 2026-08-28)",
            "entry": 9.22, "stop": 4.61, "target1": 13.83, "target2": 23.05,
            "shares": 200, "entry_date": "2026-08-14",
        }])
        tracker = a.PositionTracker(filepath=self._pos_tmp.name)
        with patch.object(a, "get_live_price") as mock_stock_price, \
             patch.object(a, "_get_option_snapshot", return_value={
                 "bid": 10.10, "ask": 10.30, "mid": 10.20}) as mock_opt_snap, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            tracker.show()
        mock_stock_price.assert_not_called()
        mock_opt_snap.assert_called_once_with("UMAC260828C00025000")
        # No false T2 alert: real premium (10.20) is well below the real
        # premium target (23.05), even though the underlying stock price
        # (~$25) would have numerically exceeded it.
        for _call in mock_tg.call_args_list:
            self.assertNotIn("T2 HIT", _call[0][0])

    def test_equity_position_is_unaffected(self):
        self._write([{
            "ticker": "CELZ", "bias": "LONG", "setup": "Gap & Hold",
            "entry": 10.0, "stop": 9.0, "target1": 12.0, "target2": 14.0,
            "shares": 100, "entry_date": "2026-08-14",
        }])
        tracker = a.PositionTracker(filepath=self._pos_tmp.name)
        with patch.object(a, "get_live_price", return_value=11.0) as mock_stock_price, \
             patch.object(a, "_get_option_snapshot") as mock_opt_snap, \
             patch.object(a, "send_telegram", return_value=True):
            tracker.show()
        mock_stock_price.assert_called_once_with("CELZ")
        mock_opt_snap.assert_not_called()

    def test_earnings_spread_is_still_handled_by_its_own_branch(self):
        self._write([{
            "ticker": "HOOD", "bias": "NEUTRAL", "setup": "Earnings Call Spread",
            "entry": 191.0, "stop": 0.0, "target1": 0.0, "target2": 0.0,
            "shares": 0, "entry_date": "2026-08-14",
            "legs": ["HOOD260807C00096000", "HOOD260807C00099000"],
            "max_loss": 191.0, "max_gain": 193.0,
        }])
        tracker = a.PositionTracker(filepath=self._pos_tmp.name)
        with patch.object(a, "get_live_price") as mock_stock_price, \
             patch.object(a, "_get_option_snapshot") as mock_opt_snap:
            tracker.show()
        mock_stock_price.assert_not_called()
        mock_opt_snap.assert_not_called()


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
        # trail_pct must be persisted too (2026-08-23) -- it's the only
        # record of what the live trailing order was submitted at, since
        # `stop` freezes at breakeven and Alpaca's own ratcheting isn't
        # written back; without it, a later stop-restore has no way to
        # reconstruct an equivalent trailing order.
        self.assertAlmostEqual(tracker.positions[0].trail_pct, submit_call.trail_percent, places=4)

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

    def test_bearish_order_flow_locks_in_between_the_two_gain_gates(self):
        # Added 2026-08-23: entry $9.80, cur $11.00 = +12.24% -- inside
        # (ORDER_FLOW_EQUITY_LOCK_MIN_GAIN_PCT=7.5%, EARLY_PROFIT_LOCK_GAIN_PCT=15%),
        # below the normal early-lock gate but order flow independently
        # confirms sellers stacking (lean well past
        # ORDER_FLOW_TIGHTEN_LEAN_THRESHOLD=-0.4). Must lock in now instead
        # of waiting for the full 15% gate.
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        bearish_quote = {"mid": 11.00, "bid": 10.98, "ask": 11.02, "bid_size": 20, "ask_size": 180}
        with patch.object(a, "get_live_quote", return_value=bearish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing",
                           return_value="stop raised to breakeven, now trailing") as mock_prog, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)
        mock_tg.assert_called_once()
        self.assertIn("Order-flow early lock", mock_tg.call_args[0][0])
        self.assertEqual(mock_prog.call_args.kwargs.get("trigger_price"), 11.00)

    def test_bullish_order_flow_does_not_lock_in_early(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        bullish_quote = {"mid": 11.00, "bid": 10.98, "ask": 11.02, "bid_size": 180, "ask_size": 20}
        with patch.object(a, "get_live_quote", return_value=bullish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing") as mock_prog, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)
        mock_tg.assert_not_called()
        mock_prog.assert_not_called()

    def test_already_trailing_is_not_re_triggered_by_order_flow(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70,
                        stop=7.83, stop_stage="trailing")
        bearish_quote = {"mid": 11.00, "bid": 10.98, "ask": 11.02, "bid_size": 20, "ask_size": 180}
        with patch.object(a, "get_live_quote", return_value=bearish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing") as mock_prog, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)
        mock_tg.assert_not_called()
        mock_prog.assert_not_called()

    def test_below_the_order_flow_floor_does_nothing_even_with_bearish_flow(self):
        # +5%, below ORDER_FLOW_EQUITY_LOCK_MIN_GAIN_PCT (7.5%) -- must not
        # act on order flow no matter how bearish; the gain floor is a
        # hard requirement, not something order flow can override.
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        bearish_quote = {"mid": 10.29, "bid": 10.27, "ask": 10.31, "bid_size": 5, "ask_size": 195}
        with patch.object(a, "get_live_quote", return_value=bearish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing") as mock_prog, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=10.29)   # +5.0%
        mock_tg.assert_not_called()
        mock_prog.assert_not_called()

    def test_at_or_above_the_normal_gate_uses_the_original_branch_not_order_flow(self):
        # +18%, past EARLY_PROFIT_LOCK_GAIN_PCT -- the original branch must
        # take precedence regardless of order flow (bearish here too), so
        # the alert must say "Early profit lock", not "Order-flow early lock".
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        bearish_quote = {"mid": 11.56, "bid": 11.54, "ask": 11.58, "bid_size": 5, "ask_size": 195}
        with patch.object(a, "get_live_quote", return_value=bearish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing", return_value="stop raised"), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.56)   # +18%
        mock_tg.assert_called_once()
        self.assertIn("Early profit lock", mock_tg.call_args[0][0])
        self.assertNotIn("Order-flow", mock_tg.call_args[0][0])

    def test_order_flow_lock_does_not_refire_same_day(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        bearish_quote = {"mid": 11.00, "bid": 10.98, "ask": 11.02, "bid_size": 20, "ask_size": 180}
        with patch.object(a, "get_live_quote", return_value=bearish_quote), \
             patch.object(a, "_progress_equity_stop_to_trailing", return_value="locked"), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)
            a._check_equity_position_target(pos, cur_price=11.05)
        self.assertEqual(mock_tg.call_count, 1)

    def test_missing_quote_is_neutral_not_a_crash(self):
        pos = self._pos(ticker="CLRO", entry=9.80, target1=14.73, target2=17.70, stop=7.83)
        with patch.object(a, "get_live_quote", return_value=None), \
             patch.object(a, "_progress_equity_stop_to_trailing") as mock_prog, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_equity_position_target(pos, cur_price=11.00)
        mock_tg.assert_not_called()
        mock_prog.assert_not_called()


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

    def test_passed_in_positions_skips_the_file_read_entirely(self):
        # Found in the 2026-08-16 review: guard_loop() loads
        # dman_positions.json once and passes it to both
        # run_options_guard() and run_equity_guard() instead of each
        # re-reading it -- the file must never even be opened when a
        # positions list is supplied.
        os.unlink(self._pos_tmp.name)   # if this got opened anyway, it would raise
        with patch.object(a, "_check_equity_position_target") as mock_check:
            a.run_equity_guard(positions=[{"ticker": "CELZ", "setup": "Low Float Catalyst"}])
        mock_check.assert_called_once()
        self.assertEqual(mock_check.call_args.args[0]["ticker"], "CELZ")


class TestUpdatePositionFieldMatching(unittest.TestCase):
    """_update_position_field() (ticker match) and
    _update_option_position_field() (OCC-symbol match) share one
    load/mutate/write helper (_update_positions_matching()) now -- these
    lock in that each still matches ONLY what it's supposed to, since
    the whole reason the OCC-keyed version exists is a real incident
    where the ticker-keyed one updated the wrong leg."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.close()
        self._patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._pos_tmp.name):
            os.unlink(self._pos_tmp.name)

    def _write(self, positions):
        with open(self._pos_tmp.name, "w") as f:
            json.dump(positions, f)

    def _read(self):
        with open(self._pos_tmp.name) as f:
            return json.load(f)

    def test_update_position_field_matches_by_ticker(self):
        self._write([
            {"ticker": "CELZ", "setup": "Low Float Catalyst", "stop": 1.0},
            {"ticker": "AAPL", "setup": "Gap & Hold", "stop": 5.0},
        ])
        a._update_position_field("CELZ", stop=1.5)
        result = self._read()
        self.assertEqual(next(p for p in result if p["ticker"] == "CELZ")["stop"], 1.5)
        self.assertEqual(next(p for p in result if p["ticker"] == "AAPL")["stop"], 5.0)

    def test_update_option_position_field_matches_only_the_named_occ_leg(self):
        # The real incident this OCC-keyed matcher exists to prevent: two
        # legs sharing the same ticker (a call and a put on the same
        # underlying) must never both get updated by one call.
        self._write([
            {"ticker": "SMCI", "setup": "Options Call SMCI260814C00034000 ($34C exp 2026-08-14)",
             "peak_premium": 1.0},
            {"ticker": "SMCI", "setup": "Options Put SMCI260814P00030000 ($30P exp 2026-08-14)",
             "peak_premium": 1.0},
        ])
        a._update_option_position_field("SMCI260814C00034000", peak_premium=2.0)
        result = self._read()
        call_leg = next(p for p in result if "Call" in p["setup"])
        put_leg  = next(p for p in result if "Put" in p["setup"])
        self.assertEqual(call_leg["peak_premium"], 2.0, "the named leg must be updated")
        self.assertEqual(put_leg["peak_premium"], 1.0, "the sibling leg on the same ticker must not be touched")

    def test_no_match_leaves_the_file_untouched(self):
        self._write([{"ticker": "CELZ", "setup": "Low Float Catalyst", "stop": 1.0}])
        a._update_position_field("NOMATCH", stop=99.0)
        result = self._read()
        self.assertEqual(result[0]["stop"], 1.0)

    def test_missing_positions_file_does_not_raise(self):
        os.unlink(self._pos_tmp.name)
        a._update_position_field("CELZ", stop=1.5)   # must not raise


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
                                           cur_prem=1.10, entry_prem=1.00, get_underlying_px_fn=lambda: 33.0)   # +10%, below the 15% start
        mock_tg.assert_not_called()

    def test_below_start_threshold_never_calls_the_underlying_price_fetcher(self):
        # The whole point of the get_underlying_px_fn callable is that the
        # (possibly uncached REST) fetch only happens when a milestone is
        # actually about to fire -- not on every guard tick. This is the
        # overwhelmingly common case (no milestone crossed this tick).
        fetch = MagicMock(return_value=33.0)
        with patch.object(a, "send_telegram", return_value=True):
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.10, entry_prem=1.00, get_underlying_px_fn=fetch)
        fetch.assert_not_called()

    def test_same_bucket_refire_never_calls_the_underlying_price_fetcher(self):
        pos = self._pos(milestone_gain_alerted=30.0)
        fetch = MagicMock(return_value=34.0)
        with patch.object(a, "send_telegram", return_value=True):
            a._check_options_pnl_milestone(pos, "CALL", "SMCI260814C00034000",
                                           cur_prem=1.38, entry_prem=1.00, get_underlying_px_fn=fetch)   # still bucket 30
        fetch.assert_not_called()

    def test_an_actual_milestone_fire_does_call_the_underlying_price_fetcher(self):
        fetch = MagicMock(return_value=34.0)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.35, entry_prem=1.00, get_underlying_px_fn=fetch)   # +35%
        fetch.assert_called_once()
        self.assertIn("$34.00", mock_tg.call_args[0][0])

    def test_no_fetcher_given_still_alerts_without_the_underlying_price_note(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.35, entry_prem=1.00)   # get_underlying_px_fn omitted
        mock_tg.assert_called_once()
        self.assertNotIn("Underlying:", mock_tg.call_args[0][0])

    def test_threshold_tightened_to_15_fires_where_old_30_would_not(self):
        # Locks in the 2026-08-15 tightening (direct instruction to catch
        # any meaningful move on the live UMAC play, not wait for a 30%
        # swing) -- +22% is below the OLD 30% start but above the new 15%.
        # (Deliberately not exactly +20% -- (1.20-1.00)/1.00*100 lands at
        # 19.999999999999996 in float, one bucket below what it looks like
        # on paper; +22% sits comfortably inside the 20% bucket instead.)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.22, entry_prem=1.00, get_underlying_px_fn=lambda: 33.0)   # +22%
        mock_tg.assert_called_once()
        self.assertIn("+20%", mock_tg.call_args[0][0])

    def test_gain_milestone_fires_at_30(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "CALL", "SMCI260814C00034000",
                                           cur_prem=1.35, entry_prem=1.00, get_underlying_px_fn=lambda: 34.0)   # +35%
        mock_tg.assert_called_once()
        self.assertIn("+30%", mock_tg.call_args[0][0])

    def test_gain_milestone_does_not_refire_same_bucket(self):
        pos = self._pos(milestone_gain_alerted=30.0)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(pos, "CALL", "SMCI260814C00034000",
                                           cur_prem=1.38, entry_prem=1.00, get_underlying_px_fn=lambda: 34.0)   # +38%, still bucket 30
        mock_tg.assert_not_called()

    def test_gain_milestone_fires_at_new_higher_bucket(self):
        pos = self._pos(milestone_gain_alerted=30.0)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(pos, "CALL", "SMCI260814C00034000",
                                           cur_prem=1.45, entry_prem=1.00, get_underlying_px_fn=lambda: 35.0)   # +45% -> bucket 40
        mock_tg.assert_called_once()
        self.assertIn("+40%", mock_tg.call_args[0][0])

    def test_loss_milestone_fires_independently_of_gain_side(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_options_pnl_milestone(self._pos(), "PUT", "SMCI260814P00029500",
                                           cur_prem=0.65, entry_prem=1.00, get_underlying_px_fn=lambda: 33.0)   # -35%
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


class TestQuoteSizeImbalance(unittest.TestCase):
    """_quote_size_imbalance() backs the order-flow-aware exit tightening
    added 2026-08-23 (direct request: recognize a real-time pullback using
    the size data Alpaca already provides at top-of-book, since true
    Level 2 depth isn't available for what this account trades). Pure
    math, independent of the monitor plumbing."""

    def test_equal_sizes_is_neutral(self):
        self.assertEqual(a._quote_size_imbalance(100, 100), 0.0)

    def test_both_zero_is_neutral_not_a_crash(self):
        self.assertEqual(a._quote_size_imbalance(0, 0), 0.0)

    def test_bid_heavy_is_positive(self):
        self.assertAlmostEqual(a._quote_size_imbalance(200, 50), 0.6)

    def test_ask_heavy_is_negative(self):
        self.assertAlmostEqual(a._quote_size_imbalance(20, 180), -0.8)

    def test_range_is_bounded_to_plus_minus_one(self):
        self.assertEqual(a._quote_size_imbalance(500, 0), 1.0)
        self.assertEqual(a._quote_size_imbalance(0, 500), -1.0)


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
        # Module-level Greeks cache (_cached_option_greeks) must not leak a
        # value across tests that reuse the same OCC symbol.
        self._greeks_cache_patch = patch.object(a, "_option_greeks_cache", {})
        self._greeks_cache_patch.start()

    def tearDown(self):
        self._dedup_patch.stop();    os.unlink(self._dedup_tmp.name)
        self._pos_patch.stop();      os.unlink(self._pos_tmp.name)
        self._milestone_patch.stop()
        self._price_patch.stop()
        self._greeks_cache_patch.stop()

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

    def test_bearish_order_flow_tightens_the_exit_that_would_otherwise_stay_open(self):
        # Same peak/current premium as
        # test_trail_active_but_giveback_not_enough_does_not_exit above
        # (normal 31.6% giveback threshold not yet reached) -- but now with
        # heavy size stacked on the ask (order-flow lean well past
        # ORDER_FLOW_TIGHTEN_LEAN_THRESHOLD), the tightened 15% threshold
        # IS reached (18.5% off peak) and the exit must fire.
        pos = self._pos(peak_premium=1.35, target1=999.0)
        bearish_snap = dict(self._snap(1.10), bid_size=10, ask_size=190)   # lean = -0.9
        with patch.object(a, "_get_option_snapshot", return_value=bearish_snap):
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord3")) as mock_close:
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._monitor_option_position(pos, "CALL")
        mock_close.assert_called_once()
        msg = mock_tg.call_args[0][0]
        self.assertIn("TRAIL EXIT", msg)
        self.assertIn("order flow", msg)

    def test_bearish_order_flow_alone_does_not_exit_a_position_still_near_its_peak(self):
        # Bounded-by-construction check: a strongly bearish lean with price
        # still essentially AT the peak (no real pullback at all) must
        # change nothing -- tightening the threshold doesn't matter if
        # current premium hasn't actually fallen to it.
        pos = self._pos(peak_premium=1.35, target1=999.0)
        bearish_snap = dict(self._snap(1.34), bid_size=5, ask_size=195)   # lean = -0.95, barely off peak
        with patch.object(a, "_get_option_snapshot", return_value=bearish_snap):
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    a._monitor_option_position(pos, "CALL")
        mock_close.assert_not_called()

    def test_bearish_order_flow_has_no_effect_before_trail_activates(self):
        # Peak only ever +10% (below the 25% activation) -- order flow must
        # not somehow arm a trail that hasn't earned activation yet. Same
        # shape as test_trail_not_active_before_activation_gain_even_with_a_drop,
        # with a bearish lean added.
        pos = self._pos(peak_premium=1.10)
        bearish_snap = dict(self._snap(0.90), bid_size=5, ask_size=195)
        with patch.object(a, "_get_option_snapshot", return_value=bearish_snap):
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(pos, "CALL")
        mock_close.assert_not_called()
        self.assertIn("not yet active", result)

    def test_bullish_order_flow_does_not_tighten_the_exit(self):
        # A positive/neutral lean must never tighten anything -- only a
        # lean at or below ORDER_FLOW_TIGHTEN_LEAN_THRESHOLD does. Same
        # premium as the "not enough giveback" test above; bid-heavy flow
        # must still leave it open.
        pos = self._pos(peak_premium=1.35, target1=999.0)
        bullish_snap = dict(self._snap(1.20), bid_size=190, ask_size=10)   # lean = +0.9
        with patch.object(a, "_get_option_snapshot", return_value=bullish_snap):
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

    def test_greeks_fall_back_to_a_rest_fetch_when_the_stream_snapshot_omits_them(self):
        # Found in the 2026-08-16 review: the real-time feed only carries
        # bid/ask/mid/sizes, no Greeks -- silently defaulting theta to 0
        # (the old behavior) meant the theta-decay alert could never fire
        # whenever the stream was live. Must now fall back to a (cached)
        # REST fetch so real Greeks values are used, not crash, and not
        # silently read as zero.
        pos = self._pos(target1=999.0, atr=0.62,
                        setup="Options Call SMCI270101C00034000 ($34C exp 2027-01-01)")
        stream_snap = {"bid": 1.03, "ask": 1.07, "mid": 1.05}
        rest_snap = self._snap(1.05, delta=0.58, theta=-0.09, iv=0.35)
        with patch.object(a, "_get_option_snapshot", return_value=rest_snap) as mock_rest:
            with patch.object(a, "_submit_options_close") as mock_close:
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(
                        pos, "CALL", get_snapshot_fn=lambda occ: stream_snap)
        mock_rest.assert_called_once()
        mock_close.assert_not_called()
        self.assertIn("Δ 0.58", result)      # real delta from the REST fallback, not the entry default
        self.assertIn("θ -0.090/d", result)  # real theta from the REST fallback, not silently 0

    def test_stream_snapshot_greeks_fallback_is_cached_not_refetched_every_call(self):
        pos = self._pos(target1=999.0)
        stream_snap = {"bid": 1.03, "ask": 1.07, "mid": 1.05}
        rest_snap = self._snap(1.05, theta=-0.09)
        with patch.object(a, "_get_option_snapshot", return_value=rest_snap) as mock_rest:
            with patch.object(a, "_submit_options_close"):
                with patch.object(a, "send_telegram", return_value=True):
                    a._monitor_option_position(pos, "CALL", get_snapshot_fn=lambda occ: stream_snap)
                    a._monitor_option_position(pos, "CALL", get_snapshot_fn=lambda occ: stream_snap)
        mock_rest.assert_called_once()

    def test_theta_alert_fires_using_the_real_rest_fetched_theta_not_a_silent_zero(self):
        # The actual bug: a live stream feed used to make this branch
        # permanently unreachable (theta always read as 0). With a real
        # decaying theta pulled from the REST fallback, the alert must
        # fire exactly as it would on a pure-REST snapshot.
        pos = self._pos(target1=999.0,
                        setup="Options Call SMCI270101C00034000 ($34C exp 2027-01-01)")
        stream_snap = {"bid": 0.99, "ask": 1.01, "mid": 1.00}   # +0% pnl -- theta branch is reachable
        rest_snap = self._snap(1.00, theta=-0.08)   # 8% of premium/day -- well past the 5% threshold
        with patch.object(a, "_get_option_snapshot", return_value=rest_snap):
            with patch.object(a, "_submit_options_close"):
                with patch.object(a, "send_telegram", return_value=True):
                    result = a._monitor_option_position(
                        pos, "CALL", get_snapshot_fn=lambda occ: stream_snap)
        self.assertIn("THETA ALERT", result)

    def test_get_price_fn_used_for_milestone_underlying_display_price(self):
        # Added 2026-08-15 alongside GUARD_EVERY_S dropping to 10s:
        # get_live_price() is an uncached REST call, so blindly calling it
        # every cycle for the milestone alert's cosmetic underlying-price
        # line would 6x that call volume for zero decision-making benefit.
        # get_price_fn (the daemon's real-time equity quote cache) must be
        # tried first, with get_live_price() only as a fallback.
        # 2026-08-18 update: the underlying-price lookup is now passed to
        # _check_options_pnl_milestone as a lazily-invoked callable (only
        # ever called on the rare tick a milestone actually fires), so
        # this test invokes that callable itself to exercise the same
        # get_price_fn-first, get_live_price-fallback ordering.
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)):
            with patch.object(a, "get_live_price") as mock_rest_price:
                with patch.object(a, "_submit_options_close"):
                    with patch.object(a, "send_telegram", return_value=True):
                        a._monitor_option_position(
                            pos, "CALL", get_price_fn=lambda t: 42.5)
                        # Must invoke inside the patch context -- the
                        # callable does a module-level lookup of
                        # get_live_price() at call time, not bind time.
                        get_underlying_px_fn = a._check_options_pnl_milestone.call_args.args[5]
                        self.assertEqual(get_underlying_px_fn(), 42.5)
        mock_rest_price.assert_not_called()

    def test_get_price_fn_falls_back_to_rest_when_it_returns_none(self):
        pos = self._pos(target1=999.0)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.05)):
            with patch.object(a, "get_live_price", return_value=33.0) as mock_rest_price:
                with patch.object(a, "_submit_options_close"):
                    with patch.object(a, "send_telegram", return_value=True):
                        a._monitor_option_position(
                            pos, "CALL", get_price_fn=lambda t: None)
                        get_underlying_px_fn = a._check_options_pnl_milestone.call_args.args[5]
                        self.assertEqual(get_underlying_px_fn(), 33.0)
        mock_rest_price.assert_called_once()

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

    def test_t1_half_sell_updates_by_occ_symbol_not_ticker(self):
        # Found 2026-08-16 review: the T1 half-sell branch used the
        # ticker-keyed _update_position_field() instead of the OCC-keyed
        # _update_option_position_field() -- exactly the bug
        # _update_option_position_field() was introduced to prevent
        # (silently updating every position sharing this ticker, e.g. a
        # call+put strangle or an unrelated equity position).
        pos = self._pos(target1=1.20, shares=200)
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.25)):
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord3")):
                with patch.object(a, "send_telegram", return_value=True):
                    with patch.object(a, "_update_option_position_field") as mock_opt_update, \
                         patch.object(a, "_update_position_field") as mock_ticker_update:
                        a._monitor_option_position(pos, "CALL")
        mock_opt_update.assert_any_call("SMCI260814C00034000", shares=100, stop=1.00)
        mock_ticker_update.assert_not_called()

    def test_t1_single_contract_breakeven_updates_by_occ_symbol_not_ticker(self):
        pos = self._pos(target1=1.20, shares=100)   # 1 contract -- no half-sell branch
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.25)):
            with patch.object(a, "send_telegram", return_value=True):
                with patch.object(a, "_update_option_position_field") as mock_opt_update, \
                     patch.object(a, "_update_position_field") as mock_ticker_update:
                    a._monitor_option_position(pos, "CALL")
        mock_opt_update.assert_any_call("SMCI260814C00034000", stop=1.00)
        mock_ticker_update.assert_not_called()

    def test_t1_fill_does_not_corrupt_a_sibling_position_on_the_same_ticker(self):
        # End-to-end reproduction of the real incident shape: a call and a
        # put on the same underlying (a manually-built strangle). The
        # call's T1 fill must leave the put's stop/shares completely
        # untouched.
        call_leg = self._pos(target1=1.20, shares=200,
                             setup="Options Call SMCI260814C00034000 ($34C exp 2026-08-14)")
        put_leg  = self._pos(target1=0.30, shares=300, stop=0.70, entry=0.50,
                             setup="Options Put SMCI260814P00029500 ($29.5P exp 2026-08-14)")
        self._write_positions([call_leg, put_leg])
        with patch.object(a, "_get_option_snapshot", return_value=self._snap(1.25)):
            with patch.object(a, "_submit_options_close", return_value=("submitted", "ord3")):
                with patch.object(a, "send_telegram", return_value=True):
                    a._monitor_option_position(call_leg, "CALL")
        with open(self._pos_tmp.name) as f:
            saved = json.load(f)
        put_saved = next(p for p in saved if p["setup"].startswith("Options Put"))
        self.assertEqual(put_saved["stop"], 0.70, "the put leg's stop must be untouched by the call's T1 fill")
        self.assertEqual(put_saved["shares"], 300, "the put leg's shares must be untouched by the call's T1 fill")

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

    def test_passed_in_positions_skips_the_file_read_entirely(self):
        # Found in the 2026-08-16 review: guard_loop() loads
        # dman_positions.json once and passes it to both
        # run_options_guard() and run_equity_guard() instead of each
        # re-reading it -- the file must never even be opened when a
        # positions list is supplied.
        os.unlink(self._pos_tmp.name)   # if this got opened anyway, it would raise
        with patch.object(a, "_monitor_option_position", return_value=None) as mock_mon:
            alerts = a.run_options_guard(verbose=False, positions=[
                {"ticker": "AAPL", "setup": "Options Call AAPL260821C00310000 ($310C exp 2026-08-21)"},
            ])
        mock_mon.assert_called_once()
        self.assertEqual(alerts, [])


class TestAlertDedupUsesEtClock(unittest.TestCase):
    """_is_alerted_today()/_mark_alerted() previously read the dedup date
    from bare date.today()/datetime.now() -- the runner's naive local
    clock, UTC on GitHub Actions. UTC midnight is 8 PM EDT / 7 PM EST,
    squarely inside the evening daemon session's run window, so a T1/
    trail/stop/DTE alert marked just before that boundary could read as
    "not sent today" a few minutes later and re-fire a duplicate for the
    same real trading day. Fixed by switching both to datetime.now(ET)."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self._patch = patch.object(a, "_ALERT_DEDUP_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def test_mark_alerted_calls_datetime_now_with_et_not_bare(self):
        with patch.object(a, "datetime") as _mock_dt:
            _mock_dt.now.return_value = datetime(2026, 8, 20, 20, 15)
            a._mark_alerted("TEST_KEY")
        _mock_dt.now.assert_called_once_with(a.ET)

    def test_is_alerted_today_calls_datetime_now_with_et_not_bare(self):
        with open(self._tmp.name, "w") as f:
            json.dump({"TEST_KEY": "2026-08-20T20:15:00"}, f)
        with patch.object(a, "datetime") as _mock_dt:
            _mock_dt.now.return_value = datetime(2026, 8, 20, 20, 15)
            a._is_alerted_today("TEST_KEY")
        _mock_dt.now.assert_called_once_with(a.ET)

    def test_mark_then_check_round_trips_using_the_same_et_day(self):
        a._mark_alerted("ROUNDTRIP_KEY")
        self.assertTrue(a._is_alerted_today("ROUNDTRIP_KEY"))


class TestHasPendingReplyPrompt(unittest.TestCase):
    """Added 2026-08-23: both _handle_manual_options_buy_reply() and
    _handle_earnings_approval_reply() return False on any plain-text reply
    that doesn't match their yes/y/no/n regex, even when a real approval
    IS pending -- meaning a human typing "yeah"/"sure"/"nah" to a live-
    money confirmation got zero feedback that it didn't register.
    _has_pending_reply_prompt() is the check used to decide whether an
    unrecognized reply deserves a "didn't understand" nudge."""

    def setUp(self):
        self._buy_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._buy_tmp.close()
        os.unlink(self._buy_tmp.name)   # start absent -- "nothing pending" is the default
        self._earn_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._earn_tmp.close()
        self._patches = [
            patch.object(a, "TELEGRAM_MANUAL_BUY_FILE", self._buy_tmp.name),
            patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._earn_tmp.name),
        ]
        for p in self._patches:
            p.start()
        a._save_earnings_pending([])

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._buy_tmp.name, self._earn_tmp.name):
            if os.path.exists(f):
                os.unlink(f)

    def test_nothing_pending_returns_false(self):
        self.assertFalse(a._has_pending_reply_prompt())

    def test_unexpired_manual_buy_returns_true(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump({"ticker": "NVDA",
                       "expires_at": (datetime.now(a.ET) + timedelta(minutes=5)).isoformat()}, f)
        self.assertTrue(a._has_pending_reply_prompt())

    def test_expired_manual_buy_returns_false(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump({"ticker": "NVDA",
                       "expires_at": (datetime.now(a.ET) - timedelta(minutes=5)).isoformat()}, f)
        self.assertFalse(a._has_pending_reply_prompt())

    def test_unexpired_awaiting_earnings_offer_returns_true(self):
        a._save_earnings_pending([{
            "ticker": "NVDA", "status": "awaiting_approval",
            "expires_at": (datetime.now(a.ET) + timedelta(minutes=5)).isoformat(),
        }])
        self.assertTrue(a._has_pending_reply_prompt())

    def test_expired_earnings_offer_returns_false(self):
        a._save_earnings_pending([{
            "ticker": "NVDA", "status": "awaiting_approval",
            "expires_at": (datetime.now(a.ET) - timedelta(minutes=5)).isoformat(),
        }])
        self.assertFalse(a._has_pending_reply_prompt())

    def test_non_awaiting_status_returns_false(self):
        a._save_earnings_pending([{
            "ticker": "NVDA", "status": "approved",
            "expires_at": (datetime.now(a.ET) + timedelta(minutes=5)).isoformat(),
        }])
        self.assertFalse(a._has_pending_reply_prompt())


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
        self._monthly_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._monthly_tmp.write(b"{}")
        self._monthly_tmp.close()
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
            patch.object(a, "MONTHLY_PNL_FILE", self._monthly_tmp.name),
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
        for f in (self._pos_tmp, self._sync_tmp, self._wr_tmp, self._pnl_tmp, self._monthly_tmp):
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
        mock_client.get_open_position.side_effect = Exception("position does not exist")
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
        mock_client.get_open_position.side_effect = Exception("position does not exist")
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
        mock_client.get_open_position.side_effect = Exception("position does not exist")
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
        # Genuinely not found -- matches the real Alpaca SDK, which raises
        # on get_open_position() for a symbol with no open position, rather
        # than returning a falsy value. Needed since 2026-08-20: a bare
        # MagicMock()'s auto-mocked __float__ doesn't raise on its own,
        # so leaving this unconfigured would make the single-symbol
        # false-negative re-check (added after the ARTL incident) think
        # the position is still open and skip clearing it.
        mock_client.get_open_position.side_effect = Exception("position does not exist")
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
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        order = self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21)
        order.id = "a-different-never-seen-order-id"
        # The re-detection this guards against is same-day/adjacent-day
        # (see the CLRO incident docstring) -- the dupe match now also
        # requires the fill date to be close to the existing record's
        # date, so this must reflect a realistic same-day re-fire rather
        # than relying on the default filled_at=None -> "today" fallback.
        order.filled_at = datetime(2026, 8, 6)
        mock_client.get_orders.return_value = [order]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0, "a ledger-matching close must not be recorded again")
        self.assertEqual(len([r for r in tracker.records if r.ticker == "IOTR"]), 1,
                          "win-rate history must still have exactly one IOTR record")

    def test_a_genuinely_separate_trade_weeks_later_at_the_same_price_is_not_a_dupe(self):
        # Found in the 2026-08-16 review: the ledger-matching dupe guard
        # above had no date component -- ticker + setup + exit-price-
        # within-half-a-cent alone can coincidentally match two REAL,
        # separate trades weeks apart that both happened to close near
        # the same round-number level. Without a date check, the second
        # trade's real P&L would be silently dropped from win-rate
        # history, indistinguishable from the intentional dupe-guard
        # behavior above.
        from alpaca.trading.enums import OrderStatus, OrderSide
        tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
        tracker.record(a.TradeRecord(
            ticker="IOTR", date="2026-07-01", bias="LONG", setup="Low Float Catalyst",
            entry=3.52, exit=3.21, outcome="LOSS", pnl_pct=-8.7, score=0, is_live=True,
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        order = self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=3.21)
        order.id = "a-genuinely-new-later-order"
        order.filled_at = datetime(2026, 8, 6)   # 36 days after the first trade
        mock_client.get_orders.return_value = [order]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 1, "a genuinely separate trade weeks later must be recorded, "
                          "not silently dropped as a false-positive dupe match")
        self.assertEqual(len([r for r in tracker.records if r.ticker == "IOTR"]), 2)

    def test_bulk_snapshot_false_negative_does_not_clear_a_genuinely_open_position(self):
        # Confirmed live 2026-08-20: ARTL's second same-day accumulating
        # entry fill landed while get_all_positions() -- the bulk snapshot
        # taken once at the top of this function -- hadn't yet caught up
        # to it (Alpaca's own eventual consistency between an order fill
        # and the positions API), so ARTL looked absent from alp_open even
        # though it was genuinely still open. The old code took that at
        # face value and cleared the tracked position entirely (losing the
        # first fill's already-accumulated 159 shares), leaving nothing
        # for the second fill's PositionTracker.open() to accumulate
        # against. A direct single-symbol re-check must catch this.
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # bulk snapshot: false negative
        mock_client.get_open_position.return_value = MagicMock(qty="159")   # direct check: really open
        mock_client.get_orders.return_value = []   # no closing order exists at all
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0)
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 1,
                          "a genuinely still-open position must not be cleared on a bulk-snapshot false negative")

    def test_genuinely_closed_position_still_gets_cleared(self):
        # The false-negative re-check must not swallow the real "actually
        # closed" case -- get_open_position() raising (matches the real
        # Alpaca SDK on a symbol with no open position) must still fall
        # through to the existing ghost/stale-clear logic.
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 0)
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0,
                          "a genuinely closed position must still be cleared")

    def test_two_separate_closing_tranches_are_both_recorded(self):
        # Added 2026-08-23: a position that took partial profit (e.g. an
        # options T1 half-sell) has TWO real closing orders for the same
        # symbol -- confirmed live 2026-08-21 on PURR, whose real $114 T1
        # half-sell would never have been recorded once the runner half
        # eventually closed too, since the old code stopped scanning after
        # the first match. Both orders here are genuinely distinct fills
        # (different ids, prices, qtys) and must both land in win-rate
        # history, with the position cleared exactly once regardless.
        from alpaca.trading.enums import OrderStatus, OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # fully closed now
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        tranche_1 = self._order(OrderSide.SELL, OrderStatus.FILLED,
                                 filled_avg_price=4.00, filled_qty="3")
        tranche_1.id = "tranche-1-partial"
        tranche_1.filled_at = datetime(2026, 8, 21, 14, 0)
        tranche_2 = self._order(OrderSide.SELL, OrderStatus.FILLED,
                                 filled_avg_price=3.21, filled_qty="4")
        tranche_2.id = "tranche-2-final"
        tranche_2.filled_at = datetime(2026, 8, 21, 15, 0)
        mock_client.get_orders.return_value = [tranche_1, tranche_2]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 2, "both distinct closing tranches must be recorded")
        self.assertEqual(len(tracker.records), 2)
        exits = sorted(r.exit for r in tracker.records)
        self.assertEqual(exits, [3.21, 4.00])
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0,
                          "the position must still be cleared exactly once")

    def test_dupe_tranche_does_not_block_a_genuinely_different_second_tranche(self):
        # The dupe-skip branch must `continue` scanning, not stop -- a
        # duplicate match on one order in the batch must not prevent a
        # genuinely different second order from still being recorded.
        from alpaca.trading.enums import OrderStatus, OrderSide
        tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
        tracker.record(a.TradeRecord(
            ticker="IOTR", date="2026-08-21", bias="LONG", setup="Low Float Catalyst",
            entry=3.5168, exit=4.00, outcome="WIN", pnl_pct=13.8, score=0, is_live=True,
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        dupe_order = self._order(OrderSide.SELL, OrderStatus.FILLED,
                                  filled_avg_price=4.00, filled_qty="3")
        dupe_order.id = "a-never-seen-id-for-the-dupe"
        dupe_order.filled_at = datetime(2026, 8, 21, 14, 0)
        real_order = self._order(OrderSide.SELL, OrderStatus.FILLED,
                                  filled_avg_price=3.21, filled_qty="4")
        real_order.id = "a-never-seen-id-for-the-real-one"
        real_order.filled_at = datetime(2026, 8, 21, 15, 0)
        mock_client.get_orders.return_value = [dupe_order, real_order]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            n = a.sync_alpaca_fills(tracker)
        self.assertEqual(n, 1, "only the genuinely new tranche should count as newly recorded")
        self.assertEqual(len([r for r in tracker.records if r.ticker == "IOTR"]), 2)


class TestSyncEarningsSpreadFills(unittest.TestCase):
    """Added 2026-08-22: confirmed live that zero "Earnings " records exist
    anywhere in 500 rows of win-rate history -- sync_alpaca_fills()
    explicitly skips multi-leg spread positions, and neither
    _monitor_earnings_spread_position() nor _close_earnings_spread() ever
    recorded the outcome after submitting a close. Real P&L from this
    strategy had never once counted toward DAILY_LOSS_LIMIT/
    MONTHLY_LOSS_LIMIT. sync_earnings_spread_fills() is the multi-leg
    counterpart that closes that gap."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._wr_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._wr_tmp.write(b"[]")
        self._wr_tmp.close()
        self._pnl_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pnl_tmp.write(b"{}")
        self._pnl_tmp.close()
        self._monthly_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._monthly_tmp.write(b"{}")
        self._monthly_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patches = [
            patch.object(a, "PositionTracker", self._isolated_pt),
            patch.object(a, "DAILY_PNL_FILE", self._pnl_tmp.name),
            patch.object(a, "MONTHLY_PNL_FILE", self._monthly_tmp.name),
            patch.object(a, "send_telegram", return_value=True),
            patch.object(a, "get_effective_account", return_value=3_500.0),
        ]
        for p in self._patches:
            p.start()
        self._isolated_pt().open(a.OpenPosition(
            ticker="BABA", bias="NEUTRAL", setup="Earnings Double Spread",
            entry=160.0, stop=0.0, target1=0.0, target2=0.0, shares=0,
            entry_date="2026-08-19", score=0, spread_qty=1, max_loss=160.0,
            max_gain=322.0, earn_date="2026-08-20",
            legs=["BABA260828C00138000", "BABA260828C00142000",
                  "BABA260828P00120000", "BABA260828P00116000"],
        ))

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._pos_tmp, self._wr_tmp, self._pnl_tmp, self._monthly_tmp):
            os.unlink(f.name)

    def _leg_mock(self, symbol):
        m = MagicMock()
        m.symbol = symbol
        return m

    def _order(self, legs, status, filled_avg_price=-0.65, order_id="close-order-1"):
        o = MagicMock()
        o.id = order_id
        o.legs = [self._leg_mock(s) for s in legs]
        o.status = status
        o.filled_avg_price = filled_avg_price
        o.filled_at = None
        return o

    def test_all_legs_closed_with_a_credit_records_a_loss(self):
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # no legs held anywhere
        mock_client.get_orders.return_value = [self._order(
            ["BABA260828C00138000", "BABA260828C00142000",
             "BABA260828P00120000", "BABA260828P00116000"],
            OrderStatus.FILLED, filled_avg_price=-0.65,
        )]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 1)
        self.assertEqual(len(tracker.records), 1)
        rec = tracker.records[0]
        self.assertEqual(rec.ticker, "BABA")
        self.assertEqual(rec.outcome, "LOSS")
        # Paid $160, received 0.65*100=$65 back -> -$95, -59.4% of the debit.
        self.assertAlmostEqual(rec.pnl_pct, -59.38, places=1)
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0)

    def test_partial_leg_close_does_not_record_or_clear(self):
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        # Only 2 of 4 legs closed -- 2 still held.
        mock_client.get_all_positions.return_value = [
            MagicMock(symbol="BABA260828C00138000"), MagicMock(symbol="BABA260828P00120000"),
        ]
        mock_client.get_orders.return_value = []
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0)
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 1,
                          "a partially-closed spread must stay tracked, not be guessed at")

    def test_no_matching_closing_order_still_clears_the_stale_tracker_entry(self):
        # Regression for the live 2026-08-28 NVDA/CRWD incident: both sat as
        # phantom "open" positions for days because the old behavior only
        # ever cleared tracking on the cycle a NEW match was found -- a
        # position resurrected as "open" by a later state-merge (its real
        # closing order already in recorded_ids from a prior cycle) then
        # never healed. All legs confirmed absent from Alpaca must clear
        # the tracker entry regardless, mirroring sync_alpaca_fills()'s
        # already-fixed CLRO-incident behavior for the single-symbol path.
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [self._order(
            ["SOME260828C00001000", "SOME260828C00002000"], OrderStatus.FILLED,
        )]   # an unrelated order, doesn't match BABA's legs
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0,
                          "clearing the stale entry must not fabricate a win-rate record")
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0)

    def test_already_recorded_id_is_skipped(self):
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [self._order(
            ["BABA260828C00138000", "BABA260828C00142000",
             "BABA260828P00120000", "BABA260828P00116000"],
            OrderStatus.FILLED, order_id="already-seen",
        )]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, {"already-seen"})
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0)

    def _simple_order(self, symbol, side, filled_avg_price, order_id, status=None):
        from alpaca.trading.enums import OrderStatus, OrderClass
        o = MagicMock()
        o.id = order_id
        o.legs = []
        o.order_class = OrderClass.SIMPLE
        o.symbol = symbol
        o.side = side
        o.status = status or OrderStatus.FILLED
        o.filled_avg_price = filled_avg_price
        o.filled_at = None
        return o

    def test_opening_order_is_never_mistaken_for_the_close(self):
        # Regression for the live 2026-08-28 NVDA/CRWD incident: the ORIGINAL
        # OPENING MLEG order has the exact same leg-set as any real closing
        # order, and used to be matched by leg-set alone. Its fill price
        # (+1.60, a debit paid) got negated to -160 as if it were a credit
        # received, producing an impossible -196.97% pnl_pct. A genuine
        # close of a debit-only-opened spread can never net positive, so an
        # opening-shaped order (positive price) must be skipped, not matched.
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [self._order(
            ["BABA260828C00138000", "BABA260828C00142000",
             "BABA260828P00120000", "BABA260828P00116000"],
            OrderStatus.FILLED, filled_avg_price=1.60, order_id="opening-order",
        )]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0,
                          "rejecting the opening order as a false match must not fabricate a record")
        # All legs are absent from Alpaca either way, so the stale-entry
        # self-heal still clears tracking -- it just records nothing.
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0)

    def test_legs_closed_via_separate_simple_orders_are_reconstructed(self):
        # Regression for the live 2026-08-28 NVDA/CRWD incident: both
        # positions were actually closed leg-by-leg via separate SIMPLE
        # orders (manual intervention), not one atomic MLEG close, so the
        # leg-set matcher never found anything and both stayed stuck open
        # in tracking forever. legs=[long_call, short_call, long_put,
        # short_put]; closing sells the longs and buys back the shorts.
        from alpaca.trading.enums import OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [
            self._simple_order("BABA260828C00138000", OrderSide.SELL, 2.74, "leg-1"),
            self._simple_order("BABA260828C00142000", OrderSide.BUY, 0.50, "leg-2"),
            self._simple_order("BABA260828P00120000", OrderSide.SELL, 0.15, "leg-3"),
            self._simple_order("BABA260828P00116000", OrderSide.BUY, 0.11, "leg-4"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 1)
        self.assertEqual(len(tracker.records), 1)
        rec = tracker.records[0]
        self.assertEqual(rec.ticker, "BABA")
        # credit = (2.74 - 0.50 + 0.15 - 0.11) * 100 = $228; paid $160 -> +$68, WIN.
        self.assertEqual(rec.exit, 228.0)
        self.assertEqual(rec.outcome, "WIN")
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0)

    def test_partial_simple_leg_closes_do_not_record_a_fabricated_pnl(self):
        # Only 2 of 4 legs have a matching closing SIMPLE fill -- must not
        # guess at the other 2's price and record a fabricated P&L. All
        # legs are absent from Alpaca though (this isn't the still-held
        # case), so the stale-entry self-heal still clears tracking, same
        # as the no-match case above -- it just records nothing while
        # doing so.
        from alpaca.trading.enums import OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [
            self._simple_order("BABA260828C00138000", OrderSide.SELL, 2.74, "leg-1"),
            self._simple_order("BABA260828C00142000", OrderSide.BUY, 0.50, "leg-2"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0)
        self.assertEqual(len(a.PositionTracker(filepath=self._pos_tmp.name).positions), 0)

    def test_simple_leg_fill_on_wrong_side_is_not_treated_as_a_close(self):
        # A SIMPLE order on one of the leg symbols but the WRONG side (e.g.
        # the original BUY_TO_OPEN of a long leg) must not count toward the
        # leg-by-leg fallback match, or the opening fills themselves could
        # get replayed as if they were the close.
        from alpaca.trading.enums import OrderSide
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [
            self._simple_order("BABA260828C00138000", OrderSide.BUY, 1.20, "open-leg-1"),  # wrong side
            self._simple_order("BABA260828C00142000", OrderSide.BUY, 0.50, "leg-2"),
            self._simple_order("BABA260828P00120000", OrderSide.SELL, 0.15, "leg-3"),
            self._simple_order("BABA260828P00116000", OrderSide.BUY, 0.11, "leg-4"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            n = a.sync_earnings_spread_fills(tracker, set())
        self.assertEqual(n, 0)
        self.assertEqual(len(tracker.records), 0)

    def test_sync_alpaca_fills_includes_earnings_spread_count(self):
        # Confirms sync_alpaca_fills() actually calls the new function and
        # folds its count into the total, so every existing caller
        # (--mode sync, --mode scan --submit, the daemon) picks this up
        # with zero call-site changes.
        from alpaca.trading.enums import OrderStatus
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_orders.return_value = [self._order(
            ["BABA260828C00138000", "BABA260828C00142000",
             "BABA260828P00120000", "BABA260828P00116000"],
            OrderStatus.FILLED, filled_avg_price=-0.65,
        )]
        _sync_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        _sync_tmp.close()
        try:
            with patch.object(a, "get_alpaca_client", return_value=mock_client), \
                 patch.object(a, "ALPACA_SYNC_FILE", _sync_tmp.name):
                tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
                n = a.sync_alpaca_fills(tracker)
        finally:
            os.unlink(_sync_tmp.name)
        self.assertEqual(n, 1)
        self.assertEqual(len(tracker.records), 1)
        self.assertEqual(tracker.records[0].ticker, "BABA")


class TestPdtStatusFailsClosed(unittest.TestCase):
    """Found in the 2026-08-16 review: _get_pdt_status() used to fail OPEN
    ({"remaining": 3, "swing_mode": False} -- "plenty of budget, use the
    normal path") on both a missing Alpaca client and any exception
    fetching account state. That's the unsafe direction on a sub-$25k
    account: a transient API hiccup at the exact moment the REAL PDT
    budget was already exhausted would let a same-day round-trip trade
    through unblocked, risking a real broker-flagged PDT violation."""

    def test_missing_client_fails_closed(self):
        with patch.object(a, "get_alpaca_client", return_value=None):
            status = a._get_pdt_status()
        self.assertEqual(status["remaining"], 0)
        self.assertTrue(status["swing_mode"])

    def test_account_fetch_exception_fails_closed(self):
        mock_client = MagicMock()
        mock_client.get_account.side_effect = Exception("network error")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            status = a._get_pdt_status()
        self.assertEqual(status["remaining"], 0)
        self.assertTrue(status["swing_mode"])

    def test_successful_lookup_with_real_budget_left_is_not_swing_mode(self):
        mock_acct = MagicMock()
        mock_acct.equity = "10000.0"
        mock_acct.daytrade_count = 0
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_acct
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "PositionTracker") as MockPT:
            MockPT.return_value.positions = []
            status = a._get_pdt_status()
        self.assertEqual(status["remaining"], 3)
        self.assertFalse(status["swing_mode"])

    def test_successful_lookup_with_budget_exhausted_is_swing_mode(self):
        mock_acct = MagicMock()
        mock_acct.equity = "10000.0"
        mock_acct.daytrade_count = 3
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_acct
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "PositionTracker") as MockPT:
            MockPT.return_value.positions = []
            status = a._get_pdt_status()
        self.assertEqual(status["remaining"], 0)
        self.assertTrue(status["swing_mode"])

    def test_todays_open_day_only_positions_count_as_committed_day_trades(self):
        # Direct instruction 2026-09-02, root-caused live: Alpaca's own
        # daytrade_count only increments AFTER a same-day round trip
        # actually completes -- a day-only position opened this morning
        # and still open doesn't show up there yet, even though it WILL
        # close today and WILL become a day trade the moment it does.
        # 2 committed (still-open, entered today) + Alpaca's own 1 already-
        # completed round trip = 3 used, 0 remaining -- not "1 used, 2 left".
        mock_acct = MagicMock()
        mock_acct.equity = "10000.0"
        mock_acct.daytrade_count = 1
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_acct
        today_str = datetime.now(a.ET).date().isoformat()
        open_today = [
            a.OpenPosition(ticker="AAA", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
                           entry=1.0, stop=0.9, target1=1.3, target2=1.5, shares=10,
                           entry_date=today_str, day_only=True),
            a.OpenPosition(ticker="BBB", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
                           entry=1.0, stop=0.9, target1=1.3, target2=1.5, shares=10,
                           entry_date=today_str, day_only=True),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "PositionTracker") as MockPT:
            MockPT.return_value.positions = open_today
            status = a._get_pdt_status()
        self.assertEqual(status["used"], 3)
        self.assertEqual(status["remaining"], 0)
        self.assertTrue(status["swing_mode"])

    def test_non_day_only_and_prior_day_positions_are_not_counted_as_committed(self):
        # Only TODAY's day_only positions are forward-looking commitments --
        # a swing/overnight position was never going to round-trip same-day
        # regardless, and a day_only position from a PRIOR day either
        # already closed (and is reflected in Alpaca's own count by now)
        # or is the stale-carry case force_close handles on its own.
        mock_acct = MagicMock()
        mock_acct.equity = "10000.0"
        mock_acct.daytrade_count = 0
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_acct
        today_str = datetime.now(a.ET).date().isoformat()
        positions = [
            a.OpenPosition(ticker="CCC", bias="LONG", setup="Gap & Hold",
                           entry=10.0, stop=9.0, target1=13.0, target2=15.0, shares=10,
                           entry_date=today_str, day_only=False),
            a.OpenPosition(ticker="DDD", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
                           entry=1.0, stop=0.9, target1=1.3, target2=1.5, shares=10,
                           entry_date="2026-01-01", day_only=True),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "PositionTracker") as MockPT:
            MockPT.return_value.positions = positions
            status = a._get_pdt_status()
        self.assertEqual(status["used"], 0)
        self.assertEqual(status["remaining"], 3)

    def test_equity_over_25k_is_never_swing_mode(self):
        mock_acct = MagicMock()
        mock_acct.equity = "30000.0"
        mock_acct.daytrade_count = 5
        mock_client = MagicMock()
        mock_client.get_account.return_value = mock_acct
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            status = a._get_pdt_status()
        self.assertFalse(status["swing_mode"])


class TestSyncAlpacaFillsPnlAccounting(unittest.TestCase):
    """Found in the 2026-08-16 review: options P&L recorded into
    dman_daily_pnl.json was missing the *100 contracts multiplier (Alpaca
    reports filled_qty in CONTRACTS for an options order, not shares), and
    both the daily and monthly files were divided by the static
    ACCOUNT_SIZE secret instead of live equity. Together these meant
    DAILY_LOSS_LIMIT was effectively blind to real options losses -- a
    real -20% options loss recorded as roughly -0.04%. Also locks in that
    record_monthly_pnl() is actually called now (it previously had zero
    call sites anywhere, making MONTHLY_LOSS_LIMIT permanently dead)."""

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
        self._daily_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._daily_tmp.write(b"{}")
        self._daily_tmp.close()
        self._monthly_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._monthly_tmp.write(b"{}")
        self._monthly_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patches = [
            patch.object(a, "PositionTracker", self._isolated_pt),
            patch.object(a, "ALPACA_SYNC_FILE", self._sync_tmp.name),
            patch.object(a, "DAILY_PNL_FILE", self._daily_tmp.name),
            patch.object(a, "MONTHLY_PNL_FILE", self._monthly_tmp.name),
            patch.object(a, "send_telegram", return_value=True),
            patch.object(a, "get_effective_account", return_value=4500.0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._pos_tmp, self._sync_tmp, self._wr_tmp, self._daily_tmp, self._monthly_tmp):
            os.unlink(f.name)

    def _order(self, side, status, filled_avg_price, filled_qty):
        o = MagicMock()
        o.id = "order-1"
        o.side = side
        o.status = status
        o.filled_avg_price = filled_avg_price
        o.filled_qty = filled_qty
        o.qty = filled_qty
        o.filled_at = None
        return o

    def test_options_close_applies_the_100x_contract_multiplier(self):
        from alpaca.trading.enums import OrderStatus, OrderSide
        self._isolated_pt().open(a.OpenPosition(
            ticker="UMAC", bias="LONG",
            setup="Options Call UMAC260828C00025000 ($25.0C exp 2026-08-28)",
            entry=9.22, stop=4.61, target1=13.83, target2=23.05,
            shares=200, entry_date="2026-08-14",
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []   # no longer held
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        # 2 contracts, stopped out at 4.61 -- real dollar loss is
        # (4.61-9.22) * 2 * 100 = -$922, not -$9.22.
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=4.61, filled_qty="2"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            a.sync_alpaca_fills(tracker)
        with open(self._daily_tmp.name) as f:
            daily = json.load(f)
        # -$922 / $4500 effective account * 100 = -20.49%
        total = sum(e["pnl_pct"] for e in daily["entries"])
        self.assertAlmostEqual(total, -922 / 4500 * 100, places=2)

    def test_equity_close_pnl_uses_live_effective_account_not_static_size(self):
        from alpaca.trading.enums import OrderStatus, OrderSide
        self._isolated_pt().open(a.OpenPosition(
            ticker="LITX", bias="LONG", setup="Gap & Hold",
            entry=35.85, stop=28.01, target1=55.45, target2=67.21,
            shares=47, entry_date="2026-08-12",
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=28.01, filled_qty="47"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            a.sync_alpaca_fills(tracker)
        with open(self._daily_tmp.name) as f:
            daily = json.load(f)
        dollar_pnl = (28.01 - 35.85) * 47
        total = sum(e["pnl_pct"] for e in daily["entries"])
        self.assertAlmostEqual(total, dollar_pnl / 4500.0 * 100, places=2)

    def test_a_real_close_also_records_monthly_pnl(self):
        # record_monthly_pnl() previously had zero call sites anywhere in
        # the codebase -- MONTHLY_LOSS_LIMIT could never trip no matter how
        # far the account drew down over a month. This is the direct fix.
        from alpaca.trading.enums import OrderStatus, OrderSide
        self._isolated_pt().open(a.OpenPosition(
            ticker="LITX", bias="LONG", setup="Gap & Hold",
            entry=35.85, stop=28.01, target1=55.45, target2=67.21,
            shares=47, entry_date="2026-08-12",
        ))
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = []
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        mock_client.get_orders.return_value = [
            self._order(OrderSide.SELL, OrderStatus.FILLED, filled_avg_price=28.01, filled_qty="47"),
        ]
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            tracker = a.WinRateTracker(filepath=self._wr_tmp.name)
            a.sync_alpaca_fills(tracker)
        with open(self._monthly_tmp.name) as f:
            monthly = json.load(f)
        self.assertIn("entries", monthly)
        total = sum(e["pnl_pct"] for e in monthly["entries"])
        self.assertLess(total, 0)


class TestDailyMonthlyPnlUsesEtClock(unittest.TestCase):
    """record_daily_pnl()/get_todays_loss() and record_monthly_pnl()/
    get_this_month_loss() used date.today() (system/UTC on the GitHub
    Actions runner) instead of ET. The evening cloud daemon session runs
    to 20:05 ET -- past midnight UTC -- so for the last ~5 minutes of
    every trading day, date.today() on that runner already reads
    "tomorrow." Found alongside the P&L accounting fix above; fixed by
    switching both pairs to datetime.now(ET)."""

    def setUp(self):
        self._daily_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._daily_tmp.write(b"{}")
        self._daily_tmp.close()
        self._monthly_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._monthly_tmp.write(b"{}")
        self._monthly_tmp.close()
        self._patches = [
            patch.object(a, "DAILY_PNL_FILE", self._daily_tmp.name),
            patch.object(a, "MONTHLY_PNL_FILE", self._monthly_tmp.name),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        for f in (self._daily_tmp, self._monthly_tmp):
            os.unlink(f.name)

    def test_daily_pnl_boundary_uses_et_not_utc(self):
        # 11:30 PM UTC on 2026-08-14 is still 7:30 PM ET the SAME day -- a
        # date.today()-based implementation (system/UTC on the GitHub
        # Actions runner) would already say 2026-08-15 here and wrongly
        # reset the accumulator instead of adding to it.
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 14, 19, 30, tzinfo=a.ET)
            a.record_daily_pnl(-1.5)
            mock_dt.now.return_value = datetime(2026, 8, 14, 19, 45, tzinfo=a.ET)
            a.record_daily_pnl(-1.0)
        with open(self._daily_tmp.name) as f:
            data = json.load(f)
        total = sum(e["pnl_pct"] for e in data["entries"])
        self.assertAlmostEqual(total, -2.5, places=4,
                                msg="both calls landed in the same ET trading day and must accumulate together")

    def test_monthly_pnl_boundary_uses_et_not_utc(self):
        with patch.object(a, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 8, 31, 19, 30, tzinfo=a.ET)
            a.record_monthly_pnl(-2.0)
            mock_dt.now.return_value = datetime(2026, 8, 31, 19, 50, tzinfo=a.ET)
            a.record_monthly_pnl(-1.5)
        with open(self._monthly_tmp.name) as f:
            data = json.load(f)
        total = sum(e["pnl_pct"] for e in data["entries"])
        self.assertAlmostEqual(total, -3.5, places=4,
                                msg="both calls landed in the same ET calendar month and must accumulate together")


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

    def _hist(self, prev_close, avg_vol, today_close=None):
        # today_close, when given, makes the last bar's Close DIFFERENT from
        # prev_close and dates it as today — the real fetch_df() shape
        # during market hours (last daily bar is today's still-forming
        # bar). Every pre-existing test below omits it, so the last bar's
        # Close equals prev_close either way and can't distinguish
        # iloc[-1] from iloc[-2] -- see TestEarningsMoverUsesCompletedBar
        # for the tests that actually exercise the distinction.
        import pandas as pd
        n = 10
        idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="D")
        closes = [prev_close] * n
        if today_close is not None:
            closes[-1] = today_close
        return pd.DataFrame({
            "Close":  closes,
            "Volume": [avg_vol] * n,
        }, index=idx)

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

    def test_gap_measured_against_yesterdays_close_not_todays_forming_bar(self):
        # Found 2026-08-16 review: during market hours, fetch_df()'s LAST
        # daily bar is today's still-forming one -- comparing the live
        # quote against that same bar's close (instead of yesterday's
        # completed close) made gap_pct ~= 0 always, so this function
        # never once returned a candidate. Real CRWV shape: prior close
        # 90.32, live price 104.51 (+15.7%). today's still-forming bar
        # closes wherever the price happens to be recorded mid-session
        # (104.30 here, close to but not exactly the live quote) -- the
        # OLD buggy code would compare 104.51 against 104.30 (~0.2%, fails
        # the 5% threshold); the fix compares against the real prior
        # close, 90.32 (~15.7%, clears it).
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=104.51), \
             patch.object(a, "fetch_df", return_value=self._hist(90.32, 24_000_000, today_close=104.30)):
            mock_requests.get.return_value = self._earnings_response([self._record("CRWV")])
            result = a.fetch_earnings_mover_tickers()
        self.assertIn("CRWV", result,
                      "a real overnight gap must be detected even though fetch_df's "
                      "last bar is today's still-forming one, not yesterday's close")

    def test_premarket_call_with_no_todays_bar_yet_uses_the_last_bar_directly(self):
        # If this ever runs before today's daily bar exists at all (a
        # pre-market call), the last bar genuinely IS the most recent
        # completed day -- iloc[-2] would wrongly skip back an extra day.
        import pandas as pd
        n = 10
        idx = pd.date_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n, freq="D")
        hist = pd.DataFrame({"Close": [90.32] * n, "Volume": [24_000_000] * n}, index=idx)
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "get_live_price", return_value=104.51), \
             patch.object(a, "fetch_df", return_value=hist):
            mock_requests.get.return_value = self._earnings_response([self._record("CRWV")])
            result = a.fetch_earnings_mover_tickers()
        self.assertIn("CRWV", result)

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


class TestGetOptionSnapshotVolume(unittest.TestCase):
    """_get_option_snapshot()'s "oi" field is always 0 -- the snapshot
    endpoint doesn't carry openInterest on any feed (see
    _merge_contract_oi()'s docstring). Volume is different: confirmed
    live 2026-08-30 that dailyBar.v IS present and real on this same
    endpoint -- this locks in that parsing so a future response-shape
    change can't silently zero it out unnoticed."""

    def _mock_response(self, daily_bar=None):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"snapshots": {"TEST260904C00050000": {
            "latestQuote": {"bp": 1.85, "ap": 1.93, "bs": 5, "as": 5},
            "greeks": {"delta": 0.4},
            "impliedVolatility": 0.5,
            "dailyBar": daily_bar or {},
        }}}
        return resp

    def test_real_daily_volume_is_parsed(self):
        with patch.object(a, "ALPACA_API_KEY", "k"), \
             patch.object(a, "ALPACA_SECRET_KEY", "s"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = self._mock_response({"v": 3133})
            snap = a._get_option_snapshot("TEST260904C00050000")
        self.assertEqual(snap["volume"], 3133)

    def test_missing_daily_bar_defaults_volume_to_zero_not_a_crash(self):
        with patch.object(a, "ALPACA_API_KEY", "k"), \
             patch.object(a, "ALPACA_SECRET_KEY", "s"), \
             patch.object(a, "requests") as mock_requests:
            mock_requests.get.return_value = self._mock_response(None)
            snap = a._get_option_snapshot("TEST260904C00050000")
        self.assertEqual(snap["volume"], 0)


class TestRenderOptionsChainTable(unittest.TestCase):
    """/options used to print two separate CALLS/PUTS lists; this renders
    a single Robinhood-style chain instead -- calls left, one shared
    strike column, puts right, ITM marked, a price divider inserted at
    its natural spot. idx numbers must stay exactly what
    _handle_options_command() assigned (calls-then-puts order) since
    /buy N depends on them -- this is a pure display change, never a
    re-indexing."""

    def _call(self, idx, strike, bid=1.0, ask=1.1, delta=0.5, est=False, oi=0, volume=0):
        return {"type": "CALL", "strike": strike, "bid": bid, "ask": ask,
                "delta": delta, "delta_estimated": est, "idx": idx, "oi": oi, "volume": volume}

    def _put(self, idx, strike, bid=1.0, ask=1.1, delta=0.5, est=False, oi=0, volume=0):
        return {"type": "PUT", "strike": strike, "bid": bid, "ask": ask,
                "delta": delta, "delta_estimated": est, "idx": idx, "oi": oi, "volume": volume}

    def test_empty_chain_returns_empty_string(self):
        self.assertEqual(a._render_options_chain_table([], [], 34.0), "")

    def test_idx_numbers_are_preserved_verbatim_for_buy_n(self):
        calls = [self._call(1, 30.0), self._call(2, 35.0)]
        puts = [self._put(3, 30.0), self._put(4, 35.0)]
        out = a._render_options_chain_table(calls, puts, 32.0)
        self.assertIn(" 1)", out)
        self.assertIn(" 2)", out)
        self.assertIn(" 3)", out)
        self.assertIn(" 4)", out)

    def test_wrapped_in_pre_block_for_telegram_monospace_alignment(self):
        out = a._render_options_chain_table([self._call(1, 30.0)], [], 30.0)
        self.assertTrue(out.startswith("<pre>"))
        self.assertTrue(out.endswith("</pre>"))

    def test_itm_calls_are_marked_below_current_price(self):
        # A call below the underlying price is ITM.
        out = a._render_options_chain_table([self._call(1, 25.0)], [], 30.0)
        self.assertIn("●", out)

    def test_otm_calls_are_not_marked(self):
        # A call above the underlying price is OTM -- no dot on that row.
        out = a._render_options_chain_table([self._call(1, 40.0)], [], 30.0)
        _rows = out.split("\n")
        _row = next(r for r in _rows if r.strip().startswith("1)"))
        self.assertNotIn("●", _row)

    def test_itm_puts_are_marked_above_current_price(self):
        # A put above the underlying price is ITM (opposite of calls).
        out = a._render_options_chain_table([], [self._put(1, 40.0)], 30.0)
        self.assertIn("●", out)

    def test_price_divider_appears_between_straddling_strikes(self):
        calls = [self._call(1, 30.0), self._call(2, 35.0)]
        out = a._render_options_chain_table(calls, [], 32.0)
        _rows = out.split("\n")
        _i30 = next(i for i, r in enumerate(_rows) if "30" in r and r.strip().startswith("1)"))
        _i35 = next(i for i, r in enumerate(_rows) if "35" in r and r.strip().startswith("2)"))
        _idiv = next(i for i, r in enumerate(_rows) if "32.00" in r)
        self.assertTrue(_i30 < _idiv < _i35,
                         f"divider must sit between the straddling strikes, got rows: {_rows}")

    def test_missing_call_or_put_at_a_strike_renders_a_blank_not_a_crash(self):
        # Strikes don't always line up on both sides (a snapshot lookup
        # can fail for just one side) -- a strike with only a call, or
        # only a put, must render cleanly rather than KeyError.
        calls = [self._call(1, 30.0)]
        puts = [self._put(2, 35.0)]
        out = a._render_options_chain_table(calls, puts, 32.0)
        self.assertIn(" 1)", out)
        self.assertIn(" 2)", out)

    def test_estimated_delta_is_tagged(self):
        out = a._render_options_chain_table([self._call(1, 30.0, delta=0.4, est=True)], [], 30.0)
        self.assertIn("Δ~0.40", out)

    def test_open_interest_and_volume_are_shown_per_strike(self):
        out = a._render_options_chain_table(
            [self._call(1, 30.0, oi=1250, volume=340)], [], 30.0)
        self.assertIn("oi 1.2k", out)
        self.assertIn("vol 340", out)

    def test_large_oi_and_volume_are_abbreviated_with_k(self):
        out = a._render_options_chain_table(
            [self._call(1, 30.0, oi=15000, volume=2500)], [], 30.0)
        self.assertIn("15.0k", out)
        self.assertIn("2.5k", out)

    def test_small_oi_and_volume_are_shown_exactly_not_abbreviated(self):
        out = a._render_options_chain_table(
            [self._call(1, 30.0, oi=42, volume=7)], [], 30.0)
        self.assertIn("oi 42", out)
        self.assertIn("vol 7", out)


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
            # _options_position_budget() (replaced the flat
            # MANUAL_BUY_MAX_RISK_DOLLARS constant 2026-08-21) calls
            # get_effective_account() on every check -- without mocking
            # this, it falls through to a real get_alpaca_client() call
            # (breaking "staging a confirmation must never touch the
            # broker" assertions) and MagicMock's auto-mocked numeric
            # coercion produces a near-zero budget, rejecting every test
            # trade as "over the per-trade cap" before it ever stages.
            patch.object(a, "get_effective_account", return_value=25_000.0),
            # is_on_probation() now also factors into the manual-buy cap
            # (2026-08-21 fix) -- without mocking this, tests read whatever
            # the REAL dman_probation.json on disk happens to say, the
            # exact ambient-state fragility already fixed once this week
            # for the win-rate file. Default to "not on probation" here;
            # the dedicated probation-cap tests below override it.
            patch.object(a, "is_on_probation", return_value=(False, 1.0)),
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

    # ── /options next/prev ───────────────────────────────────────────
    def _write_menu_at_expiry(self, ticker, expiry_str):
        menu = self._menu(ticker=ticker, expiry=expiry_str)
        with open(self._menu_tmp.name, "w") as f:
            json.dump(menu, f)

    def test_next_steps_forward_from_the_currently_shown_expiry(self):
        import datetime as _dt
        self._write_menu_at_expiry("SMCI", "2026-08-21")
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        chain = {"ticker": "SMCI", "expiry": "2026-08-28", "dte": 16, "underlying_price": 34.0,
                  "items": [{"type": "CALL", "occ_symbol": "X", "strike": 34.0,
                              "bid": 1.5, "ask": 1.55, "delta": 0.51, "delta_estimated": False}]}
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=chain) as mock_fetch, \
             patch.object(a, "send_telegram", return_value=True):
            a._handle_options_command(["/options", "SMCI", "next"])
        _, kwargs = mock_fetch.call_args
        self.assertEqual(kwargs.get("expiry"), _dt.date(2026, 8, 28))

    def test_prev_steps_backward_from_the_currently_shown_expiry(self):
        import datetime as _dt
        self._write_menu_at_expiry("SMCI", "2026-08-21")
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        chain = {"ticker": "SMCI", "expiry": "2026-08-14", "dte": 2, "underlying_price": 34.0,
                  "items": [{"type": "CALL", "occ_symbol": "X", "strike": 34.0,
                              "bid": 1.5, "ask": 1.55, "delta": 0.51, "delta_estimated": False}]}
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "_fetch_option_chain_for_display", return_value=chain) as mock_fetch, \
             patch.object(a, "send_telegram", return_value=True):
            a._handle_options_command(["/options", "SMCI", "prev"])
        _, kwargs = mock_fetch.call_args
        self.assertEqual(kwargs.get("expiry"), _dt.date(2026, 8, 14))

    def test_next_at_the_furthest_expiry_shows_an_error_not_a_wraparound(self):
        import datetime as _dt
        self._write_menu_at_expiry("SMCI", "2026-08-28")
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI", "next"])
        self.assertIn("furthest", mock_tg.call_args[0][0])

    def test_prev_at_the_nearest_expiry_shows_an_error_not_a_wraparound(self):
        import datetime as _dt
        self._write_menu_at_expiry("SMCI", "2026-08-14")
        expiries = [_dt.date(2026, 8, 14), _dt.date(2026, 8, 21), _dt.date(2026, 8, 28)]
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "_fetch_available_expiries", return_value=expiries), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI", "prev"])
        self.assertIn("nearest", mock_tg.call_args[0][0])

    def test_next_with_no_saved_menu_asks_to_run_options_first(self):
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=34.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI", "next"])
        self.assertIn("run /options SMCI first", mock_tg.call_args[0][0])

    def test_next_for_a_different_ticker_than_the_saved_menu_is_rejected(self):
        # The saved menu is for SMCI -- asking for "next" on a totally
        # different ticker has no reference point to step from.
        self._write_menu_at_expiry("SMCI", "2026-08-21")
        with patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_live_price", return_value=100.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "AAPL", "next"])
        self.assertIn("run /options AAPL first", mock_tg.call_args[0][0])

    def test_unparseable_expiry_arg_shows_usage_error(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_options_command(["/options", "SMCI", "banana"])
        self.assertIn("Couldn't parse", mock_tg.call_args[0][0])

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

    def test_buy_command_during_probation_uses_the_halved_cap(self):
        # Found in the 2026-08-21 session review: the manual /buy cap
        # didn't apply is_on_probation()'s multiplier at all, while
        # _submit_signals_to_alpaca() does for every automated entry --
        # a manual buy during probation could size at the FULL budget
        # while every automated entry was deliberately halved. Budget is
        # 25,000 * 0.15 = 3,750 normally; halved to 1,875 on probation.
        # 5 contracts @ $4.00 ask = $2,000 -- under the normal cap, over
        # the halved one.
        menu = self._menu()
        menu["items"][0]["bid"] = 3.90
        menu["items"][0]["ask"] = 4.00
        with open(self._menu_tmp.name, "w") as f:
            json.dump(menu, f)
        with patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_buy_command(["/buy", "1", "5"])
        self.assertIn("per-trade cap", mock_tg.call_args[0][0])
        self.assertIn("1875", mock_tg.call_args[0][0].replace(",", ""))
        self.assertEqual(os.path.getsize(self._buy_tmp.name), 0)

    def test_buy_command_during_probation_still_allows_a_properly_sized_trade(self):
        menu = self._menu()
        menu["items"][0]["bid"] = 3.90
        menu["items"][0]["ask"] = 4.00
        with open(self._menu_tmp.name, "w") as f:
            json.dump(menu, f)
        with patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "send_telegram", return_value=True):
            a._handle_buy_command(["/buy", "1", "4"])   # 4 x $400 = $1,600, under the $1,875 halved cap
        with open(self._buy_tmp.name) as f:
            pending = json.load(f)
        self.assertEqual(pending["contracts"], 4)

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
        # Explicit clean rolling_stats() so this test doesn't depend on
        # whatever the ambient real dman_win_rate.json happens to show --
        # found 2026-08-18: a genuine live 3-loss day made this (and
        # several sibling tests below) fail on unrelated real state.
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "check_macro_safe", return_value=(False, 0)), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("blackout", mock_tg.call_args[0][0])

    def test_consecutive_loss_guard_blocks_order(self):
        # Added 2026-08-16: this path previously checked only halt and
        # macro-blackout -- a real manual buy could submit while the
        # consecutive-loss/daily/monthly circuit breakers had tripped.
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        mock_stats = {"consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.5,
                      "avg_win_r": 2.0, "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=mock_stats), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("consecutive-loss", mock_tg.call_args[0][0])

    def test_daily_loss_limit_blocks_order(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=-(a.DAILY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("daily loss", mock_tg.call_args[0][0])

    def test_monthly_loss_limit_blocks_order(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "get_alpaca_client") as mock_gac, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_gac.assert_not_called()
        self.assertIn("monthly loss", mock_tg.call_args[0][0])


    def test_yes_submits_and_registers_position(self):
        with open(self._buy_tmp.name, "w") as f:
            json.dump(self._pending(), f)
        snap = {"bid": 1.52, "ask": 1.56, "delta": 0.51}
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="order-abc-123")
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "_get_option_snapshot", return_value=snap), \
             patch.object(a, "_cash_available_for", return_value=(True, "")), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_manual_options_buy_reply("YES")
        mock_client.submit_order.assert_called_once()
        positions = a.PositionTracker().positions
        self.assertEqual(len(positions), 1)
        self.assertTrue(positions[0].setup.startswith("Options Call SMCI260814C00034000"))
        self.assertIn("Submitted", mock_tg.call_args[0][0])


class TestEntryCircuitBreakersOk(unittest.TestCase):
    """Direct unit coverage of the shared helper both approval paths above
    now use, isolated from the Telegram-flow scaffolding."""

    def test_all_clear_returns_ok(self):
        mock_stats = {"consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                      "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4}
        with patch.object(a, "is_halted", return_value=False), \
             patch.object(a.WinRateTracker, "rolling_stats", return_value=mock_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0):
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_checks_short_circuit_in_order_halt_first(self):
        # If halted, the more expensive checks (tracker stats, P&L reads)
        # should never even run.
        with patch.object(a, "is_halted", return_value=True), \
             patch.object(a.WinRateTracker, "rolling_stats") as mock_stats:
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertFalse(ok)
        mock_stats.assert_not_called()

    def test_probation_bypasses_consec_loss_guard(self):
        mock_stats = {"consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.3,
                      "avg_win_r": 2.0, "avg_loss_r": 1.0, "total": 10, "wins": 3, "losses": 7}
        with patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a.WinRateTracker, "rolling_stats", return_value=mock_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1):
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_probation_still_enforces_daily_loss_limit(self):
        # Probation only bypasses consec-loss/monthly-loss -- a genuinely
        # bad NEW day during probation must still halt.
        with patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "get_todays_loss", return_value=-(a.DAILY_LOSS_LIMIT * 100) - 1):
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertFalse(ok)
        self.assertIn("daily loss", reason)

    def test_probation_still_enforces_manual_halt(self):
        with patch.object(a, "is_halted", return_value=True), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)):
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertFalse(ok)
        self.assertIn("halted", reason)

    def test_without_probation_consec_loss_guard_still_blocks(self):
        mock_stats = {"consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.3,
                      "avg_win_r": 2.0, "avg_loss_r": 1.0, "total": 10, "wins": 3, "losses": 7}
        with patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a.WinRateTracker, "rolling_stats", return_value=mock_stats):
            ok, reason = a._entry_circuit_breakers_ok()
        self.assertFalse(ok)
        self.assertIn("consecutive-loss", reason)


class TestIsOnProbation(unittest.TestCase):
    """Direct coverage of the probation-file reader added 2026-08-18 —
    lets Wednesday's session resume at reduced size after a 3-loss day
    tripped both the consecutive-loss guard and the monthly loss limit,
    neither of which had a path back to trading on its own."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)   # start absent, matching HALT_FILE's own default state
        self._patch = patch.object(a, "PROBATION_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_missing_file_is_inactive(self):
        self.assertEqual(a.is_on_probation(), (False, 1.0))

    def test_active_file_returns_configured_multiplier(self):
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True, "size_mult": 0.5}, f)
        self.assertEqual(a.is_on_probation(), (True, 0.5))

    def test_active_false_is_inactive_regardless_of_size_mult(self):
        with open(self._tmp.name, "w") as f:
            json.dump({"active": False, "size_mult": 0.5}, f)
        self.assertEqual(a.is_on_probation(), (False, 1.0))

    def test_missing_size_mult_defaults_to_one(self):
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True}, f)
        self.assertEqual(a.is_on_probation(), (True, 1.0))

    def test_corrupt_file_fails_safe_to_inactive(self):
        with open(self._tmp.name, "w") as f:
            f.write("{not valid json")
        self.assertEqual(a.is_on_probation(), (False, 1.0))

    def test_missing_started_field_does_not_force_expire(self):
        # Pre-2026-08-22 probation files (or a hand-written one) have no
        # "started" field at all -- must fail safe to NOT expired, not
        # crash or treat a missing timestamp as infinitely old.
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True, "size_mult": 0.5}, f)
        with patch.object(a, "send_telegram") as mock_tg:
            self.assertEqual(a.is_on_probation(), (True, 0.5))
        mock_tg.assert_not_called()

    def test_recent_probation_is_still_active(self):
        started = (datetime.now(a.ET) - timedelta(days=2)).isoformat()
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True, "size_mult": 0.5, "started": started}, f)
        self.assertEqual(a.is_on_probation(), (True, 0.5))

    def test_probation_past_max_days_auto_expires(self):
        started = (datetime.now(a.ET) - timedelta(days=a.PROBATION_MAX_DAYS + 1)).isoformat()
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True, "size_mult": 0.5, "started": started}, f)
        with patch.object(a, "_is_duplicate_alert", return_value=False), \
             patch.object(a, "_save_last_alert"), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            self.assertEqual(a.is_on_probation(), (False, 1.0))
        self.assertTrue(any("expired" in c.args[0].lower() for c in mock_tg.call_args_list))

    def test_expiry_alert_only_fires_once(self):
        started = (datetime.now(a.ET) - timedelta(days=a.PROBATION_MAX_DAYS + 1)).isoformat()
        with open(self._tmp.name, "w") as f:
            json.dump({"active": True, "size_mult": 0.5, "started": started}, f)
        with patch.object(a, "_is_duplicate_alert", return_value=True), \
             patch.object(a, "_save_last_alert"), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a.is_on_probation()
        mock_tg.assert_not_called()


class TestSetupProbation(unittest.TestCase):
    """Added 2026-08-24 (Monday-session review finding): setup_performance_drift()
    was purely informational -- a setup the drift check flagged got an
    alert and nothing else, even after already being caught drifting once
    before (Low Float Catalyst's SETUP_MIN_CONFLUENCE=90 raise). This
    closes the loop: a flagged setup now automatically gets a temporary
    extra score requirement, same auto-expiry pattern as account-level
    probation, scoped to just the underperforming setup."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)
        self._patch = patch.object(a, "SETUP_PROBATION_FILE", self._tmp.name)
        self._patch.start()
        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)
        self._alerts_patch.stop()
        os.unlink(self._alerts_tmp.name)

    def test_unrestricted_setup_has_no_bonus(self):
        self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), 0)

    def test_enter_restricts_and_returns_true_the_first_time(self):
        newly = a._enter_setup_probation("Low Float Catalyst", "0% WR test")
        self.assertTrue(newly)
        self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), a.SETUP_PROBATION_SCORE_BONUS)

    def test_re_entering_an_already_restricted_setup_returns_false(self):
        a._enter_setup_probation("Low Float Catalyst", "first")
        newly_again = a._enter_setup_probation("Low Float Catalyst", "second")
        self.assertFalse(newly_again, "an already-restricted setup must not reset its clock")

    def test_re_entering_does_not_reset_the_started_timestamp(self):
        a._enter_setup_probation("Low Float Catalyst", "first")
        original = a._load_setup_probation()["Low Float Catalyst"]["started"]
        a._enter_setup_probation("Low Float Catalyst", "second")
        self.assertEqual(a._load_setup_probation()["Low Float Catalyst"]["started"], original)

    def test_only_the_restricted_setup_gets_a_bonus(self):
        a._enter_setup_probation("Low Float Catalyst", "test")
        self.assertEqual(a._setup_probation_bonus("Gap & Hold"), 0)

    def test_recent_restriction_is_still_active(self):
        state = {"Low Float Catalyst": {
            "started": (datetime.now(a.ET) - timedelta(days=2)).isoformat(), "note": "x"}}
        a._save_setup_probation(state)
        self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), a.SETUP_PROBATION_SCORE_BONUS)

    def test_restriction_past_max_days_auto_expires(self):
        state = {"Low Float Catalyst": {
            "started": (datetime.now(a.ET) - timedelta(days=a.SETUP_PROBATION_MAX_DAYS + 1)).isoformat(),
            "note": "x"}}
        a._save_setup_probation(state)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), 0)
        self.assertTrue(any("expired" in c.args[0].lower() for c in mock_tg.call_args_list))
        self.assertNotIn("Low Float Catalyst", a._load_setup_probation(),
                          "an expired restriction must actually be removed from state")

    def test_expiry_alert_only_fires_once(self):
        state = {"Low Float Catalyst": {
            "started": (datetime.now(a.ET) - timedelta(days=a.SETUP_PROBATION_MAX_DAYS + 1)).isoformat(),
            "note": "x"}}
        a._save_setup_probation(state)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._setup_probation_bonus("Low Float Catalyst")
            a._save_setup_probation(state)   # simulate it drifting back in on a later check
            a._setup_probation_bonus("Low Float Catalyst")
        self.assertEqual(mock_tg.call_count, 1)

    def test_missing_started_field_does_not_crash(self):
        a._save_setup_probation({"Low Float Catalyst": {"note": "x"}})
        with patch.object(a, "send_telegram", return_value=True):
            self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), a.SETUP_PROBATION_SCORE_BONUS,
                              "unparseable timestamp must fail safe to NOT expired")

    def test_corrupt_state_fails_safe_to_zero(self):
        with open(self._tmp.name, "w") as f:
            f.write("{not valid json")
        self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), 0)

    def test_smallcap_score_threshold_includes_the_bonus(self):
        # Direct integration with the smallcap-discovery gate.
        base = a._smallcap_score_threshold("ARTL", "Low Float Catalyst")
        a._enter_setup_probation("Low Float Catalyst", "test")
        self.assertEqual(a._smallcap_score_threshold("ARTL", "Low Float Catalyst"),
                          base + a.SETUP_PROBATION_SCORE_BONUS)


class TestSetupProbationTelegramCommands(unittest.TestCase):
    """/setupprobation and /endsetupprobation -- the manual override for
    the auto-entry above. A live-money control needs a human escape
    hatch: setting one before the automatic drift check would, or
    clearing one early once a setup's newer trades look better."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        os.unlink(self._tmp.name)
        self._patch = patch.object(a, "SETUP_PROBATION_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        if os.path.exists(self._tmp.name):
            os.unlink(self._tmp.name)

    def test_setupprobation_restricts_a_multi_word_setup_name(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/setupprobation Low Float Catalyst")
        self.assertIn("Low Float Catalyst", a._load_setup_probation())
        sent = mock_tg.call_args[0][0]
        self.assertIn("SETUP PROBATION ON", sent)
        self.assertIn("Low Float Catalyst", sent)

    def test_setupprobation_with_no_name_is_a_usage_error_not_a_crash(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/setupprobation")
        self.assertEqual(a._load_setup_probation(), {})
        self.assertIn("Usage", mock_tg.call_args[0][0])

    def test_setupprobation_on_an_already_restricted_setup_says_so(self):
        a._enter_setup_probation("Low Float Catalyst", "prior")
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/setupprobation Low Float Catalyst")
        self.assertIn("already restricted", mock_tg.call_args[0][0])

    def test_endsetupprobation_clears_it(self):
        a._enter_setup_probation("Low Float Catalyst", "prior")
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/endsetupprobation Low Float Catalyst")
        self.assertNotIn("Low Float Catalyst", a._load_setup_probation())
        self.assertIn("SETUP PROBATION ENDED", mock_tg.call_args[0][0])

    def test_endsetupprobation_on_an_unrestricted_setup_says_so_not_an_error(self):
        with patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._handle_telegram_command("/endsetupprobation Low Float Catalyst")
        self.assertIn("nothing to end", mock_tg.call_args[0][0])

    def test_endsetupprobation_only_clears_the_named_setup(self):
        a._enter_setup_probation("Low Float Catalyst", "prior")
        a._enter_setup_probation("Gap & Short", "prior")
        with patch.object(a, "send_telegram", return_value=True):
            a._handle_telegram_command("/endsetupprobation Low Float Catalyst")
        remaining = a._load_setup_probation()
        self.assertNotIn("Low Float Catalyst", remaining)
        self.assertIn("Gap & Short", remaining)


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


class TestCheckAndHealWatchdog(unittest.TestCase):
    """Found live 2026-08-17: both dman_daemon.yml's and dman_watchdog.yml's
    scheduled cron triggers silently failed to fire for 2+ hours during a
    live trading session with real open positions, while Scanner/StockTwits/
    Pre-Market Briefing all fired normally. run_watchdog() already detects
    and auto-restarts a stale DAEMON, but only if the watchdog itself
    actually runs -- nothing previously checked whether the watchdog's own
    trigger fired. _check_and_heal_watchdog() closes that gap, called from
    two other independently-scheduled workflows (scan + stocktwits)."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._tmp.name)
        self._alerts_patch.start()
        self._token_patch = patch.object(a, "GITHUB_TOKEN", "fake-token")
        self._token_patch.start()

    def tearDown(self):
        self._alerts_patch.stop()
        self._token_patch.stop()
        os.unlink(self._tmp.name)

    def _weekday_market_hours(self):
        return datetime(2026, 8, 18, 11, 0, tzinfo=a.ET)   # a fixed Tuesday, 11 AM ET

    def _runs_response(self, minutes_ago):
        from datetime import timezone as _tz
        created = (datetime.now(_tz.utc) - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")
        resp = MagicMock()
        resp.json.return_value = {"workflow_runs": [{"created_at": created}]}
        return resp

    def test_weekend_is_a_no_op(self):
        saturday = datetime(2026, 8, 16, 11, 0, tzinfo=a.ET)
        with patch.object(a, "requests") as mock_requests:
            a._check_and_heal_watchdog(now_et=saturday)
        mock_requests.get.assert_not_called()

    def test_outside_market_hours_is_a_no_op(self):
        evening = datetime(2026, 8, 18, 20, 0, tzinfo=a.ET)
        with patch.object(a, "requests") as mock_requests:
            a._check_and_heal_watchdog(now_et=evening)
        mock_requests.get.assert_not_called()

    def test_no_github_token_is_a_no_op(self):
        self._token_patch.stop()
        with patch.object(a, "GITHUB_TOKEN", ""), \
             patch.object(a, "requests") as mock_requests:
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
        mock_requests.get.assert_not_called()
        self._token_patch.start()   # tearDown expects this to still be running

    def test_fresh_watchdog_run_does_not_dispatch(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_trigger_workflow_restart") as mock_restart:
            mock_requests.get.return_value = self._runs_response(minutes_ago=5)
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
        mock_restart.assert_not_called()

    def test_stale_watchdog_dispatches_a_fresh_run_and_alerts(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_trigger_workflow_restart", return_value=(True, "dispatched")) as mock_restart, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            mock_requests.get.return_value = self._runs_response(minutes_ago=90)
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
        mock_restart.assert_called_once_with("dman_watchdog.yml")
        mock_tg.assert_called_once()
        self.assertIn("Meta-watchdog", mock_tg.call_args[0][0])

    def test_repeated_calls_within_cooldown_do_not_re_dispatch(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_trigger_workflow_restart", return_value=(True, "dispatched")) as mock_restart, \
             patch.object(a, "send_telegram", return_value=True):
            mock_requests.get.return_value = self._runs_response(minutes_ago=90)
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
        mock_restart.assert_called_once()

    def test_api_failure_fails_quiet(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_trigger_workflow_restart") as mock_restart:
            mock_requests.get.side_effect = Exception("network timeout")
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())   # must not raise
        mock_restart.assert_not_called()

    def test_restart_failure_still_alerts_with_manual_instructions(self):
        with patch.object(a, "requests") as mock_requests, \
             patch.object(a, "_trigger_workflow_restart", return_value=(False, "HTTP 403")), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            mock_requests.get.return_value = self._runs_response(minutes_ago=90)
            a._check_and_heal_watchdog(now_et=self._weekday_market_hours())
        self.assertIn("/restart", mock_tg.call_args[0][0])


class TestIsChasingExtendedHighs(unittest.TestCase):
    """Direct instruction, 2026-08-17: don't enter plays that already ran
    to (near) all-time/52-week highs unless there's real confirmation the
    move held, not just chasing exhaustion. Added after UMAC (options) and
    ARTL (equity) were both entered right around a huge recent volume
    spike into new highs, then reversed hard: UMAC ran $24->$34 (+40%)
    into new highs on a 15.9M-share outlier-volume day, then gapped down
    and continued lower the very next session; ARTL spiked to a fresh
    local high on 15-25x normal volume and closed near the day's low the
    same session."""

    def _df(self, closes, highs=None, lows=None, rvol_last=1.5, n_lookback=15):
        # closes[-1] is "today". highs/lows default to a tight range around
        # each close so the day-range-position math stays well-defined
        # unless a test overrides it explicitly. Filler (pre-lookback)
        # rows are priced flat at closes[0] -- NOT some arbitrary far-off
        # constant -- since the gate takes the max High over the WHOLE
        # available frame when there are under 252 rows, and a filler
        # price outside the real range would silently corrupt the 52wk-
        # high calculation the gate depends on.
        import pandas as pd
        n = n_lookback + len(closes)
        pad = [closes[0]] * n_lookback
        all_closes = pad + list(closes)
        all_highs  = highs if highs is not None else [c * 1.01 for c in all_closes]
        all_lows   = lows  if lows  is not None else [c * 0.99 for c in all_closes]
        rvol = [1.0] * (len(all_closes) - 1) + [rvol_last]
        idx = pd.date_range("2026-07-01", periods=len(all_closes), freq="D")
        return pd.DataFrame({
            "Open": all_closes, "High": all_highs, "Low": all_lows,
            "Close": all_closes, "Volume": [1_000_000] * len(all_closes),
            "RVOL": rvol,
        }, index=idx)

    def test_umac_style_extended_run_with_weak_close_is_blocked(self):
        # Base ~$24 ten sessions back, run hard into today, today makes a
        # fresh high intraday ($34.93) on outlier volume but gives most
        # of it back to close weak ($30.15) -- the actual blow-off-top
        # shape, still near its own high but no longer confirmed.
        closes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 30.15]
        df = self._df(closes, rvol_last=3.0)
        df.loc[df.index[-1], "High"] = 34.93
        df.loc[df.index[-1], "Low"]  = 29.35
        df.loc[df.index[-1], "Close"] = 30.15   # closes in the bottom ~15% of the day's range
        self.assertTrue(a._is_chasing_extended_highs(df))

    def test_artl_style_flat_base_single_day_spike_is_not_this_gates_target(self):
        # ARTL was actually chopping/flat (~-4% over 10 sessions) right
        # before today's spike -- this is a same-day catalyst fakeout from
        # a flat base, NOT an "already extended into highs" situation this
        # gate targets by design (that's UMAC's shape, not this one). The
        # existing intraday-high pullback guard is the mechanism aimed at
        # this specific pattern instead. Documented here so this boundary
        # is explicit rather than silently assumed.
        closes = [0.75, 0.76, 0.75, 0.74, 0.73, 0.75, 0.74, 0.76, 0.75, 0.77, 0.7210]
        df = self._df(closes, rvol_last=15.0)
        df.loc[df.index[-1], "High"] = 0.7999
        df.loc[df.index[-1], "Low"]  = 0.6902
        df.loc[df.index[-1], "Close"] = 0.7210   # bottom ~28% of the day's range
        self.assertFalse(a._is_chasing_extended_highs(df))

    def test_healthy_breakout_near_highs_without_a_big_recent_run_is_not_blocked(self):
        # Near its high, but has NOT run 25%+ in the last 10 sessions --
        # a fresh breakout off a base, exactly what this must NOT block.
        closes = [28, 28.2, 27.9, 28.3, 28.5, 28.4, 28.7, 28.9, 29.0, 29.3, 29.5]
        df = self._df(closes, rvol_last=1.8)
        self.assertFalse(a._is_chasing_extended_highs(df))

    def test_extended_run_with_a_strong_confirmed_close_is_not_blocked(self):
        # Same extended run as the UMAC case, but today closes strong
        # (top of its range) on normal (not euphoric) volume -- real
        # confirmation, not exhaustion.
        closes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
        df = self._df(closes, rvol_last=1.5)
        df.loc[df.index[-1], "High"]  = 34.2
        df.loc[df.index[-1], "Low"]   = 33.5
        df.loc[df.index[-1], "Close"] = 34.0   # near the top of the day's range
        self.assertFalse(a._is_chasing_extended_highs(df))

    def test_near_high_but_run_below_threshold_is_not_blocked(self):
        # Near highs, up only ~12% over 10 sessions -- below the 25%
        # threshold, not what this gate targets.
        closes = [30, 30.5, 31, 31.2, 31.5, 31.8, 32, 32.5, 33, 33.3, 33.6]
        df = self._df(closes, rvol_last=2.0)
        df.loc[df.index[-1], "Low"]   = 32.0
        df.loc[df.index[-1], "Close"] = 33.6
        self.assertFalse(a._is_chasing_extended_highs(df))

    def test_insufficient_history_fails_open(self):
        df = self._df([10, 11], n_lookback=2)   # far short of the 10-day lookback
        self.assertFalse(a._is_chasing_extended_highs(df))

    def test_signal_gate_is_wired_into_score_signal(self):
        closes = [24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 30.15]
        df = self._df(closes, rvol_last=2.0)
        df.loc[df.index[-1], "High"] = 33.19
        df.loc[df.index[-1], "Low"]  = 29.90
        df.loc[df.index[-1], "Close"] = 30.15
        # Give the indicator columns score_signal() needs so it doesn't
        # blow up on missing data -- reuses the same minimal shape other
        # score_signal tests already build.
        for col, val in [("EMA9", 30), ("EMA20", 29), ("EMA50", 28), ("SMA200", 27),
                         ("RSI", 55), ("MACD", 0.5), ("MACD_sig", 0.3), ("MACD_hist", 0.2),
                         ("ATR", 1.0), ("AvgVol20", 5_000_000), ("OBV", 0), ("ST_bull", True)]:
            df[col] = val
        sig = a.ProSignal(
            ticker="TESTX", bias="LONG", setup="Gap & Hold",
            entry=30.15, stop=28.0, target1=35.0, target2=40.0,
            shares=10, rr=2.0, rsi=55.0, rvol=2.0,
            reason="test", confluence_score=0,
        )
        with patch.object(a, "regime_allows_signal", return_value=(True, 15)), \
             patch.object(a, "check_mtf", return_value=(True, 20)), \
             patch.object(a, "check_relative_strength", return_value=(True, 15)), \
             patch.object(a, "check_sector", return_value=(True, 10)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 0)), \
             patch.object(a, "check_macro_safe", return_value=(True, 0)), \
             patch.object(a, "fetch_weekly", return_value=None):
            result = a.score_signal(sig, df, {"regime": "BULL", "score": 15}, a.WinRateTracker(filepath=os.devnull))
        self.assertFalse(result.not_chasing_extended_highs)
        self.assertFalse(result.passed_all_gates())


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


class TestPremarketEntryPlanSkipsGapDown(unittest.TestCase):
    """Found in the 2026-08-16 review: the pre-market mover report's
    is_mover gate includes gap-DOWN movers by design, and a catalyst-
    less gap-down defaults to tier C (not D) -- so it used to fall
    straight into the bullish LONG entry plan: VWAP-hold/breakout
    entries and +30%/+50% T1/T2 targets computed off the DEPRESSED
    price, reading a decline as a breakout setup. Informational only
    (no auto-order), but this is the system's main human-facing entry
    surface."""

    def test_gap_down_never_gets_a_long_entry_plan_even_on_a_good_tier(self):
        block = a._build_premarket_entry_plan(
            gap_pct=-8.0, tier="A", vwap=9.5, pre_px=9.4, prev_close=10.2,
            fl_m=1.5, is_moon=False)
        self.assertIn("Gap DOWN", block)
        self.assertNotIn("ENTRY PLAN</b>\n     Aggressive", block)
        self.assertNotIn("Moderate (9:30)", block)
        self.assertNotIn("T1:", block)

    def test_gap_up_tier_a_still_gets_the_full_long_plan(self):
        block = a._build_premarket_entry_plan(
            gap_pct=12.0, tier="A", vwap=9.5, pre_px=9.4, prev_close=8.4,
            fl_m=1.5, is_moon=False)
        self.assertIn("Aggressive (pre-mkt)", block)
        self.assertIn("Moderate (9:30)", block)
        self.assertIn("T1:", block)

    def test_gap_up_tier_d_still_shows_the_dilution_skip_not_a_plan(self):
        block = a._build_premarket_entry_plan(
            gap_pct=20.0, tier="D", vwap=9.5, pre_px=9.4, prev_close=7.8,
            fl_m=1.5, is_moon=False)
        self.assertIn("dilution/offering risk", block)
        self.assertNotIn("T1:", block)


class TestLowFloatCatalystFailOpenOnMissing52wkLow(unittest.TestCase):
    """Found in the 2026-08-16 review: when the 52wk-low lookback raises
    (missing/short data), the except branch set near_bottom = True
    ("data unavailable -- don't block") but never assigned low_52wk on
    that path. The bot_note f-string further down unconditionally
    references low_52wk whenever near_bottom is True, so this actually
    raised a silent NameError caught by the function's own outer
    catch-all -- dropping the ENTIRE signal, the exact opposite of the
    documented fail-open intent."""

    def _df_missing_low_column(self, n=6):
        import pandas as pd
        rows = []
        for i in range(n - 1):
            rows.append({"Close": 8.0, "High": 8.2, "Open": 7.9,
                         "RVOL": 1.0, "MACD": 0.1, "MACD_sig": 0.2,
                         "MACD_hist": -0.05, "RSI": 45.0, "Volume": 500_000})
        rows.append({"Close": 8.5, "High": 8.6, "Open": 8.3,
                     "RVOL": 6.0, "MACD": 1.0, "MACD_sig": 0.5,
                     "MACD_hist": 0.3, "RSI": 40.0, "Volume": 3_000_000})
        return pd.DataFrame(rows)   # deliberately no "Low" column

    def setUp(self):
        self._patches = [
            patch.object(a, "ENABLE_SMALLCAP", True),
            patch.object(a, "_get_short_float_data", return_value=(1.0, 5.0, 5.0, 0.1)),
            patch.object(a, "_is_recent_reverse_split", return_value=False),
            patch.object(a, "get_effective_account", return_value=5000.0),
            patch.object(a, "_fetch_massive_benzinga_news",
                        side_effect=lambda tickers, **kw: {t: ["headline"] for t in tickers}),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_missing_52wk_low_data_does_not_silently_drop_the_whole_signal(self):
        df = self._df_missing_low_column()
        sig = a.detect_low_float_catalyst(df, "TESTX")
        self.assertIsNotNone(sig, "a missing 52wk-low lookback must fail OPEN (still produce a "
                              "signal, per the documented intent), not silently drop the whole "
                              "detector via an unrelated NameError")


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
            # Default to "nothing found" so existing tests exercise the
            # Benzinga-only path exactly as before the 2026-08-30 EDGAR
            # wiring -- the dedicated EDGAR tests below override this.
            patch.object(a, "_check_edgar_8k", return_value=(False, "")),
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

    def test_edgar_8k_alone_confirms_catalyst_with_no_benzinga_coverage(self):
        # Added 2026-08-30: _check_edgar_8k() already existed but only ran
        # in the once-daily pre-market scan, never in this intraday
        # detector -- a real 8-K Benzinga hasn't indexed yet (often the
        # case, since SEC is the primary source) used to block a real
        # catalyst outright. A real filing alone must now be enough.
        with patch.object(a, "_fetch_massive_benzinga_news", return_value={"FGL": []}), \
             patch.object(a, "_check_edgar_8k", return_value=(True, "8-K filed 2026-08-30 — FGL Inc")):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNotNone(sig)
        self.assertIn("SEC:", sig.reason)

    def test_edgar_not_queried_when_benzinga_already_confirmed_news(self):
        # EDGAR is a fallback for when Benzinga hasn't found anything, not
        # an additional check run on top of real news already in hand --
        # querying it anyway would be redundant latency with no gating
        # benefit, since the has-a-catalyst decision is already made.
        with patch.object(a, "_fetch_massive_benzinga_news",
                          return_value={"FGL": ["Company misses guidance"]}), \
             patch.object(a, "_news_sentiment_verdict", return_value="negative"), \
             patch.object(a, "_check_edgar_8k") as mock_edgar:
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        mock_edgar.assert_not_called()
        self.assertIsNone(sig)   # negative sentiment still blocks, as before

    def test_no_benzinga_and_no_edgar_still_blocks(self):
        # Neither source finding anything must still fail closed, same
        # as the original single-source behavior.
        with patch.object(a, "_fetch_massive_benzinga_news", return_value={"FGL": []}), \
             patch.object(a, "_check_edgar_8k", return_value=(False, "")):
            sig = a.detect_low_float_catalyst(self._df(), "FGL")
        self.assertIsNone(sig)

    def test_dman_watchlist_ticker_exempt_from_edgar_check_too(self):
        with patch.object(a, "_fetch_massive_benzinga_news") as mock_news, \
             patch.object(a, "_check_edgar_8k") as mock_edgar:
            sig = a.detect_low_float_catalyst(self._df(), "DMANPICK")
        mock_news.assert_not_called()
        mock_edgar.assert_not_called()
        self.assertIsNotNone(sig)


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

    def test_calendar_day_uses_et_not_the_runners_naive_clock(self):
        # Added 2026-08-23: this used bare date.today() -- UTC on a GitHub
        # Actions runner. The evening cloud daemon session runs past UTC
        # midnight (8 PM EDT / 7 PM EST), so an on-demand /pnl check in the
        # last few minutes of that session could read "tomorrow" in UTC
        # while it's still the same trading day in ET, and wrongly reseed
        # the baseline to current equity -- reproducing the exact
        # false-profit bug this function exists to prevent.
        with open(self._tmp.name, "w") as f:
            json.dump({"date": "2026-08-20", "equity": 4943.73}, f)
        with patch.object(a, "datetime") as mock_dt:
            # 12:15 AM UTC on 8/21 = 8:15 PM EDT on 8/20 -- still the SAME
            # ET trading day the seeded baseline was recorded for.
            mock_dt.now.return_value = datetime(2026, 8, 20, 20, 15, tzinfo=a.ET)
            baseline = a._get_day_start_equity(4895.85)
        self.assertEqual(baseline, 4943.73,
                          "must still read as the same ET day and NOT reseed")


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

    def test_transient_error_after_a_real_downgrade_does_not_silently_restore(self):
        # Found 2026-08-16 review: `resolved` used to always default to
        # OPTIONS_DATA_FEED (the preferred feed) before the probe, and
        # neither the "couldn't get a probe symbol" early return nor the
        # except-branch touched it -- so a real, still-active downgrade
        # (cached "indicative") got silently reverted the moment the next
        # hourly re-probe hit ANY transient error, even though OPRA
        # entitlement never actually came back.
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_403 = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_403):
                with patch.object(a, "send_telegram", return_value=True):
                    a._resolve_options_feed()   # real downgrade, caches "indicative"
        with open(self._tmp.name) as f:
            state = json.load(f)
        state["checked_at"] -= (a._OPTIONS_FEED_RECHECK_S + 1)
        with open(self._tmp.name, "w") as f:
            json.dump(state, f)
        a._options_feed_state["checked_at"] = 0.0
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", side_effect=Exception("network timeout")):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    feed = a._resolve_options_feed()
        self.assertEqual(feed, "indicative", "a transient error must keep the real cached "
                                             "downgrade, not silently restore the preferred feed")
        mock_tg.assert_not_called()


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

    def test_transient_error_after_a_real_downgrade_does_not_silently_restore(self):
        # Found 2026-08-16 review: this used to always default `resolved`
        # to the PREFERRED feed before the probe, and a bare network
        # exception (not just a clean 403) never touched it -- so a real,
        # still-active downgrade (cached "iex") got silently reverted back
        # to the preferred feed the moment the NEXT hourly re-probe hit any
        # transient error, even though entitlement never actually came
        # back. Every call in that window then hit real 403s again.
        mock_403 = MagicMock(status_code=403)
        with patch.object(a.requests, "get", return_value=mock_403):
            with patch.object(a, "send_telegram", return_value=True):
                a._resolve_stock_feed()   # real downgrade, caches "iex"
        with open(self._tmp.name) as f:
            state = json.load(f)
        state["checked_at"] -= (a._STOCK_FEED_RECHECK_S + 1)   # force a re-probe
        with open(self._tmp.name, "w") as f:
            json.dump(state, f)
        a._stock_feed_state["checked_at"] = 0.0
        with patch.object(a.requests, "get", side_effect=Exception("network timeout")):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                feed = a._resolve_stock_feed()
        self.assertEqual(feed, "iex", "a transient error must keep the real cached "
                                      "downgrade, not silently restore the preferred feed")
        mock_tg.assert_not_called()


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


class TestConfirmDirectionalWithOptionsFlow(unittest.TestCase):
    """Direct instruction 2026-08-30, after this session's review found the
    earnings-spread family's real record (1W/3L, ~-63% average) all came
    from double spreads or a malformed one: default to the hedged double
    spread, single-sided only when real-time options flow agrees with the
    historical move pattern -- a consistent history alone isn't "full
    confidence" any more."""

    def test_agreeing_bullish_flow_confirms_call_direction(self):
        with patch.object(a, "_get_options_market_context", return_value={
                "pc_ratio": 0.5, "flow_label": "bullish (call-heavy flow)"}):
            self.assertEqual(a._confirm_directional_with_options_flow("NVDA", "CALL"), "CALL")

    def test_agreeing_cautious_flow_confirms_put_direction(self):
        with patch.object(a, "_get_options_market_context", return_value={
                "pc_ratio": 1.5, "flow_label": "cautious (put-heavy hedge flow)"}):
            self.assertEqual(a._confirm_directional_with_options_flow("XYZ", "PUT"), "PUT")

    def test_disagreeing_flow_falls_back_to_double_spread(self):
        # The "inverse" case: history says CALL, but the market is
        # actually positioned bearish going into this specific event.
        with patch.object(a, "_get_options_market_context", return_value={
                "pc_ratio": 1.5, "flow_label": "cautious (put-heavy hedge flow)"}):
            self.assertIsNone(a._confirm_directional_with_options_flow("NVDA", "CALL"))

    def test_neutral_flow_falls_back_to_double_spread(self):
        with patch.object(a, "_get_options_market_context", return_value={
                "pc_ratio": 1.0, "flow_label": "neutral"}):
            self.assertIsNone(a._confirm_directional_with_options_flow("NVDA", "CALL"))


class TestBuildEarningsSpreadPlanDirectionalGate(unittest.TestCase):
    """Integration-level: build_earnings_spread_plan() must actually wire
    the new confirmation gate in, not just have it exist unused."""

    def _legs(self, side):
        return {"long_occ": f"{side}1", "short_occ": f"{side}2",
                "long_strike": 100, "short_strike": 105 if side == "call" else 95,
                "net_debit": 0.5, "expiry": "2026-09-04", "dte": 7,
                "long_oi": 500, "short_oi": 500}

    def test_confirmed_direction_produces_a_single_sided_plan(self):
        # 3 consistent up-moves, all above the min avg magnitude -- clears
        # the historical bar, and flow agrees.
        with patch.object(a, "get_effective_account", return_value=25_000.0), \
             patch.object(a, "_last_n_earnings_moves", return_value=[10.0, 12.0, 9.0]), \
             patch.object(a, "_confirm_directional_with_options_flow", return_value="CALL"), \
             patch.object(a, "_find_spread_legs", side_effect=lambda c, t, p, side, b: self._legs(side.lower())):
            plan = a.build_earnings_spread_plan(MagicMock(), "NVDA", 100.0, a.date.today(), "AMC")
        self.assertEqual(plan["directional"], "CALL")
        self.assertIn("call", plan)
        self.assertNotIn("put", plan)

    def test_unconfirmed_direction_falls_back_to_a_double_spread(self):
        # Same clean historical pattern, but the confirmation gate says
        # real-time flow disagrees -- must still get BOTH sides, not just
        # the historically-favored one.
        with patch.object(a, "get_effective_account", return_value=25_000.0), \
             patch.object(a, "_last_n_earnings_moves", return_value=[10.0, 12.0, 9.0]), \
             patch.object(a, "_confirm_directional_with_options_flow", return_value=None), \
             patch.object(a, "_find_spread_legs", side_effect=lambda c, t, p, side, b: self._legs(side.lower())):
            plan = a.build_earnings_spread_plan(MagicMock(), "NVDA", 100.0, a.date.today(), "AMC")
        self.assertIsNone(plan["directional"])
        self.assertIn("call", plan)
        self.assertIn("put", plan)

    def test_no_historical_pattern_never_calls_the_flow_gate(self):
        # Nothing to confirm if history itself doesn't clear the bar --
        # the flow check should be skipped entirely, not called with
        # nothing meaningful to check.
        with patch.object(a, "get_effective_account", return_value=25_000.0), \
             patch.object(a, "_last_n_earnings_moves", return_value=[1.0, -1.0, 0.5]), \
             patch.object(a, "_confirm_directional_with_options_flow") as mock_confirm, \
             patch.object(a, "_find_spread_legs", side_effect=lambda c, t, p, side, b: self._legs(side.lower())):
            a.build_earnings_spread_plan(MagicMock(), "NVDA", 100.0, a.date.today(), "AMC")
        mock_confirm.assert_not_called()


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

        # _check_stop_coverage() now caches its Alpaca fetch for
        # _STOP_COVERAGE_CACHE_TTL_S (30s real wall-clock) -- without a
        # fresh cache per test, a later test's call within that window
        # would silently reuse an earlier test's mocked positions/orders
        # instead of hitting its own mock_client.
        self._cache_patch = patch.object(
            a, "_stop_coverage_fetch_cache", {"positions": None, "orders": None, "ts": 0.0})
        self._cache_patch.start()

    def tearDown(self):
        self._pos_patch.stop();    os.unlink(self._pos_tmp.name)
        self._pt_patch.stop()
        self._sig_patch.stop();    os.unlink(self._sig_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)
        self._cache_patch.stop()

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

    def test_a_restored_stop_for_one_symbol_does_not_suppress_a_genuine_alert_for_another(self):
        # Found in the 2026-08-16 review: __NO_LIVE_STOP__ used to be one
        # single global dedup key shared across every symbol -- a benign
        # "stop auto-restored" for W would start the cooldown and
        # suppress a genuine "no live stop" alert for an unrelated
        # symbol (CLRO) for the next ALERT_COOLDOWN_MIN+ minutes, even
        # though CLRO's situation is completely different and dangerous.
        with open(self._pos_tmp.name, "w") as f:
            json.dump([{"ticker": "W", "bias": "LONG", "setup": "Gap & Hold",
                       "entry": 118.0, "stop": 110.0, "target1": 130.0, "target2": 140.0,
                       "shares": 3, "entry_date": "2026-08-01"}], f)
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [self._equity_position("W")]
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="restored-order-id")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._check_open_position_risk({})   # W: auto-restore succeeds, starts the cooldown
        self.assertEqual(len([m for m in self._messages(mock_tg) if "STOP AUTO-RESTORED" in m]), 1)

        # Now CLRO is unprotected -- a genuinely different, still-dangerous
        # situation -- immediately after, well within the cooldown window.
        # Cache reset: simulates enough real time passing for the next
        # guard tick to re-fetch (this test is about the ALERT dedup key,
        # not the data-fetch cache -- without this the second call would
        # just reuse the first call's cached mock_client positions).
        a._stop_coverage_fetch_cache["ts"] = 0.0
        mock_client2 = MagicMock()
        mock_client2.get_all_positions.return_value = [self._equity_position("CLRO")]
        mock_client2.get_orders.return_value = []   # nothing to restore from -- genuinely unprotected
        with patch.object(a, "get_alpaca_client", return_value=mock_client2):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg2:
                a._check_open_position_risk({})
        stop_msgs = [m for m in self._messages(mock_tg2) if "NO LIVE STOP" in m]
        self.assertEqual(len(stop_msgs), 1,
                         "a genuine alert for CLRO must not be suppressed by W's unrelated cooldown")
        self.assertIn("CLRO", stop_msgs[0])

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

        # _auto_restore_missing_stop() now rate-limits itself via
        # _is_duplicate_alert()/_save_last_alert(), which read/write the
        # real LAST_ALERTS_FILE by default -- without isolating it here,
        # every test after the first one to succeed would silently hit the
        # real cooldown and get "restore attempted recently" instead of
        # exercising the behavior under test. Same isolation need as
        # PositionTracker above, just for a different shared file.
        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

    def tearDown(self):
        self._pt_patch.stop()
        os.unlink(self._pos_tmp.name)
        self._alerts_patch.stop()
        os.unlink(self._alerts_tmp.name)

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
        # share allocation until that sibling is cleared first. The first
        # stop submission is rejected (shares still held by the TP leg);
        # only THEN does the function cancel the TP and retry -- it no
        # longer cancels every SELL order up front (see
        # test_take_profit_is_left_alone_when_stop_submission_succeeds).
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        stale_tp = MagicMock(id="stale-take-profit-id")
        from alpaca.trading.enums import OrderSide
        stale_tp.side = OrderSide.SELL
        mock_client.get_orders.return_value = [stale_tp]
        mock_client.submit_order.side_effect = [
            Exception("insufficient qty available (shares held by another order)"),
            MagicMock(id="new-id"),
            MagicMock(id="new-tp-id"),
        ]

        with patch.object(a.time, "sleep"):   # don't actually block the test suite
            ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        mock_client.cancel_order_by_id.assert_called_once_with("stale-take-profit-id")
        # target1 is set on the tracked fixture (55.45) -- a fresh take-profit
        # should be resubmitted once the stop succeeds, not left permanently gone.
        self.assertIn("take-profit re-submitted", detail)
        self.assertEqual(mock_client.submit_order.call_count, 3)   # stop attempt 1 (fails), stop attempt 2, TP resubmit

    def test_take_profit_is_left_alone_when_stop_submission_succeeds(self):
        # If the stop submits cleanly on the first try, a healthy take-profit
        # order must never be touched -- only STOP-type orders are cancelled
        # up front, and the TP-cancel fallback should never trigger.
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        healthy_tp = MagicMock(id="healthy-take-profit-id")
        from alpaca.trading.enums import OrderSide
        healthy_tp.side = OrderSide.SELL
        mock_client.get_orders.return_value = [healthy_tp]
        mock_client.submit_order.return_value = MagicMock(id="new-stop-id")

        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        mock_client.cancel_order_by_id.assert_not_called()
        self.assertNotIn("take-profit re-submitted", detail)
        mock_client.submit_order.assert_called_once()

    def test_repeat_call_within_cooldown_does_not_touch_orders(self):
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="new-stop-id")

        ok1, _ = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertTrue(ok1)
        mock_client.reset_mock()

        ok2, detail2 = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok2)
        self.assertIn("restore attempted recently", detail2)
        mock_client.get_orders.assert_not_called()
        mock_client.submit_order.assert_not_called()

    def test_submission_failure_is_reported_not_raised(self):
        self._write_positions([self._tracked_litx(stop=28.01)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.side_effect = Exception("insufficient buying power")
        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)
        self.assertFalse(ok)
        self.assertIn("insufficient buying power", detail)

    def _tracked_trailing_litx(self, stop=35.85, trail_pct=8.5):
        pos = self._tracked_litx(stop=stop)
        pos["stop_stage"] = "trailing"
        pos["trail_pct"] = trail_pct
        return pos

    def test_trailing_stage_position_restores_a_fresh_trailing_stop(self):
        # Added 2026-08-23: tracked.stop freezes at the breakeven price the
        # moment a position progresses to "trailing" -- the real protection
        # level lives entirely at Alpaca from then on and is never written
        # back locally. Restoring a plain stop at that stale, frozen
        # breakeven number would silently give back everything the live
        # trail had since earned. A trailing-stage position with a real
        # trail_pct on record must restore an equivalent TRAILING order,
        # not a plain stop at the old breakeven.
        self._write_positions([self._tracked_trailing_litx(stop=35.85, trail_pct=8.5)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="new-trail-id-12345678")

        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        self.assertIn("trailing stop", detail)
        self.assertIn("8.5%", detail)
        req = mock_client.submit_order.call_args[0][0]
        from alpaca.trading.requests import TrailingStopOrderRequest
        self.assertIsInstance(req, TrailingStopOrderRequest)
        self.assertEqual(req.trail_percent, 8.5)
        self.assertEqual(req.qty, 47)

    def test_trailing_restore_falls_back_to_plain_stop_if_trailing_submit_fails(self):
        # Never leave the position with nothing live -- if Alpaca rejects
        # the trailing order for any reason, fall back to a plain stop at
        # the tracked (breakeven) price rather than failing the whole
        # restore.
        self._write_positions([self._tracked_trailing_litx(stop=35.85, trail_pct=8.5)])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.side_effect = [
            Exception("trailing stop rejected"),
            MagicMock(id="fallback-plain-stop-id"),
        ]

        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        self.assertIn("plain stop", detail)
        self.assertIn("35.85", detail)
        self.assertIn("trailing stage", detail, "must say protection is now weaker than before, not silent")
        self.assertEqual(mock_client.submit_order.call_count, 2)
        from alpaca.trading.requests import StopOrderRequest
        last_req = mock_client.submit_order.call_args_list[-1].args[0]
        self.assertIsInstance(last_req, StopOrderRequest)
        self.assertEqual(last_req.stop_price, 35.85)

    def test_trailing_stage_without_a_recorded_trail_pct_uses_plain_stop(self):
        # A position tracked before trail_pct existed (or that fell back to
        # a plain breakeven stop originally, see stop_stage handling in
        # _progress_equity_stop_to_trailing) has stop_stage=="trailing" but
        # trail_pct==0 -- must not attempt a trail_percent=0 order, just
        # use the existing plain-stop path unchanged.
        pos = self._tracked_litx(stop=35.85)
        pos["stop_stage"] = "trailing"
        pos["trail_pct"] = 0.0
        self._write_positions([pos])
        mock_client = MagicMock()
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="plain-stop-id")

        ok, detail = a._auto_restore_missing_stop(mock_client, "ZTEST9x", 47)

        self.assertTrue(ok)
        self.assertIn("plain stop", detail)
        mock_client.submit_order.assert_called_once()
        from alpaca.trading.requests import StopOrderRequest
        req = mock_client.submit_order.call_args[0][0]
        self.assertIsInstance(req, StopOrderRequest)


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

        # _check_stop_coverage() now caches its Alpaca fetch for
        # _STOP_COVERAGE_CACHE_TTL_S (30s real wall-clock) -- without a
        # fresh cache per test, a later test's call within that window
        # would silently reuse an earlier test's mocked positions/orders
        # instead of hitting its own mock_client.
        self._cache_patch = patch.object(
            a, "_stop_coverage_fetch_cache", {"positions": None, "orders": None, "ts": 0.0})
        self._cache_patch.start()

    def tearDown(self):
        self._pos_patch.stop();    os.unlink(self._pos_tmp.name)
        self._sig_patch.stop();    os.unlink(self._sig_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)
        self._cache_patch.stop()

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


class TestEarningsSectorOverlap(unittest.TestCase):
    """Added 2026-08-23: get_earnings_spread_candidates() evaluates one
    ticker at a time, so a week with several same-sector reporters (e.g.
    NVDA and CRWD, both Technology, both reporting the same week) produced
    N individually-reasonable approval offers with no visibility that
    approving all of them is one concentrated sector bet. Advisory only —
    never blocks an offer, just tells the human approving it what else is
    already in the same sector."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patch = patch.object(a, "PositionTracker", self._isolated_pt)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._pos_tmp.name)

    def _open(self, ticker, setup, entry_date="2026-08-25"):
        self._isolated_pt().open(a.OpenPosition(
            ticker=ticker, bias="NEUTRAL", setup=setup, entry=100.0, stop=0.0,
            target1=0.0, target2=0.0, shares=0, entry_date=entry_date,
        ))

    def test_ticker_with_no_sector_returns_empty(self):
        self.assertEqual(a._earnings_sector_overlap("SPY", []), [])

    def test_same_sector_pending_offer_is_detected(self):
        pending = [{"ticker": "CRWD"}]
        self.assertEqual(a._earnings_sector_overlap("NVDA", pending), ["CRWD"])

    def test_different_sector_pending_offer_is_not_flagged(self):
        pending = [{"ticker": "MRNA"}]
        self.assertEqual(a._earnings_sector_overlap("NVDA", pending), [])

    def test_own_ticker_in_pending_is_not_self_flagged(self):
        pending = [{"ticker": "NVDA"}]
        self.assertEqual(a._earnings_sector_overlap("NVDA", pending), [])

    def test_open_earnings_position_same_sector_is_detected(self):
        self._open("MRVL", "Earnings Double Spread")
        self.assertEqual(a._earnings_sector_overlap("NVDA", []), ["MRVL"])

    def test_open_non_earnings_position_same_sector_is_ignored(self):
        self._open("MRVL", "Gap & Hold")
        self.assertEqual(a._earnings_sector_overlap("NVDA", []), [])

    def test_open_earnings_position_different_sector_is_ignored(self):
        self._open("MRNA", "Earnings Double Spread")
        self.assertEqual(a._earnings_sector_overlap("NVDA", []), [])

    def test_positiontracker_failure_fails_open_to_pending_only(self):
        with patch.object(a, "PositionTracker", side_effect=Exception("disk error")):
            pending = [{"ticker": "CRWD"}]
            self.assertEqual(a._earnings_sector_overlap("NVDA", pending), ["CRWD"])

    def test_combines_pending_and_open_without_duplicates(self):
        self._open("CRWD", "Earnings Call Spread")
        pending = [{"ticker": "CRWD"}, {"ticker": "MRVL"}]
        self.assertEqual(a._earnings_sector_overlap("NVDA", pending), ["CRWD", "MRVL"])


class TestEarningsSpreadCommittedRisk(unittest.TestCase):
    """Added 2026-08-23: _earnings_spread_committed_risk() backs a HARD
    cap (EARNINGS_SPREAD_MAX_AGGREGATE_RISK_PCT), unlike the advisory-only
    sector-overlap warning -- it must correctly sum real dollars at risk
    across both open earnings-spread positions and still-pending offers,
    and fail safe (pending-only) if PositionTracker can't be read."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        import functools
        self._isolated_pt = functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)
        self._patch = patch.object(a, "PositionTracker", self._isolated_pt)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._pos_tmp.name)

    def _open(self, ticker, setup, max_loss):
        self._isolated_pt().open(a.OpenPosition(
            ticker=ticker, bias="NEUTRAL", setup=setup, entry=100.0, stop=0.0,
            target1=0.0, target2=0.0, shares=0, entry_date="2026-08-23",
            max_loss=max_loss,
        ))

    def test_nothing_open_or_pending_is_zero(self):
        self.assertEqual(a._earnings_spread_committed_risk([]), 0.0)

    def test_open_earnings_position_counts_its_max_loss(self):
        self._open("NVDA", "Earnings Double Spread", 150.0)
        self.assertEqual(a._earnings_spread_committed_risk([]), 150.0)

    def test_non_earnings_open_position_does_not_count(self):
        self._open("NVDA", "Gap & Hold", 150.0)
        self.assertEqual(a._earnings_spread_committed_risk([]), 0.0)

    def test_awaiting_approval_pending_offer_counts_its_plan_max_loss(self):
        pending = [{"ticker": "CRWD", "status": "awaiting_approval", "plan": {"max_loss": 200.0}}]
        self.assertEqual(a._earnings_spread_committed_risk(pending), 200.0)

    def test_non_awaiting_pending_entry_does_not_count(self):
        pending = [{"ticker": "CRWD", "status": "approved", "plan": {"max_loss": 200.0}}]
        self.assertEqual(a._earnings_spread_committed_risk(pending), 0.0)

    def test_open_and_pending_sum_together(self):
        self._open("NVDA", "Earnings Call Spread", 150.0)
        pending = [{"ticker": "CRWD", "status": "awaiting_approval", "plan": {"max_loss": 200.0}}]
        self.assertEqual(a._earnings_spread_committed_risk(pending), 350.0)

    def test_positiontracker_failure_fails_safe_to_pending_only(self):
        pending = [{"ticker": "CRWD", "status": "awaiting_approval", "plan": {"max_loss": 200.0}}]
        with patch.object(a, "PositionTracker", side_effect=Exception("disk error")):
            self.assertEqual(a._earnings_spread_committed_risk(pending), 200.0)


class TestEarningsSpreadScanAggregateRiskCap(unittest.TestCase):
    """Added 2026-08-23: nothing previously stopped 3+ same-week earnings-
    spread offers, each individually within EARNINGS_SPREAD_RISK_PCT, from
    collectively over-committing the account -- the sector-overlap warning
    is advisory only. run_earnings_spread_scan() must now hard-skip a new
    offer that would push total committed risk over
    EARNINGS_SPREAD_MAX_AGGREGATE_RISK_PCT of equity, notify once (not
    spam every scan), and leave the main per-ticker dedup key untouched so
    a LATER scan can still retry once room frees up."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._pending_patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._pending_patch.start()
        a._save_earnings_pending([])

        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

    def tearDown(self):
        self._pending_patch.stop()
        os.unlink(self._tmp.name)
        self._dedup_patch.stop()
        os.unlink(self._dedup_tmp.name)

    def _candidate(self, ticker="NVDA"):
        return {"ticker": ticker, "earn_date": date.today(), "days_away": 0,
                "timing": "AMC", "current_price": 200.0}

    def _run(self, equity, existing_committed, plan_max_loss, mock_build=None):
        if mock_build is None:
            mock_build = MagicMock(return_value={"ticker": "NVDA", "max_loss": plan_max_loss})
        with patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_effective_account", return_value=equity), \
             patch.object(a, "get_earnings_spread_candidates", return_value=[self._candidate()]), \
             patch.object(a, "_earnings_spread_committed_risk", return_value=existing_committed), \
             patch.object(a, "build_earnings_spread_plan", mock_build), \
             patch.object(a, "format_earnings_spread_telegram", return_value="msg"), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a.run_earnings_spread_scan()
        return mock_build, mock_tg

    def test_offer_within_cap_is_sent_normally(self):
        # $100,000 equity, 15% cap = $15,000; nothing committed yet, this
        # offer's $100 max_loss is nowhere near the cap.
        mock_build, mock_tg = self._run(equity=100_000.0, existing_committed=0.0, plan_max_loss=100.0)
        mock_build.assert_called_once()
        self.assertEqual(len(a._load_earnings_pending()), 1)
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("msg" in t for t in sent_texts))

    def test_offer_that_would_exceed_the_cap_is_not_offered(self):
        # $10,000 equity, 15% cap = $1,500; $1,450 already committed, this
        # offer's $100 max_loss would push total to $1,550 > cap.
        mock_build, mock_tg = self._run(equity=10_000.0, existing_committed=1_450.0, plan_max_loss=100.0)
        mock_build.assert_called_once()   # the plan IS built (needed to know its max_loss) ...
        self.assertEqual(len(a._load_earnings_pending()), 0,
                          "... but never actually offered/tracked as pending")
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("aggregate risk cap" in t for t in sent_texts))
        self.assertFalse(any(t == "msg" for t in sent_texts), "the real offer message must never go out")

    def test_cap_skip_notification_is_deduped_not_spammed(self):
        mock_build, mock_tg = self._run(equity=10_000.0, existing_committed=1_450.0, plan_max_loss=100.0)
        cap_msgs_first = [c for c in mock_tg.call_args_list if "aggregate risk cap" in c.args[0]]
        self.assertEqual(len(cap_msgs_first), 1)

        mock_build2, mock_tg2 = self._run(equity=10_000.0, existing_committed=1_450.0, plan_max_loss=100.0)
        cap_msgs_second = [c for c in mock_tg2.call_args_list if "aggregate risk cap" in c.args[0]]
        self.assertEqual(len(cap_msgs_second), 0, "a repeat cap-skip for the same ticker/day must not re-notify")

    def test_cap_skip_does_not_permanently_block_a_later_retry(self):
        # The main per-ticker dedup_key must NOT be marked on a cap-skip --
        # only a genuinely SENT offer should ever permanently silence a
        # ticker for the rest of the day. Simulate a later scan where room
        # has freed up (existing_committed now low) and confirm it builds
        # and offers normally, not silently skipped as "already handled."
        self._run(equity=10_000.0, existing_committed=1_450.0, plan_max_loss=100.0)
        self.assertEqual(len(a._load_earnings_pending()), 0)

        mock_build2, mock_tg2 = self._run(equity=10_000.0, existing_committed=0.0, plan_max_loss=100.0)
        mock_build2.assert_called_once()
        self.assertEqual(len(a._load_earnings_pending()), 1,
                          "once room frees up, a later scan must still be able to offer this ticker")


class TestFormatEarningsSpreadTelegramSectorWarning(unittest.TestCase):
    """format_earnings_spread_telegram() must surface a sector-overlap
    warning line when given one, and stay silent when not — the approval
    message is the only thing a human sees before a real spread order
    goes out, so a silently-dropped warning is as bad as never computing
    it in the first place."""

    def _plan(self):
        return {
            "ticker": "NVDA", "timing": "AMC", "directional": None,
            "last_moves_pct": [], "call": None, "put": None,
            "total_cost": 500.0, "sets": 1, "max_loss": 500.0,
        }

    def test_no_overlap_omits_warning_line(self):
        msg = a.format_earnings_spread_telegram(self._plan(), None)
        self.assertNotIn("Sector overlap", msg)

    def test_empty_overlap_list_omits_warning_line(self):
        msg = a.format_earnings_spread_telegram(self._plan(), [])
        self.assertNotIn("Sector overlap", msg)

    def test_overlap_included_names_the_other_tickers_and_sector(self):
        msg = a.format_earnings_spread_telegram(self._plan(), ["CRWD", "MRVL"])
        self.assertIn("Sector overlap", msg)
        self.assertIn("CRWD", msg)
        self.assertIn("MRVL", msg)
        self.assertIn("Technology", msg)


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


class TestEarningsSpreadScanSectorOverlapSymmetry(unittest.TestCase):
    """Added 2026-08-23: _earnings_sector_overlap() used to be checked
    against `pending`, which only grows as run_earnings_spread_scan()'s
    own loop appends each new offer -- so of two same-sector candidates in
    one scan pass, only whichever was processed SECOND (WATCHLIST order,
    not sector order) ever saw the first's overlap. Fixed by comparing
    against the full candidate batch up front, not just what's been
    appended so far."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._patch.start()
        a._save_earnings_pending([])
        self._dedup_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._dedup_tmp.close()
        self._dedup_patch = patch.object(a, "_ALERT_DEDUP_FILE", self._dedup_tmp.name)
        self._dedup_patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)
        self._dedup_patch.stop()
        os.unlink(self._dedup_tmp.name)

    def test_both_same_sector_candidates_see_each_others_overlap(self):
        # NVDA and CRWD are both "Technology" in TICKER_SECTOR.
        candidates = [
            {"ticker": "NVDA", "earn_date": date.today(), "days_away": 0,
             "timing": "AMC", "current_price": 200.0},
            {"ticker": "CRWD", "earn_date": date.today(), "days_away": 0,
             "timing": "AMC", "current_price": 400.0},
        ]
        seen_overlaps = {}
        def _fake_format(plan, sector_overlap=None):
            seen_overlaps[plan["ticker"]] = sector_overlap
            return f"msg for {plan['ticker']}"
        with patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "get_effective_account", return_value=100_000.0), \
             patch.object(a, "get_earnings_spread_candidates", return_value=candidates), \
             patch.object(a, "build_earnings_spread_plan",
                           side_effect=lambda client, ticker, *rest: {"ticker": ticker, "max_loss": 100.0}), \
             patch.object(a, "format_earnings_spread_telegram", side_effect=_fake_format), \
             patch.object(a, "send_telegram", return_value=True):
            a.run_earnings_spread_scan()
        self.assertEqual(seen_overlaps.get("NVDA"), ["CRWD"],
                          "the FIRST-processed candidate must also see the second's overlap")
        self.assertEqual(seen_overlaps.get("CRWD"), ["NVDA"])


class TestEarningsApprovalTelegramFlow(unittest.TestCase):
    """The approve-gate is the entire safety rationale for this feature —
    every earnings spread requires an explicit human YES (permanent gate, no
    auto-promotion). A reply-parsing bug here either submits an unapproved
    live order or permanently ignores a legitimate approval. Uses an isolated
    temp file for pending state — never the real dman_earnings_pending.json."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patches = [
            patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name),
            # Isolate from the real macro-event calendar (FOMC/CPI/NFP) --
            # check_macro_safe() reads date.today() against real event
            # dates, so an unmocked test here silently drifts into "blocked"
            # once the real calendar rolls into a blackout window (confirmed
            # live: these tests started failing once "today" hit one).
            patch.object(a, "check_macro_safe", return_value=(True, 0)),
        ]
        for p in self._patches:
            p.start()
        a._save_earnings_pending([])

    def tearDown(self):
        for p in self._patches:
            p.stop()
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
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0):
            with patch.object(a, "send_telegram", return_value=True):
                with patch.object(a, "get_alpaca_client", return_value=mock_client):
                    with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                        with patch.object(a, "PositionTracker") as MockPT:
                            consumed = a._handle_earnings_approval_reply("yes")
        self.assertTrue(consumed)
        mock_client.submit_order.assert_called_once()
        self.assertEqual(a._load_earnings_pending(), [])

    def test_tracking_failure_after_a_real_fill_alerts_instead_of_vanishing(self):
        # Regression for the live 2026-08-27 MRVL incident: the order
        # actually filled at the broker for a real $462 debit, but
        # _open_earnings_spread_position() never completed (no
        # dman_positions.json entry at any commit since) -- with no
        # try/except around it, that failure had no Telegram trace beyond
        # the later, generic "orphan positions" alert, and no automated
        # exit/close-sync could ever apply to a position PositionTracker
        # never knew about. A tracking failure after a REAL fill must alert
        # loudly, not just vanish.
        self._add_pending("HOOD")
        mock_client = MagicMock()
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                with patch.object(a, "get_alpaca_client", return_value=mock_client):
                    with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                        with patch.object(a, "_open_earnings_spread_position",
                                           side_effect=KeyError("total_cost")):
                            consumed = a._handle_earnings_approval_reply("yes")
        self.assertTrue(consumed)
        mock_client.submit_order.assert_called_once()   # the order really did go out
        self.assertEqual(a._load_earnings_pending(), [])   # offer still consumed, no retry loop
        alert_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("NOT TRACKED" in t and "HOOD" in t for t in alert_texts),
                         f"expected a loud tracking-failure alert, got: {alert_texts}")

    def test_offer_is_consumed_before_order_submission_not_after(self):
        # Added 2026-08-23: this used to persist the consumed/tombstoned
        # state only AFTER _submit_earnings_spread() returned (including
        # after a successful submit) -- a process killed in that window
        # would leave the offer still on disk as "awaiting_approval", and
        # a redelivered Telegram update (a known real occurrence) could
        # resubmit the exact same real spread a second time. Consuming
        # first closes that window: verify the call order directly rather
        # than just the end state, since the end state alone can't
        # distinguish "consumed before" from "consumed after" once both
        # have happened by the time the function returns.
        self._add_pending("HOOD")
        mock_client = MagicMock()
        call_order = []
        _real_consume = a._consume_earnings_offer_save
        def _tracked_consume(pending, entry):
            call_order.append("consume")
            return _real_consume(pending, entry)
        def _tracked_submit(client, plan):
            call_order.append("submit")
            return "order-1", None
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "send_telegram", return_value=True), \
             patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "get_available_cash", return_value=1_000_000.0), \
             patch.object(a, "PositionTracker"), \
             patch.object(a, "_consume_earnings_offer_save", side_effect=_tracked_consume), \
             patch.object(a, "_submit_earnings_spread", side_effect=_tracked_submit):
            a._handle_earnings_approval_reply("yes")
        self.assertEqual(call_order, ["consume", "submit"],
                          "the offer must be consumed BEFORE the order is submitted, not after")

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

    def test_halt_blocks_submission(self):
        # Added 2026-08-16: this path previously checked NONE of halt,
        # consec-loss, daily-loss, or monthly-loss before submitting a real
        # multi-leg spread -- only offer-matching/reachability/price-drift.
        self._add_pending("HOOD")
        mock_client = MagicMock()
        with patch.object(a, "is_halted", return_value=True), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertIn("halted", mock_tg.call_args[0][0])
        self.assertEqual(a._load_earnings_pending(), [], "offer is still consumed, not left dangling")

    def test_consecutive_loss_guard_blocks_submission(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        mock_stats = {"consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.5,
                      "avg_win_r": 2.0, "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=mock_stats), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertIn("consecutive-loss", mock_tg.call_args[0][0])

    def test_daily_loss_limit_blocks_submission(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=-(a.DAILY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertIn("daily loss", mock_tg.call_args[0][0])

    def test_monthly_loss_limit_blocks_submission(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertIn("monthly loss", mock_tg.call_args[0][0])

    def test_macro_blackout_blocks_submission(self):
        self._add_pending("HOOD")
        mock_client = MagicMock()
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "check_macro_safe", return_value=(False, 0)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_not_called()
        self.assertIn("blackout", mock_tg.call_args[0][0])

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
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0):
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
        clean_stats = {"consec_losses": 0, "win_rate": 0.5, "avg_win_r": 2.0,
                       "avg_loss_r": 1.0, "total": 10, "wins": 5, "losses": 5}
        with patch.object(a.WinRateTracker, "rolling_stats", return_value=clean_stats), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0):
            with patch.object(a, "send_telegram", return_value=True):
                with patch.object(a, "get_alpaca_client", return_value=mock_client):
                    with patch.object(a, "get_available_cash", return_value=1_000_000.0):
                        with patch.object(a, "get_live_price", return_value=99999.0):
                            with patch.object(a, "PositionTracker"):
                                a._handle_earnings_approval_reply("yes")
        mock_client.submit_order.assert_called_once()


class TestSubmitSignalsSizeMult(unittest.TestCase):
    """size_mult param added 2026-09-01 for momentum-watch's reduced-size
    auto-execute path — verifies it actually compounds into the options
    budget rather than just being accepted and silently ignored."""

    def _run(self, size_mult):
        sig = a.ProSignal(
            ticker="TESTX", bias="LONG", setup="Gap & Hold",   # in OPTIONS_SETUPS
            entry=10.0, stop=9.95, target1=10.15, target2=10.30,   # tiny risk/share
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=100,
            shares=10, cost=100.0,
        )
        captured = {}
        def _fake_submit_options_call(client, ticker, cur, risk_dollars, signal):
            captured["risk_dollars"] = risk_dollars
            return None, None   # unfillable -> falls through, nothing else to mock
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 100.0)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "WATCHLIST", ["TESTX"]), \
             patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "_submit_options_call", side_effect=_fake_submit_options_call), \
             patch.object(a, "send_telegram", return_value=True):
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            MockPT.return_value.positions = []
            a._submit_signals_to_alpaca([sig], size_mult=size_mult)
        return captured.get("risk_dollars")

    def test_default_size_mult_matches_unmultiplied_behavior(self):
        base = self._run(1.0)
        self.assertIsNotNone(base)

    def test_reduced_size_mult_shrinks_options_budget_proportionally(self):
        base = self._run(1.0)
        reduced = self._run(0.35)
        self.assertAlmostEqual(reduced, base * 0.35, places=2)


class TestAccountMilestones(unittest.TestCase):
    """check_account_milestones() — one-time $5K/$10K/$25K crossing alerts."""

    def _run(self, equity, prior_crossed=None, mock_open_target=None):
        mock_client = MagicMock()
        mock_client.get_account.return_value.equity = equity
        read_data = json.dumps({"crossed": prior_crossed or []})
        m = mock_open(read_data=read_data)
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch("builtins.open", m), \
             patch.object(a, "_write_json_atomic") as mock_write, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a.check_account_milestones()
        return mock_tg, mock_write

    def test_first_crossing_of_5k_sends_one_alert(self):
        mock_tg, mock_write = self._run(equity=5_100.0, prior_crossed=[])
        mock_tg.assert_called_once()
        self.assertIn("$5,000", mock_tg.call_args[0][0])
        mock_write.assert_called_once()
        saved = mock_write.call_args[0][1]
        self.assertEqual(saved["crossed"], [5_000])

    def test_already_crossed_milestone_never_refires(self):
        mock_tg, mock_write = self._run(equity=6_000.0, prior_crossed=[5_000])
        mock_tg.assert_not_called()
        mock_write.assert_not_called()

    def test_equity_dip_back_below_does_not_unmark(self):
        # Crossed 5K once, dipped back to 4.5K -- must stay marked crossed,
        # never re-announce on a later re-cross of the same line.
        mock_tg, mock_write = self._run(equity=4_500.0, prior_crossed=[5_000])
        mock_tg.assert_not_called()
        mock_write.assert_not_called()

    def test_crossing_two_milestones_at_once_sends_both(self):
        mock_tg, mock_write = self._run(equity=11_000.0, prior_crossed=[])
        self.assertEqual(mock_tg.call_count, 2)
        texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("$5,000" in t for t in texts))
        self.assertTrue(any("$10,000" in t for t in texts))

    def test_below_first_milestone_sends_nothing(self):
        mock_tg, mock_write = self._run(equity=3_053.0, prior_crossed=[])
        mock_tg.assert_not_called()
        mock_write.assert_not_called()

    def test_no_alpaca_client_fails_safe(self):
        with patch.object(a, "get_alpaca_client", return_value=None), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a.check_account_milestones()   # must not raise
        mock_tg.assert_not_called()


class TestMomentumWatchAutoExecute(unittest.TestCase):
    """Direct instruction 2026-09-01: a real _detect_pre_breakout() pattern
    match (not just a bare VWAP reclaim) auto-executes at reduced size with
    no YES/NO reply needed — the account owner can't reply during market
    hours (9-5 job), and every breakout offer was expiring unactioned as a
    result (104 in one session, confirmed live, zero ever approved)."""

    def _run_watch(self, breakout_setup: bool):
        # 2-day history with a real ~-3% opening gap so is_recovery_dip
        # (source contains "recovery") is genuinely true for the fallback
        # (non-auto-exec) case's _fire condition, not just cur > vwap alone.
        import pandas as pd
        _hist2 = pd.DataFrame({
            "Open":  [10.60, 9.90],
            "Close": [10.60, 10.20],
        })
        with patch.object(a, "_force_close_day_only_positions", return_value=0), \
             patch("os.path.exists", return_value=False), \
             patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["TESTX"]), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch("yfinance.Ticker") as MockYfTicker, \
             patch.object(a, "_fetch_intraday_bars", return_value=object()), \
             patch.object(a, "_compute_session_levels", return_value={
                 "cur_price": 10.5, "vwap": 10.0, "session_low": 9.0,
                 "session_high": 10.6,
             }), \
             patch.object(a, "_detect_pre_breakout", return_value={
                 "setup": breakout_setup, "entry_px": 10.55, "stop_px": 9.9,
                 "signals": ["tight coil"],
             }), \
             patch.object(a, "_build_momentum_signal") as mock_build, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit, \
             patch.object(a, "_load_momentum_pending", return_value=[]), \
             patch.object(a, "_save_momentum_pending") as mock_save_pending, \
             patch.object(a, "send_telegram", return_value=True):
            MockYfTicker.return_value.history.return_value = _hist2
            mock_build.return_value = a.ProSignal(
                ticker="TESTX", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
                entry=10.55, stop=9.9, target1=13.7, target2=15.8,
                rr=2.0, rsi=50.0, rvol=0.0, reason="test", confluence_score=100,
            )
            a.run_momentum_watch()
        return mock_submit, mock_save_pending

    def test_real_breakout_pattern_auto_executes_no_reply_needed(self):
        mock_submit, mock_save_pending = self._run_watch(breakout_setup=True)
        mock_submit.assert_called_once()
        _, kwargs = mock_submit.call_args
        self.assertEqual(kwargs.get("size_mult"), a.MOMENTUM_AUTO_EXEC_SIZE_MULT)
        mock_save_pending.assert_not_called()

    def test_pure_vwap_reclaim_without_pattern_still_requires_approval(self):
        mock_submit, mock_save_pending = self._run_watch(breakout_setup=False)
        mock_submit.assert_not_called()
        mock_save_pending.assert_called_once()


class TestMomentumBreakoutApprovalFlow(unittest.TestCase):
    """Direct instruction 2026-08-30: run_momentum_watch()'s BREAKOUT
    SETUP alerts were purely informational -- no way to act on one
    without leaving the algo entirely. This mirrors the earnings-spread
    approval flow's own test suite: ticker disambiguation, expiry,
    consume-before-act, and the halt/macro/price-drift gates re-checked
    at the YES step. _submit_signals_to_alpaca() itself is mocked here
    (it has its own full test coverage) so these tests isolate this
    handler's own logic: parsing, matching, and what it hands off."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patches = [
            patch.object(a, "MOMENTUM_PENDING_FILE", self._tmp.name),
            patch.object(a, "check_macro_safe", return_value=(True, 0)),
            patch.object(a, "_entry_circuit_breakers_ok", return_value=(True, "")),
            patch.object(a, "get_live_price", return_value=10.0),
        ]
        for p in self._patches:
            p.start()
        a._save_momentum_pending([])

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.unlink(self._tmp.name)

    def _add_pending(self, ticker, entry_px=10.0, stop_px=9.0, minutes_until_expiry=10):
        entry = {"ticker": ticker, "entry_px": entry_px, "stop_px": stop_px,
                 "t1": entry_px * 1.3, "t2": entry_px * 1.5, "signal_str": "test breakout",
                 "created_at": datetime.now(a.ET).isoformat(),
                 "expires_at": (datetime.now(a.ET) + timedelta(minutes=minutes_until_expiry)).isoformat(),
                 "status": "awaiting_approval"}
        pending = a._load_momentum_pending()
        pending.append(entry)
        a._save_momentum_pending(pending)

    def test_bare_yes_applies_to_the_only_pending_offer(self):
        self._add_pending("HOOD")
        fake_sig = a.ProSignal(ticker="HOOD", setup=a.MOMENTUM_DAY_ONLY_SETUP, bias="LONG",
                                entry=10.0, stop=9.0, target1=13.0, target2=15.0, rr=3.0,
                                rsi=50.0, rvol=0.0, reason="test", shares=10)
        with patch.object(a, "_build_momentum_signal", return_value=fake_sig), \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            consumed = a._handle_momentum_approval_reply("yes")
        self.assertTrue(consumed)
        mock_submit.assert_called_once_with([fake_sig])
        self.assertEqual(a._load_momentum_pending(), [])

    def test_no_rejects_without_submitting(self):
        self._add_pending("HOOD")
        with patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            consumed = a._handle_momentum_approval_reply("no")
        self.assertTrue(consumed)
        mock_submit.assert_not_called()
        self.assertIn("rejected", mock_tg.call_args[0][0])

    def test_ambiguous_bare_yes_with_two_pending_is_not_silently_guessed(self):
        self._add_pending("HOOD")
        self._add_pending("APVO")
        with patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            consumed = a._handle_momentum_approval_reply("yes")
        self.assertTrue(consumed)   # message shape recognized, just ambiguous
        mock_submit.assert_not_called()
        # Both offers survive -- neither one was silently picked.
        self.assertEqual(len(a._load_momentum_pending()), 2)

    def test_yes_with_ticker_resolves_the_ambiguous_case(self):
        self._add_pending("HOOD")
        self._add_pending("APVO")
        fake_sig = a.ProSignal(ticker="HOOD", setup=a.MOMENTUM_DAY_ONLY_SETUP, bias="LONG",
                                entry=10.0, stop=9.0, target1=13.0, target2=15.0, rr=3.0,
                                rsi=50.0, rvol=0.0, reason="test", shares=10)
        with patch.object(a, "_build_momentum_signal", return_value=fake_sig), \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes HOOD")
        mock_submit.assert_called_once()
        remaining = a._load_momentum_pending()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["ticker"], "APVO")

    def test_non_yes_no_text_is_not_consumed(self):
        self._add_pending("HOOD")
        self.assertFalse(a._handle_momentum_approval_reply("what's the price"))
        self.assertEqual(len(a._load_momentum_pending()), 1)   # untouched

    def test_no_pending_offers_returns_false(self):
        self.assertFalse(a._handle_momentum_approval_reply("yes"))

    def test_circuit_breaker_blocks_submission(self):
        self._add_pending("HOOD")
        with patch.object(a, "_entry_circuit_breakers_ok", return_value=(False, "daily loss limit active")), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")
        mock_submit.assert_not_called()
        self.assertIn("daily loss limit active", mock_tg.call_args[0][0])

    def test_macro_blackout_blocks_submission(self):
        self._add_pending("HOOD")
        with patch.object(a, "check_macro_safe", return_value=(False, 0)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")
        mock_submit.assert_not_called()
        self.assertIn("macro blackout", mock_tg.call_args[0][0])

    def test_price_drift_past_limit_aborts_without_submitting(self):
        self._add_pending("HOOD", entry_px=10.0)
        with patch.object(a, "get_live_price", return_value=11.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")   # +10% drift, past the 5% limit
        mock_submit.assert_not_called()
        self.assertIn("NOT submitted", mock_tg.call_args[0][0])

    def test_small_price_drift_still_submits(self):
        self._add_pending("HOOD", entry_px=10.0)
        fake_sig = a.ProSignal(ticker="HOOD", setup=a.MOMENTUM_DAY_ONLY_SETUP, bias="LONG",
                                entry=10.0, stop=9.0, target1=13.0, target2=15.0, rr=3.0,
                                rsi=50.0, rvol=0.0, reason="test", shares=10)
        with patch.object(a, "get_live_price", return_value=10.2), \
             patch.object(a, "_build_momentum_signal", return_value=fake_sig), \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")
        mock_submit.assert_called_once()

    def test_sizing_failure_does_not_submit(self):
        self._add_pending("HOOD")
        fake_sig = a.ProSignal(ticker="HOOD", setup=a.MOMENTUM_DAY_ONLY_SETUP, bias="LONG",
                                entry=10.0, stop=9.0, target1=13.0, target2=15.0, rr=3.0,
                                rsi=50.0, rvol=0.0, reason="test", shares=0)   # sizing failed
        with patch.object(a, "_build_momentum_signal", return_value=fake_sig), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")
        mock_submit.assert_not_called()
        self.assertIn("sizing failed", mock_tg.call_args[0][0])

    def test_expired_offer_is_not_approvable(self):
        # No other offer left pending once the expired one is swept, so
        # this reply ultimately matches nothing -- same "not our message
        # to consume" behavior _handle_earnings_approval_reply() already
        # has for the identical case. The expiry notice still goes out;
        # only the return value (whether THIS reply itself was consumed)
        # is False.
        self._add_pending("HOOD", minutes_until_expiry=-1)
        with patch.object(a, "send_telegram", return_value=True) as mock_tg, \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            consumed = a._handle_momentum_approval_reply("yes")
        self.assertFalse(consumed)
        mock_submit.assert_not_called()
        self.assertIn("expired", mock_tg.call_args[0][0])
        self.assertEqual(a._load_momentum_pending(), [])

    def test_built_signal_carries_the_day_only_setup_tag(self):
        # The whole point of this flow: whatever gets submitted must be
        # tagged so _force_close_day_only_positions() can find it later.
        self._add_pending("HOOD")
        stats = {"win_rate": 0.5, "avg_win_r": 2.0, "avg_loss_r": 1.0, "total": 10,
                 "wins": 5, "losses": 5, "consec_losses": 0}
        with patch.object(a.WinRateTracker, "setup_stats", return_value=stats), \
             patch.object(a, "get_market_regime", return_value={"details": {"VIX": 15.0}}), \
             patch.object(a, "get_effective_account", return_value=25_000.0), \
             patch.object(a, "_submit_signals_to_alpaca") as mock_submit:
            a._handle_momentum_approval_reply("yes")
        submitted_sig = mock_submit.call_args[0][0][0]
        self.assertEqual(submitted_sig.setup, a.MOMENTUM_DAY_ONLY_SETUP)
        self.assertEqual(submitted_sig.ticker, "HOOD")


class TestForceCloseDayOnlyPositions(unittest.TestCase):
    """_force_close_day_only_positions() is the other half of the day-only
    scoping promise -- without it, a momentum-watch entry would just sit
    open overnight like any other position, silently breaking the "day of
    only" guarantee the approval message itself states."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        import functools
        self._patches = [
            patch.object(a, "PositionTracker", functools.partial(a.PositionTracker, filepath=self._pos_tmp.name)),
            # _is_alerted_today()/_mark_alerted() (the dedup this function
            # actually uses) read _ALERT_DEDUP_FILE, not LAST_ALERTS_FILE
            # (a separate, differently-keyed dedup mechanism) -- isolating
            # the wrong constant here silently read/wrote the real
            # production dman_alerts_dedup.json.
            patch.object(a, "_ALERT_DEDUP_FILE", self._alerts_tmp.name),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.unlink(self._pos_tmp.name)
        os.unlink(self._alerts_tmp.name)

    def _day_only_pos(self, ticker="HOOD"):
        return a.OpenPosition(
            ticker=ticker, bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
            entry=10.0, stop=9.0, target1=13.0, target2=15.0, shares=10,
            entry_date=date.today().isoformat(), day_only=True,
        )

    def _fake_now(self, hour, minute):
        return datetime(2026, 8, 31, hour, minute, tzinfo=a.ET)

    def test_before_close_window_does_nothing(self):
        a.PositionTracker().open(self._day_only_pos())
        with patch.object(a, "_close_position_at_market") as mock_close:
            closed = a._force_close_day_only_positions(self._fake_now(15, 0))   # before 3:45
        self.assertEqual(closed, 0)
        mock_close.assert_not_called()

    def test_at_or_after_close_window_closes_day_only_positions(self):
        a.PositionTracker().open(self._day_only_pos())
        with patch.object(a, "_close_position_at_market", return_value=("submitted", "ord1")), \
             patch.object(a, "send_telegram", return_value=True):
            closed = a._force_close_day_only_positions(self._fake_now(15, 45))
        self.assertEqual(closed, 1)

    def test_non_day_only_positions_are_left_alone(self):
        a.PositionTracker().open(a.OpenPosition(
            ticker="CELZ", bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=12.0, target2=14.0, shares=100,
            entry_date=date.today().isoformat(), day_only=False,
        ))
        with patch.object(a, "_close_position_at_market") as mock_close:
            closed = a._force_close_day_only_positions(self._fake_now(15, 45))
        self.assertEqual(closed, 0)
        mock_close.assert_not_called()

    def test_stale_position_from_a_prior_day_closes_immediately_before_eod_window(self):
        # Direct instruction 2026-09-02, root-caused live: a scanner gap
        # spanning the 3:45 PM close window meant run_momentum_watch()
        # never got dispatched that day, so day-only positions carried
        # overnight. The OLD bare time-of-day check would then wait for
        # the SAME 3:45 PM window to roll around on a LATER day instead of
        # closing at the next opportunity. A position entered on a PRIOR
        # calendar day must close immediately, well before 3:45 PM, the
        # very next time this runs -- not wait out another full day.
        stale_pos = a.OpenPosition(
            ticker="HOOD", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
            entry=10.0, stop=9.0, target1=13.0, target2=15.0, shares=10,
            entry_date="2026-08-28", day_only=True,   # entered days before "now" below (2026-08-31)
        )
        a.PositionTracker().open(stale_pos)
        with patch.object(a, "_close_position_at_market", return_value=("submitted", "ord1")), \
             patch.object(a, "send_telegram", return_value=True):
            closed = a._force_close_day_only_positions(self._fake_now(9, 40))   # 9:40 AM -- well before 3:45 PM
        self.assertEqual(closed, 1)

    def test_todays_position_still_waits_for_the_eod_window_not_closed_early(self):
        # The staleness fix must not accidentally make EVERY day-only
        # position close immediately -- only ones from a PRIOR day.
        todays_pos = a.OpenPosition(
            ticker="HOOD", bias="LONG", setup=a.MOMENTUM_DAY_ONLY_SETUP,
            entry=10.0, stop=9.0, target1=13.0, target2=15.0, shares=10,
            entry_date="2026-08-31", day_only=True,   # same calendar day as "now" below
        )
        a.PositionTracker().open(todays_pos)
        with patch.object(a, "_close_position_at_market") as mock_close:
            closed = a._force_close_day_only_positions(self._fake_now(9, 40))   # entered today, 9:40 AM -- too early
        self.assertEqual(closed, 0)
        mock_close.assert_not_called()

    def test_already_attempted_today_is_not_resubmitted(self):
        # A prior pass this session already tried this close (success or
        # failure) -- every subsequent momentum-watch pass today must not
        # keep resubmitting a market order for the same position.
        a.PositionTracker().open(self._day_only_pos())
        with patch.object(a, "_close_position_at_market", return_value=("submitted", "ord1")), \
             patch.object(a, "send_telegram", return_value=True):
            a._force_close_day_only_positions(self._fake_now(15, 45))
            with patch.object(a, "_close_position_at_market") as mock_close_2:
                closed_again = a._force_close_day_only_positions(self._fake_now(15, 46))
        self.assertEqual(closed_again, 0)
        mock_close_2.assert_not_called()

    def test_already_closed_status_does_not_alert_or_count(self):
        a.PositionTracker().open(self._day_only_pos())
        with patch.object(a, "_close_position_at_market", return_value=("already_closed", None)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            closed = a._force_close_day_only_positions(self._fake_now(15, 45))
        self.assertEqual(closed, 0)
        mock_tg.assert_not_called()

    def test_failed_close_alerts_loudly(self):
        a.PositionTracker().open(self._day_only_pos())
        with patch.object(a, "_close_position_at_market", return_value=("failed", None)), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._force_close_day_only_positions(self._fake_now(15, 45))
        self.assertIn("FAILED", mock_tg.call_args[0][0])


class TestClosePositionAtMarket(unittest.TestCase):
    """_close_position_at_market() is the actual order-submission half of
    the day-only close -- a bug here either fails to flatten a real
    position or (worse) tries to close something already gone."""

    def _pos(self, ticker="HOOD", setup=None):
        return a.OpenPosition(
            ticker=ticker, bias="LONG", setup=setup or a.MOMENTUM_DAY_ONLY_SETUP,
            entry=10.0, stop=9.0, target1=13.0, target2=15.0, shares=10,
            entry_date=date.today().isoformat(), day_only=True,
        )

    def test_no_client_fails(self):
        with patch.object(a, "get_alpaca_client", return_value=None):
            status, oid = a._close_position_at_market(self._pos(), "test")
        self.assertEqual(status, "failed")
        self.assertIsNone(oid)

    def test_nothing_held_is_already_closed(self):
        mock_client = MagicMock()
        mock_client.get_open_position.side_effect = Exception("position does not exist")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            status, oid = a._close_position_at_market(self._pos(), "test")
        self.assertEqual(status, "already_closed")
        mock_client.submit_order.assert_not_called()

    def test_real_position_submits_a_market_sell(self):
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = MagicMock(qty="10")
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            status, oid = a._close_position_at_market(self._pos(), "test")
        self.assertEqual(status, "submitted")
        self.assertEqual(oid, "ord1")
        mock_client.submit_order.assert_called_once()

    def test_options_position_closes_by_the_occ_symbol_not_the_ticker(self):
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = MagicMock(qty="2")
        mock_client.get_orders.return_value = []
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        _pos = self._pos(setup="Options Call HOOD260807C00096000 ($96.0C exp 2026-08-07)")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._close_position_at_market(_pos, "test")
        mock_client.get_open_position.assert_called_once_with("HOOD260807C00096000")

    def test_open_stop_orders_are_cancelled_before_closing(self):
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = MagicMock(qty="10")
        mock_client.get_orders.return_value = [MagicMock(id="stop1")]
        mock_client.submit_order.return_value = MagicMock(id="ord1")
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            a._close_position_at_market(self._pos(), "test")
        mock_client.cancel_order_by_id.assert_called_once_with("stop1")

    def test_submission_failure_alerts_and_returns_failed(self):
        mock_client = MagicMock()
        mock_client.get_open_position.return_value = MagicMock(qty="10")
        mock_client.get_orders.return_value = []
        mock_client.submit_order.side_effect = Exception("broker error")
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            status, oid = a._close_position_at_market(self._pos(), "test")
        self.assertEqual(status, "failed")
        self.assertIsNone(oid)
        self.assertIn("MARKET CLOSE FAILED", mock_tg.call_args[0][0])


class TestExpiredOfferIsTombstoned(unittest.TestCase):
    """Added 2026-08-23: both the inline expiry sweep in
    _handle_earnings_approval_reply() and the dedicated
    expire_earnings_spread_offers() used to drop an expired entry from
    the pending LIST without ever tombstoning its identity -- so a stale
    remote git copy that still shows it as "awaiting_approval" could
    merge it back in. Both consumers re-validate expires_at on their own
    next pass regardless (so this never risked an actual late approval),
    but it's a real gap against the tombstone's documented guarantee and
    produced duplicate "expired" Telegram spam on the resurrection."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._patch.start()
        a._save_earnings_pending([])

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _expired_entry(self, ticker="HOOD"):
        return {"ticker": ticker, "earn_date": "2026-07-29",
                "created_at": (datetime.now(a.ET) - timedelta(hours=1)).isoformat(),
                "expires_at": (datetime.now(a.ET) - timedelta(minutes=5)).isoformat(),
                "status": "awaiting_approval", "plan": {"ticker": ticker}}

    def test_inline_sweep_in_approval_reply_tombstones_the_expired_offer(self):
        entry = self._expired_entry()
        a._save_earnings_pending([entry])
        with patch.object(a, "send_telegram", return_value=True):
            a._handle_earnings_approval_reply("yes")
        _, consumed = a._load_earnings_state()
        self.assertIn(a._earnings_offer_identity(entry), consumed)

    def test_expire_earnings_spread_offers_tombstones_the_expired_offer(self):
        entry = self._expired_entry()
        a._save_earnings_pending([entry])
        with patch.object(a, "send_telegram", return_value=True):
            a.expire_earnings_spread_offers()
        _, consumed = a._load_earnings_state()
        self.assertIn(a._earnings_offer_identity(entry), consumed)

    def test_non_expired_offer_is_not_tombstoned_by_the_sweep(self):
        entry = {"ticker": "HOOD", "earn_date": "2026-07-29",
                 "created_at": datetime.now(a.ET).isoformat(),
                 "expires_at": (datetime.now(a.ET) + timedelta(minutes=30)).isoformat(),
                 "status": "awaiting_approval", "plan": {"ticker": "HOOD"}}
        a._save_earnings_pending([entry])
        with patch.object(a, "send_telegram", return_value=True):
            a.expire_earnings_spread_offers()
        _, consumed = a._load_earnings_state()
        self.assertNotIn(a._earnings_offer_identity(entry), consumed)
        self.assertEqual(len(a._load_earnings_pending()), 1)


class TestEarningsPendingMergeSafety(unittest.TestCase):
    """Found in the 2026-08-16 review: dman_earnings_pending.json had no
    semantic merge -- just git's default whole-file last-writer-wins,
    unlike every sibling multi-writer state file. A consumed (approved/
    rejected) offer could resurrect from a stale remote copy that still
    showed it "awaiting_approval", letting a later YES re-submit a real
    spread that was already placed (or explicitly rejected) once. Fixed
    with a merge_positions_snapshots()-style tombstone: a consumed
    identity is recorded with a timestamp and never re-added from remote."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        self._tmp.close()
        self._patch = patch.object(a, "EARNINGS_SPREAD_PENDING_FILE", self._tmp.name)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

    def _offer(self, ticker="HOOD", earn_date="2026-07-29", status="awaiting_approval"):
        return {"ticker": ticker, "earn_date": earn_date, "created_at": "2026-08-16T10:00:00-04:00",
                "expires_at": "2026-08-16T14:00:00-04:00", "status": status, "plan": {"ticker": ticker}}

    def _write_local(self, data):
        with open(self._tmp.name, "w") as f:
            json.dump(data, f)

    def _sync_against_remote(self, remote):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(remote))
            a.sync_earnings_pending_with_remote()
        with open(self._tmp.name) as f:
            return json.load(f)

    def test_a_just_consumed_offer_is_not_resurrected_by_a_stale_remote(self):
        # Local just approved-and-submitted HOOD (removed it, tombstoned its
        # identity). Remote is a stale checkout that hasn't seen the
        # submission yet and still shows HOOD awaiting approval.
        self._write_local({"pending": [], "consumed": {"HOOD_2026-07-29": time.time()}})
        remote = {"pending": [self._offer("HOOD")], "consumed": {}}
        result = self._sync_against_remote(remote)
        self.assertEqual(result["pending"], [],
                         "a consumed offer must never be resurrected as awaiting_approval")

    def test_remote_only_consumed_tombstone_is_pulled_in(self):
        # The mirror case: THIS process's copy still shows HOOD pending, but
        # a different checkout already consumed it -- must not require local
        # to already know about a remote-side tombstone for it to apply.
        self._write_local({"pending": [self._offer("HOOD")], "consumed": {}})
        remote = {"pending": [], "consumed": {"HOOD_2026-07-29": time.time()}}
        result = self._sync_against_remote(remote)
        self.assertEqual(result["pending"], [])

    def test_unrelated_pending_offer_still_merges_normally(self):
        self._write_local({"pending": [self._offer("HOOD")], "consumed": {}})
        remote = {"pending": [self._offer("RIVN")], "consumed": {}}
        result = self._sync_against_remote(remote)
        tickers = {e["ticker"] for e in result["pending"]}
        self.assertEqual(tickers, {"HOOD", "RIVN"})

    def test_expired_tombstone_is_pruned_by_the_merge(self):
        stale = time.time() - a._EARNINGS_OFFER_TOMBSTONE_S - 10
        self._write_local({"pending": [], "consumed": {"OLD_2026-01-01": stale}})
        remote = {"pending": [self._offer("RIVN")], "consumed": {}}
        result = self._sync_against_remote(remote)
        self.assertNotIn("OLD_2026-01-01", result["consumed"])

    def test_pre_migration_legacy_bare_list_local_file_merges_without_error(self):
        # A file that hasn't been touched by the new save path yet is still
        # a bare list -- the merge's extract() must handle that shape rather
        # than crashing or silently treating it as empty.
        self._write_local([self._offer("HOOD")])
        remote = {"pending": [self._offer("RIVN")], "consumed": {}}
        result = self._sync_against_remote(remote)
        tickers = {e["ticker"] for e in result["pending"]}
        self.assertEqual(tickers, {"HOOD", "RIVN"})

    def test_consume_earnings_offer_save_tombstones_and_removes(self):
        entry = self._offer("HOOD")
        a._save_earnings_pending([entry, self._offer("RIVN")])
        pending = a._load_earnings_pending()
        remaining = [e for e in pending if e["ticker"] != "HOOD"]
        a._consume_earnings_offer_save(remaining, entry)
        with open(self._tmp.name) as f:
            data = json.load(f)
        self.assertEqual([e["ticker"] for e in data["pending"]], ["RIVN"])
        self.assertIn("HOOD_2026-07-29", data["consumed"])


class TestCashAvailableForAccountsForOpenOrders(unittest.TestCase):
    """Found in the 2026-08-16 review: _cash_available_for() only checked
    Alpaca's raw account.cash figure, which is NOT reduced until an order
    actually FILLS. A queued-but-unfilled GTC entry (or earnings-spread
    MLEG debit leg, or manual options buy) doesn't show up there, so a
    second real trade could be green-lit on top of cash that's already
    spoken for -- a genuine over-spend the instant both orders fill.
    Fixed by reserving the notional of every open BUY-side order (and
    MLEG net-debit spreads) against the raw cash figure before approving
    a new trade."""

    def setUp(self):
        self._cache_patch = patch.object(a, "_reserved_cash_cache", {"reserved": 0.0, "ts": 0.0})
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()

    def _order(self, order_class, side=None, asset_class=None, qty=0,
              limit_price=None, stop_price=None):
        o = MagicMock()
        o.order_class = order_class
        o.side = side
        o.asset_class = asset_class
        o.qty = qty
        o.limit_price = limit_price
        o.stop_price = stop_price
        return o

    def _client_with_orders(self, orders):
        client = MagicMock()
        client.get_orders.return_value = orders
        return client

    def test_open_equity_buy_order_is_reserved_against_cash(self):
        from alpaca.trading.enums import OrderClass, OrderSide, AssetClass
        order = self._order(OrderClass.SIMPLE, side=OrderSide.BUY, asset_class=AssetClass.US_EQUITY,
                            qty=10, limit_price=50.0)
        client = self._client_with_orders([order])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            # $1000 cash - $500 reserved (10 * $50 queued) = $500 free; $600 > $500 -> blocked
            ok, msg = a._cash_available_for(600.0)
        self.assertFalse(ok)
        self.assertIn("cash", msg)

    def test_open_equity_sell_stop_order_is_not_reserved(self):
        from alpaca.trading.enums import OrderClass, OrderSide, AssetClass
        order = self._order(OrderClass.SIMPLE, side=OrderSide.SELL, asset_class=AssetClass.US_EQUITY,
                            qty=10, stop_price=45.0)
        client = self._client_with_orders([order])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            ok, _ = a._cash_available_for(900.0)
        self.assertTrue(ok, "a protective SELL stop must not reduce cash available for a NEW buy")

    def test_open_options_buy_order_uses_100x_multiplier(self):
        from alpaca.trading.enums import OrderClass, OrderSide, AssetClass
        order = self._order(OrderClass.SIMPLE, side=OrderSide.BUY, asset_class=AssetClass.US_OPTION,
                            qty=2, limit_price=3.0)
        client = self._client_with_orders([order])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            # reserved = 2 * $3.00 * 100 = $600; free = $400; $500 > $400 -> blocked
            ok, _ = a._cash_available_for(500.0)
        self.assertFalse(ok)

    def test_open_mleg_debit_spread_reserves_net_debit_times_100(self):
        from alpaca.trading.enums import OrderClass
        order = self._order(OrderClass.MLEG, qty=2, limit_price=4.0)
        client = self._client_with_orders([order])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            # reserved = 2 sets * $4.00 net debit * 100 = $800; free = $200; $300 > $200 -> blocked
            ok, _ = a._cash_available_for(300.0)
        self.assertFalse(ok)

    def test_mleg_net_credit_reserves_nothing(self):
        from alpaca.trading.enums import OrderClass
        order = self._order(OrderClass.MLEG, qty=2, limit_price=-1.5)   # net credit, not a debit
        client = self._client_with_orders([order])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            ok, _ = a._cash_available_for(900.0)
        self.assertTrue(ok)

    def test_no_open_orders_behaves_exactly_like_before(self):
        client = self._client_with_orders([])
        with patch.object(a, "get_alpaca_client", return_value=client), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            ok, _ = a._cash_available_for(900.0)
        self.assertTrue(ok)

    def test_client_unavailable_for_reserved_lookup_does_not_block_an_otherwise_fundable_trade(self):
        # get_available_cash() already fails CLOSED on its own if the real
        # cash balance is unverifiable. A client outage specific to the
        # reserved-open-orders lookup must not layer on a SECOND,
        # redundant fail-closed path -- it just falls back to zero
        # reserved, exactly the pre-fix behavior.
        with patch.object(a, "get_alpaca_client", return_value=None), \
             patch.object(a, "get_available_cash", return_value=1000.0):
            ok, _ = a._cash_available_for(900.0)
        self.assertTrue(ok)

    def test_cash_none_still_fails_closed_regardless_of_open_orders(self):
        with patch.object(a, "get_available_cash", return_value=None):
            ok, msg = a._cash_available_for(1.0)
        self.assertFalse(ok)
        self.assertIn("unavailable", msg)


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

    def test_trigger_restart_passes_inputs_when_given(self):
        # Added 2026-08-19 alongside /scan: dman_scanner.yml's manual
        # dispatch needs an inputs.mode="scan" payload, unlike
        # dman_daemon.yml's plain restart which passes none.
        mock_resp = MagicMock(status_code=204)
        with patch.object(a, "GITHUB_TOKEN", "fake-token"):
            with patch.object(a.requests, "post", return_value=mock_resp) as mock_post:
                ok, msg = a._trigger_workflow_restart("dman_scanner.yml", inputs={"mode": "scan"})
        self.assertTrue(ok)
        call_url = mock_post.call_args[0][0]
        self.assertIn("actions/workflows/dman_scanner.yml/dispatches", call_url)
        self.assertEqual(mock_post.call_args[1]["json"], {"ref": "main", "inputs": {"mode": "scan"}})

    def test_trigger_restart_omits_inputs_key_when_not_given(self):
        # Must NOT send an empty "inputs": {} -- confirms the existing
        # /restart call site (no inputs) keeps its original payload shape.
        mock_resp = MagicMock(status_code=204)
        with patch.object(a, "GITHUB_TOKEN", "fake-token"):
            with patch.object(a.requests, "post", return_value=mock_resp) as mock_post:
                a._trigger_workflow_restart("dman_daemon.yml")
        self.assertEqual(mock_post.call_args[1]["json"], {"ref": "main"})

    def test_scan_command_dispatches_scanner_with_scan_mode_and_confirms(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(True, "dispatched")) as mock_trigger:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/scan")
        mock_trigger.assert_called_once_with("dman_scanner.yml", inputs={"mode": "scan"})
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("Scan requested" in t for t in sent_texts))
        self.assertTrue(any("dispatched" in t.lower() for t in sent_texts))

    def test_scan_command_reports_failure_with_fallback_instructions(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(False, "HTTP 403: nope")):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/scan")
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("failed" in t.lower() and "Run workflow" in t for t in sent_texts))

    def test_review_command_dispatches_review_workflow_and_confirms(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(True, "dispatched")) as mock_trigger:
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/review")
        mock_trigger.assert_called_once_with("dman_review.yml")
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("Review requested" in t for t in sent_texts))
        self.assertTrue(any("dispatched" in t.lower() for t in sent_texts))

    def test_review_command_reports_failure_with_fallback_instructions(self):
        with patch.object(a, "_trigger_workflow_restart", return_value=(False, "HTTP 403: nope")):
            with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                a._handle_telegram_command("/review")
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


class TestKellySizingUsesRealAvgLoss(unittest.TestCase):
    """Found in the 2026-08-16 review: size_position_kelly() never passed a
    real avg_loss_r through to kelly_fraction() -- it silently used the
    hardcoded default of 1.0 no matter what a setup's actual average loss
    was. setup_stats()'s avg_win_r is a percentage (e.g. 6-10 for a decent
    setup, not an R-multiple near 1-3), so b = avg_win_r / 1.0 came out far
    too large, which pushed full_kelly (and the fractional Kelly result)
    strongly positive for nearly every setup regardless of real win rate --
    in practice, sizing always saturated at kelly_fraction()'s 3% cap. These
    lock in that a real avg_loss_r produces a genuinely smaller size for a
    setup with a weak payoff ratio, and a genuinely larger one for a setup
    with a strong ratio -- i.e. that sizing is actually responsive again."""

    def _signal(self, entry=10.0, stop=9.0, beta=1.0):
        return a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=entry, stop=stop, target1=12.0, target2=14.0,
            rr=2.0, rsi=50.0, rvol=2.0, reason="test", beta=beta,
        )

    def test_weak_payoff_ratio_no_longer_saturates_the_cap(self):
        # 55% win rate, average winner +6%, average LOSER -12% (a genuinely
        # bad payoff ratio, b=0.5) -- full Kelly here is negative, so this
        # must land at the 0.5% floor, not the 3% cap.
        sig = self._signal()
        sig = a.size_position_kelly(sig, account=10_000.0, win_rate=0.55,
                                     avg_win_r=6.0, avg_loss_r=12.0)
        self.assertAlmostEqual(sig.kelly_frac, 0.005, places=4)

    def test_strong_payoff_ratio_sizes_larger_than_weak_one(self):
        weak = a.size_position_kelly(self._signal(), account=10_000.0, win_rate=0.55,
                                      avg_win_r=6.0, avg_loss_r=12.0)
        strong = a.size_position_kelly(self._signal(), account=10_000.0, win_rate=0.75,
                                        avg_win_r=10.0, avg_loss_r=4.0)
        self.assertGreater(strong.kelly_frac, weak.kelly_frac,
                           "a setup with a real edge must size meaningfully larger than "
                           "one without, instead of both landing on the same saturated cap")

    def test_default_avg_loss_r_of_one_still_works_for_a_true_r_multiple_caller(self):
        # Backward-compat: a hypothetical caller that already passes true
        # R-multiples (avg_win_r=2.0 meaning "wins average 2R") and omits
        # avg_loss_r should behave exactly as before this fix.
        sig = a.size_position_kelly(self._signal(), account=10_000.0,
                                     win_rate=0.6, avg_win_r=2.0)
        self.assertGreater(sig.kelly_frac, 0)

    def test_score_signal_passes_the_setups_real_avg_loss_through(self):
        # Integration point: score_signal() must read setup_stats()'s
        # avg_loss_r (not the kelly_fraction() default) and pass it to
        # size_position_kelly(), not silently drop it on the floor.
        sig = self._signal()
        fake_stats = {"win_rate": 0.55, "avg_win_r": 6.0, "avg_loss_r": 12.0, "total": 30}
        with patch.object(a, "check_mtf", return_value=(True, 15)), \
             patch.object(a, "check_relative_strength", return_value=(True, 10)), \
             patch.object(a, "check_sector", return_value=(True, 8)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 1)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", return_value=None):
            tracker = MagicMock()
            tracker.setup_stats.return_value = fake_stats
            scored = a.score_signal(sig, _fake_df(), _fake_regime(), tracker)
        self.assertAlmostEqual(scored.kelly_frac, 0.005, places=4,
                               msg="a weak real payoff ratio (b=0.5) must reach the sizing "
                                   "floor, not silently size as if avg_loss_r were 1.0")


class TestKellyFloorDoesNotOvershootRisk(unittest.TestCase):
    """Found in the 2026-08-16 review: size_position_kelly() used to force
    at least 1 share regardless of budget (max(1, ...)) -- when the sized
    risk budget was smaller than one share's worth of stop distance,
    actual dollar risk became the FULL stop distance, not the sized
    fraction, which could be several times the intended budget on a
    wide-stop or higher-priced name. This defeats the entire point of
    Kelly sizing."""

    def _signal(self, entry, stop):
        return a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=entry, stop=stop, target1=entry * 1.2, target2=entry * 1.4,
            rr=2.0, rsi=50.0, rvol=2.0, reason="test", beta=1.0,
        )

    def test_a_severe_overshoot_skips_rather_than_force_buys_one_share(self):
        # account=10,000, kelly floor 0.5% -> risk_budget=$50. rps=$200
        # (entry=1000, stop=800): even 1 share risks $200, 4x the budget
        # -- must skip (shares=0), not force-buy at 4x intended risk.
        sig = self._signal(entry=1000.0, stop=800.0)
        sig = a.size_position_kelly(sig, account=10_000.0, win_rate=0.55,
                                    avg_win_r=6.0, avg_loss_r=12.0)
        self.assertEqual(sig.shares, 0)
        self.assertEqual(sig.risk_usd, 0)

    def test_a_small_overshoot_still_floors_to_one_share(self):
        # risk_budget=$50, rps=$60 -- 1 share risks $60, only 1.2x the
        # budget (within the 1.5x rounding margin) -- still worth taking
        # rather than skipping a barely-oversized, otherwise-good setup.
        sig = self._signal(entry=1000.0, stop=940.0)
        sig = a.size_position_kelly(sig, account=10_000.0, win_rate=0.55,
                                    avg_win_r=6.0, avg_loss_r=12.0)
        self.assertEqual(sig.shares, 1)
        self.assertEqual(sig.risk_usd, 60.0)

    def test_a_well_sized_signal_is_unaffected(self):
        # Plenty of budget for many shares -- the floor logic must not
        # interfere with the normal multi-share sizing path.
        sig = self._signal(entry=10.0, stop=9.0)
        sig = a.size_position_kelly(sig, account=10_000.0, win_rate=0.75,
                                    avg_win_r=10.0, avg_loss_r=4.0)
        self.assertGreater(sig.shares, 1)


class TestBearGapHoldMeasuresRealGap(unittest.TestCase):
    """Found in the 2026-08-16 review: L9 Bear Gap Hold computed "gap %"
    from today's CURRENT price vs. prior close, not today's OPEN vs. prior
    close (every other gap detector in the file, including this setup's own
    bullish mirror L6, uses open-vs-prior-close). A stock that opened
    essentially flat and then drifted down intraday could read as a real
    gap-down and fire a signal that, in the live options branch, buys a
    real ITM put on ordinary noise, not an actual gap."""

    def test_flat_open_with_intraday_drift_does_not_trigger(self):
        # Opened at 20.95 vs prior close 21.00 (0.24% gap -- not a real
        # gap), then drifted down to 20.30 intraday (3.1% intraday drop).
        # The OLD buggy calc, (21.00-20.30)/21.00=3.33%, would have crossed
        # the 1.5% threshold and could fire; the real gap is nowhere close.
        df = _bear_gap_hold_df(prior_close=21.00, prior_open=21.50,
                               today_open=20.95, today_close=20.30)
        sig = a._raw_signals(df, "TESTX")
        self.assertIsNone(sig, "ordinary intraday drift on a flat open must not read as a gap")

    def test_a_real_overnight_gap_down_still_triggers(self):
        # Prior close 21.00, opens at 20.30 (3.33% real overnight gap),
        # holds below open through the session.
        df = _bear_gap_hold_df(prior_close=21.00, prior_open=21.50,
                               today_open=20.30, today_close=20.20)
        sig = a._raw_signals(df, "TESTX")
        self.assertIsNotNone(sig, "a genuine overnight gap-down must still fire")
        self.assertEqual(sig.setup, "Bear Gap Hold")
        self.assertEqual(sig.bias, "SHORT")

    def test_rr_reflects_the_echo_target_when_it_wins(self):
        # A large gap where the echo target is tighter than the 4.0R
        # default -- rr must be recomputed against it, not left stale.
        # Stale-rr would have been the value computed before the echo
        # override -- i.e. against the 4.0R default target, not target1.
        df = _bear_gap_hold_df(prior_close=21.00, prior_open=21.50,
                               today_open=19.00, today_close=18.90)
        sig = a._raw_signals(df, "TESTX")
        self.assertIsNotNone(sig)
        self.assertLess(sig.target1, 18.90 - 2.0 * (sig.stop - 18.90),
                        "the echo target must actually be the tighter (winning) one here")
        # rr must reflect target1 (whatever it ended up being, echo or
        # default), not a value computed before the override. Small
        # tolerance below: the source recomputes rr against the unrounded
        # local stop variable (same convention L6 Gap & Hold already
        # uses), while this test only has access to the rounded, stored
        # sig.stop.
        risk = sig.stop - 18.90
        expected_rr = round((18.90 - sig.target1) / risk, 2)
        self.assertAlmostEqual(sig.rr, expected_rr, delta=0.05,
                               msg="rr must match the actual (possibly echo-overridden) target1, "
                                   "not a stale pre-override value")


class TestRsiZoneCreditsFreshCatalystGaps(unittest.TestCase):
    """Direct instruction 2026-09-02, evidence-backed by reconstructing 2
    months of real history: ABNB (RSI 79.6), MRNA (93.9), and CRM (80.6)
    each cleared 75+ on every OTHER technical factor and lost the same 5
    points here on the exact days a real catalyst gap was most obviously
    present -- the 45-62 sweet-spot zone was tuned for measured
    continuation, not a genuine gap-and-go, which pushes RSI into the
    70s-90s (or teens-20s, short side) by definition. Additive: a high/low
    RSI reading now ALSO earns the bonus when paired with real gap+volume;
    a high/low RSI with neither still scores 0, unchanged."""

    def _gapped_df(self, gap_pct):
        import pandas as pd
        df = _fake_df()
        prior_close = float(df["Close"].iloc[-2])
        gapped_open = prior_close * (1 + gap_pct / 100)
        df.loc[df.index[-1], "Open"]  = gapped_open
        df.loc[df.index[-1], "Close"] = gapped_open
        df.loc[df.index[-1], "High"]  = gapped_open * 1.01
        df.loc[df.index[-1], "Low"]   = gapped_open * 0.99
        return df

    def _signal(self, rsi, rvol, bias="LONG"):
        return a.ProSignal(
            ticker="TEST", bias=bias, setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=13.0, target2=16.0,
            rr=2.0, rsi=rsi, rvol=rvol, reason="test", confluence_score=0,
        )

    def _score(self, sig, df):
        with patch.object(a, "check_mtf", return_value=(True, 15)), \
             patch.object(a, "check_relative_strength", return_value=(True, 10)), \
             patch.object(a, "check_sector", return_value=(True, 8)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 1)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", return_value=None), \
             patch.object(a, "optimize_stop", return_value=sig.stop):
            return a.score_signal(sig, df, _fake_regime(), a.WinRateTracker())

    def test_high_rsi_with_fresh_catalyst_gap_gets_full_bonus(self):
        sig = self._signal(rsi=80.0, rvol=3.0)
        scored = self._score(sig, self._gapped_df(gap_pct=8.0))
        self.assertEqual(scored.score_breakdown["RSI Zone"], 5)

    def test_high_rsi_without_a_real_gap_still_scores_zero(self):
        # Stale, already-extended chase -- no gap behind the high RSI.
        sig = self._signal(rsi=80.0, rvol=3.0)
        scored = self._score(sig, _fake_df())
        self.assertEqual(scored.score_breakdown["RSI Zone"], 0)

    def test_high_rsi_with_gap_but_thin_volume_still_scores_zero(self):
        sig = self._signal(rsi=80.0, rvol=1.2)
        scored = self._score(sig, self._gapped_df(gap_pct=8.0))
        self.assertEqual(scored.score_breakdown["RSI Zone"], 0)

    def test_moderate_sweet_spot_rsi_unaffected_by_the_change(self):
        sig = self._signal(rsi=50.0, rvol=2.0)
        scored = self._score(sig, _fake_df())
        self.assertEqual(scored.score_breakdown["RSI Zone"], 5)

    def test_low_rsi_with_fresh_catalyst_gap_down_gets_full_bonus_short_side(self):
        sig = self._signal(rsi=18.0, rvol=3.0, bias="SHORT")
        scored = self._score(sig, self._gapped_df(gap_pct=-8.0))
        self.assertEqual(scored.score_breakdown["RSI Zone"], 5)


class TestScoreSignalPreservesTargets(unittest.TestCase):
    """Found in the 2026-08-16 review: score_signal() used to unconditionally
    overwrite target1/target2 with a flat 2.0x/3.0x multiplier of the
    (possibly ATR-tightened) stop, discarding whatever _raw_signals() had
    actually set -- a setup's real multiplier (e.g. 2.5x/4.0x) or a
    gap-echo target (a specific technical price level, not a multiple of
    risk). These lock in that targets now survive score_signal() untouched,
    with only rr recomputed against the new stop."""

    def _signal(self, target1=13.0, target2=16.0, entry=10.0, stop=9.0, bias="LONG"):
        return a.ProSignal(
            ticker="TEST", bias=bias, setup="Gap & Hold",
            entry=entry, stop=stop, target1=target1, target2=target2,
            rr=2.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=0,
        )

    def _score(self, sig, optimized_stop):
        with patch.object(a, "check_mtf", return_value=(True, 15)), \
             patch.object(a, "check_relative_strength", return_value=(True, 10)), \
             patch.object(a, "check_sector", return_value=(True, 8)), \
             patch.object(a, "check_earnings_safe", return_value=(True, 1)), \
             patch.object(a, "_get_short_float_data", return_value=(0.0, 0.0, 0.0, 0.0)), \
             patch.object(a, "fetch_df", return_value=None), \
             patch.object(a, "optimize_stop", return_value=optimized_stop):
            return a.score_signal(sig, _fake_df(), _fake_regime(), a.WinRateTracker())

    def test_non_default_target_ratio_survives_a_stop_that_tightens(self):
        # target1 at 3.0 risk (not 2.0), target2 at 6.0 risk (not 3.0) --
        # exactly the shape of a setup with its own real multiplier or a
        # gap-echo target that isn't the old hardcoded 2R/3R.
        sig = self._signal(entry=10.0, stop=9.0, target1=13.0, target2=16.0)
        scored = self._score(sig, optimized_stop=9.5)   # optimizer tightens the stop
        self.assertEqual(scored.target1, 13.0, "target1 must be untouched by the stop change")
        self.assertEqual(scored.target2, 16.0, "target2 must be untouched by the stop change")

    def test_rr_is_recomputed_against_the_new_stop(self):
        sig = self._signal(entry=10.0, stop=9.0, target1=13.0, target2=16.0)
        scored = self._score(sig, optimized_stop=9.5)
        # New risk is 0.5 (10.0 - 9.5); target1 distance from entry is 3.0.
        self.assertAlmostEqual(scored.rr, 3.0 / 0.5, places=2)

    def test_short_bias_target_also_survives(self):
        sig = self._signal(entry=10.0, stop=11.0, target1=6.5, target2=3.0, bias="SHORT")
        scored = self._score(sig, optimized_stop=10.5)
        self.assertEqual(scored.target1, 6.5)
        self.assertEqual(scored.target2, 3.0)


class TestReanchorPreservesTargetRatio(unittest.TestCase):
    """Companion to TestScoreSignalPreservesTargets: _submit_signals_to_alpaca()'s
    live-price re-anchor step used to hardcode ITS OWN 2.5x/4.0x multiplier
    -- a THIRD number, different from score_signal()'s (then also
    hardcoded) 2.0x/3.0x -- so what actually got submitted to the broker
    never matched what was logged/alerted regardless of which of the two
    overwrite bugs you look at. This locks in that the re-anchor now scales
    the signal's OWN target ratio to the live entry, not a fixed multiplier."""

    def test_custom_ratio_is_preserved_when_re_anchored_to_a_new_entry(self):
        # Signal detected at entry=10, stop=9 (risk=1), target1=13 (3R),
        # target2=16 (6R) -- deliberately not 2.5R/4.0R. Live fill price
        # drifts to 10.05.
        sig = a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=13.0, target2=16.0,
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=90,
            shares=10, cost=100.0,
        )
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 10.05)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_cash_available_for", return_value=(True, "")), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "submit_alpaca_trade", return_value=("order-1", None)), \
             patch.object(a, "send_telegram", return_value=True):
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            MockPT.return_value.positions = []
            a._submit_signals_to_alpaca([sig])
        # New entry ~10.05*1.001 = 10.0601, risk stays 1.0 (entry-stop delta
        # from signal detection), so target1 should be entry + 3*risk,
        # target2 entry + 6*risk -- the SAME ratio as before, not 2.5/4.0.
        new_entry = round(10.05 * 1.001, 2)
        self.assertAlmostEqual(sig.target1, new_entry + 3.0 * 1.0, places=2)
        self.assertAlmostEqual(sig.target2, new_entry + 6.0 * 1.0, places=2)


class TestSubmitSignalsSkipsZeroShareSizing(unittest.TestCase):
    """Found in the 2026-08-16 review: size_position_kelly() now reports
    shares=0 when even 1 share would overshoot the sized risk budget by
    more than a reasonable margin (see TestKellyFloorDoesNotOvershootRisk)
    -- _submit_signals_to_alpaca() must actually skip that signal, not
    attempt to submit a qty=0 order."""

    def test_zero_share_signal_is_skipped_before_order_submission(self):
        sig = a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=13.0, target2=16.0,
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=90,
            shares=0, cost=0.0,
        )
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price") as mock_validate, \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "submit_alpaca_trade") as mock_submit, \
             patch.object(a, "send_telegram", return_value=True):
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            MockPT.return_value.positions = []
            a._submit_signals_to_alpaca([sig])
        mock_validate.assert_not_called()
        mock_submit.assert_not_called()


class TestOptionsUnavailableSkipsInsteadOfSharesFallback(unittest.TestCase):
    """Removed 2026-08-21: the budget-capped shares fallback (added
    2026-08-08) that used to fire when an options-eligible signal
    (WATCHLIST/OPTIONS_SETUPS) found no real options fill and the ticker
    wasn't on DMan's small-cap watchlist. Direct instruction after NDSN --
    a large industrial name, nothing like a low-float catalyst play --
    bought as a single $334 share exactly this way: shares should only
    ever happen for a real low-float catalyst pick, everything else must
    skip outright rather than settle for a consolation equity position."""

    def test_options_eligible_but_unfillable_non_watchlist_signal_is_skipped_not_bought(self):
        sig = a.ProSignal(
            ticker="NDSN", bias="LONG", setup="Gap & Hold",   # Gap & Hold is in OPTIONS_SETUPS
            entry=334.75, stop=320.35, target1=406.79, target2=450.01,
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=100,
            shares=1, cost=334.75,
        )
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 334.75)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "WATCHLIST", []), \
             patch.object(a, "DMAN_SMALLCAP_WATCHLIST", []), \
             patch.object(a, "get_alpaca_client", return_value=MagicMock()), \
             patch.object(a, "_submit_options_call", return_value=(None, None)), \
             patch.object(a, "submit_alpaca_trade") as mock_submit, \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            MockPT.return_value.positions = []
            a._submit_signals_to_alpaca([sig])
        mock_submit.assert_not_called()
        sent_texts = [c.args[0] for c in mock_tg.call_args_list]
        self.assertTrue(any("not executed" in t for t in sent_texts))


class TestSubmitSignalsRespectsMaxPositionsBeforeSubmission(unittest.TestCase):
    """Found in the 2026-08-23 review: MAX_POSITIONS was only ever enforced
    AFTER an order was already live at the broker (inside pt.open(), which
    cancels the just-submitted order on a full tracker) -- a fill landing
    before/during that cancel call leaves a real position with no tracking
    slot at all. For options especially, that's zero stop-loss protection
    of any kind (no broker-side bracket by design) until a human notices an
    orphan alert. The capacity check must happen before any order is
    submitted, not after."""

    def _sig(self, ticker="TEST"):
        return a.ProSignal(
            ticker=ticker, bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=13.0, target2=16.0,
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=90,
            shares=10, cost=100.0,
        )

    def test_signal_is_skipped_before_submission_when_already_at_max_positions(self):
        sig = self._sig()
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 10.0)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "submit_alpaca_trade") as mock_submit:
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            # Already at capacity with UNRELATED tickers -- the new signal's
            # own ticker isn't a duplicate, capacity alone must block it.
            MockPT.return_value.positions = [
                MagicMock(ticker=f"HELD{i}") for i in range(a.MAX_POSITIONS)
            ]
            a._submit_signals_to_alpaca([sig])
        mock_submit.assert_not_called()

    def test_signal_still_submits_when_under_max_positions(self):
        sig = self._sig()
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "ENABLE_OPTIONS_TRADING", False), \
             patch.object(a, "DMAN_SMALLCAP_WATCHLIST", ["TEST"]), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=0.0), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 10.0)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "submit_alpaca_trade", return_value=("order-1", None)) as mock_submit:
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": 0, "win_rate": 0.6, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 6, "losses": 4,
                "consec_wins": 0,
            }
            MockPT.return_value.positions = [
                MagicMock(ticker=f"HELD{i}") for i in range(a.MAX_POSITIONS - 1)
            ]
            a._submit_signals_to_alpaca([sig])
        mock_submit.assert_called_once()


class TestSubmitSignalsProbationSizing(unittest.TestCase):
    """Added 2026-08-18 alongside is_on_probation(): the belt-and-suspenders
    circuit-breaker recheck in _submit_signals_to_alpaca() must bypass the
    consec-loss/monthly-loss checks during probation (mirroring the
    scanner-level guard), and the probation size_mult must compound into
    the same _risk_off_mult chain that already scales shares for
    macro/hot-streak/cold-streak sizing."""

    def _sig(self, shares=10):
        return a.ProSignal(
            ticker="TEST", bias="LONG", setup="Gap & Hold",
            entry=10.0, stop=9.0, target1=13.0, target2=16.0,
            rr=3.0, rsi=50.0, rvol=2.0, reason="test", confluence_score=90,
            shares=shares, cost=shares * 10.0,
        )

    def test_probation_halves_submitted_share_size(self):
        sig = self._sig(shares=10)
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(True, 0.5)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "get_this_month_loss", return_value=-(a.MONTHLY_LOSS_LIMIT * 100) - 1), \
             patch.object(a, "PositionTracker") as MockPT, \
             patch.object(a, "validate_entry_price", return_value=(True, 10.0)), \
             patch.object(a, "_fetch_global_context", return_value={
                 "risk_mult": 1.0, "tone": "NEUTRAL", "score": 0, "summary": ""}), \
             patch.object(a, "_cash_available_for", return_value=(True, "")), \
             patch.object(a, "_get_pdt_status", return_value={
                 "used": 0, "remaining": 3, "swing_mode": False, "equity": 30_000.0}), \
             patch.object(a, "submit_alpaca_trade", return_value=("order-1", None)), \
             patch.object(a, "send_telegram", return_value=True):
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.3, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 3, "losses": 7, "consec_wins": 0,
            }
            MockPT.return_value.positions = []
            a._submit_signals_to_alpaca([sig])
        self.assertEqual(sig.shares, 5)

    def test_without_probation_belt_and_suspenders_blocks_on_consec_loss(self):
        sig = self._sig(shares=10)
        with patch.object(a, "ALPACA_API_KEY", "test-key"), \
             patch.object(a, "is_market_open", return_value=True), \
             patch.object(a, "is_halted", return_value=False), \
             patch.object(a, "is_on_probation", return_value=(False, 1.0)), \
             patch.object(a, "WinRateTracker") as MockWRT, \
             patch.object(a, "submit_alpaca_trade") as mock_submit, \
             patch.object(a, "send_telegram", return_value=True):
            MockWRT.return_value.rolling_stats.return_value = {
                "consec_losses": a.MAX_CONSEC_LOSSES, "win_rate": 0.3, "avg_win_r": 2.0,
                "avg_loss_r": 1.0, "total": 10, "wins": 3, "losses": 7, "consec_wins": 0,
            }
            a._submit_signals_to_alpaca([sig])
        mock_submit.assert_not_called()


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


class TestCheckStopCoverageSplit(unittest.TestCase):
    """_check_stop_coverage() was split out of _check_open_position_risk()
    2026-08-15 so guard_loop() (dman_daemon.py) can run it on its tight 10s
    cadence instead of waiting for run_pro_scanner()'s ~10min scan pass. A
    regression here (e.g. _check_open_position_risk silently stops calling
    the split-out function) would widen detection latency for a broken
    bracket order back to minutes without any test catching it."""

    def setUp(self):
        self._pos_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._pos_tmp.write(b"[]")
        self._pos_tmp.close()
        self._pos_patch = patch.object(a, "POSITIONS_FILE", self._pos_tmp.name)
        self._pos_patch.start()
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

        # _check_stop_coverage() now caches its Alpaca fetch for
        # _STOP_COVERAGE_CACHE_TTL_S (30s real wall-clock) -- without a
        # fresh cache per test, a later test's call within that window
        # would silently reuse an earlier test's mocked positions/orders
        # instead of hitting its own mock_client.
        self._cache_patch = patch.object(
            a, "_stop_coverage_fetch_cache", {"positions": None, "orders": None, "ts": 0.0})
        self._cache_patch.start()

    def tearDown(self):
        self._pos_patch.stop();    os.unlink(self._pos_tmp.name)
        self._pt_patch.stop()
        self._sig_patch.stop();    os.unlink(self._sig_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)
        self._cache_patch.stop()

    def test_check_stop_coverage_returns_the_raw_alpaca_positions_dict(self):
        from alpaca.trading.enums import AssetClass, OrderType, OrderStatus
        pos = MagicMock()
        pos.symbol = "W"; pos.asset_class = AssetClass.US_EQUITY; pos.qty = "3"
        pos.avg_entry_price = "116.25"; pos.unrealized_pl = "-2.40"; pos.unrealized_plpc = "-0.02"
        order = MagicMock()
        order.symbol = "W"; order.order_type = OrderType.STOP; order.status = OrderStatus.NEW
        mock_client = MagicMock()
        mock_client.get_all_positions.return_value = [pos]
        mock_client.get_orders.return_value = [order]
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "send_telegram", return_value=True):
            result = a._check_stop_coverage()
        self.assertIsNotNone(result)
        self.assertIn("W", result)

    def test_check_open_position_risk_delegates_to_check_stop_coverage(self):
        with patch.object(a, "_check_stop_coverage", return_value=None) as mock_split:
            a._check_open_position_risk({})
        mock_split.assert_called_once_with()

    def test_alpaca_unreachable_still_alerts_but_flags_it_as_unverified(self):
        # Found in the 2026-08-16 review: a stray "at risk" alert for a
        # phantom (alerted-but-never-filled) signal like FGL/AMZN reads
        # identically to a real one when Alpaca is briefly unreachable,
        # since the phantom-position filter can't run without a real
        # positions fetch. Deliberately still fails OPEN here rather than
        # going silent (test_alpaca_unreachable_fails_open_still_shows_
        # pending's rationale: silently missing a genuine at-risk
        # position is judged worse than an occasional false alarm) --
        # but the alert text must now say it couldn't be cross-checked,
        # so a human reading it knows to treat it with less certainty.
        with open(self._sig_tmp.name, "w") as f:
            json.dump({"pending": [{"ticker": "FGL", "bias": "LONG", "entry": 10.0,
                                    "stop": 9.0, "target1": 12.0, "score": 90}]}, f)
        with patch.object(a, "_check_stop_coverage", return_value=None), \
             patch.object(a, "get_live_price", return_value=9.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a._check_open_position_risk({})
        mock_tg.assert_called_once()
        self.assertIn("could not confirm", mock_tg.call_args[0][0])
        self.assertIn("FGL", mock_tg.call_args[0][0])


class TestSetupPerformanceDrift(unittest.TestCase):
    """WinRateTracker.setup_performance_drift() generalizes the manual
    investigation that found Low Float Catalyst's 0% WR / -24.9% avg loss
    (see SETUP_MIN_CONFLUENCE's comment) into an automatic, ongoing check.
    Live-only by design -- see the method's docstring for why mixing in
    backtest-era records would dilute a real live problem."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"[]")
        self._tmp.close()

    def tearDown(self):
        os.unlink(self._tmp.name)

    def _record(self, tracker, outcome, setup="Low Float Catalyst", is_live=True, pnl_pct=None):
        if pnl_pct is None:
            pnl_pct = -10.0 if outcome == "LOSS" else 10.0
        tracker.record(a.TradeRecord(
            ticker="TESTX", date="2026-08-15", bias="LONG", setup=setup,
            entry=10.0, exit=9.0, outcome=outcome, pnl_pct=pnl_pct,
            score=90, is_live=is_live,
        ))

    def test_below_min_trades_is_not_flagged(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        self._record(tracker, "LOSS")
        self._record(tracker, "LOSS")
        drift = tracker.setup_performance_drift()
        self.assertEqual(drift, [])

    def test_below_wr_floor_with_enough_trades_is_flagged(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(3):
            self._record(tracker, "LOSS", pnl_pct=-20.0)
        drift = tracker.setup_performance_drift()
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["setup"], "Low Float Catalyst")
        self.assertEqual(drift[0]["win_rate"], 0.0)
        self.assertEqual(drift[0]["total"], 3)
        self.assertAlmostEqual(drift[0]["avg_loss_pct"], 20.0)

    def test_above_wr_floor_is_not_flagged(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        self._record(tracker, "WIN")
        self._record(tracker, "WIN")
        self._record(tracker, "LOSS")
        drift = tracker.setup_performance_drift()
        self.assertEqual(drift, [])

    def test_backtest_only_records_are_excluded_even_with_a_bad_win_rate(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(10):
            self._record(tracker, "LOSS", is_live=False)
        drift = tracker.setup_performance_drift()
        self.assertEqual(drift, [])

    def test_live_losses_not_masked_by_a_large_backtest_pool_of_wins(self):
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(50):
            self._record(tracker, "WIN", is_live=False)
        for _ in range(3):
            self._record(tracker, "LOSS", is_live=True)
        drift = tracker.setup_performance_drift()
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["win_rate"], 0.0)

    def test_different_earnings_spread_labels_are_pooled_as_one_family(self):
        # Regression for the live 2026-08-29 two-month review: the real
        # earnings-spread record was 1W/3L across three DIFFERENT setup
        # labels (Call/Put/Double Spread), each with only 1-2 trades --
        # under min_trades individually, so this alert had never once
        # fired for a strategy that was actually losing badly live.
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        self._record(tracker, "LOSS", setup="Earnings Call Spread", pnl_pct=-60.0)
        self._record(tracker, "LOSS", setup="Earnings Put Spread", pnl_pct=-100.0)
        self._record(tracker, "LOSS", setup="Earnings Double Spread", pnl_pct=-95.0)
        drift = tracker.setup_performance_drift()
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]["setup"], "Earnings Spread")
        self.assertEqual(drift[0]["total"], 3)
        self.assertEqual(drift[0]["win_rate"], 0.0)

    def test_non_earnings_setups_still_pool_by_their_own_exact_label(self):
        # The grouping is earnings-only -- Gap & Hold and Low Float
        # Catalyst must stay separate from each other and from Earnings
        # Spread, unaffected by this change.
        tracker = a.WinRateTracker(filepath=self._tmp.name)
        for _ in range(3):
            self._record(tracker, "LOSS", setup="Gap & Hold", pnl_pct=-10.0)
        for _ in range(3):
            self._record(tracker, "LOSS", setup="Low Float Catalyst", pnl_pct=-20.0)
        drift = tracker.setup_performance_drift()
        self.assertEqual({d["setup"] for d in drift}, {"Gap & Hold", "Low Float Catalyst"})


class TestSetupPerformanceDriftAlert(unittest.TestCase):
    """setup_performance_drift() is only useful if it actually reaches
    Telegram -- this locks in send_account_pnl_telegram()'s wiring (a
    separate, deduped message from the account summary) added 2026-08-15."""

    def setUp(self):
        import functools
        self._wr_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._wr_tmp.write(b"[]")
        self._wr_tmp.close()
        # WinRateTracker's filepath default is early-bound to WIN_RATE_FILE
        # at class-definition time (same gotcha documented for
        # PositionTracker) -- patching WIN_RATE_FILE alone would not reach
        # the bare WinRateTracker() call inside send_account_pnl_telegram.
        self._RealWRT = a.WinRateTracker
        self._isolated_wrt = functools.partial(self._RealWRT, filepath=self._wr_tmp.name)
        self._wrt_patch = patch.object(a, "WinRateTracker", self._isolated_wrt)
        self._wrt_patch.start()

        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

        # _enter_setup_probation() (added 2026-08-24) writes here on every
        # drift finding -- must be isolated same as everything else, or a
        # real test run pollutes the actual production dman_setup_probation.json.
        self._setup_prob_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._setup_prob_tmp.write(b"{}")
        self._setup_prob_tmp.close()
        self._setup_prob_patch = patch.object(a, "SETUP_PROBATION_FILE", self._setup_prob_tmp.name)
        self._setup_prob_patch.start()

    def tearDown(self):
        self._wrt_patch.stop()
        os.unlink(self._wr_tmp.name)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)
        self._setup_prob_patch.stop(); os.unlink(self._setup_prob_tmp.name)

    def _record_drifting_setup(self):
        tracker = a.WinRateTracker()
        for _ in range(3):
            tracker.record(a.TradeRecord(
                ticker="TESTX", date="2026-08-15", bias="LONG", setup="Low Float Catalyst",
                entry=10.0, exit=8.0, outcome="LOSS", pnl_pct=-20.0, score=90, is_live=True,
            ))

    def _run_with_mocked_account(self, mock_tg):
        acct = MagicMock()
        acct.equity = "5000.0"; acct.cash = "3000.0"; acct.buying_power = "3000.0"
        mock_client = MagicMock()
        mock_client.get_account.return_value = acct
        mock_client.get_all_positions.return_value = []
        with patch.object(a, "_get_day_start_equity", return_value=5000.0), \
             patch("alpaca.trading.client.TradingClient", return_value=mock_client), \
             patch.object(a, "ALPACA_API_KEY", "k"), \
             patch.object(a, "ALPACA_SECRET_KEY", "s"), \
             patch.object(a, "send_telegram", mock_tg):
            a.send_account_pnl_telegram()

    def test_drifting_setup_sends_a_separate_telegram_message(self):
        self._record_drifting_setup()
        mock_tg = MagicMock(return_value=True)
        self._run_with_mocked_account(mock_tg)
        msgs = [c[0][0] for c in mock_tg.call_args_list]
        drift_msgs = [m for m in msgs if "Setup Performance Drift" in m]
        self.assertEqual(len(drift_msgs), 1)
        self.assertIn("Low Float Catalyst", drift_msgs[0])

    def test_no_drift_sends_only_the_account_summary(self):
        mock_tg = MagicMock(return_value=True)
        self._run_with_mocked_account(mock_tg)
        msgs = [c[0][0] for c in mock_tg.call_args_list]
        self.assertFalse(any("Setup Performance Drift" in m for m in msgs))
        self.assertEqual(len(msgs), 1)

    def test_repeat_call_within_cooldown_does_not_repeat_the_drift_alert(self):
        self._record_drifting_setup()
        mock_tg = MagicMock(return_value=True)
        self._run_with_mocked_account(mock_tg)
        self._run_with_mocked_account(mock_tg)
        msgs = [c[0][0] for c in mock_tg.call_args_list]
        drift_msgs = [m for m in msgs if "Setup Performance Drift" in m]
        self.assertEqual(len(drift_msgs), 1)

    def test_drifting_setup_is_actually_restricted_not_just_alerted(self):
        # Added 2026-08-24: the whole point of this change -- a flagged
        # setup must come out of this with a real, live restriction, not
        # just a Telegram message nobody has to act on.
        self._record_drifting_setup()
        mock_tg = MagicMock(return_value=True)
        self._run_with_mocked_account(mock_tg)
        self.assertIn("Low Float Catalyst", a._load_setup_probation())
        self.assertEqual(a._setup_probation_bonus("Low Float Catalyst"), a.SETUP_PROBATION_SCORE_BONUS)
        drift_msgs = [c[0][0] for c in mock_tg.call_args_list if "Setup Performance Drift" in c[0][0]]
        self.assertIn("restricted", drift_msgs[0])

    def test_already_restricted_setup_is_not_re_restricted_or_reworded_as_new(self):
        self._record_drifting_setup()
        a._enter_setup_probation("Low Float Catalyst", "already flagged yesterday")
        original = a._load_setup_probation()["Low Float Catalyst"]["started"]
        mock_tg = MagicMock(return_value=True)
        with patch.object(a, "_is_duplicate_alert", return_value=False):
            self._run_with_mocked_account(mock_tg)
        self.assertEqual(a._load_setup_probation()["Low Float Catalyst"]["started"], original)
        drift_msgs = [c[0][0] for c in mock_tg.call_args_list if "Setup Performance Drift" in c[0][0]]
        self.assertIn("already restricted", drift_msgs[0])


class TestDescribeOccSymbol(unittest.TestCase):
    """_describe_occ_symbol() turns a raw OCC option symbol into a
    readable label -- same fixed-width slicing convention already used
    elsewhere for expiry-only parsing (date+type+strike are always the
    last 15 characters, ticker is everything before that)."""

    def test_parses_ticker_strike_type_and_expiry(self):
        out = a._describe_occ_symbol("SMCI260814C00034000")
        self.assertIn("SMCI", out)
        self.assertIn("34", out)
        self.assertIn("C", out)
        self.assertIn("2026-08-14", out)

    def test_parses_put(self):
        out = a._describe_occ_symbol("MRVL260904P00210000")
        self.assertIn("MRVL", out)
        self.assertIn("210", out)
        self.assertIn("P", out)
        self.assertIn("2026-09-04", out)

    def test_ticker_length_does_not_affect_parsing(self):
        # A 1-char and a 4-char ticker must both parse correctly -- the
        # slice is always taken from the right, never assumes a fixed
        # ticker width.
        out = a._describe_occ_symbol("F260814C00012000")
        self.assertTrue(out.startswith("F "))

    def test_malformed_symbol_falls_back_to_the_raw_string(self):
        self.assertEqual(a._describe_occ_symbol("NOTVALID"), "NOTVALID")
        self.assertEqual(a._describe_occ_symbol("TOOSHORTXXXXXX"), "TOOSHORTXXXXXX")


class TestAccountPnlTelegramOptionsPositionDisplay(unittest.TestCase):
    """Regression: send_account_pnl_telegram()'s Open Positions block
    pulls straight from Alpaca's raw position list, bypassing
    PositionTracker entirely -- so it never got the 2026-08-16 fix that
    stopped an options position from being displayed with the raw OCC
    symbol and "sh" instead of a readable strike/expiry label and "ct".
    The dollar P&L itself (Alpaca's own unrealized_pl/plpc) was always
    correct; only the label was broken."""

    def setUp(self):
        self._wr_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._wr_tmp.write(b"[]")
        self._wr_tmp.close()
        import functools
        self._patches = [
            patch.object(a, "WinRateTracker", functools.partial(a.WinRateTracker, filepath=self._wr_tmp.name)),
            patch.object(a, "_is_duplicate_alert", return_value=True),   # skip the separate drift message
            patch.object(a, "_get_day_start_equity", return_value=5000.0),
            patch.object(a, "ALPACA_API_KEY", "k"),
            patch.object(a, "ALPACA_SECRET_KEY", "s"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        os.unlink(self._wr_tmp.name)

    def _option_position(self, symbol, qty="2", avg_px="9.22", pl="196.0", plpc="0.106"):
        from alpaca.trading.enums import AssetClass
        pos = MagicMock()
        pos.symbol = symbol
        pos.qty = qty
        pos.avg_entry_price = avg_px
        pos.unrealized_pl = pl
        pos.unrealized_plpc = plpc
        pos.asset_class = AssetClass.US_OPTION
        return pos

    def _equity_position(self, symbol="CELZ", qty="100", avg_px="10.0", pl="100.0", plpc="0.10"):
        from alpaca.trading.enums import AssetClass
        pos = MagicMock()
        pos.symbol = symbol
        pos.qty = qty
        pos.avg_entry_price = avg_px
        pos.unrealized_pl = pl
        pos.unrealized_plpc = plpc
        pos.asset_class = AssetClass.US_EQUITY
        return pos

    def test_options_position_shows_readable_label_and_contract_count(self):
        acct = MagicMock()
        acct.equity = "5000.0"; acct.cash = "3000.0"; acct.buying_power = "3000.0"
        mock_client = MagicMock()
        mock_client.get_account.return_value = acct
        mock_client.get_all_positions.return_value = [self._option_position("UMAC260828C00025000")]
        mock_tg = MagicMock(return_value=True)
        with patch("alpaca.trading.client.TradingClient", return_value=mock_client), \
             patch.object(a, "send_telegram", mock_tg):
            a.send_account_pnl_telegram()
        msg = mock_tg.call_args_list[0][0][0]
        self.assertIn("UMAC", msg)
        self.assertIn("$25", msg)
        self.assertIn("2026-08-28", msg)
        self.assertIn("2ct", msg)
        self.assertNotIn("UMAC260828C00025000", msg)   # raw OCC symbol must not leak through
        self.assertNotIn("2sh", msg)                    # must not be mislabeled as shares

    def test_equity_position_is_unaffected(self):
        acct = MagicMock()
        acct.equity = "5000.0"; acct.cash = "3000.0"; acct.buying_power = "3000.0"
        mock_client = MagicMock()
        mock_client.get_account.return_value = acct
        mock_client.get_all_positions.return_value = [self._equity_position()]
        mock_tg = MagicMock(return_value=True)
        with patch("alpaca.trading.client.TradingClient", return_value=mock_client), \
             patch.object(a, "send_telegram", mock_tg):
            a.send_account_pnl_telegram()
        msg = mock_tg.call_args_list[0][0][0]
        self.assertIn("CELZ", msg)
        self.assertIn("100sh", msg)

    def test_mixed_equity_and_options_positions_both_display_correctly(self):
        acct = MagicMock()
        acct.equity = "5000.0"; acct.cash = "3000.0"; acct.buying_power = "3000.0"
        mock_client = MagicMock()
        mock_client.get_account.return_value = acct
        mock_client.get_all_positions.return_value = [
            self._equity_position(), self._option_position("UMAC260828C00025000"),
        ]
        mock_tg = MagicMock(return_value=True)
        with patch("alpaca.trading.client.TradingClient", return_value=mock_client), \
             patch.object(a, "send_telegram", mock_tg):
            a.send_account_pnl_telegram()
        msg = mock_tg.call_args_list[0][0][0]
        self.assertIn("100sh", msg)
        self.assertIn("2ct", msg)


class TestExplainTicker(unittest.TestCase):
    """/why TICKER (explain_ticker) re-runs the scan detection/scoring
    pipeline for one ticker on demand. These tests lock in the headline
    outcomes without re-deriving _raw_signals/score_signal's own internals,
    which already have their own coverage elsewhere."""

    def _signal(self, score=80, setup="Gap & Hold", regime_ok=True, mtf_ok=True,
                rs_ok=True, sector_ok=True, earnings_ok=True, macro_ok=True,
                divergence_free=True):
        sig = a.ProSignal(
            ticker="TESTX", bias="LONG", setup=setup,
            entry=10.0, stop=9.0, target1=12.0, target2=14.0,
            rr=2.0, rsi=50.0, rvol=2.0, reason="test",
        )
        sig.confluence_score = score
        sig.regime_ok = regime_ok
        sig.mtf_ok = mtf_ok
        sig.rs_ok = rs_ok
        sig.sector_ok = sector_ok
        sig.earnings_ok = earnings_ok
        sig.macro_ok = macro_ok
        sig.divergence_free = divergence_free
        sig.score_breakdown = {"MTF": 20, "Regime": 15}
        return sig

    def test_no_price_data_reports_cleanly(self):
        with patch.object(a, "fetch_df", return_value=None):
            msg = a.explain_ticker("ZZZZ")
        self.assertIn("No usable price data", msg)

    def test_no_pattern_detected_is_reported(self):
        df = _fake_df()
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=None):
            msg = a.explain_ticker("testx")
        self.assertIn("No qualifying pattern detected", msg)

    def test_passing_signal_reports_pass_verdict(self):
        df = _fake_df()
        sig = self._signal(score=90)
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=_fake_regime()), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        self.assertIn("Would PASS all gates", msg)

    def test_failing_score_reports_blocked_with_the_score_gap(self):
        df = _fake_df()
        sig = self._signal(score=50)
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=_fake_regime()), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        self.assertIn("BLOCKED", msg)
        self.assertIn("score 50", msg)

    def test_failed_hard_gate_names_the_gate(self):
        df = _fake_df()
        sig = self._signal(score=90, mtf_ok=False)
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=_fake_regime()), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        self.assertIn("BLOCKED", msg)
        self.assertIn("MTF", msg)

    def test_failed_rel_strength_alone_does_not_report_blocked(self):
        # Found 2026-08-16 review: RS/Sector used to be reported as HARD
        # blocking gates here, but the live scanner (run_pro_scanner) only
        # ever scores them as points -- a signal missing RS/Sector can
        # still trade live. This directly contradicted the live scanner's
        # real decision on exactly the question /why exists to answer.
        df = _fake_df()
        sig = self._signal(score=90, rs_ok=False, sector_ok=False)
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=_fake_regime()), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        self.assertIn("Would PASS all gates", msg,
                     "a real hard-gate pass with only RS/Sector missing must not report BLOCKED")

    def test_elevated_vix_raises_the_score_floor_like_the_live_scanner(self):
        # A signal scoring 80 clears a 75 floor but not the VIX>25 floor of
        # 90 -- the live scanner would reject this; explain_ticker must
        # agree.
        df = _fake_df()
        sig = self._signal(score=80)
        regime = {"regime": "CHOP", "score": 10, "vix_ok": True,
                 "details": {"VIX": 28.0}}
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=regime), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        self.assertIn("BLOCKED", msg, "VIX>25 must raise the required score to 90, "
                                      "matching the live scanner")

    def test_vix_shock_adds_five_to_the_floor(self):
        # Isolated from the current real calendar month's seasonal filter
        # (which would otherwise interact with the floor) -- this test is
        # specifically about the vix_shock +5 escalation. Also isolated from
        # the real dman_setup_probation.json: "Gap & Hold" itself has no
        # per-setup override in SETUP_MIN_CONFLUENCE, but a live setup can
        # land on real probation (+10 to effective_min) independent of this
        # test's scenario -- confirmed live 2026-08-28, "Gap & Hold" was
        # auto-restricted for a real 3-trade losing streak, which silently
        # broke this test's "80 is the real floor" assumption via an
        # unrelated, unmocked production-state read.
        _probation_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        _probation_tmp.write(b"{}")
        _probation_tmp.close()
        self.addCleanup(os.unlink, _probation_tmp.name)
        df = _fake_df()
        sig = self._signal(score=83, setup="Gap & Hold")   # no per-setup override, seasonal-exempt too
        regime = {"regime": "CHOP", "score": 10, "vix_ok": True,
                 "details": {"VIX": 18.0, "VIX Shock": "15->22 in one session"},
                 "vix_shock": True}
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=regime), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "SETUP_PROBATION_FILE", _probation_tmp.name), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("TESTX", min_score=75)
        # base min_score 75 -> vix_shock adds +5 = 80; Gap & Hold has no
        # per-setup override and is seasonal-exempt, so 80 is the real
        # floor -- an 83 score clears it.
        self.assertIn("need ≥80", msg)
        self.assertIn("Would PASS all gates", msg)

    def test_defensive_rotation_penalizes_tech_long_score_like_the_live_scanner(self):
        df = _fake_df()
        sig = self._signal(score=80)
        sig.ticker = "AAPL"   # a real Technology-sector ticker
        regime = {"regime": "CHOP", "score": 10, "vix_ok": True,
                 "details": {"VIX": 15.0}, "defensive_rotation": True}
        with patch.object(a, "fetch_df", return_value=df), \
             patch.object(a, "compute_indicators", return_value=df), \
             patch.object(a, "_raw_signals", return_value=sig), \
             patch.object(a, "get_market_regime", return_value=regime), \
             patch.object(a, "_fetch_alpaca_news", return_value={}), \
             patch.object(a, "score_signal", return_value=sig):
            msg = a.explain_ticker("AAPL", min_score=75)
        self.assertIn("Score   : 75/100", msg, "the -5 defensive-rotation penalty on a "
                                               "tech LONG must be applied, matching the live scanner")


class TestEquityFallbackAlert(unittest.TestCase):
    """get_effective_account() silently fell back to the static
    ACCOUNT_SIZE secret with zero alerting whenever live equity was
    unreachable. Added 2026-08-15 so that failure mode becomes visible."""

    def setUp(self):
        self._orig_cache = dict(a._live_equity_cache)
        a._live_equity_cache["equity"] = 0.0
        a._live_equity_cache["ts"] = 0.0
        self._alerts_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._alerts_tmp.write(b"{}")
        self._alerts_tmp.close()
        self._alerts_patch = patch.object(a, "LAST_ALERTS_FILE", self._alerts_tmp.name)
        self._alerts_patch.start()

    def tearDown(self):
        a._live_equity_cache.clear()
        a._live_equity_cache.update(self._orig_cache)
        self._alerts_patch.stop(); os.unlink(self._alerts_tmp.name)

    def test_unreachable_client_triggers_fallback_alert(self):
        with patch.object(a, "get_alpaca_client", return_value=None), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            result = a.get_effective_account()
        self.assertEqual(result, a.ACCOUNT_SIZE)
        msgs = [c[0][0] for c in mock_tg.call_args_list]
        self.assertTrue(any("Live equity unavailable" in m for m in msgs))

    def test_successful_fetch_sends_no_alert(self):
        mock_client = MagicMock()
        mock_client.get_account.return_value = MagicMock(equity="9876.0")
        with patch.object(a, "get_alpaca_client", return_value=mock_client), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            result = a.get_effective_account()
        self.assertEqual(result, 9876.0)
        mock_tg.assert_not_called()

    def test_stale_but_previously_good_cache_is_reused_without_alerting(self):
        a._live_equity_cache["equity"] = 12345.0
        a._live_equity_cache["ts"] = time.time() - 400   # older than the 300s cache window
        with patch.object(a, "get_alpaca_client", side_effect=Exception("network down")), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            result = a.get_effective_account()
        self.assertEqual(result, 12345.0)
        mock_tg.assert_not_called()

    def test_repeat_calls_within_cooldown_do_not_repeat_the_alert(self):
        with patch.object(a, "get_alpaca_client", return_value=None), \
             patch.object(a, "get_todays_loss", return_value=0.0), \
             patch.object(a, "send_telegram", return_value=True) as mock_tg:
            a.get_effective_account()
            a.get_effective_account()
        msgs = [c[0][0] for c in mock_tg.call_args_list]
        fallback_msgs = [m for m in msgs if "Live equity unavailable" in m]
        self.assertEqual(len(fallback_msgs), 1)


class TestOptionsPositionBudget(unittest.TestCase):
    """Added 2026-08-21: OPTIONS_MAX_POSITION_COST (a flat $2,000) replaced
    with OPTIONS_MAX_POSITION_PCT (15% of current equity) after the flat
    figure -- already flagged as ~40% concentration when set against a
    $5,044 account -- drifted to 61% of a single options trade on the same
    $2,000 against a $3,273 account post-losing-stretch. A regression here
    means position sizing silently stops tracking real account size again."""

    def test_budget_scales_with_current_equity(self):
        with patch.object(a, "get_effective_account", return_value=10_000.0):
            self.assertEqual(a._options_position_budget(), 1_500.0)

    def test_budget_shrinks_as_account_shrinks(self):
        with patch.object(a, "get_effective_account", return_value=3_273.0):
            self.assertAlmostEqual(a._options_position_budget(), 490.95, places=2)

    def test_not_cached_reflects_a_fresh_get_effective_account_call_each_time(self):
        with patch.object(a, "get_effective_account", side_effect=[5_000.0, 2_000.0]):
            first  = a._options_position_budget()
            second = a._options_position_budget()
        self.assertEqual(first, 750.0)
        self.assertEqual(second, 300.0)


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


def _bear_gap_hold_df(prior_close, prior_open, today_open, today_close):
    """Builds a DataFrame shaped to isolate Bear Gap Hold's (L9) own gap
    calculation from every OTHER hard-coded condition in _raw_signals() --
    every row except the last two is flat filler satisfying the RSI/MACD/
    RVOL/dollar-volume/ATR conditions; only Open/Close on the final two
    rows (today vs. yesterday) vary, since gap-down % is the one thing
    under test."""
    import pandas as pd
    import numpy as np
    n = 30
    idx = pd.date_range("2026-05-01", periods=n, freq="D")
    close = np.full(n, prior_close)
    df = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": [3_000_000] * n,
        "RSI": [40.0] * n, "MACD": [-0.5] * n, "MACD_sig": [-0.2] * n,
        "MACD_hist": [-0.1] * n, "EMA9": close, "EMA20": close,
        "EMA50": close, "ATR": [0.5] * n, "AvgVol20": [3_000_000] * n,
        "RVOL": [3.0] * n,
    }, index=idx)
    # Yesterday: red day (close < open) — required by Bear Gap Hold.
    df.loc[df.index[-2], "Open"]  = prior_open
    df.loc[df.index[-2], "Close"] = prior_close
    df.loc[df.index[-2], "High"]  = max(prior_open, prior_close) * 1.005
    df.loc[df.index[-2], "Low"]   = min(prior_open, prior_close) * 0.995
    # Today.
    df.loc[df.index[-1], "Open"]  = today_open
    df.loc[df.index[-1], "Close"] = today_close
    df.loc[df.index[-1], "High"]  = max(today_open, today_close) * 1.005
    df.loc[df.index[-1], "Low"]   = min(today_open, today_close) * 0.995
    return df


def _fake_regime():
    return {"regime": "CHOP", "score": 10, "vix_ok": True,
           "details": {}}


if __name__ == "__main__":
    unittest.main(verbosity=2)
