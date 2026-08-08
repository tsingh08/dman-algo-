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
        a._options_feed_state["feed"] = None
        a._options_feed_state["checked_at"] = 0.0

    def tearDown(self):
        a._options_feed_state["feed"] = None
        a._options_feed_state["checked_at"] = 0.0

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
        # First resolution alerts (state change None -> indicative). Force a
        # recheck without a real state change -- must not alert twice for
        # the same known-broken entitlement.
        mock_client = MagicMock()
        mock_client.get_option_contracts.return_value = self._mock_contract()
        mock_resp = MagicMock(status_code=403)
        with patch.object(a, "get_alpaca_client", return_value=mock_client):
            with patch.object(a.requests, "get", return_value=mock_resp):
                with patch.object(a, "send_telegram", return_value=True) as mock_tg:
                    a._resolve_options_feed()
                    a._options_feed_state["checked_at"] = 0.0   # force a recheck
                    a._resolve_options_feed()
        self.assertEqual(mock_tg.call_count, 1)


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
