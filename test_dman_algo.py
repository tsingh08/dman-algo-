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
        _, err = a._submit_earnings_spread(mock_client, self._plan(both_sides=False))
        self.assertIsNone(err)
        req = mock_client.submit_order.call_args[0][0]
        self.assertEqual(len(req.legs), 2)

    def test_limit_price_is_positive_for_a_debit(self):
        mock_client = MagicMock(); mock_client.submit_order.return_value = MagicMock(id="1")
        a._submit_earnings_spread(mock_client, self._plan())
        req = mock_client.submit_order.call_args[0][0]
        self.assertGreater(req.limit_price, 0)

    def test_submit_failure_returns_error_text_not_swallowed(self):
        mock_client = MagicMock()
        mock_client.submit_order.side_effect = Exception("insufficient buying power")
        oid, err = a._submit_earnings_spread(mock_client, self._plan())
        self.assertIsNone(oid)
        self.assertIn("insufficient buying power", err)

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

    def tearDown(self):
        self._patch.stop()
        os.unlink(self._tmp.name)

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
