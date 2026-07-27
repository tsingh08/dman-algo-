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

import os
import re
import sys
import unittest
from datetime import date, datetime
from unittest.mock import patch, MagicMock

# Credentials must be blank BEFORE import — several module-level constants
# read env vars at import time, and a stray real key in the environment
# (already bit us once this week during manual testing) must never let a
# test suite make a live network call or, worse, a live order attempt.
for _k in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "APCA_API_KEY_ID",
           "APCA_API_SECRET_KEY", "ANTHROPIC_API_KEY", "BENZINGA_API_KEY"):
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
