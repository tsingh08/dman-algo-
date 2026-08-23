"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DMan ALWAYS-ON DAEMON — the real-time layer cron can't provide          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Division of labor:                                                      ║
║    GitHub Actions (cron) → wide-net entries: full-universe scans,        ║
║        premarket, EOD P&L, roughly hourly                                ║
║    This daemon (24/5)    → fast layer, both entries and exits:           ║
║        • curated-universe signal scan every SCAN_INTERVAL_S (closes      ║
║          the up-to-55-min gap between hourly cron scans)                 ║
║        • options stop/T1/T2 enforcement every 60s (vs hourly on cron)    ║
║        • instant fill alerts via Alpaca TradingStream websocket          ║
║        • Telegram two-way commands answered in real time                 ║
║        • Alpaca fill sync every 5 min (P&L recorded within minutes)      ║
║        • git state sync so cron scans and the daemon share one brain     ║
║                                                                          ║
║  RUN:        py -3 dman_daemon.py     (or double-click start_daemon.bat) ║
║  AUTO-START: Task Scheduler → New Task → run start_daemon.bat at logon   ║
║                                                                          ║
║  REQUIRED ENV VARS (set once with setx, values from your dashboards):    ║
║    setx APCA_API_KEY_ID     "..."                                        ║
║    setx APCA_API_SECRET_KEY "..."                                        ║
║    setx TELEGRAM_TOKEN      "..."                                        ║
║    setx TELEGRAM_CHAT_ID    "..."                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO_DIR)          # all state files are repo-relative

import dman_algo as algo    # noqa: E402  (needs cwd set first)

# Real-time equity price streaming — OFF by default (2026-08-09). Confirmed
# live: equity positions (CELZ, CLRO — everything actually held) had NO
# continuous daemon-side monitoring at all, only the hourly cron. This adds
# both a WebSocket quote stream AND the guard_loop check that consumes it,
# gated behind one flag so the default daemon behavior is completely
# unchanged until it's been watched working for a session or two. Enable
# with the env var once confirmed: setx ENABLE_REALTIME_EQUITY_STREAM true
ENABLE_REALTIME_EQUITY_STREAM = os.getenv(
    "ENABLE_REALTIME_EQUITY_STREAM", "false").strip().lower() == "true"

# Real-time options quote streaming — same reasoning and same OFF-by-default
# rollout as ENABLE_REALTIME_EQUITY_STREAM above, applied to the options
# side (2026-08-12). run_options_guard() already checks every tracked
# call/put every 60s via a REST snapshot; this doesn't change that cadence,
# it feeds each check a quote that's already live in memory (a WebSocket
# push) instead of a fresh REST round-trip every single time, so a fast
# stop/trail move gets caught against fresher data without hammering the
# options snapshot endpoint 60x/hour per position. Greeks (needed only for
# the theta-decay alert, not the stop/trail/T1 price checks) stay REST-only
# — see _monitor_option_position's get_snapshot_fn docstring.
ENABLE_REALTIME_OPTIONS_STREAM = os.getenv(
    "ENABLE_REALTIME_OPTIONS_STREAM", "false").strip().lower() == "true"

# Real-time news headline streaming (2026-08-13) — every news lookup in
# dman_algo.py today (_fetch_massive_reference_news, _fetch_massive_benzinga_news,
# the Alpaca NewsClient fallback) is REST/poll-based, so a breaking headline
# on a held position or watchlist name only surfaces at the next scan_loop
# pass or hourly cron. This subscribes to Alpaca's NewsDataStream for the
# curated universe and pushes an instant Telegram alert the moment a
# relevant headline breaks — informational only, same as every other
# real-time stream here: it doesn't gate or trigger any order submission,
# the existing REST-based sentiment gate still governs new entries.
ENABLE_REALTIME_NEWS_STREAM = os.getenv(
    "ENABLE_REALTIME_NEWS_STREAM", "false").strip().lower() == "true"

GUARD_EVERY_S   = 10        # options stop/target enforcement cadence — tightened from
                            # 60s (2026-08-15, direct instruction to watch the option
                            # chain as closely as possible on the live UMAC play). Safe
                            # to tighten this far specifically BECAUSE the streamed
                            # snapshot cache (option_data_stream_loop) already updates
                            # every tick from Alpaca's OPRA feed — this cadence controls
                            # how often the DECISION logic (stop/trail/T1 checks) reads
                            # that already-live cache, not how often data is fetched, so
                            # it doesn't multiply REST calls or rate-limit risk on the
                            # happy path (stream connected). SYNC_EVERY_S (git/fill sync)
                            # is unaffected — that stays on its own separate cadence.
SYNC_EVERY_S    = 300       # fill sync + git state sync cadence
SCAN_INTERVAL_S = 600       # curated-universe signal scan cadence (10 min) —
                            # a full-scan takes a few minutes itself (large-cap
                            # pass + smallcap/Finviz discovery), so this leaves
                            # comfortable headroom rather than back-to-back runs
SCAN_RETRY_S    = 60        # after a FAILED scan, retry this soon instead of
                            # waiting the full SCAN_INTERVAL_S — every minute
                            # of downed coverage is a real missed setup
STATE_FILES = [
    "dman_positions.json", "dman_last_alerts.json", "dman_live_signals.json",
    "dman_live_outcomes.csv", "dman_alpaca_sync.json", "dman_win_rate.json",
    "dman_daily_pnl.json", "dman_monthly_pnl.json", "dman_halt.json",
    "dman_probation.json",   # manual reduced-size trading period — see is_on_probation()
    "dman_day_start_equity.json",   # today's EOD-P&L baseline — see _get_day_start_equity();
                                      # found 2026-08-21 missing from every persist list, so it
                                      # was never actually committed back and every session
                                      # started from a stale multi-day-old checkout
    "dman_telegram_state.json", "dman_smallcap_watchlist.json",
    "dman_alerts_dedup.json",   # T1/T2/stop/DTE options-alert dedup — was missing,
                                 # meant every alert re-fired across separate runs
    "dman_earnings_pending.json",   # awaiting-approval earnings spreads (permanent gate, no auto-promotion)
    "dman_rate_limit_events.json",  # today's Alpaca 429 counts — watchdog's rate-limit check reads this
    "dman_options_feed_state.json",  # same class of bug as dman_alerts_dedup.json above (confirmed
                                       # live 2026-08-08) — without this, each of the two daily daemon
                                       # sessions starts with no memory of the last OPRA-entitlement
                                       # probe and re-alerts "options data feed downgraded" on its first check
    "dman_sec_cik_map.json",   # SEC EDGAR ticker->CIK map backing the free insider-buying signal —
                                # 1-week TTL cache, pointless if it doesn't survive between sessions
    "dman_stock_feed_state.json",  # same class of bug as dman_options_feed_state.json above, for
                                     # Algo Trader Plus's stock-side SIP entitlement (added 2026-08-09)
    "dman_telegram_options_menu.json",  # the /options TICKER menu /buy N resolves an index against —
                                          # must survive a daemon restart between the two commands
    "dman_telegram_manual_buy.json",    # the staged /buy confirmation awaiting a YES/NO reply — same
                                          # reasoning as dman_earnings_pending.json above
    "dman_news_log.json",   # rolling background news log (see algo._log_news_event) — must
                              # survive a session handover for "constantly internalize news
                              # across market + extended hours" to actually mean something
                              # continuous, not a log that resets every ~3 hours
    "dman_scan_log.json",   # found 2026-08-23 missing from this list: scan_loop() writes it every
                              # ~10 min, so `git pull --rebase` in _tracked() hit an unstaged change
                              # on a file this daemon never stashed and failed outright — not a merge
                              # conflict, a hard pull failure — which stops origin/main from advancing
                              # and freezes every OTHER state file's sync for the rest of the session
                              # too, the same wedge _tracked()'s own docstring describes for a
                              # different trigger. sync_scan_log_with_remote() already exists to merge
                              # this file correctly; it just never got a chance to run.
]

EARNINGS_LOOP_TRIGGER_HHMM = 1445   # 2:45 PM ET — ~45 min buffer before the 4 PM close

# Confirmed live 2026-07-31: BOTH of that day's daemon sessions saw the
# TradingStream fail to connect on every single attempt (0 successful
# connects across two full sessions) — the storm-detection fix caught and
# contained each burst (370/676 attempts vs the prior day's 27,366/39,438,
# so the acute damage was avoided), but something kept the connection from
# ever succeeding at all. Root cause: main()'s clean exit at DAEMON_RUN_UNTIL
# just returns — Python kills daemon threads (stream_loop's thread included)
# with zero cleanup when the main thread exits, so the websocket never gets
# a real close handshake. GitHub Actions' own job-timeout and the next
# scheduled session's cancel-in-progress both do the same via SIGTERM. If
# Alpaca treats an un-closed connection as still active, every subsequent
# session's connection attempts get rejected until that stale session times
# out server-side — a plausible explanation for a lockout persisting across
# multiple sessions, independent of and in addition to the reconnect-storm
# bug fixed the day before. _active_stream/_shutdown_event let both the
# normal exit path and a SIGTERM handler close the live connection properly
# before the process actually dies.
_shutdown_event    = threading.Event()
_active_stream_lock = threading.Lock()
_active_stream = None   # the live TradingStream instance, if any

# Second, independent stream (real-time equity quotes, see
# market_data_stream_loop) — same close-before-exit reasoning as
# _active_stream above applies equally to this one; tracked separately
# since the two streams connect/disconnect on completely different
# schedules (order-fill events vs. open-position changes).
_active_market_stream_lock = threading.Lock()
_active_market_stream = None   # the live StockDataStream instance, if any

# Real-time equity quote cache fed by market_data_stream_loop, consumed by
# guard_loop's run_equity_guard() call. (price, monotonic_timestamp) per
# ticker so a stale/disconnected stream is detectable and never trusted.
_realtime_prices: dict[str, tuple[float, float]] = {}
_realtime_prices_lock = threading.Lock()
_REALTIME_PRICE_MAX_AGE_S = 30

# Third, independent stream (real-time options quotes, see
# option_data_stream_loop) — tracked separately from both streams above for
# the same reason _active_market_stream is separate from _active_stream:
# each connects/disconnects on its own schedule (open options legs changing
# is a different event than open equity positions changing).
_active_option_stream_lock = threading.Lock()
_active_option_stream = None   # the live OptionDataStream instance, if any

# Real-time options quote cache fed by option_data_stream_loop, consumed by
# guard_loop's run_options_guard(get_snapshot_fn=...) call. Keyed by OCC
# symbol -> (snapshot dict, monotonic_timestamp), same staleness guard as
# _realtime_prices above.
_realtime_option_quotes: dict[str, tuple[dict, float]] = {}
_realtime_option_quotes_lock = threading.Lock()
_REALTIME_OPTION_QUOTE_MAX_AGE_S = 30

# Fourth, independent stream (real-time news headlines, see
# news_data_stream_loop) — tracked separately for the same reason as the
# three streams above: its own connect/disconnect schedule.
_active_news_stream_lock = threading.Lock()
_active_news_stream = None   # the live NewsDataStream instance, if any

# Dedup cache for news_data_stream_loop — Alpaca's news websocket has no
# documented redelivery guarantee, but a reconnect happening mid-story
# (multiple wire updates to the same headline) is plausible, and this is
# cheap insurance against a duplicate Telegram alert either way. Plain
# dict used as an insertion-ordered set (Python 3.7+ guarantee) so the
# oldest id is O(1) to evict once the cap is hit — no need for real news
# history here, just "have we already alerted on this one."
_news_seen_ids: dict = {}
_news_seen_ids_lock = threading.Lock()
_NEWS_SEEN_IDS_MAX = 2000

# Macro/Fed-policy news extension (added 2026-08-13, direct request after
# reviewing the Thursday session): the news stream above only ever alerted
# on watchlist-ticker-tagged headlines, with no way to catch a broad
# monetary-policy story (Fed/FOMC commentary, a CPI/PPI print interpreted
# as dovish/hawkish) that isn't tagged to any single stock. get_market_regime()
# already treats TLT (rates) and UUP (dollar) as Fed-policy price proxies —
# these symbols aren't in WATCHLIST/DMAN_SMALLCAP_WATCHLIST today, so a
# headline tagged only to one of them would currently be silently dropped
# by the relevance filter even though it's exactly the kind of story this
# exists to catch. DIA is added alongside SPY/QQQ/IWM (already in
# WATCHLIST) since broad-market macro stories commonly tag all four index
# ETFs together. MACRO_NEWS_KEYWORDS is a second, independent net: a
# headline's own text is scanned regardless of which of the SUBSCRIBED
# symbols it happens to be tagged with, so a Fed story tagged alongside an
# unrelated watchlist ticker still gets flagged as macro rather than
# rendered as an ordinary single-stock alert.
#
# MACRO_NEWS_SYMBOLS/MACRO_NEWS_KEYWORDS/FAST_SCAN_MACRO_KEYWORDS moved to
# dman_algo.py 2026-08-21 (see WATCHLIST's own neighborhood there) so
# check_news_keyword_freshness() -- a standalone dman_algo.py process, no
# import of this file -- can read the SAME list a Telegram-approved
# addition/removal actually edits, instead of a copy that would silently
# drift from what the live news stream here is really matching against.
def _news_subscription_symbols() -> list[str]:
    """_curated_universe_tickers() plus the macro-proxy symbols above —
    the news stream's actual subscription list. Kept as one function so
    the subscribe call and the later reconnect-on-change poll compare
    against the exact same set."""
    return sorted(set(_curated_universe_tickers()) | set(algo.MACRO_NEWS_SYMBOLS))


def _is_macro_headline(headline: str) -> bool:
    """Case-insensitive substring match against algo.MACRO_NEWS_KEYWORDS.
    Padded keywords (e.g. "fed ", " cpi ") avoid matching mid-word
    (e.g. "federal" already has its own full-word entry; a bare "fed"
    would also match "federated", "federal-express-adjacent", etc.)."""
    h = f" {headline.lower()} "
    return any(kw in h for kw in algo.MACRO_NEWS_KEYWORDS)


# Feeds scan_loop()'s fast-scan trigger (added 2026-08-13 alongside the
# widened watchlist quote stream below): market_data_stream_loop's
# on_quote handler snapshots each curated-universe ticker's price into
# _scan_baseline_prices the first time it's seen after every scan_loop
# pass, then compares every later tick against that baseline. A move past
# FAST_SCAN_MOVE_TRIGGER_PCT sets _fast_scan_trigger, which scan_loop
# waits on instead of a fixed sleep — wakes the SAME run_pro_scanner()
# pipeline (same gates, same quality bar) early instead of waiting up to
# SCAN_INTERVAL_S. This is a detection-latency change only, not a trading-
# rule change — _submit_signals_to_alpaca()'s already-tracked guard means
# an early look at an already-held ticker is a harmless no-op.
#
# Confirmed live 2026-08-13, first day this ran: fired 123 times in one
# regular-hours session and 348 times in one after-hours session — nowhere
# near the intended "catch a genuinely fast move between 10-min cron
# cycles" rate. Two compounding causes, both fixed below:
#   1. No market_hours() gate — during closed/extended hours,
#      scan_loop's own scan (and thus the baseline-clearing that follows
#      it) never runs, so a ticker's baseline is never reset; every
#      illiquid after-hours print that happened to cross the threshold
#      fired again, forever, for nothing (no scan can even run then).
#   2. Evaluated against the FULL curated universe, including
#      DMAN_SMALLCAP_WATCHLIST — a 3% intraday swing on a sub-$50M-float
#      microcap is routine noise, not a signal; those names already get
#      dedicated, purpose-built coverage via the StockTwits monitor and
#      the smallcap catalyst gate. Restricted to the large-cap WATCHLIST,
#      where a 3% move is actually notable.
# A cooldown is also applied as a second line of defense against a
# genuinely fast, multi-ticker session still thrashing scan_loop.
FAST_SCAN_MOVE_TRIGGER_PCT = 3.0
FAST_SCAN_TRIGGER_COOLDOWN_S = 120
_last_fast_scan_trigger_ts = 0.0
_scan_baseline_prices: dict[str, float] = {}
_scan_baseline_lock = threading.Lock()
_fast_scan_trigger = threading.Event()


def _close_active_stream() -> None:
    """Best-effort clean close of whatever stream connection is currently
    live, so Alpaca doesn't carry it over as still-open into the next
    session. Safe to call even if nothing is connected."""
    with _active_stream_lock:
        s = _active_stream
    if s is not None:
        try:
            s.stop()
        except Exception:
            pass
    with _active_market_stream_lock:
        ms = _active_market_stream
    if ms is not None:
        try:
            ms.stop()
        except Exception:
            pass
    with _active_option_stream_lock:
        os_ = _active_option_stream
    if os_ is not None:
        try:
            os_.stop()
        except Exception:
            pass
    with _active_news_stream_lock:
        ns = _active_news_stream
    if ns is not None:
        try:
            ns.stop()
        except Exception:
            pass


def _get_realtime_price(ticker: str) -> float | None:
    """Real-time streamed mid-quote for `ticker`, or None if the stream
    isn't running / hasn't reported one / it's gone stale. Passed as
    run_equity_guard()'s get_price_fn — that function already falls back
    to a REST lookup on None, so a dead or cold stream never blocks a
    check, it just makes that one check as slow as it always was."""
    with _realtime_prices_lock:
        entry = _realtime_prices.get(ticker)
    if entry is None:
        return None
    price, ts = entry
    if time.monotonic() - ts > _REALTIME_PRICE_MAX_AGE_S:
        return None
    return price


def _curated_universe_tickers() -> list[str]:
    """WATCHLIST + DMAN_SMALLCAP_WATCHLIST + every currently-held ticker
    (the "ticker" field is always the underlying regardless of setup type
    — plain equity, options call/put, or earnings spread — so no
    setup-prefix filtering is needed here, unlike the options-leg helper
    below). Shared subscription list for both real-time streams that
    benefit from broad coverage rather than just open positions:
    market_data_stream_loop() (equity quotes — widened 2026-08-13 from
    open-positions-only so a watchlist name's fast move can be caught
    without waiting for the next scan_loop cycle) and
    news_data_stream_loop() (breaking headlines). Bounded (~200 symbols)
    and changes rarely, unlike _current_option_occ_symbols() which churns
    on every contract-level open/close."""
    tickers = set(algo.WATCHLIST) | set(algo.DMAN_SMALLCAP_WATCHLIST)
    try:
        with open(algo.POSITIONS_FILE) as f:
            positions = json.load(f)
        for pos in positions:
            t = pos.get("ticker", "")
            if t:
                tickers.add(t)
    except Exception:
        pass
    return sorted(tickers)


def _get_realtime_option_snapshot(occ_symbol: str) -> dict | None:
    """Real-time streamed bid/ask/mid snapshot for `occ_symbol`, or None if
    the stream isn't running / hasn't reported one / it's gone stale.
    Passed as run_options_guard()'s get_snapshot_fn — that function already
    falls back to a REST snapshot on None, so a dead or cold stream never
    blocks a check, it just makes that one check as slow as it always was.
    Mirrors _get_realtime_price() exactly."""
    with _realtime_option_quotes_lock:
        entry = _realtime_option_quotes.get(occ_symbol)
    if entry is None:
        return None
    snap, ts = entry
    if time.monotonic() - ts > _REALTIME_OPTION_QUOTE_MAX_AGE_S:
        return None
    return snap


def _current_option_occ_symbols() -> list[str]:
    """OCC symbols for every currently-tracked options leg (naked calls/
    puts and earnings-spread legs alike) — the options stream's
    subscription list. Contract-level, so (unlike _curated_universe_tickers())
    this changes on every open/close and can't be folded into that
    shared helper."""
    try:
        with open(algo.POSITIONS_FILE) as f:
            positions = json.load(f)
    except Exception:
        return []
    symbols = set()
    for pos in positions:
        setup = pos.get("setup", "")
        if setup.startswith("Earnings "):
            for sym in pos.get("legs", []):
                if sym:
                    symbols.add(sym)
        elif setup.startswith(("Options Call ", "Options Put ")):
            # algo._position_identity() already extracts the OCC symbol
            # for exactly this purpose -- found in the 2026-08-16 review:
            # this used to reimplement the same parsing inline, one of
            # three duplicated copies in the project (the other two in
            # dman_algo.py: sync_alpaca_fills(), the orphan-position
            # detector). If the setup-string format ever changes, the
            # canonical helper gets fixed and copies like this one
            # silently don't.
            symbols.add(algo._position_identity(pos.get("ticker", ""), setup))
    return sorted(symbols)


def _handle_sigterm(signum, frame) -> None:
    """
    SIGTERM has no default cleanup behavior in Python (unlike SIGINT, which
    raises KeyboardInterrupt) — without this handler, a job-timeout or the
    next session's cancel-in-progress kills the process with literally zero
    chance for the websocket to close cleanly. This buys that close
    handshake the ~seconds GitHub Actions grants before escalating to
    SIGKILL.
    """
    log(f"received signal {signum} — closing stream connection before exit")
    _shutdown_event.set()
    _close_active_stream()


def log(msg: str) -> None:
    print(f"[{datetime.now(algo.ET).strftime('%a %H:%M:%S')}] {msg}", flush=True)


def market_hours() -> bool:
    now = datetime.now(algo.ET)
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 925 <= t <= 1605


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_DIR,
                          capture_output=True, text=True, timeout=90)


def _existing(paths: list[str]) -> list[str]:
    return [p for p in paths if os.path.exists(p)]


def _tracked(paths: list[str]) -> list[str]:
    """Subset of `paths` git already knows about (committed, in HEAD or the
    index) regardless of whether they currently exist on disk right now.

    Added 2026-08-16 after reproducing a real, permanent git_sync() wedge:
    _existing() alone can't see a file that WAS committed but has since
    been deleted locally — e.g. dman_telegram_manual_buy.json, removed by
    os.remove() on EVERY YES/NO reply to /buy, or dman_halt.json, removed
    by os.remove() on /resume. A deleted-but-still-tracked file is an
    uncommitted change git cares about, and `git pull --rebase` refuses to
    run at all while any such change is unstaged — every git_sync() call
    after that point fails its pre-flight silently (only a `log()` line),
    origin/main stops advancing, and the daemon stops publishing state
    (raised stops, closed positions, halt flags) for the rest of the
    session. Used at the staging step (git add -A) so a deletion actually
    gets committed instead of sitting as a permanent unstaged change.
    """
    if not paths:
        return []
    try:
        result = _git("ls-files", "--", *paths)
        if result.returncode != 0:
            return []
        return [line for line in result.stdout.splitlines() if line]
    except Exception:
        return []


def _restore_corrupted_state_files(paths: list[str]) -> list[str]:
    """
    Guarantee nothing invalid ever gets staged. Any .json file that fails to
    parse — most commonly literal `<<<<<<<` conflict markers left behind by
    a `git stash pop` that couldn't auto-merge — is restored from the last
    good commit (`git checkout -- file`) and excluded from this cycle's add
    list. This actually happened in production: a stash-pop conflict on
    dman_alpaca_sync.json (a one-line timestamp change on both sides) landed
    raw conflict markers in the file, which the old code then committed and
    pushed as-is. Losing one cycle's update to a single file is bounded and
    self-healing (the next tick recomputes it fresh); publishing broken JSON
    to the shared repo is not — every future read of that file falls back to
    an empty default until someone notices, silently discarding sync history.

    Found in the 2026-08-16 review: this only ever validated .json files.
    dman_live_outcomes.csv is in STATE_FILES but isn't JSON, so the exact
    same stash-pop-conflict failure mode there sailed straight through
    unvalidated and would get staged/committed/pushed with raw conflict
    markers still in it -- corrupting the win-rate ledger's CSV export
    the same way an unvalidated JSON file corrupts its own state. Checked
    for the literal `<<<<<<<`/`>>>>>>>` marker text rather than parsing as
    a structured CSV (this file's rows are free-text ticker/setup fields,
    so a stricter structural check risks false positives on legitimate
    content) — those two specific markers are unambiguous git artifacts
    that can't occur naturally in trade data.
    """
    ok: list[str] = []
    for p in paths:
        if p.endswith(".json"):
            try:
                with open(p) as f:
                    json.load(f)
                ok.append(p)
            except Exception as exc:
                log(f"  ⚠️  {p} failed JSON validation ({exc}) — restoring last "
                    f"good commit, skipping this cycle's update to that file")
                _git("checkout", "--", p)
            continue
        if p.endswith(".csv"):
            try:
                with open(p, "r", errors="replace") as f:
                    content = f.read()
                if "<<<<<<<" in content or ">>>>>>>" in content:
                    raise ValueError("unresolved git conflict marker found")
                ok.append(p)
            except Exception as exc:
                log(f"  ⚠️  {p} failed CSV validation ({exc}) — restoring last "
                    f"good commit, skipping this cycle's update to that file")
                _git("checkout", "--", p)
            continue
        ok.append(p)
    return ok


_git_lock = threading.Lock()


def git_sync() -> None:
    """
    Best-effort two-way state sync with the repo so cron scans and this
    daemon share one brain. Pull first (new positions opened by Actions),
    then push local changes (halts, raised stops, telegram offset).
    Alpaca remains the source of truth for actual holdings, so an
    occasional failed sync degrades gracefully.

    Every git invocation below filters STATE_FILES through _existing()
    first: `git add a.json missing.json` fails ATOMICALLY on the missing
    pathspec and silently stages NOTHING (not even a.json) — dman_halt.json
    and dman_telegram_state.json only exist after the first /halt or bot
    command, so passing the raw list here would silently no-op every sync.

    Called from two threads now (guard_loop's periodic sync and scan_loop's
    immediate post-submission sync) — _git_lock serializes them so two
    concurrent git processes on the same local repo can't collide on
    .git/index.lock.
    """
    if not _git_lock.acquire(timeout=60):
        log("git sync skipped — another sync already in progress past timeout")
        return
    try:
        _present = _existing(STATE_FILES)
        if _present:
            _git("stash", "push", "--include-untracked", "--", *_present)
        pull = _git("pull", "--rebase", "origin", "main")
        if pull.returncode != 0:
            _git("rebase", "--abort")
        if _present:
            pop = _git("stash", "pop")
            if pop.returncode != 0:
                # Conflict: git left <<<<<<< markers in whichever file(s)
                # couldn't auto-merge. Resolve just those files from the last
                # good commit, then the stash entry is fully spent (some
                # files applied cleanly by pop itself, the rest reset by us)
                # — drop it so conflicted stashes don't pile up run over run.
                log("stash pop conflict — restoring corrupted file(s) from last commit")
                _restore_corrupted_state_files(_existing(STATE_FILES))
                _git("stash", "drop")

        # Merge + stage + commit + push, retried up to 3x — each attempt
        # re-runs EVERY semantic merge fresh, right before its own commit.
        #
        # Found live 2026-08-17: the previous version ran these merges
        # ONCE, then on a rejected push (another session pushed a real
        # change in the meantime — two overlapping daemon sessions,
        # caused by that same morning's separate scheduling-reliability
        # incident, both racing to push) did a raw `git pull --rebase` +
        # blind retry. That rebases the ALREADY-COMMITTED, now-stale
        # merge result onto the new origin/main via a line-based git
        # merge — it never re-runs the SEMANTIC merge to reconcile
        # against whatever actually changed on the other side. A real
        # ARTL position, tracked by one daemon session via
        # PositionTracker.open(), was silently dropped from
        # dman_positions.json this exact way the same night this fix was
        # written: the losing session's stale, unremerged retry
        # overwrote the winning session's real addition. Each attempt
        # below re-runs every sync_*_with_remote() call fresh before
        # trying to push again, so a rejected push always gets a genuine
        # re-reconciliation against whatever is ACTUALLY on origin/main,
        # never a replay of a decision made against data that's since
        # changed. Cheap to repeat — each merge already no-ops when
        # there's genuinely nothing new to reconcile.
        #
        # Semantic merge for dman_positions.json specifically: this
        # daemon (60s cadence) and the hourly cron scanner both
        # independently raise stops to breakeven / reduce shares on the
        # same open positions from separate concurrency groups — git's
        # line-based stash-pop/rebase merge can silently keep the LESS-
        # protective side. algo.sync_positions_with_remote() re-
        # reconciles against origin/main using a rule that can't regress
        # protection, regardless of whether the stash pop above was clean.
        #
        # The other syncs are the SAME class of bug, confirmed in
        # production on 2026-07-27: dman_scan_log.json went a full
        # trading day with zero new entries despite 8+ genuinely
        # successful scans, because it's rewritten by both the daemon
        # (every 10 min) and the cron scanner (hourly) so often that a
        # whole-file conflict resolution was silently discarding
        # whichever side's entry lost the race.
        for _attempt in range(3):
            for _sync_fn in (algo.sync_positions_with_remote, algo.sync_scan_log_with_remote,
                            algo.sync_win_rate_with_remote, algo.sync_live_signals_with_remote,
                            algo.sync_alpaca_sync_state_with_remote, algo.sync_news_log_with_remote,
                            algo.sync_daily_pnl_with_remote, algo.sync_monthly_pnl_with_remote,
                            algo.sync_earnings_pending_with_remote):
                try:
                    _sync_fn()
                except Exception as exc:
                    log(f"{_sync_fn.__name__} error (non-fatal): {exc}")

            # Re-check existence every attempt — the stash pop, pull, or
            # merge above may have created/removed files.
            # _restore_corrupted_state_files is a second, unconditional
            # safety net here — catches corruption from any source, not
            # just a flagged stash conflict (e.g. a process killed
            # mid-write leaving a truncated file).
            _present = _existing(STATE_FILES)
            _present = _restore_corrupted_state_files(_present)
            # Union with _tracked(): a STATE_FILES entry that's been deleted
            # locally (still tracked in git, absent from _present) must still
            # reach `git add -A` so its deletion gets staged — see _tracked()'s
            # docstring for the real incident this fixes. `-A` (not a bare add)
            # is what actually stages a removal for a path git already knows
            # about; a path that's neither existing nor tracked is deliberately
            # excluded from the pathspec (git add -A errors atomically — stages
            # NOTHING — if any single pathspec element matches nothing at all).
            _stage_paths = sorted(set(_present) | set(_tracked(STATE_FILES)))
            if _stage_paths:
                _git("add", "-A", "--", *_stage_paths)
            staged = _git("diff", "--staged", "--quiet")
            if staged.returncode == 0:
                break   # nothing to push this attempt — already in sync

            _git("-c", "user.email=github-actions[bot]@users.noreply.github.com",
                 "-c", "user.name=github-actions[bot]",
                 "commit", "-m", "chore: daemon state sync [skip ci]")
            push = _git("push", "origin", "HEAD:main")
            if push.returncode == 0:
                break   # published successfully

            log(f"git sync: push rejected (attempt {_attempt + 1}/3) — "
                "re-fetching and re-merging against current origin/main before retrying")
            # fetch + reset --soft (not `pull --rebase`) deliberately:
            # --soft only moves the branch pointer, leaving the index and
            # working tree exactly as they are — it asks git to do NO
            # content-level merge at all, so there's no line-based
            # rebase conflict to hit in the first place. The actual
            # reconciliation happens at the application layer on the
            # next loop iteration, when the sync_*_with_remote() calls
            # re-read the now-current origin/main and re-merge it against
            # whatever's still sitting in the working tree (which still
            # has every local change this process ever made, uncommitted-
            # but-staged relative to the new HEAD).  `pull --rebase` was
            # tried here first and confirmed live to hit exactly this
            # kind of conflict on a JSON file mid-race — see this
            # function's docstring for the full incident.
            fetch_retry = _git("fetch", "origin", "main")
            if fetch_retry.returncode != 0:
                log("git sync: fetch failed on retry — giving up this cycle, next sync will catch up")
                break
            _git("reset", "--soft", "origin/main")
        else:
            log("git sync: push still rejected after 3 attempts — giving up this cycle, next sync will catch up")
    except Exception as exc:
        log(f"git sync error (non-fatal): {exc}")
    finally:
        _git_lock.release()


def telegram_loop() -> None:
    """Real-time Telegram command handling — long-polls 24/7."""
    log("Telegram command loop started (/help from your phone)")
    while True:
        try:
            n = algo._process_telegram_commands(timeout=25)
            if n:
                log(f"handled {n} Telegram command(s)")
        except Exception as exc:
            log(f"telegram loop error: {exc}")
            time.sleep(10)


def guard_loop() -> None:
    """Options exit enforcement during market hours. When
    ENABLE_REALTIME_EQUITY_STREAM is on, also runs the equity counterpart
    (run_equity_guard) on the same GUARD_EVERY_S cadence — see that
    function's docstring for why equity positions needed this at all.
    When ENABLE_REALTIME_OPTIONS_STREAM is on, run_options_guard() is fed
    the streamed quote cache (option_data_stream_loop) instead of relying
    solely on its own REST fallback — same cadence, fresher data.
    Also runs _check_stop_coverage() (orphan-position + broker-side
    live-stop check, split out of _check_open_position_risk 2026-08-15) on
    this same tight cadence instead of only via run_pro_scanner()'s ~10min
    scan pass, so a broken bracket order (e.g. a stop stuck HELD) gets
    caught and auto-restored in seconds instead of minutes.

    git_sync() and the fill sync run on their OWN thread (sync_loop(),
    below) rather than inline here — found in the 2026-08-16 review: they
    used to run on THIS thread every SYNC_EVERY_S (5 min), roughly 10
    blocking git subprocesses plus a REST-heavy fill sync, stalling every
    check in this function (stop enforcement, T1/T2, orphan detection)
    for however long that took. A slow git push/rebase-retry (network
    hiccup, a real merge conflict) reintroduced exactly the latency the
    10s cadence tightening was meant to remove, for the duration of a
    network round-trip -- on a bad day, real minutes with no live stop
    enforcement.
    """
    log(f"Options guard loop started ({GUARD_EVERY_S}s cadence during market hours)"
        + (", equity guard also active" if ENABLE_REALTIME_EQUITY_STREAM else "")
        + (", options stream feeding checks" if ENABLE_REALTIME_OPTIONS_STREAM else ""))
    was_open = False
    while True:
        try:
            if market_hours():
                if not was_open:
                    log("Market open — guard active")
                    algo.send_telegram("🛡 <b>Daemon active</b> — real-time options "
                                       "guard + fill stream running for today's session.")
                    was_open = True
                # Loaded once and passed to both guards -- found in the
                # 2026-08-16 review: run_options_guard() and
                # run_equity_guard() each independently re-read and
                # re-parsed the same small dman_positions.json every 10s
                # guard tick. (_check_stop_coverage() below still does its
                # own PositionTracker() read -- different shape, left as
                # its own fetch rather than risk that larger refactor.)
                try:
                    with open(algo.POSITIONS_FILE) as _pf:
                        _guard_positions = json.load(_pf)
                except Exception:
                    _guard_positions = []
                _snap_fn = _get_realtime_option_snapshot if ENABLE_REALTIME_OPTIONS_STREAM else None
                _und_price_fn = _get_realtime_price if ENABLE_REALTIME_EQUITY_STREAM else None
                alerts = algo.run_options_guard(verbose=False, get_snapshot_fn=_snap_fn,
                                                get_price_fn=_und_price_fn, positions=_guard_positions)
                for a in alerts:
                    log(a.split("\n")[0].replace("<b>", "").replace("</b>", ""))
                try:
                    algo._check_stop_coverage()
                except Exception as exc:
                    log(f"stop coverage check error: {exc}")
                if ENABLE_REALTIME_EQUITY_STREAM:
                    try:
                        algo.run_equity_guard(get_price_fn=_get_realtime_price, positions=_guard_positions)
                    except Exception as exc:
                        log(f"equity guard error: {exc}")
            else:
                if was_open:
                    log("Market closed — guard idle")
                    was_open = False
            time.sleep(GUARD_EVERY_S)
        except Exception as exc:
            log(f"guard loop error: {exc}")
            time.sleep(GUARD_EVERY_S)


def sync_loop() -> None:
    """
    git state sync + Alpaca fill sync, on its own thread — decoupled from
    guard_loop()'s tight stop-checking cadence so a slow git push/rebase
    or a REST-heavy fill sync can never stall stop/T1/T2 enforcement. See
    guard_loop()'s docstring for the full reasoning. git_sync() itself is
    already safe to call from multiple threads (_git_lock serializes it
    against scan_loop's own immediate post-submission sync calls), so
    adding this third caller needs no new locking.
    """
    log(f"Sync loop started ({SYNC_EVERY_S}s cadence during market hours)")
    was_open = False
    while True:
        try:
            if market_hours():
                was_open = True
                git_sync()
                try:
                    n = algo.sync_alpaca_fills(algo.WinRateTracker())
                    if n:
                        log(f"synced {n} closed trade(s)")
                except Exception as exc:
                    log(f"fill sync error: {exc}")
            else:
                if was_open:
                    git_sync()   # final push of the day's state
                    was_open = False
            time.sleep(SYNC_EVERY_S)
        except Exception as exc:
            log(f"sync loop error: {exc}")
            time.sleep(SYNC_EVERY_S)


def scan_loop() -> None:
    """
    Periodic curated-universe signal scan — closes the coverage gap between
    hourly cron scans. The cron scanner only checks for new setups roughly
    once an hour (plus the 9:45 AM Gap & Hold gate); a fast-forming,
    catalyst-driven move (an earnings gap, a sudden news-driven breakout)
    can go undetected for up to 55 minutes even though this daemon is
    already running continuously with fast, reliable Alpaca SIP access.
    This uses that idle time to re-scan the curated (fast, large-cap +
    small-cap watchlist) universe every SCAN_INTERVAL_S seconds during
    market hours — same run_pro_scanner()/quality gates as the cron
    scanner, just checked more often. This changes frequency, not
    standards: no score threshold, setup gate, or risk rule is touched.

    Submission-race safety: PositionTracker's already_tracked check in
    _submit_signals_to_alpaca() prevents re-submitting a ticker that's
    already an open position, but that check only sees whatever
    dman_positions.json looked like at the start of THIS process's git
    pull — the daemon and the hourly cron are separate processes with
    separate checkouts. To keep that race window as small as possible
    (rather than the full 5-minute periodic sync interval), this loop
    calls git_sync() immediately after any submission attempt instead of
    waiting for guard_loop()'s next scheduled sync.

    Fast-scan trigger (added 2026-08-13): the 30s poll below now waits on
    _fast_scan_trigger instead of a plain sleep. market_data_stream_loop's
    widened watchlist quote stream sets that event the moment any curated-
    universe ticker moves FAST_SCAN_MOVE_TRIGGER_PCT+ since this loop's
    last pass, pulling next_scan_due to now so the SAME run_pro_scanner()
    gates get checked immediately instead of up to SCAN_INTERVAL_S late.
    Only changes when a look happens, never what counts as a real signal.
    """
    log(f"Signal scan loop started (curated universe, every {SCAN_INTERVAL_S}s "
        f"during market hours, {SCAN_RETRY_S}s retry after a failed scan, "
        f"fast-triggered early on a {FAST_SCAN_MOVE_TRIGGER_PCT:.0f}%+ watchlist move)")
    next_scan_due = 0.0
    while True:
        try:
            if market_hours() and time.time() >= next_scan_due:
                log("Running periodic curated-universe scan...")
                scan_ok = False
                try:
                    # CRITICAL: this daemon is a long-running process, unlike
                    # the cron scanner's one-shot execution. algo._cache has
                    # no TTL — without clearing it, every scan after the
                    # first would silently reuse hours-old price snapshots,
                    # making repeated scanning pointless. The codebase's own
                    # pre-existing --mode watch loop already solved this
                    # exact problem the same way (see its own _cache.clear()
                    # call, commented "force fresh data") — this loop needs
                    # the identical fix for the identical reason.
                    algo._cache.clear()
                    signals = algo.run_pro_scanner(
                        algo.WATCHLIST, use_ai=False, universe_label="daemon-curated",
                        include_dynamic_smallcap=False,   # broad Finviz net already covered hourly by cron
                    )
                    if signals:
                        log(f"{len(signals)} signal(s) found — submitting")
                        algo._submit_signals_to_alpaca(signals)
                        git_sync()   # push updated positions immediately, don't wait for the periodic sync
                    else:
                        log("no qualifying signals this pass")
                    scan_ok = True
                except Exception as exc:
                    log(f"scan error: {exc}")
                # Fresh baseline for the fast-scan trigger now that a look
                # just happened, whether it was itself fast-triggered or on
                # the normal schedule — a move measured from a stale
                # baseline would just keep re-firing on the same old move.
                with _scan_baseline_lock:
                    _scan_baseline_prices.clear()
                # Self-healing: on a failed scan, come back soon instead of
                # waiting the full interval — every minute of downed
                # coverage is a real missed setup. On success, the normal
                # cadence avoids hammering the API for no reason.
                if scan_ok:
                    next_scan_due = time.time() + SCAN_INTERVAL_S
                else:
                    log(f"retrying scan in {SCAN_RETRY_S}s (not waiting the full {SCAN_INTERVAL_S}s interval)")
                    next_scan_due = time.time() + SCAN_RETRY_S
            if _fast_scan_trigger.wait(timeout=30):
                _fast_scan_trigger.clear()
                log(f"Fast-scan trigger: a watchlist ticker moved "
                    f"{FAST_SCAN_MOVE_TRIGGER_PCT:.0f}%+ since the last pass — scanning now")
                next_scan_due = 0.0
        except Exception as exc:
            log(f"scan loop error: {exc}")
            time.sleep(30)


def stream_loop() -> None:
    """
    Alpaca TradingStream — instant Telegram alert on every fill.

    CRITICAL: never call stream.run() unbounded. Confirmed live 2026-07-30:
    alpaca-py's TradingStream._run_forever() (installed alpaca-py==0.43.x)
    retries on ANY websockets.WebSocketException — including a sustained
    HTTP 429 from Alpaca — with only a 10ms sleep between attempts, and
    never raises back out of stream.run(), so this function's own
    try/except+sleep(15) never got a chance to run. One ordinary disconnect
    (ping timeout, brief network blip — ordinary and expected) triggered a
    reconnect-flood fast enough that Alpaca's abuse detection locked out
    further attempts, and the flood then kept re-triggering the lockout for
    the rest of the session: 27,366 failed reconnects over 2h07m in one
    daemon run, 39,438 in another the same day — ~3.5 hours of the single
    busiest earnings day of the week. That reconnect storm burns the
    account's shared Alpaca API budget, and is the confirmed root cause of
    that day's earnings-spread scan finding "no liquid leg pair" on every
    single candidate (AAPL included) — the option-snapshot REST calls were
    being starved by the same lockout, not a real liquidity problem.

    Fix: run stream.run() in a watched background thread. Wrap _start_ws so
    we know the wall-clock time of every real connection attempt. If 20+
    attempts land within a 30s window with no successful (re)connect in
    between, that's the storm signature — stop the stream, throw the
    TradingStream instance away (don't try to resume a poisoned one), and
    back off for real (15s, doubling, capped at 5 min) before building a
    fresh one. A stable connection never accumulates attempts, so this
    can't false-positive on normal operation.
    """
    try:
        from alpaca.trading.stream import TradingStream
    except ImportError:
        log("alpaca-py stream unavailable — fill alerts disabled")
        return

    async def on_update(data) -> None:
        try:
            ev = str(getattr(data, "event", ""))
            o  = getattr(data, "order", None)
            if ev in ("fill", "partial_fill") and o is not None:
                sym  = getattr(o, "symbol", "?")
                side = str(getattr(o, "side", "?")).split(".")[-1].upper()
                qty  = getattr(o, "filled_qty", "?")
                px   = getattr(o, "filled_avg_price", "?")
                log(f"FILL {sym} {side} ×{qty} @ ${px}")
                algo.send_telegram(f"⚡ <b>FILL</b> — {sym} {side} ×{qty} @ ${px}"
                                   + (" (partial)" if ev == "partial_fill" else ""))
        except Exception as exc:
            log(f"fill handler error: {exc}")

    STORM_WINDOW_S      = 30
    STORM_ATTEMPT_COUNT = 20
    STORM_ALERT_COOLDOWN_S = 1800   # don't spam Telegram every backoff cycle
    backoff = 15
    last_storm_alert = 0.0

    global _active_stream
    while True:
        if _shutdown_event.is_set():
            log("stream loop stopping — shutdown requested")
            return
        attempt_times: list[float] = []
        connected = threading.Event()
        stopped   = threading.Event()
        stream    = None
        try:
            stream = TradingStream(algo.ALPACA_API_KEY, algo.ALPACA_SECRET_KEY,
                                   paper=algo.ALPACA_PAPER)
            stream.subscribe_trade_updates(on_update)
            with _active_stream_lock:
                _active_stream = stream

            # Every mutable object this pair of closures touches is passed
            # as a default-argument (early-bound at definition time), not
            # read as a free variable off stream_loop()'s own locals.
            # Found 2026-08-16 review: only `_orig` had this treatment —
            # attempt_times/connected/stream were read by NAME, so they
            # shared ONE cell across every loop iteration (Python closures
            # capture by reference, not by value). If a zombie thread from
            # iteration N is still blocked inside stream.run() (the exact
            # failure mode this whole function exists to guard against —
            # see the module docstring above) when iteration N+1 begins
            # and reassigns attempt_times/connected/stream to fresh
            # objects, the zombie's own _watched_start_ws/_run_stream
            # calls would silently operate on iteration N+1's objects
            # instead of its own — falsely arming the NEW stream's storm
            # detector with the OLD stream's reconnect attempts, and
            # letting the OLD stream's eventual stopped.set() fire on the
            # NEW iteration's Event.
            _orig_start_ws = stream._start_ws
            async def _watched_start_ws(_orig=_orig_start_ws, _attempts=attempt_times, _connected=connected):
                _attempts.append(time.monotonic())
                await _orig()
                _connected.set()
            stream._start_ws = _watched_start_ws

            def _run_stream(_stream=stream, _stopped=stopped):
                try:
                    _stream.run()
                finally:
                    _stopped.set()

            t = threading.Thread(target=_run_stream, daemon=True)
            t.start()

            storm = False
            while t.is_alive() and not stopped.is_set():
                if _shutdown_event.is_set():
                    break
                time.sleep(3)
                now = time.monotonic()
                attempt_times[:] = [a for a in attempt_times if now - a < STORM_WINDOW_S]
                if len(attempt_times) >= STORM_ATTEMPT_COUNT:
                    storm = True
                    break
                if connected.is_set():
                    connected.clear()
                    backoff = 15
                    log("TradingStream connected — instant fill alerts on")

            if _shutdown_event.is_set():
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                log("stream loop stopped cleanly (shutdown)")
                return

            if storm:
                log(f"stream reconnect storm detected ({len(attempt_times)} attempts in "
                    f"{STORM_WINDOW_S}s, likely rate-limited) — stopping and backing "
                    f"off {backoff}s")
                if time.monotonic() - last_storm_alert > STORM_ALERT_COOLDOWN_S:
                    algo.send_telegram(
                        "⚠️ Fill-alert stream hit a reconnect storm (rate-limited) — "
                        "backing off. Fills still get recorded via the normal scan/guard "
                        "sync, just not instantly.")
                    last_storm_alert = time.monotonic()
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            # Thread exited cleanly (stop() called elsewhere, or a non-WebSocketException
            # escaped _run_forever) without ever storming — treat as a normal disconnect.
            # Found 2026-08-16 review: unlike the other three stream loops,
            # this branch never called stream.stop()/waited on `stopped`
            # before looping back around to build a fresh TradingStream —
            # leaking the old stream object un-stopped every ordinary
            # disconnect, and using the full `backoff` (>=15s, doubling)
            # here instead of a short fixed wait.
            try:
                stream.stop()
            except Exception:
                pass
            stopped.wait(timeout=10)
            time.sleep(backoff)
        except Exception as exc:
            log(f"stream error: {exc} — reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            with _active_stream_lock:
                if _active_stream is stream:
                    _active_stream = None


def market_data_stream_loop() -> None:
    """
    Real-time equity quote stream — feeds _realtime_prices so guard_loop's
    run_equity_guard() call can check T1/stop progression against a fresh
    streamed price instead of a fresh REST call every single 60s check.
    Off by default (ENABLE_REALTIME_EQUITY_STREAM) — added 2026-08-09
    alongside run_equity_guard() itself; see that function's docstring
    for why equity positions needed continuous daemon-side monitoring at
    all (they previously had none — only the hourly cron checked them).

    Widened 2026-08-13 from open-positions-only to the full curated
    universe (_curated_universe_tickers(): WATCHLIST + smallcap watchlist
    + positions) — an Alpaca account gets exactly one live equity data
    connection per feed, so broadening coverage means widening THIS
    stream's subscription rather than opening a second one. The extra
    ticks for non-position symbols aren't wasted: they feed
    scan_baseline_prices/_fast_scan_trigger below, letting scan_loop react
    to a fast watchlist move without waiting for its next scheduled pass.

    Mirrors stream_loop()'s reconnect-storm guard exactly — same failure
    mode (an ordinary disconnect triggering a reconnect flood that gets
    the account's Alpaca API budget abuse-locked) is possible on any
    Alpaca websocket, not just TradingStream. See that function's
    docstring for the full incident writeup (27,366 / 39,438 failed
    reconnects across two sessions on 2026-07-30) this guards against.

    Subscription model: rather than mutating a live stream's subscription
    list (cross-thread call safety into alpaca-py's asyncio internals
    isn't something to bet a live position's monitoring on without much
    more testing time than tonight has), this reconnects with a fresh
    symbol list whenever the curated universe changes. It changes rarely
    enough that this is simple, safe, and not a reconnect-storm risk on
    its own — polled at most once per POSITION_POLL_S.
    """
    try:
        from alpaca.data.live import StockDataStream
        from alpaca.data.enums import DataFeed
    except ImportError:
        log("alpaca-py market data stream unavailable — real-time equity prices disabled")
        return

    async def on_quote(q) -> None:
        global _last_fast_scan_trigger_ts
        try:
            sym = getattr(q, "symbol", None)
            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            if not (sym and bid > 0 and ask > 0):
                return
            mid = round((bid + ask) / 2, 4)
            with _realtime_prices_lock:
                _realtime_prices[sym] = (mid, time.monotonic())
            # Fast-scan trigger: large-cap WATCHLIST only, market hours only
            # (see FAST_SCAN_MOVE_TRIGGER_PCT's docstring above for why —
            # smallcap noise and after-hours illiquidity both false-fired
            # this constantly on 2026-08-13, the day it shipped).
            if market_hours() and sym in algo.WATCHLIST:
                with _scan_baseline_lock:
                    base = _scan_baseline_prices.get(sym)
                    if base is None:
                        _scan_baseline_prices[sym] = mid
                    elif base > 0 and abs(mid - base) / base * 100 >= FAST_SCAN_MOVE_TRIGGER_PCT:
                        _scan_baseline_prices[sym] = mid   # reset so this fires again only on a FURTHER move
                        now_mono = time.monotonic()
                        if now_mono - _last_fast_scan_trigger_ts >= FAST_SCAN_TRIGGER_COOLDOWN_S:
                            _last_fast_scan_trigger_ts = now_mono
                            _fast_scan_trigger.set()
        except Exception as exc:
            log(f"quote handler error: {exc}")

    STORM_WINDOW_S         = 30
    STORM_ATTEMPT_COUNT    = 20
    STORM_ALERT_COOLDOWN_S = 1800   # don't spam Telegram every backoff cycle
    POSITION_POLL_S        = 60     # how often to check for a changed universe
    backoff = 15
    last_storm_alert = 0.0

    global _active_market_stream
    while True:
        if _shutdown_event.is_set():
            log("market data stream loop stopping — shutdown requested")
            return

        tickers = _curated_universe_tickers()
        if not tickers:
            # Nothing to stream — check back soon without opening a
            # connection for zero symbols.
            time.sleep(POSITION_POLL_S)
            continue

        attempt_times: list[float] = []
        connected = threading.Event()
        stopped   = threading.Event()
        stream    = None
        try:
            # Found in the 2026-08-16 review: this hardcoded DataFeed.SIP
            # instead of calling algo._resolve_stock_feed() (the options
            # stream does this correctly for its own OPRA/indicative
            # choice a few hundred lines below). A SIP entitlement lapse
            # would silently degrade this stream with no downgrade alert
            # -- the exact class of gap _resolve_stock_feed() exists to
            # close for every OTHER stock-data call site in the project.
            _feed = DataFeed.SIP if algo._resolve_stock_feed() == "sip" else DataFeed.IEX
            stream = StockDataStream(algo.ALPACA_API_KEY, algo.ALPACA_SECRET_KEY,
                                     feed=_feed)
            stream.subscribe_quotes(on_quote, *tickers)
            with _active_market_stream_lock:
                _active_market_stream = stream

            # Force-bind attempt_times/connected/stream via default args —
            # see stream_loop()'s matching comment for the full zombie-
            # thread closure-sharing bug this prevents.
            _orig_start_ws = stream._start_ws
            async def _watched_start_ws(_orig=_orig_start_ws, _attempts=attempt_times, _connected=connected):
                _attempts.append(time.monotonic())
                await _orig()
                _connected.set()
            stream._start_ws = _watched_start_ws

            def _run_stream(_stream=stream, _stopped=stopped):
                try:
                    _stream.run()
                finally:
                    _stopped.set()

            t = threading.Thread(target=_run_stream, daemon=True)
            t.start()

            storm = False
            last_poll = time.monotonic()
            while t.is_alive() and not stopped.is_set():
                if _shutdown_event.is_set():
                    break
                time.sleep(3)
                now = time.monotonic()
                attempt_times[:] = [a for a in attempt_times if now - a < STORM_WINDOW_S]
                if len(attempt_times) >= STORM_ATTEMPT_COUNT:
                    storm = True
                    break
                if connected.is_set():
                    connected.clear()
                    backoff = 15
                    log(f"market data stream connected — real-time equity quotes on "
                        f"{len(tickers)} symbol(s) (curated watchlist + open positions)")
                if now - last_poll > POSITION_POLL_S:
                    last_poll = now
                    if _curated_universe_tickers() != tickers:
                        log("curated universe changed — reconnecting stream with updated symbols")
                        break

            if _shutdown_event.is_set():
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                log("market data stream stopped cleanly (shutdown)")
                return

            if storm:
                log(f"market data stream reconnect storm detected ({len(attempt_times)} "
                    f"attempts in {STORM_WINDOW_S}s, likely rate-limited) — stopping and "
                    f"backing off {backoff}s")
                if time.monotonic() - last_storm_alert > STORM_ALERT_COOLDOWN_S:
                    algo.send_telegram(
                        "⚠️ Real-time equity price stream hit a reconnect storm — backing "
                        "off. Equity T1/stop checks fall back to REST + the hourly cron "
                        "in the meantime, nothing stops working.")
                    last_storm_alert = time.monotonic()
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            # Thread exited cleanly — either the curated universe changed
            # (see the poll check above) or an ordinary disconnect. Either
            # way, loop back around and reconnect with a freshly-derived
            # symbol list.
            try:
                stream.stop()
            except Exception:
                pass
            stopped.wait(timeout=10)
            time.sleep(1)
        except Exception as exc:
            log(f"market data stream error: {exc} — reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            with _active_market_stream_lock:
                if _active_market_stream is stream:
                    _active_market_stream = None


def option_data_stream_loop() -> None:
    """
    Real-time options quote stream for open calls/puts/spread legs — feeds
    _realtime_option_quotes so guard_loop's run_options_guard() call can
    check stop/trail/T1 against a fresh streamed quote instead of a fresh
    REST snapshot every single 60s check. Off by default
    (ENABLE_REALTIME_OPTIONS_STREAM), added 2026-08-12 as the options
    counterpart to market_data_stream_loop() — same reconnect-storm guard,
    same subscription-changes-trigger-reconnect model, same reasoning
    throughout; see that function's docstring for the full incident
    writeup this pattern guards against. Only bid/ask/mid/sizes stream —
    Greeks stay REST-only (see _monitor_option_position's get_snapshot_fn
    docstring in dman_algo.py), which is fine since Greeks don't gate the
    stop/trail/T1 price checks this exists to speed up.
    """
    try:
        from alpaca.data.live import OptionDataStream
        from alpaca.data.enums import OptionsFeed
    except ImportError:
        log("alpaca-py options data stream unavailable — real-time option quotes disabled")
        return

    async def on_quote(q) -> None:
        try:
            sym = getattr(q, "symbol", None)
            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            # Found in the 2026-08-16 review: this only guarded ask <= 0,
            # unlike market_data_stream_loop()'s equity on_quote() above
            # (requires bid > 0 AND ask > 0). A single transient one-sided
            # tick (no market maker currently quoting a bid on a thin
            # contract -- not "the option is actually worthless") could
            # sit as this cache's authoritative view for up to 30s, and
            # the options guard's stop-check logic sells against bid --
            # reading a real bid of 0 as "at or below stop" and triggering
            # a market sell on a book that was never actually that thin.
            if not sym or bid <= 0 or ask <= 0:
                return
            mid = round((bid + ask) / 2, 2)
            snap = {
                "bid": round(bid, 2), "ask": round(ask, 2), "mid": mid,
                "bid_size": int(getattr(q, "bid_size", 0) or 0),
                "ask_size": int(getattr(q, "ask_size", 0) or 0),
                "spread_pct": round((ask - bid) / mid, 3) if mid > 0 else 1.0,
            }
            with _realtime_option_quotes_lock:
                _realtime_option_quotes[sym] = (snap, time.monotonic())
        except Exception as exc:
            log(f"option quote handler error: {exc}")

    STORM_WINDOW_S         = 30
    STORM_ATTEMPT_COUNT    = 20
    STORM_ALERT_COOLDOWN_S = 1800   # don't spam Telegram every backoff cycle
    POSITION_POLL_S        = 60     # how often to check for a changed leg set
    backoff = 15
    last_storm_alert = 0.0

    global _active_option_stream
    while True:
        if _shutdown_event.is_set():
            log("option data stream loop stopping — shutdown requested")
            return

        occ_symbols = _current_option_occ_symbols()
        if not occ_symbols:
            # Nothing to stream — check back soon without opening a
            # connection for zero symbols.
            time.sleep(POSITION_POLL_S)
            continue

        feed = OptionsFeed.OPRA if algo._resolve_options_feed() == "opra" else OptionsFeed.INDICATIVE

        attempt_times: list[float] = []
        connected = threading.Event()
        stopped   = threading.Event()
        stream    = None
        try:
            stream = OptionDataStream(algo.ALPACA_API_KEY, algo.ALPACA_SECRET_KEY, feed=feed)
            stream.subscribe_quotes(on_quote, *occ_symbols)
            with _active_option_stream_lock:
                _active_option_stream = stream

            # Force-bind attempt_times/connected/stream via default args —
            # see stream_loop()'s matching comment for the full zombie-
            # thread closure-sharing bug this prevents.
            _orig_start_ws = stream._start_ws
            async def _watched_start_ws(_orig=_orig_start_ws, _attempts=attempt_times, _connected=connected):
                _attempts.append(time.monotonic())
                await _orig()
                _connected.set()
            stream._start_ws = _watched_start_ws

            def _run_stream(_stream=stream, _stopped=stopped):
                try:
                    _stream.run()
                finally:
                    _stopped.set()

            t = threading.Thread(target=_run_stream, daemon=True)
            t.start()

            storm = False
            last_poll = time.monotonic()
            while t.is_alive() and not stopped.is_set():
                if _shutdown_event.is_set():
                    break
                time.sleep(3)
                now = time.monotonic()
                attempt_times[:] = [a for a in attempt_times if now - a < STORM_WINDOW_S]
                if len(attempt_times) >= STORM_ATTEMPT_COUNT:
                    storm = True
                    break
                if connected.is_set():
                    connected.clear()
                    backoff = 15
                    log(f"option data stream connected — real-time option quotes on "
                        f"({len(occ_symbols)} symbol(s): {occ_symbols}, feed={feed.value})")
                if now - last_poll > POSITION_POLL_S:
                    last_poll = now
                    if _current_option_occ_symbols() != occ_symbols:
                        log("open options legs changed — reconnecting stream with updated symbols")
                        break

            if _shutdown_event.is_set():
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                log("option data stream stopped cleanly (shutdown)")
                return

            if storm:
                log(f"option data stream reconnect storm detected ({len(attempt_times)} "
                    f"attempts in {STORM_WINDOW_S}s, likely rate-limited) — stopping and "
                    f"backing off {backoff}s")
                if time.monotonic() - last_storm_alert > STORM_ALERT_COOLDOWN_S:
                    algo.send_telegram(
                        "⚠️ Real-time options quote stream hit a reconnect storm — backing "
                        "off. Options stop/trail/T1 checks fall back to REST in the "
                        "meantime, nothing stops working.")
                    last_storm_alert = time.monotonic()
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            # Thread exited cleanly — either the leg set changed (see the
            # poll check above) or an ordinary disconnect. Either way, loop
            # back around and reconnect with a freshly-derived symbol list.
            try:
                stream.stop()
            except Exception:
                pass
            stopped.wait(timeout=10)
            time.sleep(1)
        except Exception as exc:
            log(f"option data stream error: {exc} — reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            with _active_option_stream_lock:
                if _active_option_stream is stream:
                    _active_option_stream = None


def news_data_stream_loop() -> None:
    """
    Real-time news headline stream for the curated universe. Off by
    default (ENABLE_REALTIME_NEWS_STREAM), added 2026-08-13 as the news
    counterpart to market_data_stream_loop() / option_data_stream_loop()
    — same reconnect-storm guard, same subscription-changes-trigger-
    reconnect model; see market_data_stream_loop()'s docstring for the
    full incident writeup this pattern guards against.

    Every relevant headline is logged to the background news log
    (algo._log_news_event) regardless of relevance tier — added
    2026-08-15, direct instruction to have the algo "constantly
    internalize" news across market + extended hours instead of losing
    each headline the moment its alert scrolls past. Telegram alerting
    itself is near-silent by the same instruction ("I don't need much of
    it on my phone"): only macro/Fed-scale and held-position headlines
    actually page the phone now; a routine watchlist headline still logs,
    it just doesn't alert.

    For a non-negative catalyst on a large-cap WATCHLIST ticker not
    already held, this ALSO wakes scan_loop() early via the shared
    _fast_scan_trigger (added 2026-08-14, direct instruction to make sure
    news actually helps get into plays, not just alert about them) — see
    on_news's inline comment for the exact scope/reasoning. This never
    skips a gate: it only makes run_pro_scanner()'s existing
    MTF/sector/RS/macro-safe/confluence/news_boost checks run sooner,
    never changes what they allow.

    Subscribes to _news_subscription_symbols() (_curated_universe_tickers()
    — ~200 symbols: WATCHLIST + smallcap watchlist + open positions — plus
    MACRO_NEWS_SYMBOLS, the Fed/rates/dollar proxies get_market_regime()
    already tracks) rather than Alpaca's "*" firehose — keeps volume tied
    to names the algo can actually act on plus the handful of symbols that
    carry macro-policy news, same scoping choice as the equity/options
    streams. MACRO_NEWS_KEYWORDS is a second, symbol-independent net on
    top: a headline's own text is checked regardless of which subscribed
    symbol it's tagged with, so a Fed/CPI/PPI story tagged alongside an
    unrelated watchlist ticker still surfaces as macro news.
    """
    try:
        from alpaca.data.live import NewsDataStream
    except ImportError:
        log("alpaca-py news data stream unavailable — real-time news alerts disabled")
        return

    def _already_seen(news_id) -> bool:
        with _news_seen_ids_lock:
            if news_id in _news_seen_ids:
                return True
            _news_seen_ids[news_id] = True
            if len(_news_seen_ids) > _NEWS_SEEN_IDS_MAX:
                _news_seen_ids.pop(next(iter(_news_seen_ids)))
            return False

    def _handle_news_sync(n) -> None:
        global _last_fast_scan_trigger_ts
        try:
            news_id  = getattr(n, "id", None)
            headline = (getattr(n, "headline", "") or "").strip()
            symbols  = getattr(n, "symbols", None) or []
            source   = (getattr(n, "source", "") or "").strip()
            url      = (getattr(n, "url", "") or "").strip()
            if news_id is None or not headline or not symbols:
                return
            if _already_seen(news_id):
                return
            is_macro = _is_macro_headline(headline) or any(s in algo.MACRO_NEWS_SYMBOLS for s in symbols)
            try:
                with open(algo.POSITIONS_FILE) as f:
                    held = {p.get("ticker", "") for p in json.load(f)}
            except Exception:
                held = set()
            relevant = [s for s in symbols if s in held or s in algo.WATCHLIST
                        or s in algo.DMAN_SMALLCAP_WATCHLIST or s in algo.MACRO_NEWS_SYMBOLS]
            if not relevant and not is_macro:
                return
            held_hit = [s for s in relevant if s in held]

            # Fast-scan trigger extension (2026-08-14, direct instruction to
            # make sure news actually helps get into plays, not just alert
            # about them): a non-negative catalyst on a large-cap WATCHLIST
            # ticker not already held wakes scan_loop early, same mechanism
            # market_data_stream_loop's price-based trigger already uses
            # (same shared cooldown/market-hours gate). Restricted to
            # WATCHLIST, not DMAN_SMALLCAP_WATCHLIST or the macro proxies —
            # smallcap names already have their own dedicated catalyst gate
            # in detect_low_float_catalyst(), and a macro/index headline
            # isn't a single ticker to newly enter on. This never skips a
            # gate: run_pro_scanner() re-checks MTF/sector/RS/macro-safe/
            # confluence/its own (now sentiment-aware) news_boost exactly as
            # it would on the next scheduled pass — this only changes WHEN
            # that check happens, not what it allows.
            # Added 2026-08-21: a headline matching FAST_SCAN_MACRO_KEYWORDS
            # (Trump/Jensen Huang-style unscheduled statements, not a
            # calendar-known data print) also wakes the scanner early, on
            # top of the existing non-macro watchlist-hit trigger below --
            # this is exactly the "get in ahead of the gap-up" case: the
            # headline itself isn't tied to one ticker, but the sector it
            # moves (semis, drone/defense on a tariff, etc.) IS covered by
            # WATCHLIST, and waiting for the next cron slot gives that
            # advantage away for nothing. Same cooldown/lock, same "never
            # skips a gate" guarantee as every other trigger source here.
            _headline_lower = f" {headline.lower()} "
            _is_fast_scan_macro = any(kw in _headline_lower for kw in algo.FAST_SCAN_MACRO_KEYWORDS)
            if not held_hit and market_hours() and (_is_fast_scan_macro or not is_macro):
                _watchlist_hits = [s for s in relevant if s in algo.WATCHLIST]
                _worth_a_look = _is_fast_scan_macro or (
                    _watchlist_hits and algo._news_sentiment_verdict(_watchlist_hits[0]) != "negative")
                if _worth_a_look:
                    # Found in the 2026-08-16 review: market_data_stream_loop's
                    # own trigger (above) holds _scan_baseline_lock across
                    # this exact check-and-set on _last_fast_scan_trigger_ts;
                    # this one didn't lock at all. Two threads (this one and
                    # the market-data stream's) can both read the stale
                    # cooldown timestamp before either writes, both pass, and
                    # both fire the trigger -- the same class of double-fire
                    # bug FAST_SCAN_TRIGGER_COOLDOWN_S exists to prevent.
                    # Reuses the same lock rather than a new one: they guard
                    # the same shared timestamp and must serialize against
                    # each other, not just within themselves.
                    with _scan_baseline_lock:
                        now_mono = time.monotonic()
                        if now_mono - _last_fast_scan_trigger_ts >= FAST_SCAN_TRIGGER_COOLDOWN_S:
                            _last_fast_scan_trigger_ts = now_mono
                            _fast_scan_trigger.set()

            # Background knowledge-base log (2026-08-15, direct instruction
            # to have the algo "constantly internalize" news rather than
            # only flash a headline past in an alert and lose it) — EVERY
            # relevant headline gets logged here, alerted or not. Tag
            # reflects which relevance bucket matched, for filtering later.
            # Sentiment attached at write-time (not macro — a macro/index
            # headline isn't about one ticker's sentiment) so the log
            # itself is the data source _news_sentiment_breadth() reads
            # later, no separate re-fetch needed at query time.
            _log_tag = "macro" if is_macro else ("held" if held_hit else "watchlist")
            _log_syms = relevant or symbols
            _log_sentiment = None if is_macro else algo._news_sentiment_verdict(_log_syms[0])
            algo._log_news_event(_log_syms, headline, source=source, tag=_log_tag, sentiment=_log_sentiment)

            # Telegram alerting quieted to near-silent (2026-08-15, direct
            # instruction — "I don't need much of it on my phone"): only
            # held-position and macro/Fed-scale news actually pings the
            # phone now. A routine watchlist headline (no held-position or
            # macro relevance) still gets logged above, just doesn't alert
            # — it's exactly the kind of volume that was the complaint.
            if not (is_macro or held_hit):
                return

            tag = "🏛 <b>Macro/Fed news</b> —" if is_macro else "🔴 <b>HELD POSITION</b> —"
            sym_str = ', '.join(relevant[:5]) if relevant else ', '.join(symbols[:5])
            msg = f"{tag} {sym_str}\n{headline}"
            if source:
                msg += f"\n<i>{source}</i>"
            if held_hit:
                msg += f"\n💬 <code>/options {held_hit[0]}</code> to review"
            algo.send_telegram(msg)
        except Exception as exc:
            log(f"news handler error: {exc}")

    # _news_queue/_news_consumer_task (used by on_news() below, nonlocal)
    # are declared per reconnect-loop iteration further down, not here --
    # each iteration gets a fresh event loop via alpaca-py's stream.run(),
    # and an asyncio.Queue/Task from a prior loop can't be awaited on a
    # new one. See on_news()'s docstring.

    async def _news_consumer_loop(queue: "asyncio.Queue") -> None:
        while True:
            n = await queue.get()
            await asyncio.to_thread(_handle_news_sync, n)

    async def on_news(n) -> None:
        """
        Alpaca's NewsDataStream drives this callback from its own asyncio
        event loop -- the SAME loop that sends websocket pings and parses
        incoming frames. _handle_news_sync() does real blocking work
        (a positions-file read, up to two synchronous sentiment-lookup
        HTTP calls, a full JSON news-log rewrite via algo._log_news_event,
        a blocking Telegram POST) that would stall that loop if run
        inline. A cluster of headlines (a Fed print, exactly the scenario
        this stream exists to catch) could block it long enough for
        alpaca-py to drop the connection and force a reconnect right when
        news matters most.

        Just enqueues and returns immediately; a single background
        consumer task drains the queue one headline at a time via
        asyncio.to_thread(), off this event loop. One consumer, not
        asyncio.to_thread() called directly from here, deliberately:
        calling it here would let alpaca-py dispatch several headlines
        onto SEPARATE worker threads concurrently, and
        algo._log_news_event()'s read-modify-write JSON file has no lock
        -- exactly the same in-process lost-update race this project
        already fixed at the cross-process level for every other
        multi-writer state file. Serializing through one queue/consumer
        keeps headline handling in the same one-at-a-time order it always
        ran in, just off the event loop instead of blocking it.
        """
        nonlocal _news_queue, _news_consumer_task
        if _news_queue is None:
            _news_queue = asyncio.Queue()
            _news_consumer_task = asyncio.get_event_loop().create_task(_news_consumer_loop(_news_queue))
        await _news_queue.put(n)

    STORM_WINDOW_S         = 30
    STORM_ATTEMPT_COUNT    = 20
    STORM_ALERT_COOLDOWN_S = 1800   # don't spam Telegram every backoff cycle
    UNIVERSE_POLL_S        = 300    # curated universe rarely changes — check less often than leg-level streams
    backoff = 15
    last_storm_alert = 0.0

    global _active_news_stream
    while True:
        if _shutdown_event.is_set():
            log("news data stream loop stopping — shutdown requested")
            return

        symbols = _news_subscription_symbols()
        if not symbols:
            time.sleep(UNIVERSE_POLL_S)
            continue

        attempt_times: list[float] = []
        connected = threading.Event()
        stopped   = threading.Event()
        stream    = None
        _news_queue = None            # fresh queue/consumer task bound to this
        _news_consumer_task = None    # iteration's event loop -- see on_news()
        try:
            stream = NewsDataStream(algo.ALPACA_API_KEY, algo.ALPACA_SECRET_KEY)
            stream.subscribe_news(on_news, *symbols)
            with _active_news_stream_lock:
                _active_news_stream = stream

            # Force-bind attempt_times/connected/stream via default args —
            # see stream_loop()'s matching comment for the full zombie-
            # thread closure-sharing bug this prevents.
            _orig_start_ws = stream._start_ws
            async def _watched_start_ws(_orig=_orig_start_ws, _attempts=attempt_times, _connected=connected):
                _attempts.append(time.monotonic())
                await _orig()
                _connected.set()
            stream._start_ws = _watched_start_ws

            def _run_stream(_stream=stream, _stopped=stopped):
                try:
                    _stream.run()
                finally:
                    _stopped.set()

            t = threading.Thread(target=_run_stream, daemon=True)
            t.start()

            storm = False
            last_poll = time.monotonic()
            while t.is_alive() and not stopped.is_set():
                if _shutdown_event.is_set():
                    break
                time.sleep(3)
                now = time.monotonic()
                attempt_times[:] = [a for a in attempt_times if now - a < STORM_WINDOW_S]
                if len(attempt_times) >= STORM_ATTEMPT_COUNT:
                    storm = True
                    break
                if connected.is_set():
                    connected.clear()
                    backoff = 15
                    log(f"news data stream connected — real-time headlines on {len(symbols)} symbol(s)")
                if now - last_poll > UNIVERSE_POLL_S:
                    last_poll = now
                    if _news_subscription_symbols() != symbols:
                        log("curated universe changed — reconnecting news stream with updated symbols")
                        break

            if _shutdown_event.is_set():
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                log("news data stream stopped cleanly (shutdown)")
                return

            if storm:
                log(f"news data stream reconnect storm detected ({len(attempt_times)} "
                    f"attempts in {STORM_WINDOW_S}s, likely rate-limited) — stopping and "
                    f"backing off {backoff}s")
                if time.monotonic() - last_storm_alert > STORM_ALERT_COOLDOWN_S:
                    algo.send_telegram(
                        "⚠️ Real-time news stream hit a reconnect storm — backing off. "
                        "Breaking-news alerts are paused; the existing REST-based "
                        "catalyst/sentiment checks in the scan pipeline keep working.")
                    last_storm_alert = time.monotonic()
                try:
                    stream.stop()
                except Exception:
                    pass
                stopped.wait(timeout=10)
                time.sleep(backoff)
                backoff = min(backoff * 2, 300)
                continue

            # Thread exited cleanly — either the curated universe changed
            # (see the poll check above) or an ordinary disconnect. Either
            # way, loop back around and reconnect with a freshly-derived
            # symbol list.
            try:
                stream.stop()
            except Exception:
                pass
            stopped.wait(timeout=10)
            time.sleep(1)
        except Exception as exc:
            log(f"news data stream error: {exc} — reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            with _active_news_stream_lock:
                if _active_news_stream is stream:
                    _active_news_stream = None


def earnings_loop() -> None:
    """
    Once daily, ~75 min before the 4 PM close, scans WATCHLIST for tickers
    reporting earnings today/tomorrow and sends a Telegram approval request
    for each new one — this gate is PERMANENT, not a trial: every earnings
    spread requires an explicit human YES before it submits, no auto-
    promotion to autonomous. A fixed trigger time rather than "as soon as
    market opens" because the entry timing rule is "before today's close"
    either way (AMC-today or BMO-tomorrow), so there's no need to rush it at
    9:30, and running this once daily (not on guard_loop's 60s cadence)
    keeps it cheap.
    """
    log(f"Earnings spread loop started (fires once daily at "
        f"{EARNINGS_LOOP_TRIGGER_HHMM // 100:02d}:{EARNINGS_LOOP_TRIGGER_HHMM % 100:02d} ET, "
        f"approval always required — no autonomous submission)")
    last_run_date = ""
    while True:
        try:
            now = datetime.now(algo.ET)
            hhmm = now.hour * 100 + now.minute
            today_str = now.date().isoformat()
            if (getattr(algo, "ENABLE_EARNINGS_SPREADS", False) and market_hours()
                    and hhmm >= EARNINGS_LOOP_TRIGGER_HHMM
                    and last_run_date != today_str):
                last_run_date = today_str
                algo.run_earnings_spread_scan()
            algo.expire_earnings_spread_offers()
        except Exception as exc:
            log(f"earnings loop error: {exc}")
        time.sleep(60)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    if not algo.ALPACA_API_KEY or not algo.ALPACA_SECRET_KEY:
        print("\n  ❌ APCA_API_KEY_ID / APCA_API_SECRET_KEY not set.")
        print("     Set them once (values from app.alpaca.markets → API Keys):")
        print('       setx APCA_API_KEY_ID     "your-key-id"')
        print('       setx APCA_API_SECRET_KEY "your-secret"')
        print("     …then open a NEW terminal and rerun.\n")
        sys.exit(1)
    if not algo.TELEGRAM_TOKEN or not algo.TELEGRAM_CHAT_ID:
        print("\n  ⚠️  TELEGRAM_TOKEN / TELEGRAM_CHAT_ID not set — commands and")
        print("     alerts disabled. Set them with setx (same values as GitHub secrets).\n")

    # Cloud mode (GitHub Actions): DAEMON_RUN_UNTIL=HHMM ET makes the session
    # exit cleanly so the next scheduled session takes over seamlessly.
    run_until = os.getenv("DAEMON_RUN_UNTIL", "").strip()
    cloud     = bool(run_until)

    if cloud:
        from datetime import date as _date
        if _date.today() in algo._MARKET_HOLIDAYS:
            log("NYSE holiday — cloud daemon session skipped")
            return
        if datetime.now(algo.ET).weekday() >= 5:
            log("Weekend — cloud daemon session skipped")
            return

    log(f"DMan daemon starting — repo {REPO_DIR}"
        + (f"  (cloud session until {run_until} ET)" if cloud else ""))
    git_sync()
    algo._register_telegram_commands()
    # Snapshot today's day-start equity baseline as early as possible.
    # Found in the 2026-08-21 session review: _get_day_start_equity() was
    # ONLY ever called from inside send_account_pnl_telegram() itself,
    # which only runs once, near 4 PM ET (the EOD summary) -- meaning the
    # "day start" baseline and the "current" equity it's compared against
    # were captured in the SAME call, so day_pl = equity - equity == 0
    # every single day regardless of what actually happened. The morning
    # daemon session (the first one to start most days, ~9:23 AM ET) is
    # the natural place to set the real baseline; _get_day_start_equity()
    # already no-ops if today's baseline is already set, so this is safe
    # to call from every session without double-snapshotting the day.
    try:
        _eq_now = algo.get_effective_account()
        if _eq_now > 0:
            algo._get_day_start_equity(_eq_now)
    except Exception as _eq_exc:
        log(f"day-start equity snapshot failed (non-fatal): {_eq_exc}")
    algo.send_telegram("🤖 <b>DMan daemon ONLINE</b>"
                       + (f" [cloud session → {run_until[:2]}:{run_until[2:]} ET]" if cloud else "")
                       + " — real-time entries, exits, fill stream, and phone "
                         "commands active. Send /help for commands.")

    threads = [
        threading.Thread(target=telegram_loop, daemon=True, name="telegram"),
        threading.Thread(target=guard_loop,    daemon=True, name="guard"),
        threading.Thread(target=sync_loop,     daemon=True, name="sync"),
        threading.Thread(target=stream_loop,   daemon=True, name="stream"),
        threading.Thread(target=scan_loop,     daemon=True, name="scan"),
        threading.Thread(target=earnings_loop, daemon=True, name="earnings"),
    ]
    if ENABLE_REALTIME_EQUITY_STREAM:
        threads.append(threading.Thread(target=market_data_stream_loop,
                                        daemon=True, name="market-data-stream"))
    if ENABLE_REALTIME_OPTIONS_STREAM:
        threads.append(threading.Thread(target=option_data_stream_loop,
                                        daemon=True, name="option-data-stream"))
    if ENABLE_REALTIME_NEWS_STREAM:
        threads.append(threading.Thread(target=news_data_stream_loop,
                                        daemon=True, name="news-data-stream"))
    for t in threads:
        t.start()

    hb = 0
    try:
        while True:
            time.sleep(60)
            if cloud:
                now = datetime.now(algo.ET)
                if now.hour * 100 + now.minute >= int(run_until):
                    log(f"Reached {run_until} ET — clean session end "
                        "(next scheduled session takes over)")
                    _shutdown_event.set()
                    _close_active_stream()
                    time.sleep(2)   # let the websocket close handshake actually go out
                    git_sync()
                    return
            hb += 1
            if hb % 60 == 0:
                log("heartbeat — daemon alive"
                    + ("" if all(t.is_alive() for t in threads) else " ⚠️ A THREAD DIED"))
    except KeyboardInterrupt:
        log("daemon stopped by user")
        _shutdown_event.set()
        _close_active_stream()
        time.sleep(2)
        algo.send_telegram("🔌 <b>DMan daemon OFFLINE</b> — real-time guard stopped. "
                           "Cron scans still run hourly as backup.")


if __name__ == "__main__":
    main()
