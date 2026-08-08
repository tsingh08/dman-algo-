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

GUARD_EVERY_S   = 60        # options stop/target enforcement cadence
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


def _restore_corrupted_json(paths: list[str]) -> list[str]:
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
    """
    ok: list[str] = []
    for p in paths:
        if not p.endswith(".json"):
            ok.append(p)
            continue
        try:
            with open(p) as f:
                json.load(f)
            ok.append(p)
        except Exception as exc:
            log(f"  ⚠️  {p} failed JSON validation ({exc}) — restoring last "
                f"good commit, skipping this cycle's update to that file")
            _git("checkout", "--", p)
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
                _restore_corrupted_json(_existing(STATE_FILES))
                _git("stash", "drop")

        # Semantic merge for dman_positions.json: this daemon (60s cadence)
        # and the hourly cron scanner both independently raise stops to
        # breakeven / reduce shares on the same open positions from separate
        # concurrency groups — git's line-based stash-pop/rebase merge can
        # silently keep the LESS-protective side. algo.sync_positions_with_remote()
        # re-reconciles against origin/main using a rule that can't regress
        # protection, regardless of whether the stash pop above was clean.
        #
        # The other four are the SAME class of bug, confirmed in production
        # on 2026-07-27: dman_scan_log.json went a full trading day with
        # zero new entries despite 8+ genuinely successful scans, because
        # it's rewritten by both the daemon (every 10 min) and the cron
        # scanner (hourly) so often that a whole-file conflict resolution
        # was silently discarding whichever side's entry lost the race.
        for _sync_fn in (algo.sync_positions_with_remote, algo.sync_scan_log_with_remote,
                        algo.sync_win_rate_with_remote, algo.sync_live_signals_with_remote,
                        algo.sync_alpaca_sync_state_with_remote):
            try:
                _sync_fn()
            except Exception as exc:
                log(f"{_sync_fn.__name__} error (non-fatal): {exc}")

        # Stage and push any local state changes (re-check existence — the
        # stash pop, pull, or merge above may have created/removed files).
        # _restore_corrupted_json is a second, unconditional safety net here
        # — catches corruption from any source, not just a flagged stash
        # conflict (e.g. a process killed mid-write leaving a truncated file).
        _present = _existing(STATE_FILES)
        _present = _restore_corrupted_json(_present)
        if _present:
            _git("add", "--", *_present)
        staged = _git("diff", "--staged", "--quiet")
        if staged.returncode != 0:          # something staged
            _git("-c", "user.email=github-actions[bot]@users.noreply.github.com",
                 "-c", "user.name=github-actions[bot]",
                 "commit", "-m", "chore: daemon state sync [skip ci]")
            push = _git("push", "origin", "HEAD:main")
            if push.returncode != 0:
                pull2 = _git("pull", "--rebase", "origin", "main")
                if pull2.returncode != 0:
                    _git("rebase", "--abort")
                _git("push", "origin", "HEAD:main")
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
    """Options exit enforcement + periodic fill sync during market hours."""
    log("Options guard loop started (60s cadence during market hours)")
    last_sync = 0.0
    was_open  = False
    while True:
        try:
            if market_hours():
                if not was_open:
                    log("Market open — guard active")
                    algo.send_telegram("🛡 <b>Daemon active</b> — real-time options "
                                       "guard + fill stream running for today's session.")
                    was_open = True
                alerts = algo.run_options_guard(verbose=False)
                for a in alerts:
                    log(a.split("\n")[0].replace("<b>", "").replace("</b>", ""))
                if time.time() - last_sync > SYNC_EVERY_S:
                    git_sync()
                    try:
                        n = algo.sync_alpaca_fills(algo.WinRateTracker())
                        if n:
                            log(f"synced {n} closed trade(s)")
                    except Exception as exc:
                        log(f"fill sync error: {exc}")
                    last_sync = time.time()
            else:
                if was_open:
                    log("Market closed — guard idle")
                    git_sync()   # final push of the day's state
                    was_open = False
            time.sleep(GUARD_EVERY_S)
        except Exception as exc:
            log(f"guard loop error: {exc}")
            time.sleep(GUARD_EVERY_S)


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
    """
    log(f"Signal scan loop started (curated universe, every {SCAN_INTERVAL_S}s "
        f"during market hours, {SCAN_RETRY_S}s retry after a failed scan)")
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
                # Self-healing: on a failed scan, come back soon instead of
                # waiting the full interval — every minute of downed
                # coverage is a real missed setup. On success, the normal
                # cadence avoids hammering the API for no reason.
                if scan_ok:
                    next_scan_due = time.time() + SCAN_INTERVAL_S
                else:
                    log(f"retrying scan in {SCAN_RETRY_S}s (not waiting the full {SCAN_INTERVAL_S}s interval)")
                    next_scan_due = time.time() + SCAN_RETRY_S
            time.sleep(30)
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

            _orig_start_ws = stream._start_ws
            async def _watched_start_ws(_orig=_orig_start_ws):
                attempt_times.append(time.monotonic())
                await _orig()
                connected.set()
            stream._start_ws = _watched_start_ws

            def _run_stream():
                try:
                    stream.run()
                finally:
                    stopped.set()

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
            time.sleep(backoff)
        except Exception as exc:
            log(f"stream error: {exc} — reconnecting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
        finally:
            with _active_stream_lock:
                if _active_stream is stream:
                    _active_stream = None


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
    algo.send_telegram("🤖 <b>DMan daemon ONLINE</b>"
                       + (f" [cloud session → {run_until[:2]}:{run_until[2:]} ET]" if cloud else "")
                       + " — real-time entries, exits, fill stream, and phone "
                         "commands active. Send /help for commands.")

    threads = [
        threading.Thread(target=telegram_loop, daemon=True, name="telegram"),
        threading.Thread(target=guard_loop,    daemon=True, name="guard"),
        threading.Thread(target=stream_loop,   daemon=True, name="stream"),
        threading.Thread(target=scan_loop,     daemon=True, name="scan"),
        threading.Thread(target=earnings_loop, daemon=True, name="earnings"),
    ]
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
