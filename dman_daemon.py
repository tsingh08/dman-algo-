"""
╔══════════════════════════════════════════════════════════════════════════╗
║  DMan ALWAYS-ON DAEMON — the real-time layer cron can't provide          ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Division of labor:                                                      ║
║    GitHub Actions (cron) → ENTRIES: scheduled scans, premarket, EOD P&L  ║
║    This daemon (24/5)    → EXITS + CONTROL:                              ║
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

import os
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

GUARD_EVERY_S = 60          # options stop/target enforcement cadence
SYNC_EVERY_S  = 300         # fill sync + git state sync cadence
STATE_FILES = [
    "dman_positions.json", "dman_last_alerts.json", "dman_live_signals.json",
    "dman_live_outcomes.csv", "dman_alpaca_sync.json", "dman_win_rate.json",
    "dman_daily_pnl.json", "dman_monthly_pnl.json", "dman_halt.json",
    "dman_telegram_state.json", "dman_smallcap_watchlist.json",
]


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
    """
    try:
        _present = _existing(STATE_FILES)
        if _present:
            _git("stash", "push", "--include-untracked", "--", *_present)
        pull = _git("pull", "--rebase", "origin", "main")
        if pull.returncode != 0:
            _git("rebase", "--abort")
        if _present:
            _git("stash", "pop")
        # Stage and push any local state changes (re-check existence — the
        # stash pop or pull may have created/removed files)
        _present = _existing(STATE_FILES)
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


def stream_loop() -> None:
    """Alpaca TradingStream — instant Telegram alert on every fill."""
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

    while True:
        try:
            stream = TradingStream(algo.ALPACA_API_KEY, algo.ALPACA_SECRET_KEY,
                                   paper=algo.ALPACA_PAPER)
            stream.subscribe_trade_updates(on_update)
            log("TradingStream connected — instant fill alerts on")
            stream.run()          # blocks until disconnect
        except Exception as exc:
            log(f"stream error: {exc} — reconnecting in 15s")
        time.sleep(15)


def main() -> None:
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
                       + " — real-time exits, fill stream, and phone commands "
                         "active. Send /help for commands.")

    threads = [
        threading.Thread(target=telegram_loop, daemon=True, name="telegram"),
        threading.Thread(target=guard_loop,    daemon=True, name="guard"),
        threading.Thread(target=stream_loop,   daemon=True, name="stream"),
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
                    git_sync()
                    return
            hb += 1
            if hb % 60 == 0:
                log("heartbeat — daemon alive"
                    + ("" if all(t.is_alive() for t in threads) else " ⚠️ A THREAD DIED"))
    except KeyboardInterrupt:
        log("daemon stopped by user")
        algo.send_telegram("🔌 <b>DMan daemon OFFLINE</b> — real-time guard stopped. "
                           "Cron scans still run hourly as backup.")


if __name__ == "__main__":
    main()
