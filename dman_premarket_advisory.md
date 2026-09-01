# DMan Pre-Market Advisory — 2026-09-01

## Live Performance
- **Total live trades:** 25
- **Overall live WR:** 60.0% (15W / 10L) | backtest baseline: 60.5% ✓ on track
- **Gap & Hold:** 75.0% WR (9W / 3L, 12 trades) | baseline 63.1% → **+11.9 pts above baseline** ✓
- **Bear Gap Hold (SHORT):** 50.0% WR (1W / 1L, 2 trades) | insufficient sample (<8 trades)
- **MACD Cross:** DISABLED (`ENABLE_MACD_CROSS = False`) — not tracked
- **Vol Breakdown:** DISABLED (`ENABLE_VOL_BREAKOUT = False`) — not tracked
- **Low Float Catalyst:** 44.4% WR (4W / 5L, 9 trades) | no backtest baseline (forward-test only); approaching 40% alert floor
- **Morning Runner:** 100% WR (1 trade) | insufficient sample
- **Day 2 Continuation:** 0% WR (1 trade) | insufficient sample

---

## Filter Status
- **Seasonal weak months: ACTIVE** — September (month 9) is in `SEASONAL_WEAK_MONTHS = {1, 7, 8, 9, 12}` (line 501). Algo should auto-throttle new signal acceptance today.
- **Global min score:** 75 (`MIN_CONFLUENCE`, line 72)
- **Per-setup overrides:** Low Float Catalyst → 90, Morning Runner → 85, Vol Breakdown → 85, Gap & Short → 82, MACD Bear → 82, OS Bounce → 82 (lines 73–93)
- **Volatile tickers:** Extra 88-pt floor for RIOT, GME, COIN, MSTR, Chinese ADRs, etc. (line 116)
- **Monthly loss limit:** 4% of account (`MONTHLY_LOSS_LIMIT = 0.04`, line 448)
- **Consecutive loss guard:** Halts after 3 consecutive losses (header / line 21)

---

## Pending Signals
| Ticker | Setup | Bias | Entry | Stop | T1 | T2 | Score | Signal Date |
|--------|-------|------|-------|------|-----|-----|-------|-------------|
| HWM | Bear Gap Hold | SHORT | 245.80 | 253.30 | 227.06 | 215.82 | 100 | 2026-08-31 |

> **Note:** HWM signal is from 8/31 (yesterday). Only 2 live Bear Gap Hold trades on record (insufficient sample). Verify whether price/setup is still valid before acting.

---

## Suggestions (REVIEW ONLY — no auto-apply)

1. **[COMPLIANCE FLAG — Low Float Catalyst score floor]** All ARTL trades reviewed against the configured 90-pt minimum for this setup:
   - ARTL 2026-08-12: score **65** → WIN (BE) — below 90 floor
   - ARTL 2026-08-18: score **55** → LOSS — below 90 floor
   - ARTL 2026-08-20: score **47** → LOSS — significantly below 90 floor
   - ARTL 2026-08-17 (exit 8/27): score **88** → LOSS — below 90 floor
   - 0 of 4 ARTL trades met the configured floor. Consider adding ARTL to a per-ticker blacklist until the compliance gap is investigated.

2. **[Monthly loss limit verification]** `dman_monthly_pnl.json` contains 3 entries from late August totalling approx. **-19.7 pct** (individual entries: -14.476%, -5.404%, +0.164%). If these represent account-level daily P&L, they far exceed the 4% monthly halt threshold. Verify whether the monthly halt triggered for August and confirm it has correctly **reset for September** (new calendar month).

3. **[HWM pending signal — caution warranted]** Bear Gap Hold has only 2 live trades and 50% WR. September is a seasonal weak month. Recommend waiting for the seasonal filter gate confirmation before accepting any new signals today, including HWM.

4. **[Position sizing in seasonal weak month]** September triggers the seasonal filter. Consider manually enforcing reduced position sizing (e.g., 50% of normal) for any signals that do pass the gate this month, consistent with the algo's weak-month posture.

---

## Risk Flags

1. **SEASONAL FILTER ACTIVE**: September is a designated weak month. The algo should suppress or reduce new entries. Confirm `_seasonal_active` logic is running correctly today (see lines 14784, 15056).

2. **Low Float Catalyst near alert floor**: 44.4% WR over 9 live trades approaches the 40% performance alert floor (`SETUP_PERFORMANCE_ALERT_WR_FLOOR`, line 100). ARTL was the primary driver of losses in this setup (4 trades, 0 clean wins). Score compliance was broken on all 4 ARTL trades.

3. **Recent 4-consecutive-loss streak (resolved)**: Losses streaked from 8/21 through 8/27 (ARTL ×2, NDSN, ARTL). Streak broken by 3 wins on 8/28 (PURR +39.68%, BITX BE, BITU BE). The Consecutive Loss Guard (3-loss halt) should have fired mid-streak — confirm it triggered and reset properly.

4. **August monthly PnL anomaly**: The `dman_monthly_pnl.json` entries (−14.476% on 8/28, −5.404% and +0.164% on 8/29) appear to represent account-level daily changes, not individual trade P&L, and were logged while the PURR trade (a +39.68% trade gain) also closed on 8/28. This inconsistency warrants review — likely reflects sizing/concentration effects from options positions (`ENABLE_OPTIONS_TRADING = True`, `OPTIONS_MAX_POSITION_PCT = 0.15`).

5. **No flags on Gap & Hold**: This setup is the algo's core edge and is performing well (+11.9 pts above backtest baseline). PURR (39.68% gain to T2) is a strong real-world data point.
