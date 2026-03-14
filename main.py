"""
main.py
-------
Entry point for the Polymarket trading bot.

Wires together all modules and runs the main trading loop with:
  - Graceful shutdown on SIGINT / SIGTERM
  - Structured JSON logging to file + human-readable console output
  - Configurable scan interval
  - Status heartbeat printed every N cycles
  - All config driven by environment variables (no code changes needed)

Environment variables (.env):
    # Polymarket
    POLYMARKET_PRIVATE_KEY       Your wallet private key
    POLYMARKET_PROXY_ADDRESS     Your Polymarket proxy / funder address
    POLYMARKET_SIGNATURE_TYPE    1 = email/Magic, 2 = MetaMask (default: 1)

    # Anthropic
    ANTHROPIC_API_KEY            Your Anthropic API key

    # Risk
    BANKROLL                     Total USDC allocated to the bot   (default: 1000)
    KELLY_FRACTION               Fractional Kelly multiplier        (default: 0.25)
    MAX_BET_USDC                 Hard cap per trade in USDC         (default: 100)
    MAX_DAILY_LOSS_PCT           Kill-switch drawdown threshold      (default: 0.05)
    MAX_OPEN_POSITIONS           Max simultaneous positions          (default: 10)
    MIN_CONFIDENCE               low | medium | high                 (default: medium)

    # Strategy
    DRY_RUN                      true / false — safe default is true (default: true)
    SCAN_INTERVAL_SECONDS        Seconds between scan cycles         (default: 300)
    SCAN_LIMIT                   Max markets fetched per cycle       (default: 50)
    VOLUME_MIN                   Min 24h volume filter in USDC       (default: 5000)
    LIQUIDITY_MIN                Min liquidity filter in USDC        (default: 2000)
    TAG_ID                       Polymarket tag ID to focus on       (default: none)
    USE_LIMIT_ORDERS             true / false                        (default: true)
    USE_WEB_SEARCH               true / false — Claude news fetch    (default: true)
    HEARTBEAT_EVERY_N_CYCLES     Print status every N cycles         (default: 5)
    LOG_FILE                     Path for JSON log file              (default: bot.log)
"""

import json
import logging
import logging.handlers
import signal
import sys
import time
from datetime import datetime, timezone, timedelta

from dotenv import load_dotenv

from env_helpers import env_int, env_str, env_float, env_bool
from polymarket_client import PolymarketClient
from claude_analyst import ClaudeAnalyst
from risk_manager import RiskManager, RiskConfig
from strategy import Strategy, StrategyConfig

load_dotenv()


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging(log_file: str, log_level: str = "INFO"):
    """
    Dual-output logging:
      - Console: human-readable with colour-friendly format
      - File:    one JSON object per line (machine-parseable for analysis)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(console)

    # JSON file handler (rotates at 10 MB, keeps 5 backups)
    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            return json.dumps({
                "ts":      datetime.now(timezone.utc).isoformat(),
                "level":   record.levelname,
                "logger":  record.name,
                "message": record.getMessage(),
            })

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)

# ── Config builder ─────────────────────────────────────────────────────────────

def _build_risk_config() -> RiskConfig:
    return RiskConfig(
        bankroll               = env_float("BANKROLL", 1_000.0),
        kelly_fraction         = env_float("KELLY_FRACTION", 0.25),
        max_bet_usdc           = env_float("MAX_BET_USDC", 100.0),
        max_daily_loss_pct     = env_float("MAX_DAILY_LOSS_PCT", 0.05),
        max_open_positions     = env_int  ("MAX_OPEN_POSITIONS", 10),
        min_confidence         = env_str  ("MIN_CONFIDENCE", "medium"),
    )

def _build_strategy_config() -> StrategyConfig:
    tag_id_raw    = env_str("TAG_ID", "")
    horizon_hours = env_int("HORIZON_HOURS", 48)
    hours_to_resolution = env_int("MIN_HOURS_TO_RESOLUTION", 2)
    end_date_max  = (
            datetime.now(timezone.utc) + timedelta(hours=horizon_hours)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_date_min  = (
            datetime.now(timezone.utc) + timedelta(hours=hours_to_resolution)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    return StrategyConfig(
        dry_run              = env_bool ("DRY_RUN", True),
        scan_limit           = env_int  ("SCAN_LIMIT", 50),
        volume_min           = env_float("VOLUME_MIN", 5_000.0),
        liquidity_min        = env_float("LIQUIDITY_MIN", 2_000.0),
        end_date_max         = end_date_max,
        end_date_min         = end_date_min,
        tag_id               = int(tag_id_raw) if tag_id_raw.isdigit() else None,
        use_limit_orders     = env_bool ("USE_LIMIT_ORDERS", True),
        delay_between_trades = env_float("TRADE_DELAY_SEC", 2.0),
    )


# ── Shutdown handler ───────────────────────────────────────────────────────────

class _GracefulShutdown:
    """Catches SIGINT / SIGTERM and sets a flag the main loop checks."""
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT,  self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logging.getLogger("main").warning("Signal %s received — shutting down after current cycle…", sig_name)
        self.requested = True


# ── Startup banner ─────────────────────────────────────────────────────────────

def _print_banner(risk_cfg: RiskConfig, strat_cfg: StrategyConfig, scan_interval: int):
    mode = "🔴  LIVE TRADING" if not strat_cfg.dry_run else "🟡  DRY-RUN (no orders sent)"
    print("\n" + "═" * 65)
    print("  POLYMARKET TRADING BOT")
    print("═" * 65)
    print(f"  Mode:            {mode}")
    print(f"  Bankroll:        ${risk_cfg.bankroll:,.2f} USDC")
    print(f"  Kelly fraction:  {risk_cfg.kelly_fraction:.0%}")
    print(f"  Max bet:         ${risk_cfg.max_bet_usdc:.2f} USDC")
    print(f"  Max positions:   {risk_cfg.max_open_positions}")
    print(f"  Kill-switch:     >{risk_cfg.max_daily_loss_pct:.0%} daily loss")
    print(f"  Min confidence:  {risk_cfg.min_confidence}")
    print(f"  Scan interval:   {scan_interval}s")
    print(f"  Horizon:         {env_int('HORIZON_HOURS', 48)}h (markets resolving within)")
    print(f"  Volume filter:   >${risk_cfg.min_confidence}")
    print(f"  Volume filter:   >${strat_cfg.volume_min:,.0f}")
    print(f"  Liquidity filter:>${strat_cfg.liquidity_min:,.0f}")
    print(f"  Tag filter:      {strat_cfg.tag_id or 'none'}")
    print("═" * 65 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log_file  = env_str("LOG_FILE", "bot.log")
    log_level = env_str("LOG_LEVEL", "INFO")
    scan_interval  = env_int("SCAN_INTERVAL_SECONDS", 300)
    heartbeat_every = env_int("HEARTBEAT_EVERY_N_CYCLES", 5)
    use_web_search  = env_bool("USE_WEB_SEARCH", True)

    _setup_logging(log_file, log_level)
    logger = logging.getLogger("main")

    risk_cfg  = _build_risk_config()
    strat_cfg = _build_strategy_config()

    _print_banner(risk_cfg, strat_cfg, scan_interval)

    if not strat_cfg.dry_run:
        logger.warning("=" * 55)
        logger.warning("  LIVE MODE — real orders will be sent to Polymarket")
        logger.warning("=" * 55)

    # ── Initialise components ──────────────────────────────────────────────────
    logger.info("Initialising components…")

    try:
        client  = PolymarketClient(read_only=strat_cfg.dry_run)
        analyst = ClaudeAnalyst(use_web_search=use_web_search)
        risk    = RiskManager(risk_cfg)
        strategy = Strategy(client, analyst, risk, strat_cfg)
    except EnvironmentError as exc:
        logger.critical("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("Failed to initialise: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("All components ready. Starting main loop.")

    # ── Main loop ──────────────────────────────────────────────────────────────
    shutdown = _GracefulShutdown()
    cycle    = 0

    while not shutdown.requested:
        cycle += 1
        cycle_start = time.monotonic()
        logger.info("━━━ Cycle %d ━━━", cycle)

        try:
            records = strategy.run_once()

            approved = [r for r in records if r.approved]
            rejected = [r for r in records if not r.approved]
            logger.info(
                "Cycle %d done: %d approved, %d rejected",
                cycle, len(approved), len(rejected),
            )

        except Exception as exc:
            logger.error("Unhandled error in cycle %d: %s", cycle, exc, exc_info=True)

        # Heartbeat status print
        if cycle % heartbeat_every == 0:
            status = risk.status_report()
            logger.info(
                "── Heartbeat (cycle %d) ──  exposure=$%.2f (%.0f%%)  "
                "open=%d  daily_pnl=$%.2f  halted=%s",
                cycle,
                status["total_exposure"],
                status["exposure_pct"] * 100,
                status["open_markets"],
                status["daily_pnl"],
                status["halted"],
                )
            strategy.print_trade_log(last_n=10)

        # Kill-switch check
        if risk.is_halted:
            logger.critical(
                "Kill-switch is active — bot halted. "
                "Fix the issue and call risk.resume() to restart, "
                "or restart the process after reviewing the logs."
            )
            break

        # Sleep until next cycle, checking for shutdown every second
        elapsed  = time.monotonic() - cycle_start
        sleep_remaining = max(0, scan_interval - elapsed)
        logger.info("Next cycle in %.0fs…", sleep_remaining)

        for _ in range(int(sleep_remaining)):
            if shutdown.requested or risk.is_halted:
                break
            time.sleep(1)

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("Shutting down…")

    # Cancel any open resting limit orders before exiting
    if not strat_cfg.dry_run:
        try:
            logger.info("Cancelling all open orders…")
            client.cancel_all_orders()
        except Exception as exc:
            logger.error("Failed to cancel open orders on shutdown: %s", exc)

    # Final status
    status = risk.status_report()
    logger.info(
        "Final status | cycles=%d | daily_pnl=$%.2f | "
        "exposure=$%.2f | open_positions=%d",
        cycle,
        status["daily_pnl"],
        status["total_exposure"],
        status["open_markets"],
    )

    strategy.print_trade_log()
    logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    main()