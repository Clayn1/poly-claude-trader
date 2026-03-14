"""
strategy.py
-----------
Wires together the PolymarketClient, ClaudeAnalyst, and RiskManager into a
complete trading strategy loop.

Responsibilities:
  - Scan markets based on configurable filters
  - Request analysis from Claude
  - Gate every trade through the RiskManager
  - Execute approved trades via the PolymarketClient
  - Log all decisions (including no-trades) with full reasoning

This module is intentionally stateless — all state lives in RiskManager.
Call Strategy.run_once() from your main loop.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from polymarket_client import PolymarketClient, MarketInfo
from claude_analyst import ClaudeAnalyst, AnalysisResult
from risk_manager import RiskManager, RiskConfig, TradeDecision

logger = logging.getLogger("strategy")


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    Market scanning and execution parameters.

    Attributes:
        scan_limit:          Max markets to fetch per scan cycle.
        volume_min:          Server-side volume filter passed to Gamma API.
        liquidity_min:       Server-side liquidity filter passed to Gamma API.
        end_date_max:        Only consider markets resolving before this ISO date.
                             e.g. "2025-06-01T00:00:00Z". None = no limit.
        tag_id:              Optional Polymarket tag to narrow the scan category.
        order_by:            Field to sort markets by ("volume" | "liquidity").
        use_limit_orders:    True = place limit orders at mid; False = market orders.
        limit_order_offset:  How far inside the spread to place the limit (0.01 = 1¢).
        delay_between_trades: Seconds to wait between consecutive order placements.
        dry_run:             If True, log trades but never send orders to the exchange.
    """
    scan_limit: int              = 50
    volume_min: float            = 5_000.0
    liquidity_min: float         = 2_000.0
    end_date_max: Optional[str]  = None
    end_date_min: Optional[str]  = None
    tag_id: Optional[int]        = None
    order_by: str                = "volume"
    use_limit_orders: bool       = True
    limit_order_offset: float    = 0.01
    delay_between_trades: float  = 2.0
    dry_run: bool                = True       # SAFE DEFAULT — flip to False to go live


@dataclass
class TradeRecord:
    """Audit log entry for a single trade decision."""
    timestamp: str
    condition_id: str
    question: str
    action: str
    approved: bool
    size_usdc: float
    entry_price: float
    edge: float
    confidence: str
    rejection_reason: str
    dry_run: bool
    order_response: dict = field(default_factory=dict)

    def log_line(self) -> str:
        status = "DRY-RUN" if self.dry_run else ("FILLED" if self.approved else "REJECTED")
        return (
                f"[{self.timestamp}] {status:<10} | {self.action:<8} | "
                f"${self.size_usdc:>7.2f} @ {self.entry_price:.4f} | "
                f"edge={self.edge:+.1%} conf={self.confidence:<6} | "
                f"{self.question[:55]}"
                + (f" | ✗ {self.rejection_reason}" if not self.approved else "")
        )


# ── Strategy ───────────────────────────────────────────────────────────────────

class Strategy:
    """
    End-to-end trading strategy.

    Usage:
        strategy = Strategy(
            client  = PolymarketClient(),
            analyst = ClaudeAnalyst(),
            risk    = RiskManager(RiskConfig(bankroll=1000)),
            config  = StrategyConfig(dry_run=False),
        )

        # In your main loop:
        while True:
            records = strategy.run_once()
            time.sleep(300)
    """

    def __init__(
            self,
            client:  PolymarketClient,
            analyst: ClaudeAnalyst,
            risk:    RiskManager,
            config:  StrategyConfig = None,
    ):
        self.client  = client
        self.analyst = analyst
        self.risk    = risk
        self.config  = config or StrategyConfig()
        self.trade_log: list[TradeRecord] = []

        mode = "DRY-RUN" if self.config.dry_run else "LIVE"
        logger.info("Strategy initialised in %s mode", mode)

    # ── Market scanning ───────────────────────────────────────────────────────

    def _scan_markets(self) -> list[MarketInfo]:
        """Fetch candidate markets from Polymarket using current strategy filters."""
        logger.info("Scanning markets…")
        markets = self.client.get_active_markets(
            limit         = self.config.scan_limit,
            order         = self.config.order_by,
            ascending     = False,
            volume_min    = self.config.volume_min,
            liquidity_min = self.config.liquidity_min,
            end_date_max  = self.config.end_date_max,
            end_date_min  = self.config.end_date_min,
            tag_id        = self.config.tag_id,
            closed        = False,
        )
        logger.info("Found %d candidate markets", len(markets))
        return markets

    # ── Entry price calculation ───────────────────────────────────────────────

    def _entry_price(self, market: MarketInfo, action: str) -> float:
        """
        Determine the limit order price for a trade.

        For limit orders we place slightly inside the spread to improve
        fill probability while still getting a better price than market.
        For market orders we use the current best ask.
        """
        if action == "BUY_YES":
            mid = market.yes_mid
            ask = market.yes_price
        else:
            mid = market.no_mid
            ask = market.no_price

        if self.config.use_limit_orders:
            # Place limit at mid + small offset toward the ask
            price = mid + self.config.limit_order_offset
            price = min(price, ask)          # never cross the spread
        else:
            price = ask

        return round(max(0.01, min(0.99, price)), 4)

    # ── Position resolution ──────────────────────────────────────────────────

    def _check_resolved_positions(self, positions: list) -> None:
        """
        Scan open positions and close any whose markets have resolved.
        Calculates P&L and notifies the RiskManager so exposure is freed up.
        """
        for pos in positions:
            try:
                market = self.client.get_market_by_id(pos.condition_id)
                if not market or not market.closed:
                    continue

                # Binary settlement: winner gets $1.00 per share, loser $0.00
                if pos.side == "YES":
                    # We held YES — won if market resolved YES (price → 1.0)
                    resolved_yes = market.yes_mid >= 0.99
                    pnl = (1.0 - pos.avg_price) * pos.size if resolved_yes else -pos.avg_price * pos.size
                else:
                    # We held NO — won if market resolved NO (price → 1.0)
                    resolved_no = market.no_mid >= 0.99
                    pnl = (1.0 - pos.avg_price) * pos.size if resolved_no else -pos.avg_price * pos.size

                self.risk.record_close(pos.condition_id, pnl)
                logger.info(
                    "Position resolved: %s %s | avg_entry=%.4f | pnl=$%.2f",
                    pos.condition_id[:12], pos.side, pos.avg_price, pnl,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to check resolution for %s: %s",
                    pos.condition_id[:12], exc,
                )

    # ── Order execution ───────────────────────────────────────────────────────

    def _execute_trade(
            self,
            market: MarketInfo,
            analysis: AnalysisResult,
            decision: TradeDecision,
    ) -> TradeRecord:
        """
        Place the order (or log it in dry-run mode) and return a TradeRecord.
        """
        action      = decision.action
        token_id    = market.yes_token_id if action == "BUY_YES" else market.no_token_id
        entry_price = self._entry_price(market, action)
        now         = datetime.now(timezone.utc).isoformat()
        order_resp  = {}

        # Hard guard — belt-and-suspenders on top of the config check.
        # If the client was initialised read-only, placing an order would
        # raise PermissionError anyway, but we abort here first so the
        # TradeRecord is clearly marked as a dry-run.
        if self.config.dry_run or self.client.read_only:
            if not self.config.dry_run:
                logger.warning(
                    "Client is read-only but dry_run=False — treating as dry-run. "
                    "Set DRY_RUN=false AND provide credentials to trade live."
                )
            logger.info(
                "[DRY-RUN] Would %s $%.2f @ %.4f — %s",
                action, decision.size_usdc, entry_price, market.question[:60],
            )
        else:
            try:
                if self.config.use_limit_orders:
                    order_resp = self.client.place_limit_order(
                        token_id   = token_id,
                        side       = "BUY",
                        price      = entry_price,
                        size       = decision.size_usdc,
                    )
                else:
                    order_resp = self.client.place_market_order(
                        token_id = token_id,
                        side     = "BUY",
                        amount   = decision.size_usdc,
                    )

                # Record fill with risk manager
                self.risk.record_fill(
                    condition_id = market.condition_id,
                    size_usdc    = decision.size_usdc,
                    entry_price  = entry_price,
                )

                logger.info(
                    "Order placed: %s $%.2f @ %.4f → %s",
                    action, decision.size_usdc, entry_price, order_resp,
                )

            except Exception as exc:
                logger.error("Order placement failed for %s: %s", market.condition_id[:12], exc)
                order_resp = {"error": str(exc)}

        record = TradeRecord(
            timestamp        = now,
            condition_id     = market.condition_id,
            question         = market.question,
            action           = action,
            approved         = True,
            size_usdc        = decision.size_usdc,
            entry_price      = entry_price,
            edge             = analysis.edge,
            confidence       = analysis.confidence,
            rejection_reason = "",
            dry_run          = self.config.dry_run,
            order_response   = order_resp,
        )
        return record

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_once(self) -> list[TradeRecord]:
        """
        Execute one full scan-analyse-trade cycle.

        Flow:
          1. Check kill-switch
          2. Fetch current positions and USDC balance
          3. Scan for candidate markets
          4. For each market: analyse → risk-check → execute
          5. Return all TradeRecords from this cycle (approved and rejected)

        Returns a list of TradeRecord for this cycle.
        """
        cycle_records: list[TradeRecord] = []
        now = datetime.now(timezone.utc).isoformat()

        # 1. Kill-switch check
        if self.risk.is_halted:
            logger.critical("Bot is halted — skipping cycle. Call risk.resume() to restart.")
            return []

        # 2. Account state
        # In dry-run mode skip authenticated CLOB calls entirely —
        # the read-only client would raise PermissionError, aborting the
        # cycle before _execute_trade's dry_run guard is ever reached.
        if self.config.dry_run:
            balance   = self.risk.config.bankroll   # simulate full bankroll
            positions = []                          # no real positions to track
        else:
            balance   = self.client.get_usdc_balance()
            positions = self.client.get_positions()
            # Check if any held positions have resolved and free up exposure
            self._check_resolved_positions(positions)
        status = self.risk.status_report()

        logger.info(
            "Cycle start | balance=$%.2f | exposure=$%.2f (%.0f%%) | "
            "open_positions=%d | daily_pnl=$%.2f",
            balance,
            status["total_exposure"],
            status["exposure_pct"] * 100,
            status["open_markets"],
            status["daily_pnl"],
            )

        # 3. Market scan
        markets = self._scan_markets()
        if not markets:
            logger.info("No markets returned from scan — nothing to do.")
            return []

        # 4. Analyse and trade
        for market in markets:

            # Skip markets we already hold a position in
            held_ids = {p.condition_id for p in positions}
            if market.condition_id in held_ids:
                logger.debug("Skipping %s — already holding position", market.condition_id[:12])
                continue

            # Claude analysis
            analysis = self.analyst.analyse(market)
            if analysis is None:
                continue

            # Risk gate
            available_liquidity = market.liquidity
            decision = self.risk.evaluate(analysis, positions, available_liquidity)

            if not decision.approved:
                logger.debug("Trade rejected: %s", decision)
                cycle_records.append(TradeRecord(
                    timestamp        = now,
                    condition_id     = market.condition_id,
                    question         = market.question,
                    action           = analysis.recommended_action,
                    approved         = False,
                    size_usdc        = 0.0,
                    entry_price      = 0.0,
                    edge             = analysis.edge,
                    confidence       = analysis.confidence,
                    rejection_reason = decision.rejection_reason,
                    dry_run          = self.config.dry_run,
                ))
                continue

            # Execute
            record = self._execute_trade(market, analysis, decision)
            cycle_records.append(record)
            self.trade_log.append(record)

            logger.info(record.log_line())

            # Rate-limit between orders
            time.sleep(self.config.delay_between_trades)

        # Cycle summary
        approved = sum(1 for r in cycle_records if r.approved)
        rejected = len(cycle_records) - approved
        logger.info(
            "Cycle complete: %d approved, %d rejected | %s",
            approved, rejected, now,
        )

        return cycle_records

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_trade_log(self, last_n: int = 20):
        """Print the last N trade records to stdout."""
        records = self.trade_log[-last_n:]
        print(f"\n{'─' * 100}")
        print(f"  TRADE LOG  (last {len(records)} entries)")
        print(f"{'─' * 100}")
        for r in records:
            print(r.log_line())
        print(f"{'─' * 100}\n")