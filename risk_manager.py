"""
risk_manager.py
---------------
Position sizing, exposure limits, and kill-switch logic for the trading bot.

Responsibilities:
  - Size positions using fractional Kelly Criterion
  - Enforce per-trade, per-market, and total portfolio limits
  - Track daily P&L and halt trading if drawdown threshold is breached
  - Validate every trade before it reaches the order placement layer

All monetary values are in USDC.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional

from claude_analyst import AnalysisResult
from polymarket_client import Position

logger = logging.getLogger("risk_manager")


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """
    Tunable risk parameters. Adjust these to match your risk appetite
    before going live.

    Attributes:
        bankroll:               Total USDC capital allocated to the bot.
        kelly_fraction:         Fraction of full Kelly to use (0.25 = quarter-Kelly).
                                Full Kelly maximises growth but has huge variance;
                                0.25–0.5 is typical for real trading.
        min_bet_usdc:           Minimum order size. Orders below this are skipped
                                (Polymarket has its own minimum; stay above it).
        max_bet_usdc:           Hard cap on any single order in USDC.
        max_position_pct:       Max % of bankroll in any single market (both sides).
        max_total_exposure_pct: Max % of bankroll deployed across ALL open positions.
        max_daily_loss_pct:     Kill-switch threshold — halt if daily loss exceeds this
                                fraction of bankroll.
        max_open_positions:     Hard cap on the number of simultaneous open positions.
        min_edge:               Minimum edge required (redundant guard on top of analyst).
        min_confidence:         Minimum Claude confidence level to allow a trade.
        min_liquidity_usdc:     Skip markets where available liquidity is below this.
    """
    bankroll: float               = 1_000.0
    kelly_fraction: float         = 0.25
    min_bet_usdc: float           = 5.0
    max_bet_usdc: float           = 100.0
    max_position_pct: float       = 0.10     # 10% of bankroll per market
    max_total_exposure_pct: float = 0.50     # 50% of bankroll total
    max_daily_loss_pct: float     = 0.05     # 5% daily drawdown kills the bot
    max_open_positions: int       = 10
    min_edge: float               = 0.05
    min_confidence: str           = "medium"  # "low" | "medium" | "high"
    min_liquidity_usdc: float     = 1_000.0


@dataclass
class TradeDecision:
    """
    Output of RiskManager.evaluate() — either approved with a size, or rejected
    with a reason.
    """
    approved: bool
    size_usdc: float = 0.0
    rejection_reason: str = ""
    kelly_size: float = 0.0       # full fractional-Kelly suggestion before capping
    condition_id: str = ""
    action: str = ""              # "BUY_YES" | "BUY_NO"

    def __str__(self):
        if self.approved:
            return (
                f"APPROVED  {self.action} ${self.size_usdc:.2f} USDC  "
                f"(kelly=${self.kelly_size:.2f}, market={self.condition_id[:12]}…)"
            )
        return f"REJECTED  {self.rejection_reason}  (market={self.condition_id[:12]}…)"


# ── Risk Manager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Validates and sizes every trade before it reaches the exchange.

    Usage:
        rm = RiskManager(RiskConfig(bankroll=2000))
        decision = rm.evaluate(analysis, current_positions, available_liquidity)
        if decision.approved:
            client.place_limit_order(token_id, "BUY", price, decision.size_usdc)
            rm.record_fill(condition_id, decision.size_usdc, entry_price)
    """

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self._daily_pnl: float = 0.0
        self._daily_pnl_date: date = datetime.now(timezone.utc).date()
        self._halted: bool = False
        self._halt_reason: str = ""
        # condition_id → usdc_deployed (for exposure tracking)
        self._open_exposure: dict[str, float] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def total_exposure(self) -> float:
        return sum(self._open_exposure.values())

    @property
    def daily_pnl(self) -> float:
        self._roll_daily_pnl_if_needed()
        return self._daily_pnl

    # ── Kelly sizing ──────────────────────────────────────────────────────────

    def _kelly_size(self, edge: float, win_prob: float) -> float:
        """
        Fractional Kelly Criterion for a binary bet.

        Full Kelly: f* = (b*p - q) / b
          where b = odds on a $1 bet (i.e. 1/p_market - 1),
                p = our estimated win probability,
                q = 1 - p

        We then multiply by kelly_fraction (e.g. 0.25) and bankroll.
        """
        # win_prob is our estimate; derive market-implied prob from edge
        market_prob = win_prob - edge
        market_prob = max(0.01, min(0.99, market_prob))

        # Decimal odds on a $1 stake
        b = (1.0 / market_prob) - 1.0
        p = win_prob
        q = 1.0 - p

        full_kelly_fraction = (b * p - q) / b
        full_kelly_fraction = max(0.0, full_kelly_fraction)   # never negative

        raw_size = full_kelly_fraction * self.config.kelly_fraction * self.config.bankroll
        return round(raw_size, 2)

    # ── Core evaluation ───────────────────────────────────────────────────────

    def evaluate(
            self,
            analysis: AnalysisResult,
            current_positions: list[Position],
            available_liquidity: float = float("inf"),
    ) -> TradeDecision:
        """
        Decide whether to trade and how much.

        Checks (in order):
          1. Kill-switch — bot is halted
          2. Analysis quality — action, confidence, edge
          3. Duplicate position — already long/short this market
          4. Open position count limit
          5. Total portfolio exposure limit
          6. Per-market position limit
          7. Liquidity check
          8. Kelly sizing and bet-size bounds

        Returns a TradeDecision (approved=True/False).
        """
        cid = analysis.condition_id

        # 1. Kill-switch
        if self._halted:
            return TradeDecision(False, rejection_reason=f"Bot halted: {self._halt_reason}", condition_id=cid)

        # 2. Analysis quality guards
        if analysis.recommended_action == "PASS":
            return TradeDecision(False, rejection_reason="Analyst recommended PASS", condition_id=cid, action=analysis.recommended_action)

        confidence_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = confidence_rank.get(self.config.min_confidence, 1)
        if confidence_rank.get(analysis.confidence, 0) < min_rank:
            return TradeDecision(
                False,
                rejection_reason=f"Confidence too low ({analysis.confidence} < {self.config.min_confidence})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        if abs(analysis.edge) < self.config.min_edge:
            return TradeDecision(
                False,
                rejection_reason=f"Edge too small ({analysis.edge:+.1%} < {self.config.min_edge:.1%})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 3. Duplicate position check
        open_sides = {p.condition_id: p.side for p in current_positions}
        if cid in open_sides:
            return TradeDecision(
                False,
                rejection_reason=f"Already have open {open_sides[cid]} position in this market",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 4. Open position count
        if len(current_positions) >= self.config.max_open_positions:
            return TradeDecision(
                False,
                rejection_reason=f"Max open positions reached ({self.config.max_open_positions})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 5. Total portfolio exposure
        max_total = self.config.bankroll * self.config.max_total_exposure_pct
        if self.total_exposure >= max_total:
            return TradeDecision(
                False,
                rejection_reason=f"Total exposure limit reached (${self.total_exposure:.0f} / ${max_total:.0f})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 6. Per-market exposure
        max_per_market = self.config.bankroll * self.config.max_position_pct
        existing_market_exposure = self._open_exposure.get(cid, 0.0)
        if existing_market_exposure >= max_per_market:
            return TradeDecision(
                False,
                rejection_reason=f"Per-market limit reached (${existing_market_exposure:.0f} / ${max_per_market:.0f})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 7. Liquidity check
        if available_liquidity < self.config.min_liquidity_usdc:
            return TradeDecision(
                False,
                rejection_reason=f"Insufficient market liquidity (${available_liquidity:.0f} < ${self.config.min_liquidity_usdc:.0f})",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        # 8. Size the trade
        # For BUY_YES use claude_yes_prob; for BUY_NO invert it
        if analysis.recommended_action == "BUY_YES":
            win_prob = analysis.claude_yes_prob
            edge     = analysis.edge
        else:
            win_prob = 1.0 - analysis.claude_yes_prob
            edge     = -analysis.edge

        kelly_size = self._kelly_size(edge, win_prob)

        # Apply hard caps
        size = kelly_size
        size = max(size, self.config.min_bet_usdc)
        size = min(size, self.config.max_bet_usdc)

        # Don't exceed remaining room in per-market limit
        room_in_market = max_per_market - existing_market_exposure
        size = min(size, room_in_market)

        # Don't exceed remaining total exposure room
        room_in_total = max_total - self.total_exposure
        size = min(size, room_in_total)

        # Don't take more than available liquidity
        size = min(size, available_liquidity * 0.10)   # max 10% of market liquidity per order

        size = round(size, 2)

        if size < self.config.min_bet_usdc:
            return TradeDecision(
                False,
                rejection_reason=f"Final size ${size:.2f} below minimum ${self.config.min_bet_usdc:.2f} after limits",
                condition_id=cid,
                action=analysis.recommended_action,
            )

        logger.info(
            "Trade approved: %s %s $%.2f (kelly=$%.2f) — %s",
            analysis.recommended_action, cid[:12], size, kelly_size, analysis.question[:50],
        )

        return TradeDecision(
            approved=True,
            size_usdc=size,
            kelly_size=kelly_size,
            condition_id=cid,
            action=analysis.recommended_action,
        )

    # ── State tracking ────────────────────────────────────────────────────────

    def record_fill(self, condition_id: str, size_usdc: float, entry_price: float):
        """
        Call this after a successful order fill to update exposure tracking.
        """
        self._open_exposure[condition_id] = (
                self._open_exposure.get(condition_id, 0.0) + size_usdc
        )
        logger.info(
            "Fill recorded: %s +$%.2f @ %.4f | total exposure: $%.2f",
            condition_id[:12], size_usdc, entry_price, self.total_exposure,
        )

    def record_close(self, condition_id: str, pnl_usdc: float):
        """
        Call this when a position is resolved or manually closed.
        Updates daily P&L and removes the position from exposure tracking.
        """
        self._roll_daily_pnl_if_needed()
        self._daily_pnl += pnl_usdc
        self._open_exposure.pop(condition_id, None)

        logger.info(
            "Position closed: %s | pnl=$%.2f | daily_pnl=$%.2f",
            condition_id[:12], pnl_usdc, self._daily_pnl,
        )

        self._check_kill_switch()

    def _roll_daily_pnl_if_needed(self):
        """Reset daily P&L at UTC midnight."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_pnl_date:
            logger.info(
                "New trading day — resetting daily P&L (was $%.2f)", self._daily_pnl
            )
            self._daily_pnl = 0.0
            self._daily_pnl_date = today

    def _check_kill_switch(self):
        """Halt the bot if daily loss exceeds the configured threshold."""
        max_daily_loss = -self.config.bankroll * self.config.max_daily_loss_pct
        if self._daily_pnl < max_daily_loss:
            reason = (
                f"Daily loss ${abs(self._daily_pnl):.2f} exceeded limit "
                f"${abs(max_daily_loss):.2f} ({self.config.max_daily_loss_pct:.0%} of bankroll)"
            )
            self._halted = True
            self._halt_reason = reason
            logger.critical("KILL SWITCH TRIGGERED — %s", reason)

    def resume(self, override_reason: str = ""):
        """Manually resume trading after a kill-switch halt."""
        self._halted = False
        self._halt_reason = ""
        logger.warning("Trading resumed manually. Override reason: %s", override_reason or "none given")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def status_report(self) -> dict:
        """Return a snapshot of current risk state."""
        self._roll_daily_pnl_if_needed()
        return {
            "halted":           self._halted,
            "halt_reason":      self._halt_reason,
            "daily_pnl":        round(self._daily_pnl, 2),
            "total_exposure":   round(self.total_exposure, 2),
            "open_markets":     len(self._open_exposure),
            "bankroll":         self.config.bankroll,
            "exposure_pct":     round(self.total_exposure / self.config.bankroll, 4),
            "daily_pnl_pct":    round(self._daily_pnl / self.config.bankroll, 4),
        }