"""
polymarket_client.py
--------------------
A clean wrapper around py-clob-client for the Polymarket CLOB API.
Handles authentication, market data fetching, order management,
position tracking, and account info.

Requirements:
    pip install py-clob-client python-dotenv requests

Environment variables (.env):
    POLYMARKET_PRIVATE_KEY     - Your wallet private key
    POLYMARKET_PROXY_ADDRESS   - Your Polymarket proxy/funder address
    POLYMARKET_SIGNATURE_TYPE  - 1 (email/Magic) or 2 (MetaMask/browser wallet)
"""

import json
import os
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    BookParams,
    OrderArgs,
    MarketOrderArgs,
    OrderType,
    TradeParams,
    OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY, SELL
from py_clob_client.exceptions import PolyApiException

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("polymarket_client")

# ── Constants ──────────────────────────────────────────────────────────────────

HOST = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"   # market metadata
DATA_API  = "https://data-api.polymarket.com"    # user positions & activity
CHAIN_ID = 137                                    # Polygon mainnet

MIN_LIQUIDITY_USDC = 500     # ignore markets thinner than this
MIN_VOLUME_USDC = 1_000      # ignore markets with less total volume
MAX_RETRIES = 3
RETRY_DELAY = 2              # seconds between retries


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class MarketInfo:
    """Flattened view of a Polymarket binary market."""
    condition_id: str
    question: str
    description: str
    end_date: str
    active: bool
    closed: bool
    # YES token
    yes_token_id: str
    yes_price: float          # current best ask (0-1)
    yes_mid: float            # midpoint (implied probability)
    # NO token
    no_token_id: str
    no_price: float
    no_mid: float
    # Market stats
    volume: float
    liquidity: float
    # Raw category tags
    tags: list[str]
    neg_risk: bool


@dataclass
class OrderbookSnapshot:
    """Top-N levels of the orderbook for a token."""
    token_id: str
    timestamp: float
    best_bid: float
    best_ask: float
    mid: float
    spread: float
    bids: list[dict]   # [{"price": x, "size": y}, ...]
    asks: list[dict]


@dataclass
class Position:
    """Current open position in a market."""
    token_id: str
    condition_id: str
    side: str           # "YES" or "NO"
    size: float
    avg_price: float
    current_price: float
    unrealized_pnl: float


# ── Client ─────────────────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Wraps py-clob-client with higher-level helpers for a trading bot.

    Usage (authenticated):
        client = PolymarketClient()
        markets = client.get_active_markets()

    Usage (read-only, no private key needed):
        client = PolymarketClient(read_only=True)
    """

    def __init__(self, read_only: bool = False):
        self.read_only = read_only

        if read_only:
            self._clob = ClobClient(HOST)
            logger.info("PolymarketClient initialised in READ-ONLY mode")
        else:
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
            proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS")
            sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

            if not private_key or not proxy_address:
                raise EnvironmentError(
                    "Set POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS in .env"
                )

            self._clob = ClobClient(
                HOST,
                key=private_key,
                chain_id=CHAIN_ID,
                signature_type=sig_type,
                funder=proxy_address,
            )
            # Derive (or create) L2 API credentials
            self._clob.set_api_creds(self._clob.create_or_derive_api_creds())
            logger.info(
                "PolymarketClient initialised (signature_type=%d, funder=%s…)",
                sig_type,
                proxy_address[:8],
            )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _retry(self, fn, *args, **kwargs):
        """
        Call fn(*args, **kwargs) with simple retry logic.

        404 errors are never retried — they mean the resource doesn't exist
        and retrying will never help (and wastes ~10s per call at RETRY_DELAY=2).
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except PolyApiException as exc:
                if exc.status_code == 404:
                    raise   # permanent — don't retry
                if attempt == MAX_RETRIES:
                    raise
                logger.warning("Attempt %d failed (%s); retrying…", attempt, exc)
                time.sleep(RETRY_DELAY * attempt)
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                logger.warning("Attempt %d failed (%s); retrying…", attempt, exc)
                time.sleep(RETRY_DELAY * attempt)

    def _gamma_get(self, path: str, params: dict = None) -> dict | list:
        """HTTP GET against the Gamma markets API."""
        import json as _json
        url = f"{GAMMA_API}{path}"
        # Normalise Python booleans to lowercase strings — the Gamma API requires
        # "true"/"false", but urlencode() and requests produce "True"/"False".
        if params:
            params = {
                k: str(v).lower() if isinstance(v, bool) else v
                for k, v in params.items()
            }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        # Use json.loads(resp.text) instead of resp.json() — the latter fails when
        # the server omits the "application/json" Content-Type header or includes a
        # BOM/unexpected encoding that the requests decoder rejects, even when the
        # body itself is perfectly valid JSON.
        try:
            return _json.loads(resp.text)
        except _json.JSONDecodeError as exc:
            logger.error(
                "Gamma API returned non-JSON response (status=%d): %s… — %s",
                resp.status_code, resp.text[:200], exc,
            )
            raise

    def _extract_price(self, raw) -> float:
        """
        Normalise CLOB price responses to a plain float.

        get_price()    returns {'price': '0.65'}
        get_midpoint() returns {'mid': '0.65'}
        Older versions returned the value directly as a string or float.
        This helper handles all three cases.
        """
        if isinstance(raw, dict):
            val = raw.get("price") or raw.get("mid") or 0
        else:
            val = raw
        return float(val or 0)

    # ── Market discovery ───────────────────────────────────────────────────────

    def get_active_markets(
            self,
            # Pagination
            limit: int = 100,
            offset: int = 0,
            order: str = "volume",          # field to sort by, e.g. "volume", "liquidity"
            ascending: bool = False,
            # Server-side filters (pushed to the API — no wasted bandwidth)
            liquidity_min: float = MIN_LIQUIDITY_USDC,
            liquidity_max: float = None,
            volume_min: float = MIN_VOLUME_USDC,
            volume_max: float = None,
            end_date_min: str = None,       # ISO-8601, e.g. "2025-06-01T00:00:00Z"
            end_date_max: str = None,
            start_date_min: str = None,
            start_date_max: str = None,
            # Lookup by identity
            condition_ids: list[str] = None,
            clob_token_ids: list[str] = None,
            slugs: list[str] = None,
            question_ids: list[str] = None,
            # Category / tag filters
            tag_id: int = None,             # single tag id to filter by
            related_tags: bool = None,      # include markets with related tags
            # Sports-specific
            sports_market_types: list[str] = None,
            game_id: str = None,
            # Misc
            closed: bool = False,
            rewards_min_size: float = None, # only markets with active rewards above this
            cyom: bool = None,              # "Create Your Own Market" flag
    ) -> list[MarketInfo]:
        """
        Fetch markets from the Gamma API, delegating as much filtering as
        possible to the server, then enrich each result with live CLOB prices.

        Key server-side filters (save bandwidth & avoid client-side loops):
          - liquidity_min / liquidity_max
          - volume_min / volume_max
          - end_date_min / end_date_max  → target markets resolving in a window
          - tag_id                       → focus on a category (e.g. politics, crypto)
          - condition_ids / slugs        → fetch specific known markets
          - rewards_min_size             → only incentivised markets

        Returns a list of MarketInfo objects enriched with live bid/ask data.
        """
        params: dict = {
            "active":    not closed,   # active=True for open markets, False when fetching closed
            "closed":    closed,
            "archived":  False,
            "limit":     limit,
            "offset":    offset,
            "order":     order,
            "ascending": ascending,
        }

        # Numeric range filters — let the API do the heavy lifting
        if liquidity_min is not None:
            params["liquidity_num_min"] = liquidity_min
        if liquidity_max is not None:
            params["liquidity_num_max"] = liquidity_max
        if volume_min is not None:
            params["volume_num_min"] = volume_min
        if volume_max is not None:
            params["volume_num_max"] = volume_max

        # Date window filters — useful for targeting near-term resolutions
        if end_date_min is not None:
            params["end_date_min"] = end_date_min
        if end_date_max is not None:
            params["end_date_max"] = end_date_max
        if start_date_min is not None:
            params["start_date_min"] = start_date_min
        if start_date_max is not None:
            params["start_date_max"] = start_date_max

        # Identity lookups — fetch a known set of markets directly
        if condition_ids:
            params["condition_ids"] = condition_ids
        if clob_token_ids:
            params["clob_token_ids"] = clob_token_ids
        if slugs:
            params["slug"] = slugs
        if question_ids:
            params["question_ids"] = question_ids

        # Category / tag filters
        if tag_id is not None:
            params["tag_id"]       = tag_id
        if related_tags is not None:
            params["related_tags"] = related_tags

        # Sports filters
        if sports_market_types:
            params["sports_market_types"] = sports_market_types
        if game_id:
            params["game_id"] = game_id

        # Misc
        if rewards_min_size is not None:
            params["rewards_min_size"] = rewards_min_size
        if cyom is not None:
            params["cyom"] = str(cyom).lower()

        logger.info("Fetching markets with params: %s", params)
        raw = self._gamma_get("/markets", params=params)

        markets: list[MarketInfo] = []
        for m in raw:
            try:
                # clobTokenIds is a JSON-stringified list e.g. '["0xabc...", "0xdef..."]'
                # Index 0 = YES token, index 1 = NO token
                raw_ids = m.get("clobTokenIds", "[]")
                token_ids = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids

                if len(token_ids) < 2:
                    continue

                yes_id = token_ids[0]
                no_id  = token_ids[1]

                if not yes_id or not no_id:
                    continue

                # Live pricing from CLOB — failures are non-fatal, fall back to 0.0
                try:
                    yes_price = self._extract_price(self._retry(self._clob.get_price, yes_id, side="BUY"))
                    yes_mid   = self._extract_price(self._retry(self._clob.get_midpoint, yes_id))
                    no_price  = self._extract_price(self._retry(self._clob.get_price, no_id, side="BUY"))
                    no_mid    = self._extract_price(self._retry(self._clob.get_midpoint, no_id))
                except PolyApiException as price_exc:
                    if price_exc.status_code == 404:
                        # Market resolved or book removed — skip it
                        logger.debug("No orderbook for %s — skipping", m.get("conditionId", "?")[:12])
                        continue
                    logger.warning(
                        "CLOB price fetch failed for %s (%s) — using 0.0: %s",
                        m.get("conditionId", "?")[:12], m.get("question", "?")[:50], price_exc,
                    )
                    yes_price = yes_mid = no_price = no_mid = 0.0
                except Exception as price_exc:
                    logger.warning(
                        "CLOB price fetch failed for %s (%s) — using 0.0: %s",
                        m.get("conditionId", "?")[:12], m.get("question", "?")[:50], price_exc,
                    )
                    yes_price = yes_mid = no_price = no_mid = 0.0

                markets.append(MarketInfo(
                    condition_id = m.get("conditionId", ""),
                    question     = m.get("question", ""),
                    description  = m.get("description", ""),
                    end_date     = m.get("endDate", ""),
                    active       = bool(m.get("active", False)),
                    closed       = bool(m.get("closed", False)),
                    yes_token_id = yes_id,
                    yes_price    = yes_price,
                    yes_mid      = yes_mid,
                    no_token_id  = no_id,
                    no_price     = no_price,
                    no_mid       = no_mid,
                    volume       = float(m.get("volume", 0) or 0),
                    liquidity    = float(m.get("liquidity", 0) or 0),
                    tags         = m.get("tags", []),
                    neg_risk     = bool(m.get("negRisk", False)),
                ))

            except Exception as exc:
                logger.warning("Skipping market %s: %s", m.get("condition_id", "?"), exc)

        logger.info("Returning %d markets", len(markets))
        return markets

    def get_market_by_id(self, condition_id: str) -> Optional[MarketInfo]:
        """Fetch a single market by condition_id."""
        raw_list = self._gamma_get("/markets", params={"condition_ids": condition_id})
        if not raw_list:
            return None
        markets = self.get_active_markets.__wrapped__ if hasattr(
            self.get_active_markets, "__wrapped__"
        ) else None
        # Re-use the full parsing logic via a single-item fetch
        for m in raw_list[:1]:
            try:
                tokens    = m.get("tokens", [])
                yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), tokens[0])
                no_token  = next((t for t in tokens if t.get("outcome") == "No"),  tokens[1])
                yes_id    = yes_token["token_id"]
                no_id     = no_token["token_id"]
                return MarketInfo(
                    condition_id = m.get("conditionId", ""),
                    question     = m.get("question", ""),
                    description  = m.get("description", ""),
                    end_date     = m.get("endDate", ""),
                    active       = bool(m.get("active")),
                    closed       = bool(m.get("closed")),
                    yes_token_id = yes_id,
                    yes_price    = self._extract_price(self._clob.get_price(yes_id, side="BUY")),
                    yes_mid      = self._extract_price(self._clob.get_midpoint(yes_id)),
                    no_token_id  = no_id,
                    no_price     = self._extract_price(self._clob.get_price(no_id, side="BUY")),
                    no_mid       = self._extract_price(self._clob.get_midpoint(no_id)),
                    volume       = float(m.get("volume", 0) or 0),
                    liquidity    = float(m.get("liquidity", 0) or 0),
                    tags         = m.get("tags", []),
                    neg_risk     = bool(m.get("negRisk", False)),
                )
            except Exception as exc:
                logger.error("Failed to parse market %s: %s", condition_id, exc)
        return None

    # ── Orderbook ──────────────────────────────────────────────────────────────

    def get_orderbook(self, token_id: str, depth: int = 5) -> OrderbookSnapshot:
        """
        Return the top `depth` bid/ask levels for a given token.
        """
        book = self._retry(self._clob.get_order_book, token_id)

        def _top(levels, n):
            return [{"price": float(l.price), "size": float(l.size)} for l in levels[:n]]

        bids = sorted(_top(book.bids, depth), key=lambda x: x["price"], reverse=True)
        asks = sorted(_top(book.asks, depth), key=lambda x: x["price"])

        best_bid = bids[0]["price"] if bids else 0.0
        best_ask = asks[0]["price"] if asks else 1.0
        mid      = (best_bid + best_ask) / 2
        spread   = best_ask - best_bid

        return OrderbookSnapshot(
            token_id  = token_id,
            timestamp = time.time(),
            best_bid  = best_bid,
            best_ask  = best_ask,
            mid       = mid,
            spread    = spread,
            bids      = bids,
            asks      = asks,
        )

    def get_mid_price(self, token_id: str) -> float:
        """
        Quick mid-price lookup (no full orderbook needed).

        Returns 0.0 if the orderbook no longer exists (404) — this happens
        when a market has resolved and the CLOB has removed its book.
        """
        try:
            return self._extract_price(self._retry(self._clob.get_midpoint, token_id))
        except PolyApiException as exc:
            if exc.status_code == 404:
                logger.debug("No orderbook for token %s (market resolved?)", token_id[:16])
                return 0.0
            raise

    # ── Order placement ────────────────────────────────────────────────────────

    def _check_authenticated(self):
        if self.read_only:
            raise PermissionError("Client is in read-only mode. Provide credentials to trade.")

    def place_limit_order(
            self,
            token_id: str,
            side: str,           # "BUY" or "SELL"
            price: float,        # 0.01 – 0.99 in 0.01 increments
            size: float,         # USDC amount
            order_type: str = "GTC",
    ) -> dict:
        """
        Place a resting limit order.

        Args:
            token_id:   YES or NO token id of the market.
            side:       "BUY" or "SELL"
            price:      Limit price as a decimal probability (e.g. 0.65).
            size:       Order size in USDC.
            order_type: "GTC" (default) or "GTD".

        Returns:
            Raw API response dict.
        """
        self._check_authenticated()

        raw_side = BUY if side.upper() == "BUY" else SELL
        ot = OrderType.GTC if order_type.upper() == "GTC" else OrderType.GTD

        order_args = OrderArgs(
            token_id = token_id,
            price    = round(price, 4),
            size     = round(size, 2),
            side     = raw_side,
        )
        signed = self._clob.create_order(order_args)
        resp   = self._retry(self._clob.post_order, signed, ot)

        logger.info(
            "Limit %s order placed: token=%s price=%.4f size=%.2f → %s",
            side, token_id[:12], price, size, resp,
        )
        return resp

    def place_market_order(
            self,
            token_id: str,
            side: str,
            amount: float,
    ) -> dict:
        """
        Place an immediate Fill-Or-Kill market order.

        Args:
            token_id: YES or NO token id.
            side:     "BUY" or "SELL"
            amount:   USDC amount to spend/receive.

        Returns:
            Raw API response dict.
        """
        self._check_authenticated()

        raw_side = BUY if side.upper() == "BUY" else SELL
        mo = MarketOrderArgs(
            token_id   = token_id,
            amount     = round(amount, 2),
            side       = raw_side,
            order_type = OrderType.FOK,
        )
        signed = self._clob.create_market_order(mo)
        resp   = self._retry(self._clob.post_order, signed, OrderType.FOK)

        logger.info(
            "Market %s order placed: token=%s amount=%.2f → %s",
            side, token_id[:12], amount, resp,
        )
        return resp

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single open order by its order_id."""
        self._check_authenticated()
        resp = self._retry(self._clob.cancel, order_id=order_id)
        logger.info("Cancelled order %s → %s", order_id, resp)
        return resp

    def cancel_all_orders(self) -> dict:
        """Cancel every open order on the account."""
        self._check_authenticated()
        resp = self._retry(self._clob.cancel_all)
        logger.info("Cancelled all orders → %s", resp)
        return resp

    # ── Account & positions ────────────────────────────────────────────────────

    def get_open_orders(self, market_id: Optional[str] = None) -> list[dict]:
        """
        Return all open resting orders, optionally filtered by market.

        Args:
            market_id: Optional condition_id to filter by.
        """
        self._check_authenticated()
        params = OpenOrderParams(market=market_id) if market_id else OpenOrderParams()
        orders = self._retry(self._clob.get_orders, params)
        return orders or []

    def get_trades(
            self,
            token_id: Optional[str] = None,
            after: Optional[str] = None,
            before: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve recent trade history.

        TradeParams supports: id, maker_address, market, asset_id, before, after.
        There is no limit field — use before/after (ISO-8601 timestamps) to
        window results if you need to page through a large history.

        Args:
            token_id: Optional token (asset_id) to filter trades by.
            after:    Optional ISO-8601 timestamp — return trades after this time.
            before:   Optional ISO-8601 timestamp — return trades before this time.
        """
        self._check_authenticated()
        params = TradeParams(
            asset_id = token_id,
            after    = after,
            before   = before,
        )
        trades = self._retry(self._clob.get_trades, params)
        return trades or []

    def get_positions(self) -> list[Position]:
        """
        Fetch currently held positions from the Polymarket Data API.

        Uses GET data-api.polymarket.com/positions?user=<proxy_address>
        This endpoint only returns tokens you actively hold, so there is no
        risk of calling get_mid_price on resolved/deleted orderbooks.

        No authentication required — the proxy address is public.
        """
        self._check_authenticated()

        proxy_address = os.getenv("POLYMARKET_PROXY_ADDRESS", "")
        if not proxy_address:
            logger.warning("POLYMARKET_PROXY_ADDRESS not set — cannot fetch positions")
            return []

        try:
            resp = requests.get(
                f"{DATA_API}/positions",
                params={"user": proxy_address, "sizeThreshold": "0.01"},
                timeout=10,
            )
            resp.raise_for_status()
            raw_positions = resp.json()
        except Exception as exc:
            logger.warning("Failed to fetch positions from Data API: %s", exc)
            return []

        positions = []
        for pos in raw_positions:
            tid  = pos.get("asset", "")
            size = float(pos.get("size", 0))

            if not tid or size < 0.01:
                continue

            avg_price     = float(pos.get("avgPrice", 0))
            current_price = self.get_mid_price(tid)
            unrealized    = size * (current_price - avg_price)
            side          = pos.get("outcome", "Yes").upper()

            positions.append(Position(
                token_id       = tid,
                condition_id   = pos.get("conditionId", ""),
                side           = side,
                size           = round(size, 4),
                avg_price      = round(avg_price, 4),
                current_price  = round(current_price, 4),
                unrealized_pnl = round(unrealized, 4),
            ))

        return positions

    def get_usdc_balance(self) -> float:
        """
        Return available USDC balance of the proxy/funder wallet in USDC.

        Important: get_balance() returns the *signer key* wallet balance, which
        is typically 0 for email/Magic logins — the actual funds live in the
        proxy wallet (POLYMARKET_PROXY_ADDRESS). We query that address directly
        via get_balance_allowance() with AssetType.COLLATERAL (= USDC).
        The returned value is in wei (micro-USDC), so we divide by 1e6.
        """
        self._check_authenticated()
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            result = self._clob.get_balance_allowance(params)
            bal_wei = result.get("balance", "0") if isinstance(result, dict) else result
            return float(bal_wei or 0) / 1_000_000
        except Exception as exc:
            logger.warning("Could not fetch balance: %s", exc)
            return 0.0

    # ── Convenience helpers ────────────────────────────────────────────────────

    def summarise_market(self, market: MarketInfo) -> str:
        """Return a human-readable one-liner for a market."""
        return (
            f"[{market.condition_id[:10]}…] {market.question[:80]} | "
            f"YES={market.yes_mid:.2%}  NO={market.no_mid:.2%} | "
            f"Vol=${market.volume:,.0f}  Liq=${market.liquidity:,.0f}"
        )


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Read-only demo — no private key required
    client = PolymarketClient(read_only=True)

    print("\n=== Top 5 active markets ===")
    markets = client.get_active_markets(limit=20)
    for m in markets[:5]:
        print(client.summarise_market(m))

    if markets:
        sample = markets[0]
        print(f"\n=== Orderbook for YES token of first market ===")
        book = client.get_orderbook(sample.yes_token_id, depth=3)
        print(f"  Best bid: {book.best_bid:.4f}")
        print(f"  Best ask: {book.best_ask:.4f}")
        print(f"  Mid:      {book.mid:.4f}")
        print(f"  Spread:   {book.spread:.4f}")
        print(f"  Top bids: {book.bids}")
        print(f"  Top asks: {book.asks}")