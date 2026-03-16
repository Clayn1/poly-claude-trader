"""
claude_analyst.py
-----------------
Uses Claude to analyse Polymarket binary markets and produce calibrated
probability estimates with structured reasoning.

Each analysis call:
  1. Optionally fetches fresh news context via Haiku + web_search (cheap)
  2. Asks Sonnet to reason step-by-step before committing to a probability
  3. Caches the system prompt to cut repeated input token costs by ~90%
  4. Returns a typed AnalysisResult with: estimated probability, confidence,
     edge vs. market, recommended action, and full reasoning chain

Model strategy:
  - News fetch  → claude-haiku-4-5-20251001  ($1/$5 per MTok)   fast, cheap
  - Analysis    → claude-sonnet-4-6           ($3/$15 per MTok)  good reasoning
  - System prompt cached on every analysis call (~90% input savings)

Environment variables (.env):
    ANTHROPIC_API_KEY   - Your Anthropic API key

Requirements:
    pip install anthropic python-dotenv
"""

import os
import json
import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

from env_helpers import env_int
from polymarket_client import MarketInfo

load_dotenv()

logger = logging.getLogger("claude_analyst")

# ── Constants ──────────────────────────────────────────────────────────────────

# Main analysis model — good reasoning at 5x lower cost than Opus
ANALYSIS_MODEL = "claude-sonnet-4-6"

# News fetch model — just summarising web results, no deep reasoning needed
NEWS_MODEL = "claude-haiku-4-5-20251001"

MAX_TOKENS = 1_024   # shorter JSON response is all we need — halves output cost

# Minimum edge (Claude prob - market prob) to recommend a trade
# Lower = more trades, higher = more selective. Start at 3% and tune from data.
MIN_EDGE_TO_TRADE = 0.03        # 3 percentage points

# Cache TTL — re-use analyses within this window to save API calls
CACHE_TTL_SECONDS = 60 * 45    # 45 minutes (was 30)

# Keywords that indicate a pure price-bracket market where news is useless
_PRICE_MARKET_KEYWORDS = (
    "price of", "above $", "below $", "reach $", "exceed $",
    "higher than $", "lower than $", "over $", "under $",
    "at least $", "at most $",
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Structured output from a single Claude market analysis.

    Attributes:
        condition_id:        Polymarket market identifier.
        question:            The market question as asked.
        market_yes_prob:     Current market-implied YES probability (mid).
        claude_yes_prob:     Claude's estimated true YES probability (0–1).
        confidence:          Claude's self-reported confidence: "low" | "medium" | "high".
        edge:                claude_yes_prob - market_yes_prob (positive = YES underpriced).
        recommended_action:  "BUY_YES" | "BUY_NO" | "PASS".
        reasoning:           Full chain-of-thought from Claude.
        key_factors:         Bullet list of the most important factors Claude identified.
        risks:               Key risks / ways Claude could be wrong.
        news_context:        Web search snippets used, if any.
        model:               Model string used for analysis.
        analysed_at:         UTC timestamp of the analysis.
        cached:              True if this result came from the in-memory cache.
    """
    condition_id: str
    question: str
    market_yes_prob: float
    claude_yes_prob: float
    confidence: str
    edge: float
    recommended_action: str
    reasoning: str
    key_factors: list[str]
    risks: list[str]
    news_context: str = ""
    model: str = ANALYSIS_MODEL
    analysed_at: str = ""
    cached: bool = False

    def __post_init__(self):
        if not self.analysed_at:
            self.analysed_at = datetime.now(timezone.utc).isoformat()

    def is_tradeable(self) -> bool:
        return (
                self.recommended_action != "PASS"
                and abs(self.edge) >= MIN_EDGE_TO_TRADE
                and self.confidence in ("medium", "high")
        )

    def summary(self) -> str:
        return (
            f"[{self.condition_id[:10]}…] {self.question[:70]}\n"
            f"  Market: {self.market_yes_prob:.1%}  →  Claude: {self.claude_yes_prob:.1%}  "
            f"(edge={self.edge:+.1%}, confidence={self.confidence})\n"
            f"  Action: {self.recommended_action}  |  Tradeable: {self.is_tradeable()}"
        )


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional prediction market analyst specialising in binary outcome markets.
Your task is to estimate the true probability of the YES outcome for a given market question.

Core principles:
- Think like a superforecaster: use base rates, reference classes, and available evidence.
- Be calibrated: a 70% probability means you expect to be right 70% of the time.
- Distinguish between what you know and what you're uncertain about.
- Actively look for reasons you might be wrong before committing to a number.
- Do NOT anchor to the current market price — form your own view first.

Output format:
You MUST respond with a single valid JSON object and nothing else. Schema:
{
  "reasoning": "<step-by-step analysis, 200-400 words>",
  "key_factors": ["<factor 1>", "<factor 2>", ...],
  "risks": ["<risk 1>", "<risk 2>", ...],
  "claude_yes_prob": <float between 0.01 and 0.99>,
  "confidence": "<low|medium|high>"
}

Confidence definitions:
  high   - Strong evidence, clear base rates, low ambiguity in resolution criteria
  medium - Moderate evidence, some uncertainty in key variables
  low    - Thin evidence, highly uncertain, or ambiguous resolution criteria
"""

# The system prompt is identical on every call — wrap it once with cache_control
# so Anthropic stores the processed tokens and charges ~10% on subsequent hits.
CACHED_SYSTEM_PROMPT = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


def _build_user_prompt(market: MarketInfo, news_context: str = "") -> str:
    hours_to_end = _hours_until(market.end_date)
    news_section = (
        f"\n\n## Recent News & Context\n{news_context}"
        if news_context
        else "\n\n## Recent News & Context\nNone retrieved."
    )

    return f"""\
## Market Details

**Question:** {market.question}

**Description / Resolution criteria:**
{market.description or "Not provided."}

**Resolves:** {market.end_date} ({hours_to_end:.0f} hours from now)

**Current market prices:**
- YES mid-price (implied probability): {market.yes_mid:.1%}
- NO  mid-price (implied probability): {market.no_mid:.1%}
- 24h volume: ${market.volume:,.0f}
- Liquidity:  ${market.liquidity:,.0f}

**Tags:** {", ".join(str(t) for t in market.tags) if market.tags else "None"}
{news_section}

---

Analyse this market and provide your probability estimate for the YES outcome.
Remember: form your view independently before considering the market price.
Respond ONLY with the JSON object described in the system prompt.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hours_until(iso_date: str) -> float:
    """Return hours between now and an ISO-8601 date string. Returns 0 on parse failure."""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        delta = dt - datetime.now(timezone.utc)
        return max(delta.total_seconds() / 3600, 0)
    except Exception:
        return 0.0


def _cache_key(condition_id: str) -> str:
    return hashlib.md5(condition_id.encode()).hexdigest()


def _is_price_market(question: str) -> bool:
    """Return True if the market is a pure price-bracket bet (news won't help)."""
    q = question.lower()
    return any(kw in q for kw in _PRICE_MARKET_KEYWORDS)


def _determine_action(edge: float, confidence: str) -> str:
    """
    Translate edge + confidence into a trading action.

    Low-confidence trades are allowed — Kelly sizing will naturally
    produce a smaller bet for weaker signals.
    """
    if edge >= MIN_EDGE_TO_TRADE:
        return "BUY_YES"
    if edge <= -MIN_EDGE_TO_TRADE:
        return "BUY_NO"
    return "PASS"


def _parse_claude_response(raw: str) -> dict:
    """
    Extract and parse the JSON object from Claude's response.

    Handles three common failure modes:
      1. Markdown fences (```json ... ```)
      2. Truncated response — MAX_TOKENS cut the JSON mid-stream; we extract
         whatever valid fields are present using regex fallback
      3. Unescaped special characters inside string values

    Always returns a dict. On total failure returns a safe fallback that
    results in a PASS action rather than crashing the cycle.
    """
    text = raw.strip()

    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    # Happy path
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Recovery 1: find the outermost { ... } block and try again
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Recovery 2: response was truncated — extract whatever fields exist via regex
    # This handles MAX_TOKENS truncation mid-JSON
    import re
    result: dict = {}

    prob_match = re.search(r'"claude_yes_prob"\s*:\s*([0-9.]+)', text)
    if prob_match:
        result["claude_yes_prob"] = float(prob_match.group(1))

    conf_match = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', text)
    if conf_match:
        result["confidence"] = conf_match.group(1)

    reason_match = re.search(r'"reasoning"\s*:\s*"(.*?)"(?=\s*,|\s*})', text, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1)[:500]

    if "claude_yes_prob" in result:
        logger.warning("Used regex fallback to extract partial JSON — response may be truncated")
        result.setdefault("confidence", "low")
        result.setdefault("reasoning", "Truncated response — partial extraction")
        result.setdefault("key_factors", [])
        result.setdefault("risks", [])
        return result

    # Total failure — return safe PASS defaults
    logger.error("Could not extract any fields from Claude response — returning PASS defaults")
    return {
        "claude_yes_prob": 0.5,
        "confidence":      "low",
        "reasoning":       "JSON parse failed entirely",
        "key_factors":     [],
        "risks":           [],
    }


# ── Analyst ────────────────────────────────────────────────────────────────────

class ClaudeAnalyst:
    """
    Wraps the Anthropic API to produce structured market analyses.

    Features:
    - Two-tier model routing: Haiku for news, Sonnet for analysis
    - System prompt caching (~90% savings on repeated input tokens)
    - In-memory result cache (TTL-based, avoids redundant API calls)
    - Skips markets too close to resolution
    - Full chain-of-thought reasoning extracted alongside the final probability

    Usage:
        analyst = ClaudeAnalyst()
        result  = analyst.analyse(market)
        if result.is_tradeable():
            print(result.summary())
    """

    def __init__(self, use_web_search: bool = True):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set in environment.")

        self._client = anthropic.Anthropic(api_key=api_key)
        self.use_web_search = use_web_search

        # { cache_key: (AnalysisResult, expiry_timestamp) }
        self._cache: dict[str, tuple[AnalysisResult, float]] = {}

        logger.info(
            "ClaudeAnalyst initialised (analysis=%s, news=%s, web_search=%s)",
            ANALYSIS_MODEL, NEWS_MODEL, use_web_search,
        )

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _get_cached(self, condition_id: str) -> Optional[AnalysisResult]:
        key = _cache_key(condition_id)
        entry = self._cache.get(key)
        if entry and time.time() < entry[1]:
            logger.debug("Cache hit for %s", condition_id[:12])
            result = entry[0]
            result.cached = True
            return result
        return None

    def _set_cache(self, condition_id: str, result: AnalysisResult):
        key = _cache_key(condition_id)
        self._cache[key] = (result, time.time() + CACHE_TTL_SECONDS)

    def invalidate_cache(self, condition_id: str):
        """Force re-analysis on next call for a specific market."""
        self._cache.pop(_cache_key(condition_id), None)

    def clear_cache(self):
        """Wipe the entire analysis cache."""
        self._cache.clear()

    # ── Web search ─────────────────────────────────────────────────────────────

    def _fetch_news_context(self, market: MarketInfo) -> str:
        """
        Ask Haiku to search for relevant context for a market.
        Returns a plain-text summary of what was found.

        Uses NEWS_MODEL (Haiku) rather than ANALYSIS_MODEL — web search
        is just retrieval and summarisation, not deep reasoning.
        Cost: ~5x cheaper per call than using Sonnet here.
        """
        search_prompt = (
            f"Search for recent news and information relevant to this prediction market "
            f"question: '{market.question}'. "
            f"Summarise the most relevant facts in 150-200 words. "
            f"Focus on concrete evidence that would help forecast the probability of YES."
        )

        try:
            response = self._client.messages.create(
                model=NEWS_MODEL,
                max_tokens=512,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": search_prompt}],
            )

            context_parts = [
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            ]
            context = "\n\n".join(context_parts).strip()
            logger.debug(
                "News context fetched (%d chars) for %s",
                len(context), market.condition_id[:12],
            )
            return context

        except Exception as exc:
            logger.warning(
                "Web search failed for %s: %s", market.condition_id[:12], exc
            )
            return ""

    # ── Core analysis ──────────────────────────────────────────────────────────

    def analyse(self, market: MarketInfo, force_refresh: bool = False) -> Optional[AnalysisResult]:
        """
        Analyse a single market and return a structured AnalysisResult.

        Returns None if:
          - The market is too close to resolution (< MIN_HOURS_TO_RESOLUTION)
          - The Claude API call fails

        Args:
            market:        MarketInfo from polymarket_client.
            force_refresh: Bypass cache and re-analyse even if cached.
        """
        # 1. Skip markets resolving too soon
        hours_left = _hours_until(market.end_date)
        hours_to_resolution = env_int("MIN_HOURS_TO_RESOLUTION", 2)
        if hours_left < hours_to_resolution:
            logger.info(
                "Skipping %s — resolves in %.1fh (< %dh threshold)",
                market.condition_id[:12], hours_left, hours_to_resolution,
            )
            return None

        # 2. Return cached result if fresh
        if not force_refresh:
            cached = self._get_cached(market.condition_id)
            if cached:
                return cached

        logger.info("Analysing market: %s", market.question[:80])

        # 3. Fetch news context via Haiku — skip for price-bracket markets
        #    ("Will ETH be above $2000?") since current price is all that matters
        #    and news context won't change the analysis meaningfully.
        news_context = ""
        if self.use_web_search and not _is_price_market(market.question):
            news_context = self._fetch_news_context(market)

        # 4. Build the analysis prompt
        user_prompt = _build_user_prompt(market, news_context)

        # 5. Call Sonnet with cached system prompt for the structured analysis.
        #    CACHED_SYSTEM_PROMPT includes cache_control so Anthropic stores the
        #    processed tokens — subsequent calls hit the cache at ~10% of input cost.
        raw_text = ""
        try:
            response = self._client.messages.create(
                model=ANALYSIS_MODEL,
                max_tokens=MAX_TOKENS,
                system=CACHED_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            raw_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

            parsed = _parse_claude_response(raw_text)

        except anthropic.APIError as exc:
            logger.error(
                "Anthropic API error for %s: %s", market.condition_id[:12], exc
            )
            return None

        # 6. Build the result
        claude_prob = float(parsed.get("claude_yes_prob", 0.5))
        claude_prob = max(0.01, min(0.99, claude_prob))   # clamp to (0.01, 0.99)

        edge   = claude_prob - market.yes_mid
        action = _determine_action(edge, parsed.get("confidence", "low"))

        result = AnalysisResult(
            condition_id       = market.condition_id,
            question           = market.question,
            market_yes_prob    = market.yes_mid,
            claude_yes_prob    = claude_prob,
            confidence         = parsed.get("confidence", "low"),
            edge               = edge,
            recommended_action = action,
            reasoning          = parsed.get("reasoning", ""),
            key_factors        = parsed.get("key_factors", []),
            risks              = parsed.get("risks", []),
            news_context       = news_context,
            model              = ANALYSIS_MODEL,
        )

        # 7. Cache and return
        self._set_cache(market.condition_id, result)
        logger.info(
            "Analysis complete — %s | edge=%s | action=%s",
            market.condition_id[:12], f"{edge:+.1%}", action,
        )
        return result

    def analyse_batch(
            self,
            markets: list[MarketInfo],
            min_edge_to_log: float = 0.0,
            delay_between: float = 1.0,
    ) -> list[AnalysisResult]:
        """
        Analyse a list of markets sequentially.

        Args:
            markets:          List of MarketInfo from polymarket_client.
            min_edge_to_log:  Only log markets where |edge| >= this threshold.
            delay_between:    Seconds to wait between API calls (rate-limit courtesy).

        Returns:
            List of AnalysisResult (excludes None results from skipped markets).
        """
        results = []
        total = len(markets)

        for i, market in enumerate(markets, 1):
            logger.info("Batch progress: %d/%d — %s", i, total, market.question[:60])

            result = self.analyse(market)
            if result is None:
                continue

            results.append(result)

            if abs(result.edge) >= min_edge_to_log:
                logger.info(result.summary())

            if i < total:
                time.sleep(delay_between)

        tradeable = [r for r in results if r.is_tradeable()]
        logger.info(
            "Batch complete: %d/%d analysed, %d tradeable opportunities found",
            len(results), total, len(tradeable),
        )
        return results

    def get_top_opportunities(
            self,
            markets: list[MarketInfo],
            top_n: int = 5,
    ) -> list[AnalysisResult]:
        """
        Analyse all markets and return the top N by absolute edge,
        filtered to only tradeable results.
        """
        results = self.analyse_batch(markets, min_edge_to_log=MIN_EDGE_TO_TRADE)
        tradeable = [r for r in results if r.is_tradeable()]
        return sorted(tradeable, key=lambda r: abs(r.edge), reverse=True)[:top_n]


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from polymarket_client import PolymarketClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    pm      = PolymarketClient(read_only=True)
    analyst = ClaudeAnalyst(use_web_search=True)

    markets = pm.get_active_markets(volume_min=5_000, liquidity_min=2_000, limit=5)

    if not markets:
        print("No markets returned — check your filters.")
    else:
        print(f"\nAnalysing {len(markets)} markets…\n")
        for market in markets:
            result = analyst.analyse(market)
            if result:
                print(result.summary())
                print(f"  Reasoning snippet: {result.reasoning[:200]}…\n")