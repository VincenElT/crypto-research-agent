"""Deterministic risk filters.

Python-only checks (position size, cooldown, max drawdown, spread/volatility guards).
If any check fails, the trade is rejected even if both AI agents agree.
AI agents will never override these filters. Live trading is strictly disabled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from crypto_research.exceptions import RiskFilterError
from crypto_research.indicators import IndicatorSnapshot
from crypto_research.schemas import ScoutProposal, SkepticReview


@dataclass(frozen=True)
class RiskLimits:
    """Configurable deterministic risk parameters."""

    max_position_size_usd: float = 1000.0
    max_portfolio_allocation_pct: float = 0.10
    cooldown_bars: int = 5
    max_drawdown_pct: float = 0.15
    max_volatility_pct: float = 0.08
    min_scout_confidence: float = 0.50
    require_skeptic_approval: bool = True


@dataclass(frozen=True)
class PortfolioState:
    """Simulated portfolio and history needed for deterministic risk evaluation.

    Spot Trading Semantics:
    - BUY increases position_usd and consumes cash_usd.
    - SELL reduces position_usd and returns cash_usd (closing existing long).
    - Shorting without inventory is forbidden in spot mode.
    """

    cash_usd: float = 10_000.0
    current_position_usd: float = 0.0
    peak_equity_usd: float = 10_000.0
    current_equity_usd: float = 10_000.0
    bars_since_last_trade: int = 999


@dataclass(frozen=True)
class RiskEvaluation:
    """Deterministic risk evaluation outcome."""

    passed: bool
    proposed_action: str
    approved_size_usd: float
    rejection_reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_risk(
    *,
    proposal: ScoutProposal,
    critique: SkepticReview,
    snapshot: IndicatorSnapshot,
    recent_candles: pd.DataFrame,
    portfolio: PortfolioState | None = None,
    limits: RiskLimits | None = None,
) -> RiskEvaluation:
    """Evaluate deterministic Python risk limits against closed-candle market data.

    Returns:
        RiskEvaluation containing whether all checks passed, approved size,
        and explicit rejection reasons if any check failed.
    """
    portfolio = portfolio or PortfolioState()
    limits = limits or RiskLimits()

    if recent_candles.empty:
        raise RiskFilterError("Cannot evaluate risk: candle data is empty.")

    checks: dict[str, bool] = {}
    rejection_reasons: list[str] = []
    details: dict[str, Any] = {}

    # 1. Fail-closed portfolio integrity validation
    portfolio_valid = True
    if not math.isfinite(portfolio.current_equity_usd) or portfolio.current_equity_usd <= 0:
        portfolio_valid = False
        rejection_reasons.append(
            f"Current portfolio equity ({portfolio.current_equity_usd:.2f} USD) must be positive and finite."
        )
    if not math.isfinite(portfolio.peak_equity_usd) or portfolio.peak_equity_usd <= 0:
        portfolio_valid = False
        rejection_reasons.append(
            f"Peak portfolio equity ({portfolio.peak_equity_usd:.2f} USD) must be positive and finite."
        )
    if not math.isfinite(portfolio.cash_usd) or portfolio.cash_usd < 0:
        portfolio_valid = False
        rejection_reasons.append(
            f"Portfolio cash ({portfolio.cash_usd:.2f} USD) cannot be negative or non-finite."
        )
    if not math.isfinite(portfolio.current_position_usd) or portfolio.current_position_usd < 0:
        portfolio_valid = False
        rejection_reasons.append(
            f"Portfolio position ({portfolio.current_position_usd:.2f} USD) cannot be negative or non-finite in spot mode."
        )
    checks["portfolio_state_valid"] = portfolio_valid

    # 2. Fail-closed market price integrity validation
    price_valid = math.isfinite(snapshot.close) and snapshot.close > 0
    checks["price_valid"] = price_valid
    if not price_valid:
        rejection_reasons.append(
            f"Reference close price ({snapshot.close}) must be positive and finite."
        )

    # 3. Action tradability
    if proposal.action == "HOLD":
        checks["action_tradable"] = False
        rejection_reasons.append("Scout proposed HOLD; no trade action to execute.")
    elif proposal.action in ("BUY", "SELL"):
        checks["action_tradable"] = True
    else:
        checks["action_tradable"] = False
        rejection_reasons.append(f"Unsupported action: {proposal.action}")

    # 4. Skeptic critique requirement
    if limits.require_skeptic_approval:
        skeptic_passed = critique.verdict == "APPROVE" and critique.agrees_with_action
        checks["skeptic_approved"] = skeptic_passed
        if not skeptic_passed:
            rejection_reasons.append(
                f"Skeptic did not approve proposal (verdict={critique.verdict!r}, "
                f"agrees_with_action={critique.agrees_with_action})."
            )
    else:
        checks["skeptic_approved"] = True

    # 5. Minimum Scout confidence guard
    confidence_passed = proposal.confidence >= limits.min_scout_confidence
    checks["confidence_threshold_met"] = confidence_passed
    if not confidence_passed:
        rejection_reasons.append(
            f"Scout confidence {proposal.confidence:.2f} is below minimum {limits.min_scout_confidence:.2f}."
        )

    # 6. Trade cooldown guard
    cooldown_passed = portfolio.bars_since_last_trade >= limits.cooldown_bars
    checks["cooldown_passed"] = cooldown_passed
    details["bars_since_last_trade"] = portfolio.bars_since_last_trade
    if not cooldown_passed:
        rejection_reasons.append(
            f"Cooldown active: {portfolio.bars_since_last_trade} bars since last trade "
            f"(required: {limits.cooldown_bars})."
        )

    # 7. Portfolio drawdown guard
    if portfolio.peak_equity_usd > 0:
        drawdown = max(0.0, (portfolio.peak_equity_usd - portfolio.current_equity_usd) / portfolio.peak_equity_usd)
    else:
        drawdown = 1.0
    details["drawdown_pct"] = drawdown
    drawdown_passed = drawdown <= limits.max_drawdown_pct
    checks["drawdown_within_limit"] = drawdown_passed
    if not drawdown_passed:
        rejection_reasons.append(
            f"Drawdown {drawdown:.2%} exceeds maximum allowed limit {limits.max_drawdown_pct:.2%}."
        )

    # 8. Bar spread / volatility guard on timestamp-aligned candle
    aligned_candles = recent_candles[recent_candles["timestamp"] <= snapshot.as_of]
    if aligned_candles.empty or aligned_candles.iloc[-1]["timestamp"] != snapshot.as_of:
        raise RiskFilterError(
            f"Recent candles do not align with snapshot as_of timestamp {snapshot.as_of.isoformat()}."
        )
    last_candle = aligned_candles.iloc[-1]
    candle_close = float(last_candle["close"])
    candle_high = float(last_candle["high"])
    candle_low = float(last_candle["low"])

    if (
        not math.isfinite(candle_close)
        or not math.isfinite(candle_high)
        or not math.isfinite(candle_low)
        or candle_close <= 0
        or candle_high <= 0
        or candle_low <= 0
        or candle_high < candle_low
    ):
        checks["volatility_within_limit"] = False
        rejection_reasons.append(
            f"Invalid candle price values on bar (close={candle_close}, high={candle_high}, low={candle_low})."
        )
    else:
        candle_range = (candle_high - candle_low) / candle_close
        details["candle_range_pct"] = candle_range
        volatility_passed = candle_range <= limits.max_volatility_pct
        checks["volatility_within_limit"] = volatility_passed
        if not volatility_passed:
            rejection_reasons.append(
                f"Market volatility {candle_range:.2%} exceeds maximum threshold {limits.max_volatility_pct:.2%}."
            )

    # 9. Position sizing calculation & Spot Trading semantics
    approved_size_usd = 0.0
    if proposal.action == "SELL":
        if portfolio.current_position_usd <= 0.0:
            checks["has_position_to_sell"] = False
            rejection_reasons.append(
                "Cannot execute SELL: no open long position exists to close in spot mode."
            )
        else:
            checks["has_position_to_sell"] = True
            approved_size_usd = min(limits.max_position_size_usd, portfolio.current_position_usd)
            checks["capital_available"] = True
    elif proposal.action == "BUY":
        if portfolio.current_equity_usd > 0:
            max_allowed_position_usd = portfolio.current_equity_usd * limits.max_portfolio_allocation_pct
            remaining_capacity_usd = max(0.0, max_allowed_position_usd - portfolio.current_position_usd)
            details["max_allowed_position_usd"] = max_allowed_position_usd
            details["remaining_capacity_usd"] = remaining_capacity_usd

            if remaining_capacity_usd <= 0.0:
                checks["allocation_capacity_available"] = False
                rejection_reasons.append(
                    f"Position allocation limit reached: current position {portfolio.current_position_usd:.2f} USD "
                    f"already meets or exceeds max allocation limit {max_allowed_position_usd:.2f} USD "
                    f"({limits.max_portfolio_allocation_pct:.1%})."
                )
            else:
                checks["allocation_capacity_available"] = True
                target_size_usd = min(limits.max_position_size_usd, remaining_capacity_usd)
                details["calculated_target_size_usd"] = target_size_usd

                if target_size_usd > portfolio.cash_usd:
                    checks["capital_available"] = False
                    rejection_reasons.append(
                        f"Insufficient cash ({portfolio.cash_usd:.2f} USD) for target size {target_size_usd:.2f} USD."
                    )
                else:
                    checks["capital_available"] = True
                    approved_size_usd = target_size_usd

    if approved_size_usd <= 0.0 and len(rejection_reasons) == 0:
        rejection_reasons.append("Approved trade size must be positive.")

    passed = len(rejection_reasons) == 0
    final_size_usd = approved_size_usd if passed else 0.0

    return RiskEvaluation(
        passed=passed,
        proposed_action=proposal.action,
        approved_size_usd=final_size_usd,
        rejection_reasons=rejection_reasons,
        checks=checks,
        details=details,
    )
