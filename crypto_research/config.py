"""Load configuration from environment variables.

Secrets are read from `.env` and must never be printed. Exchange API keys
are intentionally not part of this version: only public market data is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from crypto_research.exceptions import ConfigurationError

if TYPE_CHECKING:
    from crypto_research.risk import RiskLimits

DEFAULT_EXCHANGE_ID = "binance"
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "1d"
DEFAULT_CANDLE_LIMIT = 400
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_SQLITE_DB_PATH = "crypto_research.db"
MIN_CANDLES_FOR_SMA_200 = 200

DEFAULT_MAX_POSITION_SIZE_USD = 1000.0
DEFAULT_MAX_PORTFOLIO_ALLOCATION_PCT = 0.10
DEFAULT_MIN_TRADE_SIZE_USD = 10.0
DEFAULT_COOLDOWN_BARS = 5
DEFAULT_MAX_DRAWDOWN_PCT = 0.15
DEFAULT_MAX_VOLATILITY_PCT = 0.08
DEFAULT_MIN_SCOUT_CONFIDENCE = 0.50
DEFAULT_REQUIRE_SKEPTIC_APPROVAL = True

DEFAULT_PAPER_INITIAL_CASH_USD = 10_000.0
DEFAULT_PAPER_FEE_RATE = 0.001
DEFAULT_PAPER_SLIPPAGE_PCT = 0.0005
DEFAULT_PAPER_ACCOUNT_ID = "default"


@dataclass(frozen=True)
class Settings:
    """Runtime settings for market-data research and risk parameters."""

    exchange_id: str
    symbol: str
    timeframe: str
    candle_limit: int
    openai_model: str
    openai_api_key_configured: bool
    sqlite_db_path: str = DEFAULT_SQLITE_DB_PATH
    max_position_size_usd: float = DEFAULT_MAX_POSITION_SIZE_USD
    max_portfolio_allocation_pct: float = DEFAULT_MAX_PORTFOLIO_ALLOCATION_PCT
    min_trade_size_usd: float = DEFAULT_MIN_TRADE_SIZE_USD
    cooldown_bars: int = DEFAULT_COOLDOWN_BARS
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT
    max_volatility_pct: float = DEFAULT_MAX_VOLATILITY_PCT
    min_scout_confidence: float = DEFAULT_MIN_SCOUT_CONFIDENCE
    require_skeptic_approval: bool = DEFAULT_REQUIRE_SKEPTIC_APPROVAL
    paper_initial_cash_usd: float = DEFAULT_PAPER_INITIAL_CASH_USD
    paper_fee_rate: float = DEFAULT_PAPER_FEE_RATE
    paper_slippage_pct: float = DEFAULT_PAPER_SLIPPAGE_PCT
    paper_account_id: str = DEFAULT_PAPER_ACCOUNT_ID

    def to_risk_limits(self) -> RiskLimits:
        """Construct a validated RiskLimits instance from settings."""
        from crypto_research.risk import RiskLimits

        return RiskLimits(
            max_position_size_usd=self.max_position_size_usd,
            max_portfolio_allocation_pct=self.max_portfolio_allocation_pct,
            min_trade_size_usd=self.min_trade_size_usd,
            cooldown_bars=self.cooldown_bars,
            max_drawdown_pct=self.max_drawdown_pct,
            max_volatility_pct=self.max_volatility_pct,
            min_scout_confidence=self.min_scout_confidence,
            require_skeptic_approval=self.require_skeptic_approval,
        )


def _parse_bool(value: str, default: bool) -> bool:
    v = value.strip().lower()
    if not v:
        return default
    if v in ("true", "1", "yes", "t"):
        return True
    if v in ("false", "0", "no", "f"):
        return False
    raise ConfigurationError(f"Invalid boolean setting value: {value!r}")


def _parse_float(name: str, raw: str, default: float) -> float:
    if not raw.strip():
        return default
    try:
        val = float(raw)
        return val
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a float, got {raw!r}.") from exc


def _parse_int(name: str, raw: str, default: int) -> int:
    if not raw.strip():
        return default
    try:
        val = int(raw)
        return val
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}.") from exc


def load_settings(*, dotenv_path: str | None = None) -> Settings:
    """Load settings from the environment after optionally reading a `.env` file.

    Args:
        dotenv_path: Optional path to a `.env` file. If omitted, python-dotenv
            searches the current working directory.

    Returns:
        Parsed settings. The OpenAI key itself is never stored on this object
        so it cannot be accidentally logged.

    Raises:
        ConfigurationError: If numeric settings are invalid.
    """

    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=True)
    else:
        load_dotenv()

    candle_limit = _parse_int("CANDLE_LIMIT", os.getenv("CANDLE_LIMIT", str(DEFAULT_CANDLE_LIMIT)), DEFAULT_CANDLE_LIMIT)
    if candle_limit < MIN_CANDLES_FOR_SMA_200:
        raise ConfigurationError(
            f"CANDLE_LIMIT must be at least {MIN_CANDLES_FOR_SMA_200} so SMA 200 "
            f"can be computed. Got {candle_limit}."
        )

    max_position_size = _parse_float(
        "MAX_POSITION_SIZE_USD",
        os.getenv("MAX_POSITION_SIZE_USD", str(DEFAULT_MAX_POSITION_SIZE_USD)),
        DEFAULT_MAX_POSITION_SIZE_USD,
    )
    if max_position_size <= 0:
        raise ConfigurationError(f"MAX_POSITION_SIZE_USD must be positive, got {max_position_size}.")

    max_allocation = _parse_float(
        "MAX_PORTFOLIO_ALLOCATION_PCT",
        os.getenv("MAX_PORTFOLIO_ALLOCATION_PCT", str(DEFAULT_MAX_PORTFOLIO_ALLOCATION_PCT)),
        DEFAULT_MAX_PORTFOLIO_ALLOCATION_PCT,
    )
    if not (0.0 < max_allocation <= 1.0):
        raise ConfigurationError(f"MAX_PORTFOLIO_ALLOCATION_PCT must be in (0, 1], got {max_allocation}.")

    min_trade_size = _parse_float(
        "MIN_TRADE_SIZE_USD",
        os.getenv("MIN_TRADE_SIZE_USD", str(DEFAULT_MIN_TRADE_SIZE_USD)),
        DEFAULT_MIN_TRADE_SIZE_USD,
    )
    if min_trade_size < 0:
        raise ConfigurationError(f"MIN_TRADE_SIZE_USD cannot be negative, got {min_trade_size}.")

    cooldown = _parse_int(
        "COOLDOWN_BARS",
        os.getenv("COOLDOWN_BARS", str(DEFAULT_COOLDOWN_BARS)),
        DEFAULT_COOLDOWN_BARS,
    )
    if cooldown < 0:
        raise ConfigurationError(f"COOLDOWN_BARS cannot be negative, got {cooldown}.")

    max_drawdown = _parse_float(
        "MAX_DRAWDOWN_PCT",
        os.getenv("MAX_DRAWDOWN_PCT", str(DEFAULT_MAX_DRAWDOWN_PCT)),
        DEFAULT_MAX_DRAWDOWN_PCT,
    )
    if not (0.0 < max_drawdown <= 1.0):
        raise ConfigurationError(f"MAX_DRAWDOWN_PCT must be in (0, 1], got {max_drawdown}.")

    max_volatility = _parse_float(
        "MAX_VOLATILITY_PCT",
        os.getenv("MAX_VOLATILITY_PCT", str(DEFAULT_MAX_VOLATILITY_PCT)),
        DEFAULT_MAX_VOLATILITY_PCT,
    )
    if max_volatility <= 0:
        raise ConfigurationError(f"MAX_VOLATILITY_PCT must be positive, got {max_volatility}.")

    min_confidence = _parse_float(
        "MIN_SCOUT_CONFIDENCE",
        os.getenv("MIN_SCOUT_CONFIDENCE", str(DEFAULT_MIN_SCOUT_CONFIDENCE)),
        DEFAULT_MIN_SCOUT_CONFIDENCE,
    )
    if not (0.0 <= min_confidence <= 1.0):
        raise ConfigurationError(f"MIN_SCOUT_CONFIDENCE must be in [0, 1], got {min_confidence}.")

    require_skeptic = _parse_bool(
        os.getenv("REQUIRE_SKEPTIC_APPROVAL", str(DEFAULT_REQUIRE_SKEPTIC_APPROVAL)),
        DEFAULT_REQUIRE_SKEPTIC_APPROVAL,
    )

    paper_cash = _parse_float(
        "PAPER_INITIAL_CASH_USD",
        os.getenv("PAPER_INITIAL_CASH_USD", str(DEFAULT_PAPER_INITIAL_CASH_USD)),
        DEFAULT_PAPER_INITIAL_CASH_USD,
    )
    if paper_cash <= 0:
        raise ConfigurationError(f"PAPER_INITIAL_CASH_USD must be positive, got {paper_cash}.")

    paper_fee = _parse_float(
        "PAPER_FEE_RATE",
        os.getenv("PAPER_FEE_RATE", str(DEFAULT_PAPER_FEE_RATE)),
        DEFAULT_PAPER_FEE_RATE,
    )
    if not (0.0 <= paper_fee < 1.0):
        raise ConfigurationError(f"PAPER_FEE_RATE must be in [0, 1), got {paper_fee}.")

    paper_slippage = _parse_float(
        "PAPER_SLIPPAGE_PCT",
        os.getenv("PAPER_SLIPPAGE_PCT", str(DEFAULT_PAPER_SLIPPAGE_PCT)),
        DEFAULT_PAPER_SLIPPAGE_PCT,
    )
    if not (0.0 <= paper_slippage < 1.0):
        raise ConfigurationError(f"PAPER_SLIPPAGE_PCT must be in [0, 1), got {paper_slippage}.")

    paper_account = os.getenv("PAPER_ACCOUNT_ID", DEFAULT_PAPER_ACCOUNT_ID).strip() or DEFAULT_PAPER_ACCOUNT_ID

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    return Settings(
        exchange_id=os.getenv("EXCHANGE_ID", DEFAULT_EXCHANGE_ID).strip() or DEFAULT_EXCHANGE_ID,
        symbol=os.getenv("SYMBOL", DEFAULT_SYMBOL).strip() or DEFAULT_SYMBOL,
        timeframe=os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME).strip() or DEFAULT_TIMEFRAME,
        candle_limit=candle_limit,
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL,
        openai_api_key_configured=bool(openai_key),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", DEFAULT_SQLITE_DB_PATH).strip() or DEFAULT_SQLITE_DB_PATH,
        max_position_size_usd=max_position_size,
        max_portfolio_allocation_pct=max_allocation,
        min_trade_size_usd=min_trade_size,
        cooldown_bars=cooldown,
        max_drawdown_pct=max_drawdown,
        max_volatility_pct=max_volatility,
        min_scout_confidence=min_confidence,
        require_skeptic_approval=require_skeptic,
        paper_initial_cash_usd=paper_cash,
        paper_fee_rate=paper_fee,
        paper_slippage_pct=paper_slippage,
        paper_account_id=paper_account,
    )
