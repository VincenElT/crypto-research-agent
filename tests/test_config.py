"""Unit tests for environment configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from crypto_research.config import load_settings
from crypto_research.exceptions import ConfigurationError


def test_load_settings_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EXCHANGE_ID=kraken\nSYMBOL=ETH/USDT\nTIMEFRAME=1d\nCANDLE_LIMIT=300\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EXCHANGE_ID", raising=False)
    monkeypatch.delenv("SYMBOL", raising=False)
    monkeypatch.delenv("CANDLE_LIMIT", raising=False)
    settings = load_settings(dotenv_path=str(env_file))
    assert settings.exchange_id == "kraken"
    assert settings.symbol == "ETH/USDT"
    assert settings.candle_limit == 300
    assert settings.openai_api_key_configured is False
    assert settings.max_position_size_usd == 1000.0


def test_rejects_small_candle_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CANDLE_LIMIT=50\n", encoding="utf-8")
    monkeypatch.delenv("CANDLE_LIMIT", raising=False)
    with pytest.raises(ConfigurationError, match="at least 200"):
        load_settings(dotenv_path=str(env_file))


def test_custom_risk_limits_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CANDLE_LIMIT=300\n"
        "MAX_POSITION_SIZE_USD=2500.0\n"
        "MAX_PORTFOLIO_ALLOCATION_PCT=0.20\n"
        "COOLDOWN_BARS=10\n"
        "MAX_DRAWDOWN_PCT=0.25\n"
        "MAX_VOLATILITY_PCT=0.12\n"
        "MIN_SCOUT_CONFIDENCE=0.65\n"
        "REQUIRE_SKEPTIC_APPROVAL=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CANDLE_LIMIT", raising=False)
    monkeypatch.delenv("MAX_POSITION_SIZE_USD", raising=False)
    monkeypatch.delenv("MAX_PORTFOLIO_ALLOCATION_PCT", raising=False)
    monkeypatch.delenv("COOLDOWN_BARS", raising=False)
    monkeypatch.delenv("MAX_DRAWDOWN_PCT", raising=False)
    monkeypatch.delenv("MAX_VOLATILITY_PCT", raising=False)
    monkeypatch.delenv("MIN_SCOUT_CONFIDENCE", raising=False)
    monkeypatch.delenv("REQUIRE_SKEPTIC_APPROVAL", raising=False)
    settings = load_settings(dotenv_path=str(env_file))
    assert settings.max_position_size_usd == 2500.0
    assert settings.max_portfolio_allocation_pct == 0.20
    assert settings.cooldown_bars == 10
    assert settings.max_drawdown_pct == 0.25
    assert settings.max_volatility_pct == 0.12
    assert settings.min_scout_confidence == 0.65
    assert settings.require_skeptic_approval is False

    limits = settings.to_risk_limits()
    assert limits.max_position_size_usd == 2500.0
    assert limits.require_skeptic_approval is False


def test_invalid_risk_limits_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file_alloc = tmp_path / ".env_alloc"
    env_file_alloc.write_text("CANDLE_LIMIT=300\nMAX_PORTFOLIO_ALLOCATION_PCT=1.5\n", encoding="utf-8")
    monkeypatch.delenv("CANDLE_LIMIT", raising=False)
    monkeypatch.delenv("MAX_PORTFOLIO_ALLOCATION_PCT", raising=False)
    with pytest.raises(ConfigurationError, match="MAX_PORTFOLIO_ALLOCATION_PCT"):
        load_settings(dotenv_path=str(env_file_alloc))

    monkeypatch.delenv("MAX_PORTFOLIO_ALLOCATION_PCT", raising=False)
    env_file_cd = tmp_path / ".env_cd"
    env_file_cd.write_text("CANDLE_LIMIT=300\nCOOLDOWN_BARS=-1\n", encoding="utf-8")
    monkeypatch.delenv("COOLDOWN_BARS", raising=False)
    with pytest.raises(ConfigurationError, match="COOLDOWN_BARS"):
        load_settings(dotenv_path=str(env_file_cd))
