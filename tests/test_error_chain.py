"""Tests for flattening nested network/TLS error messages."""

from __future__ import annotations

from crypto_research.market_data import _exception_chain_text


def test_exception_chain_includes_nested_ssl_message() -> None:
    root = ValueError("certificate verify failed")
    wrapped = RuntimeError("binance GET https://api.binance.com")
    wrapped.__cause__ = root
    text = _exception_chain_text(wrapped)
    assert "binance GET" in text
    assert "certificate verify failed" in text
