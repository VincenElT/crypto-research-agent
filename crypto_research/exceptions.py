"""Domain errors for market data and research pipeline failures."""


class CryptoResearchError(Exception):
    """Base error for this project."""


class ConfigurationError(CryptoResearchError):
    """Raised when environment configuration is invalid."""


class MarketDataError(CryptoResearchError):
    """Raised when public market data cannot be fetched or is unusable."""


class InsufficientHistoryError(MarketDataError):
    """Raised when there are not enough closed candles for indicators."""


class AgentResponseError(CryptoResearchError):
    """Raised when an AI agent call fails or fails schema validation."""


class RiskFilterError(CryptoResearchError):
    """Raised when deterministic risk evaluation encounters invalid input."""


class PaperTradingError(CryptoResearchError):
    """Raised when paper-trade candidate generation fails."""


class StorageError(CryptoResearchError):
    """Raised when SQLite storage operations fail."""
