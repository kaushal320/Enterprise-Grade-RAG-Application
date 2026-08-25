"""Custom exception types for UI and backend connectivity."""

class UIError(Exception):
    """Base exception class for UI errors."""

class BackendConnectionError(UIError):
    """Raised when connecting to backend fails."""

class BackendTimeoutError(UIError):
    """Raised when backend response times out."""
