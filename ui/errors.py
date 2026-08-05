"""Custom exception types for UI and backend connectivity."""

class UIError(Exception):
    """Base exception class for UI errors."""
    pass

class BackendConnectionError(UIError):
    """Raised when connecting to backend fails."""
    pass

class BackendTimeoutError(UIError):
    """Raised when backend response times out."""
    pass
