class MktoolsError(Exception):
    """Base exception for mktools."""


class PathValidationError(MktoolsError, ValueError):
    """Raised when a path violates a validation rule."""


class UnsupportedFileTypeError(MktoolsError, ValueError):
    """Raised when no loader exists for the requested file type."""


class UnsafeArchiveError(MktoolsError, ValueError):
    """Raised when an archive contains unsafe or unsupported members."""
