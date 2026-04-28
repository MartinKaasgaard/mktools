"""Exceptions used by :mod:`mktools.repo`."""


class RepoError(Exception):
    """Base error for mktools.repo."""


class RepoParseError(RepoError):
    """Repository URL or spec could not be parsed."""


class RepoProviderError(RepoError):
    """Provider-specific failure."""


class RepoNotFoundError(RepoError):
    """Repository, path, or file was not found."""


class RepoIntegrityError(RepoError):
    """Downloaded content failed integrity validation."""
