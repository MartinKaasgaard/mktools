"""Repository browsing and download helpers.

Typical usage
-------------

>>> from mktools import repo
>>> tree = repo.browse("https://github.com/alecuba16/fuhrlander", ref="master", path="dataset", recursive=True)
>>> plan = repo.select(tree, target="dir", path="dataset")
>>> result = repo.get(plan, dest="data/fuhrlander")
"""

from .api import browse, fetch, get, plan, select
from .errors import RepoError, RepoIntegrityError, RepoNotFoundError, RepoParseError, RepoProviderError
from .models import FetchResult, RepoEntry, RepoSpec, RepoTree, SelectionPlan

__all__ = [
    "browse",
    "select",
    "plan",
    "get",
    "fetch",
    "RepoError",
    "RepoIntegrityError",
    "RepoNotFoundError",
    "RepoParseError",
    "RepoProviderError",
    "FetchResult",
    "RepoEntry",
    "RepoSpec",
    "RepoTree",
    "SelectionPlan",
]
