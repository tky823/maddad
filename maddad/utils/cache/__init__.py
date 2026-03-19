# ported from https://github.com/tky823/Audyn/blob/c1aed30b3ce09d94ea76029416fe392efe9cf209/audyn/utils/cache/__init__.py
import os


def get_cache_dir() -> str:
    """Get cache directory for maddad.

    .. note::

        You can set cache directory by setting ``MADDAD_CACHE_DIR`` environment variable.

    Returns:
        str: Cache directory path.

    """

    _home_dir = os.path.expanduser("~")
    cache_dir = os.getenv("MADDAD_CACHE_DIR") or os.path.join(_home_dir, ".cache", "maddad")

    return cache_dir


def get_model_cache_dir() -> str:
    """Get model cache directory for maddad.

    Returns:
        str: Cache directory path.

    """

    cache_dir = get_cache_dir()
    model_cache_dir = os.path.join(cache_dir, "models")

    return model_cache_dir
