import os


def get_cache_dir() -> str:
    _home_dir = os.path.expanduser("~")
    cache_dir = os.getenv("MADDAD_TESTING_CACHE_DIR")
    cache_dir = cache_dir or os.path.join(_home_dir, ".cache", "maddad_testing")

    return cache_dir
