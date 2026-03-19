from .cache import get_cache_dir, get_model_cache_dir

__all__ = [
    "maddad_cache_dir",
    "model_cache_dir",
]

maddad_cache_dir = get_cache_dir()
model_cache_dir = get_model_cache_dir()
