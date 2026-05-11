from lore.core.cache import TieredCache
from logging import Logger

def test_l1_cache_creation_deletion():
    cache = TieredCache(Logger())