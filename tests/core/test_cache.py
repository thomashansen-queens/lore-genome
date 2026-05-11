from lore.core.cache import TieredCache
from logging import Logger

# TODO
def test_l1_cache_creation_deletion(tmp_path):
    cache = TieredCache(Logger("Test Logger"), tmp_path)