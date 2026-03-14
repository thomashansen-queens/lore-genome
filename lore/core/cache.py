"""
Two-tier LRU (least recently used) cache for compute-expensive results.
"""

import functools
import hashlib
import inspect
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
from typing import Callable, OrderedDict, Any
import zipfile
import logging


class TieredCache:
    """
    Two-tier LRU (least recently used) cache for compute-expensive results.
    L1: In-memory cache for storing compute-expensive results during Task 
    execution. Allows quick retrieval of already-computed results for preview 
    and/or rapid iteration. Evicts oldest cached items.
    L2: Content-addressable storage (CAS) on disk for sharing between Workers.
    NOTE: Not currently thread-safe. Will improve in the future.
    """
    def __init__(
        self,
        logger: logging.Logger,
        cache_dir: Path,
        max_bytes: int = 500 * 1024 * 1024,
        max_disk: int = 5 * 1024 * 1024 * 1024,
    ):
        # RAM cache: A simple dictionary living in server RAM.
        self.logger = logger
        self.dir = cache_dir
        self._max_bytes = max_bytes
        self._current_bytes = 0
        self._store: OrderedDict[str, Any] = OrderedDict()

        # CAS cache: A directory containing files named by their content hash
        # TODO: Implement disk eviction algorithm (e.g. check cas_dir size and delete oldest .pkl)
        self._max_disk = max_disk
        self.cas_dir = cache_dir / "cas"
        self.cas_dir.mkdir(parents=True, exist_ok=True)

    def _estimate_size(self, obj: Any) -> int:
        """Accurately weighs strings, primitive, and common data structures"""
        # 1. Pandas DataFrames
        if hasattr(obj, "memory_usage"):
            try:
                # deep=True looks inside actual object sizes
                return int(obj.memory_usage(deep=True).sum())
            except Exception as e:
                pass

        # 2. Handle iterables (shallow estimate)
        if isinstance(obj, (list, tuple, set)):
            return sys.getsizeof(obj) + sum(sys.getsizeof(i) for i in obj)
        if isinstance(obj, dict):
            return sys.getsizeof(obj) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())

        # 3. Fallback for primitives
        return sys.getsizeof(obj)

    def _make_key(self, session_id: str, prefix: str, kwargs: dict) -> str | None:
        """Creates a unique fingerprint for this exact function call."""
        try:
            # Serialize data into a pure string. sort_keys guarantees that same 
            # produces same regardless of dict key order
            serialized = json.dumps(kwargs, sort_keys=True)
        except TypeError as e:
            self.logger.warning("String-serializing uncacheable kwargs for %s: %s", prefix, e)
            serialized = str(kwargs)

        key_raw = f"{session_id}_{prefix}_{serialized}"
        return hashlib.md5(key_raw.encode()).hexdigest()

    def _promote_to_ram(self, key: str, result: Any) -> None:
        """Helper to store results in RAM and enfore size limits."""
        item_size = self._estimate_size(result)
        self._store[key] = result
        self._current_bytes += item_size
        self._store.move_to_end(key)

        # Evict until we are back under the memory limit
        while self._current_bytes > self._max_bytes and self._store:
            evicted_key, evicted_result = self._store.popitem(last=False)
            evicted_size = self._estimate_size(evicted_result)
            self._current_bytes -= evicted_size
            self.logger.debug("L1 cache evicted '%s' freeing %d bytes",
                              evicted_key, evicted_size)

        return result

    def get_or_compute(
        self,
        session_id: str,
        prefix: str,
        compute_fn: Callable,
        cache_kwargs: dict,
    ) -> Any:
        """
        Check L1 (RAM) > L2 (Disk) > Compute and cache. Evict if needed.
        Usage: FUTURE: Need to write a usage example here
        """
        key = self._make_key(session_id, prefix, cache_kwargs)
        if key is None:
            return compute_fn()  # uncacheable, just compute

        # 1. L1 (RAM) cache hit
        if key in self._store:
            self.logger.debug("Cache HIT (RAM) for key: %s", key)
            self._store.move_to_end(key)
            return self._store[key]

        # 2. L2 (Disk) cache hit
        cas_file = self.cas_dir / f"{key}.pkl"
        if cas_file.exists():
            self.logger.debug("Cache HIT (Disk) for key: %s", key)
            try:
                with open(cas_file, "rb") as f:
                    result = pickle.load(f)

                # Promote to L1 RAM
                self._promote_to_ram(key, result)
                return result
            except Exception as e:
                self.logger.warning("Failed to load L2 (CAS) for key %s: %s", key, e)

        # 3. Cache miss - compute result
        self.logger.debug("Cache MISS for %s. Computing...", prefix)
        result = compute_fn()

        # 4. Store in L2 (CAS) cache
        try:
            with open(cas_file, "wb") as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.warning("Failed to save L2 (CAS) for key %s: %s", key, e)

        # 5. Save to L1 (RAM) cache
        self._promote_to_ram(key, result)

        return result

    def get_stats(self) -> dict:
        """Returns aggregate L1 and L2 cache stats."""
        pkl_files = list(self.cas_dir.glob("*.pkl"))
        return {
            "l1_count": len(self._store),
            "l1_bytes": self._current_bytes,
            "l1_max_bytes": self._max_bytes,
            "l2_count": len(pkl_files),
            "l2_bytes": sum(f.stat().st_size for f in pkl_files),
            "l2_max_bytes": self._max_disk,
            "cas_dir": self.cas_dir,
        }

    def inspect_cas_item(self, key: str) -> dict:
        """
        Loads the pickle for a CAS entry and returns introspective metadata.
        Does not store the result in L1 — read-only inspection only.
        """
        cas_file = (self.cas_dir / f"{key}.pkl").resolve()
        if not cas_file.exists():
            return {"error": f"No CAS entry found for key: {key}"}
        if not cas_file.is_relative_to(self.cas_dir.resolve()):
            return {"error": f"Invalid key: {key}"}

        try:
            with open(cas_file, "rb") as f:
                obj = pickle.load(f)
        except Exception as e:
            return {"error": str(e)}

        info: dict = {
            "error": None,
            "type": type(obj).__qualname__,
            "module": type(obj).__module__,
        }

        # Length (lists, dicts, strings, DataFrames, etc.)
        if hasattr(obj, "__len__"):
            try:
                info["length"] = len(obj)
            except Exception:
                pass

        # Dict: show first 20 keys
        if isinstance(obj, dict):
            info["keys"] = list(obj.keys())[:20]
            info["keys_truncated"] = len(obj) > 20

        # List/tuple: element type
        elif isinstance(obj, (list, tuple)) and obj:
            info["element_type"] = type(obj[0]).__qualname__

        # Array-like shape (numpy, pandas DataFrame/Series)
        shape = getattr(obj, "shape", None)
        if isinstance(shape, tuple):
            info["shape"] = shape

        # DataFrame columns + dtypes
        columns = getattr(obj, "columns", None)
        if columns is not None:
            info["columns"] = list(columns)

        dtypes = getattr(obj, "dtypes", None)
        if dtypes is not None:
            info["dtypes"] = {str(k): str(v) for k, v in dtypes.items()}

        # Zip file check
        if isinstance(obj, bytes):
            try:
                with zipfile.ZipFile(io.BytesIO(obj)) as zf:
                    info["zip_namelist"] = zf.namelist()
            except zipfile.BadZipFile:
                pass

        # Repr preview (capped to avoid rendering huge strings)
        try:
            info["repr_preview"] = repr(obj)[:400]
        except Exception:
            info["repr_preview"] = None

        return info

    def get_cas_item(self, key: str) -> dict | None:
        """Returns file-stat metadata for a single CAS entry, without opening it."""
        cas_file = self.cas_dir / f"{key}.pkl"
        if not cas_file.exists():
            return None
        stat = cas_file.stat()
        return {
            "key": key,
            "path": cas_file,
            "size_bytes": stat.st_size,
            "last_modified": stat.st_mtime,
            "last_accessed": stat.st_atime,
            "created": stat.st_ctime,
            "in_ram": key in self._store,
        }

    def list_cas_items(self) -> list[dict]:
        """Returns all L2 CAS items with metadata, sorted newest first."""
        items = []
        for file in self.cas_dir.glob("*.pkl"):
            stat = file.stat()
            items.append({
                "key": file.stem,
                "size_bytes": stat.st_size,
                "last_modified": stat.st_mtime,
                "in_ram": file.stem in self._store,
            })
        return sorted(items, key=lambda x: x["last_modified"], reverse=True)

    def clear_cache(self):
        """Nukes the memoization CAS directory. Use with caution!"""
        path = self.cas_dir.resolve()

        expected_path = self.dir.resolve() / "cas"
        if path != expected_path:
            self.logger.error(
                "Cache clear attempted on unexpected path: %s. This is two-key "
                "nuclear system: If you change the cache directory, you must " \
                "also update the expected_path var in clear_cache to prevent " \
                "accidents.", path
            )
            return

        # Clear L1 RAM cache
        self._store.clear()
        self._current_bytes = 0

        # Clear L2 disk cache and recreate the directory
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("🧹 Cache cleared at %s", path.absolute())


def memoize(prefix: str | None = None, ignore: str | list[str] | None = None):
    """
    Decorator to cache compute-expensive helper functions in a Task.
    The decorated function MUST accept `ctx` as its first argument.

    Args:
        prefix: A string prefix to namespace this cache entry. If None, the function name will be used.
        ignore: A list of kwarg names to exclude from hashing. Useful for hashable args (e.g. file handles, DataFrames).

    Usage:
        @memoize(prefix="my_helper", ignore="arg2")
        def my_helper(ctx, arg1, arg2):
            # Scary expensive calculation here
            return result
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(ctx, *args, **kwargs):
            actual_prefix = prefix or func.__name__

            # 1. Bind all arguments to figure out exactly what the user passed
            sig = inspect.signature(func)
            bound = sig.bind(ctx, *args, **kwargs)
            bound.apply_defaults()

            # 2. Strip out 'ctx' from the kwargs we send to the hasher
            ignore_list = ignore or []
            if isinstance(ignore_list, str):
                ignore_list = [ignore_list]
            cache_kwargs = {
                k: v for k, v in bound.arguments.items()
                if k != 'ctx' and k not in ignore_list
            }

            # 3. Create a zero-argument thunk to execute the function if it's a cache miss
            def _compute_thunk():
                return func(ctx, *args, **kwargs)

            # 4. Hand off to the TieredCache
            return ctx.runtime.cache.get_or_compute(
                session_id=ctx.session_id,
                prefix=actual_prefix,
                compute_fn=_compute_thunk,
                cache_kwargs=cache_kwargs,
            )
        return wrapper
    return decorator
