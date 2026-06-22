"""
Reader Registry
"""
from .base import BaseReader
import logging

logger = logging.getLogger(__name__)


class ReaderRegistry:
    """
    A registry mapping file extensions to Reader classes. Core readers are added
    via `_register_core`; plugins use the `register` decorator (exposed as
    `lore.reader`).
    """
    def __init__(self):
        # Nested map: extension: {reader_key: reader_cls}
        self._readers: dict[str, dict[str, type[BaseReader]]] = {}

    @staticmethod
    def _clean(ext: str) -> str:
        """Normalize an extension to a bare, lowercase key (e.g. '.CSV' -> 'csv')."""
        return ext.lower().strip().lstrip(".")

    def register(self, extensions: list[str] | str):
        """Decorator to register a plugin Reader class for one or more extensions."""
        def wrapper(cls: type[BaseReader]) -> type[BaseReader]:
            if not issubclass(cls, BaseReader):
                raise TypeError("Reader '{cls.__name__}' must inherit from BaseReader.")

            exts = [extensions] if isinstance(extensions, str) else extensions
            self._register_core(cls, exts)
            return cls
        return wrapper

    def _register_core(self, reader_cls: type[BaseReader], extensions: list[str]) -> None:
        """
        Private method to add core readers (e.g. text) to the registry.
        """
        self._add(reader_cls, extensions)

    def _add(self, reader_cls: type[BaseReader], extensions: list[str]) -> None:
        """Internal registry method to add a new Reader to the dict"""
        reader_key = reader_cls.__name__
        for ext in extensions:
            clean_ext = self._clean(ext)
            if clean_ext not in self._readers:
                self._readers[clean_ext] = {}
            if reader_key in self._readers[clean_ext]:
                logger.warning(
                    "Overwriting existing Reader for extension '%s' and key '%s'",
                    clean_ext, reader_key
                )
            self._readers[clean_ext][reader_key] = reader_cls

    def get_reader(self, extension: str, reader_key: str | None = None) -> type | None:
        """
        Get the Reader class registered for a given file extension, or None.
        """
        clean_ext = self._clean(extension)
        ext_readers = self._readers.get(clean_ext)

        # 1. No reader
        if not ext_readers:
            from .raw import RawReader
            return RawReader

        # 2. Explicit reader key provided
        if reader_key:
            reader_cls = ext_readers.get(reader_key)
            if not reader_cls:
                logger.warning(
                    "No Reader found for extension '%s' with key '%s'. "
                    "Available keys: %s. Falling back to RawReader.",
                    clean_ext, reader_key, list(ext_readers.keys())
                )
                from .raw import RawReader
                return RawReader
            return reader_cls

        # 3. Default routing: Grab the last registered reader for this ext
        last_key = list(ext_readers.keys())[-1]
        return ext_readers[last_key]
