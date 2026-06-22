"""
Reader Registry
"""


class ReaderRegistry:
    """
    A registry mapping file extensions to Reader classes. Core readers are added
    via `_register_core`; plugins use the `register` decorator (exposed as
    `lore.reader`).
    """
    def __init__(self):
        self._readers: dict[str, type] = {}

    @staticmethod
    def _clean(ext: str) -> str:
        """Normalize an extension to a bare, lowercase key (e.g. '.CSV' -> 'csv')."""
        return ext.lower().strip().lstrip(".")

    def register(self, extensions: list[str] | str):
        """Decorator to register a plugin Reader class for one or more extensions."""
        def decorator(cls):
            exts = [extensions] if isinstance(extensions, str) else extensions
            for ext in exts:
                self._readers[self._clean(ext)] = cls
            return cls
        return decorator

    def _register_core(self, reader_cls: type, extensions: list[str]) -> None:
        """
        Protected internal method to add core readers (e.g. text) to the registry.
        """
        for ext in extensions:
            self._readers[self._clean(ext)] = reader_cls

    def get_reader(self, extension: str) -> type | None:
        """
        Get the Reader class registered for a given file extension, or None.
        """
        return self._readers.get(self._clean(extension))
