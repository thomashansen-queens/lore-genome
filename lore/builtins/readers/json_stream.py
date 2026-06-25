"""
JSON via ijson (streaming enabled).

stdlib json parser is all-or-nothing, so it raises NotImplementedError when
trying to stream json. ijson streams top-level JSON arrays, keeping large
datasets out of RAM. JSON Lines and other shapes (e.g. a single top-level
object) defer to the inherited TableReader behavior.
"""
import ijson

import lore
from lore.core.readers.table import TableReader


@lore.reader(extensions=["json"])
class JsonStreamReader(TableReader):
    """
    Streams top-level JSON array elements with ijson.
    Inherits the rest from TableReader.
    """
    def get_metadata(self) -> dict:
        base_meta = self.get_base_metadata()
        base_meta["can_stream"] = True
        return base_meta

    def stream(self, config: dict | None = None, **kwargs):
        # ijson.items(f, "item") silently yields nothing for a non-array, so
        # confirm a top-level array first; defer other shapes (single object,
        # etc.) to read_full() via the NotImplementedError contract.
        if self._first_nonspace_byte() != b"[":
            raise NotImplementedError(
                f"'{self.path.name}' is not a JSON array; read_full() handles it."
            )

        with open(self.path, "rb") as f:
            yield from ijson.items(f, "item")

    def _first_nonspace_byte(self) -> bytes:
        """Peek the first non-whitespace byte without reading the whole file."""
        with open(self.path, "rb") as f:
            while chunk := f.read(4096):
                stripped = chunk.lstrip()
                if stripped:
                    return stripped[:1]
        return b""
