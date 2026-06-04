"""
The AdapterRegistry is a global registry for used by LoRē find available
adapters, based on what data needs to go into and out of them.
"""

import logging
from typing import Callable, TYPE_CHECKING, TypeVar

from lore.core.adapters.base import BaseAdapter

if TYPE_CHECKING:
    from lore.core.artifacts import Artifact
    from lore.core.topology.traits import DataTrait

logger = logging.getLogger(__name__)


C = TypeVar("C", bound=type["BaseAdapter"])


class AdapterRegistry:
    """
    Global registry for Adapters. Tasks can query this to find an Adapter for
    a given Artifact and data type.

    - Use the `register` method as a decorator to add new Adapters
    - Use the `get`, `get_for_artifact`, or `get_for_type` methods to access

    This registry is instantiated as a global singleton 'adapter_registry"
    which itself contains a dict of instantiated singleton Adapters. This
    actually allows adapters to maintain internal state if needed, though
    there is no current application for this.
    """

    def __init__(self):
        self._adapters: dict[str, BaseAdapter] = {}

    def __getitem__(self, key: str) -> BaseAdapter:
        """Allows dict-like access to Adapter definitions"""
        if key not in self._adapters:
            raise KeyError(f"Adapter with key '{key}' not found.")
        return self._adapters[key]

    def register(self, cls: C | None = None, *, key: str | None = None) -> Callable[[C], C] | C:
        """
        Class decorator for registering Adapters.
        Supports both naked (@adapter) and called (@adapter()) usage.

        Example of a naked decorator applying default registration logic:
            @adapter_registry
            class MyAdapter(BaseAdapter):
                ...

        Example of a called decorator, specifying a custom key:
            @adapter_registry(key="Adapter for Tom's old sequencing machine")
            class AncientSangerReadsAdapter(BaseAdapter):
                ...
        """

        def wrapper(adapter_cls: C) -> C:
            if not issubclass(adapter_cls, BaseAdapter):
                raise TypeError(
                    f"Cannot register {adapter_cls.__name__}. "
                    f"Adapters must inherit from BaseAdapter."
                )

            # Instantiate, register, and optionally override default key
            self.add(adapter_cls(), custom_key=key)
            return adapter_cls

        # If called with parentheses, return the wrapper
        if cls is None:
            return wrapper

        # If naked decorator, cls is passed directly
        return wrapper(cls)

    def add(self, adapter: BaseAdapter, custom_key: str | None = None) -> None:
        """
        Manually register an instance of an Adapter. Used internally for
        fundamental adapters like CsvAdapter. For plug-ins, the decorator
        is more ergonomic.
        """
        adapter_key = custom_key or adapter.__class__.__name__

        if not (adapter.accepted_types or adapter.accepted_formats):
            logger.warning(
                f"Adapter '{adapter_key}' does not specify any accepted types or formats. "
                "It will not be discoverable by Tasks. Please set 'accepted_types' and/or "
                "'accepted_formats'."
            )

        if adapter_key in self._adapters:
            raise KeyError(f"An adapter with key '{adapter_key}' is already registered.")

        self._adapters[adapter_key] = adapter
        logger.debug(f"Registered adapter '{adapter_key}'")

    def get(self, key: str) -> BaseAdapter | None:
        """Safe get method that returns None if adapter is not found"""
        return self._adapters.get(key, None)

    def get_for_artifact(
        self, artifact: "Artifact", must_provide: "str | DataTrait" = "*",
    ) -> list[BaseAdapter]:
        """
        Finds all adapters that can bridge the gap between this Artifact and the
        Task's data requirements
        """
        return self.get_for_type(artifact.data_type, artifact.extension, must_provide)

    def get_for_type(
        self, data_type: str, extension: str, must_provide: "str | DataTrait" = "*",
    ) -> list[BaseAdapter]:
        """
        Finds all adapters that can bridge the gap between this physical
        Artifact and the Task's Logical Requirement.
        """

        matches = []
        for adapter in self._adapters.values():
            # 1. Artifact compatibility: Can I read this?
            format_match = (
                extension == "*"
                or "*" in adapter.accepted_formats
                or extension in adapter.accepted_formats
            )
            type_match = (
                data_type == "*"
                or "*" in adapter.accepted_types
                or data_type in adapter.accepted_types
            )
            if not (format_match and type_match):
                continue

            # 2. Logical compatibility: Can I provide this?
            if must_provide == "*":
                matches.append(adapter)
            elif isinstance(must_provide, DataTrait):
                if must_provide.is_satisfied_by(data_type, [adapter]):
                    matches.append(adapter)
            elif isinstance(must_provide, str):
                if adapter.provides(must_provide):
                    matches.append(adapter)

        # 3. Optimization: Sort by 'Expertise'
        def sort_score(a):
            """
            Prioritize Adapters that explicitly provide rather than '*' with
            semantic type (1.1) scored higher than file format (1.0)
            """
            score = 0
            if extension in a.accepted_formats:
                score += 1.0
            if data_type in a.accepted_types:
                score += 1.1
            return score

        matches.sort(key=sort_score, reverse=True)
        return matches
