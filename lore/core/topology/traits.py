"""
Traits are contracts for input/output slots that define expected data types.
The semantic matching system in LoRē starts with these traits, and if a trait 
does not exist for a term, that term is treated as a literal string.

In simple English, a trait hijacks narrow keywords like "table" or "alignment" 
to give them broader meaning in the matching engine.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from lore.core.adapters import BaseAdapter


class DataTrait(ABC):
    """
    Base class for defining semantic data requirements.

    A trait "hijacks" a keyword (e.g. 'tabular'): wherever that exact keyword
    appears in a Task's accepted_data, the matcher evaluates this trait's
    is_satisfied_by() instead of a literal type match. The keyword attribute
    exists so the class name is arbitrary, but the serialized keyword, as a
    plain string, is what gets used.
    """
    keyword: ClassVar[str]

    @abstractmethod
    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        """
        Evaluates whether a provided data type or its adapters can satisfy the
        data requirements of this trait.
        """

    def __str__(self) -> str:
        """How the trait appears in the UI, logs, and json_schema_extra."""
        return self.keyword

    def __repr__(self) -> str:
        """How the trait appears in the console and debug logs as a Python object"""
        return f"<DataTrait:{self.keyword}>"


class AnyTrait(DataTrait):
    """A wildcard trait that accepts any data type."""
    keyword = "*"

    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        return True


class TabularTrait(DataTrait):
    """Accepts native tables or anything that can be adapted to a table."""
    keyword = "tabular"
    TARGET_TYPES = {"table", "tabular", "dataframe"}
    NATIVE_TYPES = {"table", "tabular", "dataframe", "csv", "tsv"}

    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        # 1. Native match: provided type is already a table
        if provided_type in self.NATIVE_TYPES:
            return True

        # 2. Adapter match: can any adapter convert this type to a table?
        for adapter in (adapters or []):
            # Adapter class with a provides() method (uninstantiated)
            if isinstance(adapter, type) and issubclass(adapter, BaseAdapter):
                instance = adapter()
            else:
                instance = adapter

            # Instantiated adapter
            if any(instance.provides(t) for t in self.TARGET_TYPES):
                return True

        return False


# Mini-registry for traits: keyword -> trait
# The matcher resolves accepted_data strings against this, so a keyword always
# wins over a literal type of the same name.
# TODO: Set up a true plugin-enabled trait registry
TRAITS: dict[str, DataTrait] = {AnyTrait.keyword: AnyTrait(), TabularTrait.keyword: TabularTrait()}


def register_trait(trait: DataTrait) -> None:
    """Register a trait so its keyword is recognised by the matcher."""
    TRAITS[trait.keyword] = trait


def resolve_trait(name: object) -> DataTrait | None:
    """Return the trait that hijacks `name`, or None if it's a literal type."""
    return TRAITS.get(name) if isinstance(name, str) else None
