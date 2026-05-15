"""
Traits are contracts for input/output slots that define expected data types.
The semantic matching system in LoRē starts with these traits, and if a trait 
does not exist for a term, that term is treated as a literal string.

In simple English, a trait hijacks narrow keywords like "table" or "alignment" 
to give them broader meaning in the matching engine.
"""

from abc import ABC, abstractmethod

from lore.core.adapters import BaseAdapter, TableAdapter


class DataTrait(ABC):
    """Base class for defining semantic data requiremenets."""
    @abstractmethod
    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        """
        Evaluates whether a provided data type or its adapters can satisfy the 
        data requirements of this trait.
        """

    def __str__(self) -> str:
        """How the trait appears in the UI and Jinja templates."""
        # Converts e.g. "TabularTrait" -> "TABULAR"
        return self.__class__.__name__.replace("Trait", "").upper()

    def __repr__(self) -> str:
        """How the trait appears in terminal logs and debuggers."""
        return f"<DataTrait:{self.__str__()}>"


class AnyTrait(DataTrait):
    """A wildcard trait that accepts any data type."""
    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        return True


class TabularTrait(DataTrait):
    """Accepts native tables or anything that can be adapted to a table."""
    def is_satisfied_by(self, provided_type: str, adapters: list[BaseAdapter]) -> bool:
        # 1. Native match - is the provided type already a table?
        if provided_type in ["table", "tabular", "dataframe", "csv", "tsv"]:
            return True

        # 2. Adapter match - can any adapter convert this type to a table?
        for adapter in (adapters or []):
            # Adapter class with a provides() method
            if isinstance(adapter, type):
                if issubclass(adapter, TableAdapter):
                    return True

            # Instantiated TableAdapter
            elif isinstance(adapter, TableAdapter):
                return True

        return False


# Instances of common traits to be exposed through the DSL for use in Task definitions
ANY = AnyTrait()
TABULAR = TabularTrait()
