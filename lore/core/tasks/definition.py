"""
A TaskDefinition is a contract that LoRē uses to orchestrate workflows. It
defines the inputs, outputs, and handler function for a Task.
"""
from dataclasses import dataclass
from enum import StrEnum
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import Any, Callable, Literal, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from lore.core.adapters import BaseAdapter


class PreviewMode(StrEnum):
    """Defines the level of preview data to return for a Task in the UI."""
    NONE = "none"  # No preview data; only metadata and validation status
    LIVE = "live"  # Live preview with real-time data
    FULL = "full"  # Full preview with complete data (may be slow)
    DRY_RUN = "dry_run"  # Simulate execution without running the handler (for testing)

    @property
    def executes_handler(self) -> bool:
        """Convenience property to check if a preview executes the handler function."""
        return self in {self.LIVE, self.FULL}

    @property
    def is_allowed(self) -> bool:
        """Checks if the task should return any type of preview data."""
        return self != self.NONE

    @property
    def is_live(self) -> bool:
        """Whether the UI should auto-refresh the preview as inputs change."""
        return self == self.LIVE


PreviewModeLiteral = Literal["none", "live", "full", "dry_run"]


@dataclass(frozen=True)
class TaskDefinition:
    """
    Each Task is a contract to perform a single unit of work in LoRē, with defined inputs, outputs,
    and a handler function that does the science.

    Attributes:
        key: Unique identifier for the Task (e.g. "ncbi.fetch_genome_reports").
        name: Human-readable name of the Task. (e.g. "Fetch Genome Reports from NCBI").
        handler: Python function that implements the Task logic. (ctx, config) -> result.
        input_model: Pydantic model defining the Inputs for the Task.
        output_model: Pydantic model defining the Outputs for the Task
        description: Human-readable description of the Task.
        category: Category or grouping for the Task (e.g. "NCBI", "Phylogeny").
        icon: Emoji or symbol representing the Task visually. TODO: De-unicodify this, use SVGs
        preview_mode: Permissions for UI previews ("none", "live", "full", "dry_run")
    """
    key: str
    handler: Callable
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    description: str = ""
    name: str = ""
    category: str = ""
    icon: str = "⚡"
    preview_mode: PreviewMode = PreviewMode.NONE

    def field_meta(self, key: str, is_output: bool = False) -> tuple[FieldInfo, dict[str, Any]]:
        """
        Validates and extracts metadata for a given input field in a TaskDefinition.
        :returns: tuple[FieldInfo, json_schema_extra_dict]
        """
        model = self.output_model if is_output else self.input_model

        # No output model (e.g. it's an exporter)
        if not model:
            return FieldInfo(annotation=None), {}

        field_info = model.model_fields.get(key)

        # 1. Graceful fallback (e.g. missing TaskDefinition)
        if field_info is None:
            dummy_field = FieldInfo(annotation=None)
            return dummy_field, {}

        # 2. Extract enriched field metadata from json_schema_extra
        extra = getattr(field_info, "json_schema_extra", None)
        if extra is None or not isinstance(extra, dict):
            return field_info, {}
        return field_info, extra

    @property
    def primary_output_key(self) -> str | None:
        """Scans the output model for the field marked is_primary=True"""
        for key in self.output_model.model_fields.keys():
            meta = self.output_model.model_fields[key].json_schema_extra
            if meta is None:
                continue
            if meta.get("is_primary"):
                return key

        return None

    def get_adapters_for_output(self, output_key: str) -> list["BaseAdapter"]:
        """
        Convenience accessor to find valid Adapters for a specific theoretical output.
        Mirrors the behavior of Artifact.get_adapters()
        """
        from lore.core.adapters import adapter_registry

        field = self.output_model.model_fields.get(output_key)
        if not field:
            return []

        _, meta = self.field_meta(output_key, is_output=True)
        data_type = meta.get("data_type", "unknown")

        return adapter_registry.get_for_type(data_type=data_type, extension="*")
