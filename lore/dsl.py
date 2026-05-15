"""
LoRe Genome Public DSL (Domain Specific Language).
This is the single import point for all plugin authors writing Tasks or Adapters.
"""
# 1. Task Inputs & Configuration
from lore.core.tasks.parameters import (
    TaskInput,
    ValueInput,
    TaskOutput,
    ArtifactInput,
    Widget,
    Cardinality,
    Materialization,
    Passthrough,
)
from lore.core.tasks.registry import task_registry
from lore.core.settings import config_registry

# 2. Topology & trait matching
from lore.core.topology.traits import ANY, TABULAR

# 3. Execution State
from lore.core.execution.context import (
    ExecutionContext,
)
from lore.core.cache import memoize

# 4. Adapters
from lore.core.adapters import (
    adapter_registry,
    AdapterPreview,
    BaseAdapter,
    ImageAdapter,
    TableAdapter,
)

# 5. Aliases
adapter = adapter_registry
task = task_registry.register
config = config_registry.register

# 6. Direct Enum access
# Cardinality for ArtifactInputs
OPTIONAL = Cardinality.OPTIONAL_SINGLE
SINGLE = Cardinality.SINGLE
MULTIPLE = Cardinality.MULTIPLE
OPTIONAL_MULTIPLE = Cardinality.OPTIONAL_MULTIPLE

# Materialization for ArtifactInputs
ARTIFACT = Materialization.ARTIFACT
PATH = Materialization.PATH
RAW = Materialization.RAW
ADAPTED = Materialization.ADAPTED
RAW_STREAM = Materialization.RAW_STREAM
ADAPTED_STREAM = Materialization.ADAPTED_STREAM
PREVIEW = Materialization.PREVIEW

# Widgets for GUI
ARTIFACT_MULTI = Widget.ARTIFACT_MULTI
ARTIFACT_SINGLE = Widget.ARTIFACT_SINGLE
CHECKBOX = Widget.CHECKBOX
CHECKBOX_GROUP = Widget.CHECKBOX_GROUP
DATE = Widget.DATE
DATETIME = Widget.DATETIME
FLOAT = Widget.FLOAT
INTEGER = Widget.INTEGER
RADIO = Widget.RADIO
SEGMENTED_RADIO = Widget.SEGMENTED_RADIO
SELECT = Widget.SELECT
SLIDER = Widget.SLIDER
TEXT = Widget.TEXT
TEXTAREA = Widget.TEXTAREA


# Public API
__all__ = [
    "task_registry",
    "TaskInput",
    "ValueInput",
    "ArtifactInput",
    "TaskOutput",
    "Widget",
    "Cardinality",
    "Materialization",
    "Passthrough",
    # Traits
    "ANY", "TABULAR",
    # Execution
    "ExecutionContext",
    "memoize",
    # Adapter layer
    "AdapterPreview",
    "BaseAdapter",
    "ImageAdapter",
    "TableAdapter",
    # Decorator aliases
    "adapter",
    "task",
    "config",
    # Enums
    "OPTIONAL", "SINGLE", "MULTIPLE", "OPTIONAL_MULTIPLE",
    "ARTIFACT", "PATH", "RAW", "ADAPTED", "RAW_STREAM", "ADAPTED_STREAM", "PREVIEW",
    # Widgets
    "ARTIFACT_MULTI", "ARTIFACT_SINGLE", "CHECKBOX", "CHECKBOX_GROUP", "DATE", "DATETIME",
    "FLOAT", "INTEGER", "RADIO", "SEGMENTED_RADIO", "SELECT", "SLIDER", "TEXT", "TEXTAREA",
]
