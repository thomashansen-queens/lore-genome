"""
LoRē Genome Public DSL (Domain Specific Language).
This is the single import point for all plugin authors writing Tasks 
or Adapters. Workflows do not have any DSL-specific imports (yet).
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
    # ImageAdapter,
    TabularAdapter,
    CsvAdapter,
    JsonAdapter,
    TextAdapter,
)

# 5. Decorators
adapter = adapter_registry.register
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

# Public API
__all__ = [
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
    "TabularAdapter",
    "CsvAdapter",
    "JsonAdapter",
    "TextAdapter",
    # Decorator aliases
    "adapter",
    "task",
    "config",
    # Input configuration Enums (cardinality/'select', materliaziation/'load_as')
    "OPTIONAL", "SINGLE", "MULTIPLE", "OPTIONAL_MULTIPLE",
    "ARTIFACT", "PATH", "RAW", "ADAPTED", "RAW_STREAM", "ADAPTED_STREAM", "PREVIEW",
]
