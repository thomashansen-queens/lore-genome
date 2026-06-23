"""
LoRē Genome Public DSL (Domain Specific Language).
This is the single import point for all plugin authors writing Tasks 
or Adapters. Workflows do not have any DSL-specific imports (yet).
"""
# 1. Task Inputs & Configuration
from lore.core.tasks import (
    TaskInput,
    ValueInput,
    TaskOutput,
    ArtifactInput,
    Widget,
    Cardinality,
    Materialization,
    Passthrough,
    PreviewMode,
    CardinalityLiteral,
    MaterializationLiteral,
    PreviewModeLiteral,
    WidgetLiteral,
    task_registry,
)
from lore.core.settings import config_registry

# 2. Topology & trait matching (replaced with plain strings!)
# from lore.core.topology.traits import ANY, TABULAR

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

# 5. Readers
from lore.core.readers import (
    reader_registry,
    BaseReader,
    TextReader,
    TableReader,
    ImageReader,
)

# 6. Decorators
reader = reader_registry.register
adapter = adapter_registry.register
task = task_registry.register
config = config_registry.register

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
    "PreviewMode",
    # Execution
    "ExecutionContext",
    "memoize",
    # Reader layer
    "BaseReader",
    "TextReader",
    "TableReader",
    "ImageReader",
    "reader_registry",
    # Adapter layer
    "AdapterPreview",
    "BaseAdapter",
    "TabularAdapter",
    "CsvAdapter",
    "JsonAdapter",
    "TextAdapter",
    # Decorator aliases
    "reader",
    "adapter",
    "task",
    "config",
    # Input configuration Enums (cardinality/'select', materliaziation/'load_as')
    "CardinalityLiteral",
    "MaterializationLiteral",
    "PreviewModeLiteral",
    "WidgetLiteral",
]
