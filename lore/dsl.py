"""
LoRe Genome Public DSL (Domain Specific Language).
This is the single import point for all plugin authors writing Tasks or Adapters.
"""
# 1. Task Inputs & Configuration
from lore.core.tasks.dsl import (
    TaskInput,
    ValueInput,
    TaskOutput,
    ArtifactInput,
    Widget,
    Cardinality,
    Materialization,
)
from lore.core.tasks.registry import task_registry

# 2. Execution State
from lore.core.execution.context import (
    ExecutionContext,
)
from lore.core.cache import memoize

# 3. FUTURE: Adapters
# from lore.core.adapters import Adapter, adapter_registry

# 4. Aliases
# Function
task = task_registry.register

# Direct Enum access
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


# The __all__ list strictly defines your Public API. 
__all__ = [
    "task_registry",
    "TaskInput",
    "ValueInput",
    "ArtifactInput",
    "TaskOutput",
    "Widget",
    "Cardinality",
    "Materialization",
    "ExecutionContext",
    "memoize",
    # aliases
    "task",
    "OPTIONAL", "SINGLE", "MULTIPLE", "OPTIONAL_MULTIPLE",
    "ARTIFACT", "PATH", "RAW", "ADAPTED", "RAW_STREAM", "ADAPTED_STREAM", "PREVIEW",
]
