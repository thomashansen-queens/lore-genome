"""
Core task management, data models, and the global registry.
"""
# 1. Definition
from .definition import (
    TaskDefinition,
    PreviewMode,
    PreviewModeLiteral,
)

# 2. Instance
from .instance import (
    AdapterStrategy,
    AdapterConfig,
    ExecutionConfig,
    TaskConfig,
    Task,
    TaskResults,
)

# 3. Parameters
from .parameters import (
    Widget,
    WidgetLiteral,
    Cardinality,
    CardinalityLiteral,
    Materialization,
    MaterializationLiteral,
    TaskInput,
    ArtifactInput,
    ValueInput,
    Passthrough,
    TaskOutput,
)

# 4. Registry
from .registry import (
    TaskRegistry, 
    task_registry,
)

# 5. State
from .state import (
    TaskIntegrity,
    TaskStatus,
)

# 6. Publicize
__all__ = [
    "TaskDefinition",
    "PreviewMode",
    "PreviewModeLiteral",

    "AdapterStrategy",
    "AdapterConfig",
    "ExecutionConfig",
    "TaskConfig",
    "Task",
    "TaskResults",

    "Widget",
    "WidgetLiteral",
    "Cardinality",
    "CardinalityLiteral",
    "Materialization",
    "MaterializationLiteral",
    "TaskInput",
    "ArtifactInput",
    "ValueInput",
    "Passthrough",
    "TaskOutput",

    "TaskRegistry",
    "task_registry",

    "TaskIntegrity",
    "TaskStatus",
]
