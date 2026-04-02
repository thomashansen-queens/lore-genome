"""
Core task management, data models, and the global registry.
"""
# 1. Models
from .models import (
    Task, 
    TaskDefinition, 
    TaskConfig, 
    AdapterConfig, 
    ExecutionConfig,
    TaskResults,
    TaskStatus,
    TaskIntegrity,
)

# 2. Registry
from .registry import (
    TaskRegistry, 
    task_registry,
)

# 3. DSL Primitives (Assuming these live in your internal dsl.py)
from .parameters import (
    TaskInput,
    ValueInput,
    ArtifactInput,
    TaskOutput,
    Widget,
    Cardinality,
    Materialization,
)

# 4. Publicize
__all__ = [
    # Models
    "Task",
    "TaskDefinition",
    "TaskConfig",
    "AdapterConfig",
    "ExecutionConfig",
    "TaskResults",
    "TaskStatus",
    "TaskIntegrity",

    # Registry
    "TaskRegistry",
    "task_registry",

    # Internal DSL Primitives
    "TaskInput",
    "ValueInput",
    "ArtifactInput",
    "TaskOutput",
    "Widget",
    "Cardinality",
    "Materialization",
]