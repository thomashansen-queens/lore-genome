"""
Tests for the core Task module, including Task models
"""

import pytest
from pydantic import BaseModel

from lore.core.bindings import LiteralBinding, ReferenceBinding
from lore.core.tasks.models import Task, TaskStatus
from lore.core.tasks.registry import task_registry

# --- Dummy Task ---

class DummyValidationModel(BaseModel):
    e_value: float
    query_files: list[str]

# --- Tests ---

def test_task_airlock_on_init():
    """
    Ensures raw primitive are wrapped into appropriate Bindings upon Task creation.
    """
    raw_inputs = {
        "e_value": 1e-5,
        "query_files": ["file1.fasta", "file2.fasta"],
    }

    task = Task(
        id="test-1",
        registry_key="dummy_task",
        inputs=raw_inputs,
    )
