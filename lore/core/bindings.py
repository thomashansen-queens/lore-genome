"""
Bindings define how Task inputs are connected to outputs of other Tasks or to 
user-provided values.
"""

from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, TypeAdapter


class LiteralBinding(BaseModel):
    """A concrete input value or existing Artifact ID (e.g. uploaded file)."""
    type: Literal["literal"] = "literal"
    value: Any


class ReferenceBinding(BaseModel):
    """
    A reference to an upstream step's output. This is what builds the DAG
    edges. If artifact_id is unspecified (unpinned), it will be resolved at
    runtime by the materializer per the Task's input model.
    """
    type: Literal["reference"] = "reference"
    source_id: str
    output_key: str
    artifact_id: str | None = None


class UserInputBinding(BaseModel):
    """
    An explicit placeholder for a value that must be filled in by the user.
    value can be set to a default.
    """
    type: Literal["user_input"] = "user_input"
    input_key: str
    value: Any | None = None


Binding = Annotated[
    Union[LiteralBinding, ReferenceBinding, UserInputBinding],
    Field(discriminator="type")
]

# --- Errors ---

class UnresolvedReferenceError(ValueError):
    """Raised when a Task cannot validate because it is waiting on an upstream Step."""
    pass


class MissingUserInputError(ValueError):
    """Raised when a Task cannot validate because a UserInputBinding is unfulfilled."""
    pass

# --- Coercion ---


def wrap_in_bindings(inputs: dict[str, Any] | None) -> dict[str, list[Binding]]:
    """
    Convert a dict of input values into engine Bindings. If the value is already a Binding (or a 
    dict representation of one) is is passed through.
    """
    binding_parser = TypeAdapter(Binding)

    binding_inputs = {}
    for k, v in (inputs or {}).items():
        items = v if isinstance(v, list) else [v]

        parsed_list = []
        for item in items:
            # 1. An already instantiated Binding object
            if isinstance(item, (LiteralBinding, ReferenceBinding, UserInputBinding)):
                # Intercept empty LiteralBindings
                if isinstance(item, LiteralBinding) and item.value in (None, ""):
                    continue
                parsed_list.append(item)

            # 2. A dict that can be parsed as a binding (e.g. from a manifest or front end)
            elif isinstance(item, dict) and item.get("type") in {"literal", "reference", "user_input"}:
                # Intercept JSON dicts with empty literals
                if item.get("type") == "literal" and item.get("value") in (None, ""):
                    continue
                parsed_list.append(binding_parser.validate_python(item))

            # 3. Raw primitive: Wrap it in a LiteralBinding
            else:
                # Intercept empty literals (Allow False and 0)
                if item is not None and item != "":
                    parsed_list.append(LiteralBinding(value=item))

        binding_inputs[k] = parsed_list

    return binding_inputs
