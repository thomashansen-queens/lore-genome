"""
Matcher functions for finding Artifacts to satisfy Task input requirements.
Bridges TaskDefinition schemas to the available Artifacts in a Session
"""

from typing import Any, TYPE_CHECKING

from lore.core.tasks import task_registry

if TYPE_CHECKING:
    from lore.core.artifacts import Artifact
    from lore.core.sessions import Session
    from lore.core.tasks import TaskDefinition


def find_artifacts_for_field(session: "Session", field_extra: dict) -> list["Artifact"]:
    """
    Returns a list of Artifacts able to satisfy an input field's data type requirements.
    """
    from lore.core.adapters import TableAdapter
    if not field_extra.get("is_artifact"):
        return []

    accepted_data = set(field_extra.get("accepted_data", ["*"]))

    valid_artifacts = []

    # 1. True wildcard (field accepts anything)
    if "*" in accepted_data:
        return session.list_artifacts()

    for artifact in session.list_artifacts():
        # 2. Semantic check (does the file match any accepted types?)
        resolvable_types = artifact.resolvable_types()
        if accepted_data & resolvable_types:
            valid_artifacts.append(artifact)
            continue

        # 3. Does a slice of a table-like artifact match?
        table_adapters = [a for a in artifact.get_adapters() if isinstance(a, TableAdapter)]
        if table_adapters:
            available_cols = set()

            # A. Dynamic schema: Columns (e.g. from arbitrary CSVs)
            available_cols.update(artifact.metadata.get("columns", []))
            available_cols.update(artifact.metadata.get("keys", []))

            # B. Static schema: Adapter-provided
            for adapter in table_adapters:
                schema = getattr(adapter, 'schema', {})
                available_cols.update(schema.keys())

            # Is either schema capable of satisfying the accepted data requirements?
            if accepted_data & available_cols:
                valid_artifacts.append(artifact)

    return valid_artifacts


def find_artifact_candidates(session: "Session", task_def: "TaskDefinition") -> dict[str, list["Artifact"]]:
    """
    Find valid Artifacts for every field in a Task.

    :returns: dict[field_name: [valid_artifacts]]
    """
    candidates = {}

    for field_name in task_def.input_model.model_fields.keys():
        _, extra = task_def.field_meta(field_name)
        if extra.get("is_artifact"):
            candidates[field_name] = find_artifacts_for_field(session, extra)

    return candidates


def map_artifacts_to_task_inputs(session: "Session", task_def: "TaskDefinition", source_artifact_ids: list[str]) -> dict[str, Any]:
    """
    Given a TaskDefinition and a list of candidate Artifact IDs, determine 
    the best mapping of Artifacts to Task inputs.
    TODO: Add complexity and elegance to how artifacts are assigned to input slots

    :returns: dict[field_name: artifact_id | list[artifact_id]]
    """
    if not source_artifact_ids:
        return {}

    artifacts = [a for aid in source_artifact_ids if (a := session.get_artifact(aid)) is not None]
    mapping = {}

    for key in task_def.input_model.model_fields.keys():
        _, extra = task_def.field_meta(key)

        accepted_data = set(extra.get("accepted_data", ["*"]))
        is_multiple = extra.get("cardinality", "single") in ("multiple", "pair", "two_or_more")

        for artifact in artifacts:
            resolvable = artifact.resolvable_types()

            if "*" in accepted_data or accepted_data.intersection(resolvable):
                if is_multiple:
                    mapping.setdefault(key, []).append(artifact.id)
                else:
                    if key not in mapping:
                        mapping[key] = artifact.id
                    break

    return mapping

def find_valid_upstream_outputs(
    session: "Session",
    current_task_id: str | None,
    field_extra: dict,
) -> list[dict[str, Any]]:
    """
    Looks at other Tasks in the Session and checks their output schemas
    to see if they can satisft the current field's data requirements.
    TODO: Actually check upstream. Currently checks all other Tasks.

    Returns: list[dict]: [{"task": Task, "valid_outputs": list[str]}]
    """
    if not field_extra.get("is_artifact"):
        return []

    accepted_data = set(field_extra.get("accepted_data", ["*"]))
    valid_upstream = []

    for task in session.list_tasks():
        # Prevent self-binding
        if current_task_id and task.id == current_task_id:
            continue

        upstream_def = task_registry.get(task.registry_key)
        if not upstream_def or not upstream_def.output_model:
            continue

        valid_outputs = []

        for out_key in upstream_def.output_model.model_fields.keys():
            _, out_extra = upstream_def.field_meta(out_key, is_output=True)

            produced_type = out_extra.get("data_type", "unknown")

            if "*" in accepted_data or produced_type in accepted_data:
                valid_outputs.append(out_key)

        if valid_outputs:
            valid_upstream.append({
                "task": task,
                "valid_outputs": valid_outputs
            })

    return valid_upstream
