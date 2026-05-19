"""
Concrete matching functions for finding Artifacts to satisfy Task 
input requirements.

This matcher module considers how task outputs (Artifacts and primitives)
can be matched to task inputs within the Session. Topology is not considered 
at this level. A separate matcher module in the 'topology' layer exists to 
consider upstream/downstream relationships and other factors (e.g. cycles).
"""

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from lore.core.artifacts import Artifact
    from lore.core.sessions import Session
    from lore.core.tasks import TaskDefinition


def find_artifacts_for_field(session: "Session", field_extra: dict) -> list["Artifact"]:
    """
    Returns a list of Artifacts able to satisfy an input field's data type requirements.
    """
    from lore.core.adapters import TableAdapter
    from lore.core.topology.traits import DataTrait

    if not field_extra.get("is_artifact"):
        return []

    accepted_data = field_extra.get("accepted_data", ["*"])

    # 1. True wildcard (field accepts anything)
    if "*" in accepted_data:
        return session.list_artifacts()

    trait_requirements = [d for d in accepted_data if isinstance(d, DataTrait)]
    semantic_requirements = {d for d in accepted_data if not isinstance(d, DataTrait)}
    valid_artifacts = []

    for artifact in session.list_artifacts():
        art_adapters = artifact.get_adapters()

        # 2. Requirement is a trait - e.g. lore.TABULAR
        if any(
            trait.is_satisfied_by(artifact.data_type, art_adapters)
            for trait in trait_requirements
        ):
            valid_artifacts.append(artifact)
            continue

        # 3. Requirement is a literal string - e.g. "json"
        if not semantic_requirements:
            continue

        provided_types = set(artifact.resolvable_types())

        # 4. Does a slice of a table-like artifact match?
        table_adapters = [a for a in art_adapters if isinstance(a, TableAdapter)]
        if table_adapters:
            # A. Dynamic schema: Columns (e.g. from arbitrary CSVs)
            provided_types.update(artifact.metadata.get("columns", []))
            provided_types.update(artifact.metadata.get("keys", []))

            # B. Static schema: Adapter-provided
            for adapter in table_adapters:
                schema = getattr(adapter, "schema", {})
                provided_types.update(schema.keys())

        # 5. Is either schema capable of satisfying the accepted data requirements?
        if semantic_requirements & provided_types:
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
