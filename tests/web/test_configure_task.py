from types import SimpleNamespace

from typing import Annotated

from pydantic import BaseModel, Field

from lore.core.artifacts.artifact import Artifact
from lore.core.bindings import LiteralBinding, ReferenceBinding
from lore.core.tasks import (
    ArtifactInput,
    Cardinality,
    Materialization,
    Task,
    TaskDefinition,
)
from lore.web.deps import templates
from lore.web.utils import configure_task
from lore.web.utils.configure_task import _build_field_ui_state


class ArtifactTaskInputs(BaseModel):
    source: Annotated[
        str,
        ArtifactInput(
            accepted_data="*",
            load_as=Materialization.ADAPTED,
        ).to_field_info(),
    ]


class ArtifactTaskOutputs(BaseModel):
    result: str = Field(json_schema_extra={"data_type": "json"})


class MultipleArtifactTaskInputs(BaseModel):
    source: Annotated[
        list[str],
        ArtifactInput(
            accepted_data="*",
            select=Cardinality.MULTIPLE,
            load_as=Materialization.ADAPTED,
        ).to_field_info(),
    ]


def _render_task_input(task_def, ui_state, field_name="source"):
    return templates.env.get_template("components/task_input.html").render(
        context_type="session",
        task=None,
        field_info=task_def.input_model.model_fields[field_name],
        field_name=field_name,
        ui_state=ui_state,
    )


def test_single_artifact_literal_renders_adapter_selector(temp_session, tmp_path):
    artifact_path = tmp_path / "source.json"
    artifact_path.write_text('[{"id": 1}]', encoding="utf-8")
    artifact = temp_session.register_artifact(
        artifact_path,
        name="Source",
        data_type="json",
    )
    task_def = TaskDefinition(
        key="test.single_artifact_input",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )

    ui_state = _build_field_ui_state(
        container=temp_session,
        task_def=task_def,
        field_name="source",
        task=None,
        raw_prefill=artifact.id,
        avail_tasks=[],
        candidates=[artifact],
        mode_override=None,
    )
    rendered = _render_task_input(task_def, ui_state)

    assert ui_state["adapter_enabled"]
    assert 'name="task_config[input_adapters][source]"' in rendered
    assert f'value="{ui_state["adapter_value"]}"' in rendered
    assert 'data-adapter-control hidden' not in rendered


def test_task_output_reference_renders_adapter_selector(
    temp_session, monkeypatch
):
    task_def = TaskDefinition(
        key="test.task_output_input",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    upstream = SimpleNamespace(
        id="upstream-task",
        registry_key="test.upstream",
        outputs={},
        resolve_output_type=lambda output_key, container: {"data_type": "json"},
    )
    upstream_def = TaskDefinition(
        key="test.upstream",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    monkeypatch.setattr(
        configure_task.task_registry,
        "get_safe",
        lambda task_key: upstream_def if task_key == "test.upstream" else None,
    )
    current_task = Task(
        id="current-task",
        registry_key=task_def.key,
        inputs={
            "source": [
                ReferenceBinding(source_id="upstream-task", output_key="result")
            ]
        },
        exec_config={},
    )
    ui_state = _build_field_ui_state(
        container=temp_session,
        task_def=task_def,
        field_name="source",
        task=current_task,
        raw_prefill=None,
        avail_tasks=[{"task": upstream, "valid_outputs": ["result"]}],
        candidates=[],
        mode_override=None,
    )
    rendered = _render_task_input(task_def, ui_state)

    assert ui_state["binding_mode"] == "reference"
    assert ui_state["adapter_enabled"]
    assert 'name="task_config[input_adapters][source]"' in rendered
    assert 'data-adapters=' in rendered
    assert 'data-adapter-control hidden' not in rendered


def test_multiple_artifact_adapter_choices_are_intersection(
    temp_session, tmp_path, monkeypatch
):
    class SharedAdapter:
        name = "SharedAdapter"

    class FirstOnlyAdapter:
        name = "FirstOnlyAdapter"

    class SecondOnlyAdapter:
        name = "SecondOnlyAdapter"

    shared = SharedAdapter()
    first_only = FirstOnlyAdapter()
    second_only = SecondOnlyAdapter()
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for index, path in enumerate(paths):
        path.write_text(f'[{{"id": {index + 1}}}]', encoding="utf-8")
    artifacts = [
        temp_session.register_artifact(path, name=path.stem, data_type="json")
        for path in paths
    ]
    adapters_by_id = {
        artifacts[0].id: [shared, first_only],
        artifacts[1].id: [shared, second_only],
    }
    monkeypatch.setattr(
        Artifact,
        "get_adapters",
        lambda artifact: adapters_by_id[artifact.id],
    )
    task_def = TaskDefinition(
        key="test.multiple_artifact_input",
        handler=lambda ctx, **kwargs: None,
        input_model=MultipleArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )

    current_task = Task(
        id="multi-task",
        registry_key=task_def.key,
        inputs={
            "source": [
                LiteralBinding(value=artifact.id)
                for artifact in artifacts
            ]
        },
    )
    ui_state = _build_field_ui_state(
        container=temp_session,
        task_def=task_def,
        field_name="source",
        task=current_task,
        raw_prefill=None,
        avail_tasks=[],
        candidates=artifacts,
        mode_override=None,
    )
    rendered = _render_task_input(task_def, ui_state)

    assert [adapter["value"] for adapter in ui_state["adapter_choices"]] == [
        "SharedAdapter"
    ]
    dropdown = rendered.split("data-adapter-select", 1)[1].split("</select>", 1)[0]
    assert "SharedAdapter" in dropdown
    assert "FirstOnlyAdapter" not in dropdown
    assert "SecondOnlyAdapter" not in dropdown
    assert 'data-adapter-control hidden' not in rendered
