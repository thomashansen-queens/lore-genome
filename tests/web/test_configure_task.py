import re
from types import SimpleNamespace
from typing import Annotated

from pydantic import BaseModel, Field

from lore.core.adapters import BaseAdapter, JsonAdapter, TextAdapter
from lore.core.adapters.tabular import TabularAdapter
from lore.core.artifacts.artifact import Artifact
from lore.core.bindings import LiteralBinding, ReferenceBinding
from lore.core.execution.preview import PreviewOutput, PreviewPayload
from lore.core.tasks import (
    ArtifactInput,
    Cardinality,
    Materialization,
    Task,
    TaskDefinition,
)
from lore.web.deps import templates
from lore.web.features.artifacts.routes import _viewer_completeness
from lore.web.features.tasks import routes as task_routes
from lore.web.features.tasks.routes import _display_complete_for_data
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


class ConstrainedArtifactTaskInputs(BaseModel):
    source: Annotated[
        str,
        ArtifactInput(
            accepted_data="*",
            adapt_as=[JsonAdapter, TextAdapter],
            load_as=Materialization.ADAPTED,
        ).to_field_info(),
    ]


def _render_task_input(
    task_def,
    ui_state,
    field_name="source",
    input_preview_api_url=None,
):
    return templates.env.get_template("components/task_input.html").render(
        context_type="session",
        task=None,
        input_preview_api_url=input_preview_api_url,
        field_info=task_def.input_model.model_fields[field_name],
        field_name=field_name,
        ui_state=ui_state,
    )


def test_configure_page_uses_auto_preview_setting(
    client, temp_runtime, monkeypatch
):
    task_def = TaskDefinition(
        key="test.auto_preview_setting",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    monkeypatch.setitem(task_routes.task_registry._tasks, task_def.key, task_def)
    session = temp_runtime.create_session(name="Auto-preview setting")
    configure_url = f"/sessions/{session.id}/tasks/configure/{task_def.key}"

    enabled_response = client.get(configure_url)
    assert enabled_response.status_code == 200
    assert 'data-auto-preview-selected-artifacts="true"' in enabled_response.text

    monkeypatch.setattr(
        temp_runtime.settings, "auto_preview_selected_artifacts", False
    )
    disabled_response = client.get(configure_url)
    assert disabled_response.status_code == 200
    assert 'data-auto-preview-selected-artifacts="false"' in disabled_response.text


def test_input_artifact_preview_displays_last_artifact_by_default():
    previews = [
        {
            "artifact": SimpleNamespace(ui_name="First"),
            "preview": SimpleNamespace(
                data="first data",
                metadata={},
                adapter_name="RawAdapter",
                view_mode="raw",
            ),
        },
        {
            "artifact": SimpleNamespace(ui_name="Last"),
            "preview": SimpleNamespace(
                data="last data",
                metadata={},
                adapter_name="RawAdapter",
                view_mode="raw",
            ),
        },
    ]

    rendered = templates.env.get_template(
        "features/tasks/fragments/input_preview.html"
    ).render(previews=previews)

    assert 'class="btn-tab secondary active"\n                        data-input-preview-target="input-artifact-tab-2"' in rendered
    assert re.search(
        r'id="input-artifact-tab-1"\s+class="input-artifact-tab column fill"\s+'
        r'style="display: none;"',
        rendered,
    )
    assert re.search(
        r'id="input-artifact-tab-2"\s+class="input-artifact-tab column fill"',
        rendered,
    )
    assert "last data" in rendered


def test_display_complete_for_adapted_preview_data() -> None:
    metadata = {"file_eof_hit": True, "total_rows": 12}

    assert _display_complete_for_data(metadata) is True
    assert _viewer_completeness(metadata)["display_complete"] is True

    capped = {"file_eof_hit": True, "total_rows": 12, "ui_limit_hit": True}
    assert _display_complete_for_data(capped) is False
    assert _viewer_completeness(capped)["display_complete"] is False

    ram_limited = {"file_eof_hit": True, "ram_limit_hit": True}
    assert _display_complete_for_data(ram_limited) is False
    assert _viewer_completeness(ram_limited)["display_complete"] is False


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
    rendered = _render_task_input(
        task_def,
        ui_state,
        input_preview_api_url="/preview",
    )

    assert ui_state["adapter_enabled"]
    assert 'name="task_config[input_adapters][source]"' in rendered
    assert f'value="{ui_state["adapter_value"]}"' in rendered
    assert 'data-adapter-control hidden' not in rendered
    assert 'data-preview-url="/preview"' in rendered
    assert 'data-adapter-preview' in rendered


def test_adapter_priority_orders_choices_and_selects_default(
    temp_session, tmp_path, monkeypatch
):
    class NormalPriorityAdapter(BaseAdapter):
        pass

    class HighPriorityAdapter(BaseAdapter):
        priority = 1

    artifact_path = tmp_path / "source.json"
    artifact_path.write_text('[{"id": 1}]', encoding="utf-8")
    artifact = temp_session.register_artifact(
        artifact_path,
        name="Source",
        data_type="json",
    )
    monkeypatch.setattr(
        Artifact,
        "get_adapters",
        lambda self: [NormalPriorityAdapter(), HighPriorityAdapter()],
    )
    task_def = TaskDefinition(
        key="test.priority_artifact_input",
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

    assert [choice["value"] for choice in ui_state["adapter_choices"]] == [
        "HighPriorityAdapter",
        "NormalPriorityAdapter",
    ]
    assert ui_state["adapter_value"] == "HighPriorityAdapter"


def test_adapt_as_filters_and_orders_artifact_adapter_choices(
    temp_session, tmp_path, monkeypatch
):
    artifact_path = tmp_path / "source.json"
    artifact_path.write_text('[{"id": 1}]', encoding="utf-8")
    artifact = temp_session.register_artifact(
        artifact_path,
        name="Source",
        data_type="json",
    )
    available_adapters = [TextAdapter(), JsonAdapter(), BaseAdapter()]
    monkeypatch.setattr(
        Artifact,
        "get_adapters",
        lambda self: available_adapters if self.id == artifact.id else [],
    )
    task_def = TaskDefinition(
        key="test.constrained_artifact_input",
        handler=lambda ctx, **kwargs: None,
        input_model=ConstrainedArtifactTaskInputs,
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

    assert [choice["value"] for choice in ui_state["adapter_choices"]] == [
        "JsonAdapter",
        "TextAdapter",
    ]
    assert ui_state["adapter_value"] == "JsonAdapter"
    assert re.search(r'value="JsonAdapter"\s+selected', rendered)


def test_adapt_as_filters_upstream_adapter_choices(temp_session, monkeypatch):
    task_def = TaskDefinition(
        key="test.constrained_reference_input",
        handler=lambda ctx, **kwargs: None,
        input_model=ConstrainedArtifactTaskInputs,
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
    monkeypatch.setattr(
        configure_task.adapter_registry,
        "get_for_type",
        lambda **kwargs: [TextAdapter(), JsonAdapter()],
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

    assert [
        choice["value"] for choice in ui_state["adapter_choices"]
    ] == ["JsonAdapter", "TextAdapter"]
    assert ui_state["adapter_value"] == "JsonAdapter"


def test_input_artifact_preview_endpoint_renders_adapter_output(
    client, temp_runtime, tmp_path, monkeypatch
):
    task_def = TaskDefinition(
        key="test.input_artifact_preview",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    monkeypatch.setattr(
        configure_task.task_registry,
        "get",
        lambda task_key: task_def if task_key == task_def.key else None,
    )
    artifact_path = tmp_path / "source.json"
    artifact_path.write_text('[{"id": 1}]', encoding="utf-8")
    session = temp_runtime.create_session(name="Input artifact preview")
    with session:
        artifact = session.register_artifact(
            artifact_path,
            name="Preview source",
            data_type="json",
        )

    adapter = artifact.get_adapters()[0]
    response = client.post(
        f"/sessions/{session.id}/tasks/configure/{task_def.key}/input-preview",
        json={
            "artifact_ids": [artifact.id],
            "adapter_name": adapter.name,
        },
    )

    assert response.status_code == 200
    assert "Preview source" in response.text
    assert adapter.name in response.text


def test_task_output_preview_uses_adapted_display_completeness(
    client, temp_runtime, monkeypatch
):
    task_def = TaskDefinition(
        key="test.task_output_adapted_completeness",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    monkeypatch.setattr(
        configure_task.task_registry,
        "get",
        lambda task_key: task_def if task_key == task_def.key else None,
    )
    monkeypatch.setitem(
        task_routes.task_registry._tasks,
        task_def.key,
        task_def,
    )
    monkeypatch.setattr(task_routes, "form_json_to_dict", lambda *args: {})
    monkeypatch.setattr(task_routes, "infer_bindings_from_raw", lambda *args: {})

    preview_payload = PreviewPayload(
        task_key=task_def.key,
        is_dry_run=False,
        resolved_inputs={},
        execution_config={},
        output_previews={
            "result": PreviewOutput(
                data=[{"summary": "All source data represented"}],
                data_type="json",
                result_complete=True,
                display_complete=False,
                truncation_reason="sampled",
                io_metadata={
                    "extension": "json",
                    "file_eof_hit": False,
                    "total_rows": None,
                },
            )
        },
    )
    monkeypatch.setattr(
        temp_runtime,
        "preview_task",
        lambda **kwargs: preview_payload,
    )

    class SummaryAdapter(TabularAdapter):
        name = "SummaryAdapter"

        def adapt(self, raw_data, config=None, **kwargs):
            return [{"summary": "All source data represented"}]

    monkeypatch.setattr(
        task_routes.adapter_registry,
        "get_for_type",
        lambda **kwargs: [SummaryAdapter()],
    )

    session = temp_runtime.create_session(name="Task output preview completeness")
    response = client.post(
        f"/sessions/{session.id}/tasks/configure/{task_def.key}/preview",
        json={"task_config": {}},
    )

    assert response.status_code == 200
    assert "All source data represented" in response.text
    assert "1 cols ×" in response.text
    assert "unknown" in response.text
    assert "PREVIEW (Click to load full)" in response.text
    assert "Showing: Top 1" in response.text


def test_input_artifact_preview_full_strategy_loads_remaining_rows(
    client, temp_runtime, tmp_path, monkeypatch
):
    task_def = TaskDefinition(
        key="test.input_artifact_full_preview",
        handler=lambda ctx, **kwargs: None,
        input_model=ArtifactTaskInputs,
        output_model=ArtifactTaskOutputs,
    )
    monkeypatch.setattr(
        configure_task.task_registry,
        "get",
        lambda task_key: task_def if task_key == task_def.key else None,
    )
    monkeypatch.setattr(temp_runtime.settings, "preview_peek_limit", 10)

    artifact_path = tmp_path / "source.csv"
    artifact_path.write_text(
        "id\n" + "".join(f"row-{index}\n" for index in range(12)),
        encoding="utf-8",
    )
    session = temp_runtime.create_session(name="Input artifact full preview")
    with session:
        artifact = session.register_artifact(
            artifact_path,
            name="Preview source",
            data_type="csv",
        )

    adapter = next(
        adapter for adapter in artifact.get_adapters()
        if adapter.name == "CsvAdapter"
    )
    preview_url = (
        f"/sessions/{session.id}/tasks/configure/{task_def.key}/input-preview"
    )
    peek_response = client.post(
        preview_url,
        json={"artifact_ids": [artifact.id], "adapter_name": adapter.name},
    )
    full_response = client.post(
        preview_url,
        json={
            "artifact_ids": [artifact.id],
            "adapter_name": adapter.name,
            "strategy": "full",
        },
    )

    assert peek_response.status_code == 200
    assert "row-11" not in peek_response.text
    assert "Showing: Top" in peek_response.text
    assert "PREVIEW (Click to load full)" in peek_response.text
    assert full_response.status_code == 200
    assert "row-11" in full_response.text
    assert "Showing: Top" not in full_response.text
    assert "PREVIEW (Click to load full)" not in full_response.text
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
