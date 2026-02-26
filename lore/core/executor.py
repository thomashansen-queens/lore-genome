"""
Task execution logic. Is the engine behind Session and TaskRecord.
Uses Sandwich pattern (load -> run -> save) to manage task lifecycle within a session.
Allows long-running tasks to run without holding locks on the session manifest.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from pathlib import Path
import logging
import tempfile

from lore.core.tasks import task_registry, Materialization, Cardinality

if TYPE_CHECKING:
    from lore.core.artifacts import Artifact
    from lore.core.runtime import Runtime
    from lore.core.session import Session
    from lore.core.tasks import Task, TaskDefinition

# --- Input resolution ---

def _extract_field_meta(task_def: "TaskDefinition", key: str) -> tuple[Any, dict[str, Any]]:
    """
    Validates and extracts metadata for a given input field in a TaskDefinition.
    """
    model_fields = task_def.input_model.model_fields
    field_info = model_fields.get(key)
    if field_info is None:
        raise ValueError(f"Input key '{key}' not defined in TaskDefinition input model.")

    extra = field_info.json_schema_extra
    if extra is None or not isinstance(extra, dict):
        raise ValueError(f"Input key '{key}' missing JSON schema extra metadata for resolution.")

    return field_info, extra


def _resolve_task_inputs(s: "Session", task_def, raw_inputs: dict) -> tuple[dict, dict]:
    """
    Resolves raw input references (i.e. Artifacts) to actual data based on the 
    TaskDefinition's DSL instructions (Materialization, series extraction) and
    available Adapters.
    """
    resolved = {}
    input_artifacts_snapshot = {}  # key: input_name, value: list[Artifact] or Artifact

    for key, value in raw_inputs.items():
        # 1. Get input field metadata from TaskDefinition
        field_info, extra = _extract_field_meta(task_def, key)

        if not extra.get("is_artifact"):
            # primitive (e.g. dict, str, int), pass through
            resolved[key] = value
            continue

        # 2. Extract Metadata from DSL
        load_as = extra.get("load_as")
        cardinality = extra.get("cardinality")
        if not load_as or not cardinality:
            raise ValueError(f"Task input '{key}' missing 'load_as' or 'cardinality'")
        accepted_data = extra.get("accepted_data", [])

        # 3. Resolve IDs to Artifacts (bulk fetch then snapshot)
        # Duck typing: If input is an artifact ID, treat it as one! Otherwise, 
        # allow users to manually enter input for fields that accept artifacts
        resolved_list = []
        artifacts = []
        task_inputs = value if isinstance(value, list) else ([value] if value else [])
        for val in task_inputs:
            if not val:
                continue  # val is falsy (e.g. None, empty)
            try:
                artifacts.append(s.get_artifact(val))
            except ValueError:
                resolved_list.append(val)
        input_artifacts_snapshot[key] = artifacts

        # 4. Process each Artifact according to the instructions
        processed_artifacts = []  # auto-concatenate if multiple items
        for a in artifacts:
            item_data = _materialize_single_artifact(s, a, load_as, accepted_data)
            processed_artifacts.append(item_data)

        # 5. Handle packaging & concatenation
        # Auto-concatenate if it's a series type and multiple items are allowed
        if Cardinality(cardinality).allows_multiple:
            flattened = resolved_list.copy()  # start with non-input artifacts
            for item in processed_artifacts:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            resolved[key] = flattened
        else:
            if resolved_list:
                resolved[key] = resolved_list[0]  # manual input takes precedence
            else:
                resolved[key] = processed_artifacts[0] if processed_artifacts else None

    return resolved, input_artifacts_snapshot


def _materialize_single_artifact(
    s: "Session", artifact: "Artifact", load_as: str, accepted_data: list[str],
) -> Any:
    """
    Helper to Materialize an Artifact into real data per DSL instructions
    If loading as CONTENT, will prioritize the narrowest type of accepted data
    i.e. Series > Adapted > Raw
    """
    m = Materialization(load_as)
    path = s.get_artifact_path(artifact.id)

    if m == Materialization.PATH:
        return str(path)

    if m == Materialization.STREAM:
        # FUTURE: Handler MUST close this stream or we have a leak
        return path.open("rb")

    if m == Materialization.PREVIEW:
        content, _ = s.preview_artifact(artifact.id, mode="bytes")
        return content

    if m == Materialization.CONTENT:
        # FUTURE: handle this more gracefully for large files
        raw_data = s.load_artifact_data(artifact.id)

        # Try to provide a series
        adapters = artifact.get_adapters()
        for adapter in adapters:
            for accepted in accepted_data:
                if accepted in adapter.get_keys():
                    series = adapter.get_series(raw_data, accepted)
                    if series is not None:
                        return series

        # If no series, adapt to table
        for adapter in adapters:
            for accepted in accepted_data:
                if adapter.provides(accepted):
                    adapted = adapter.adapt(raw_data)
                    if adapted is not None:
                        return adapted

        # Fallback to raw content if no adapters or can't adapt to target
        return raw_data

    return artifact.id  # fallback to ID if no instructions


class TaskResults:
    """A simple container for Task outputs."""
    _allowed_keys: tuple[str, ...]

    def __init__(self, output_keys: list[str]):
        # Create list of allowed attribute keys based on TaskDefinition
        super().__setattr__("_allowed_keys", tuple(output_keys))
        for key in output_keys:
            object.__setattr__(self, key, None)  # bypass guard to set initial None values

    def __setattr__(self, name: str, value: Any):
        # Once initialized, only allow setting attribs defined by TaskDefinition
        if name == "_allowed_keys":
            if hasattr(self, "_allowed_keys"):
                raise AttributeError("_allowed_keys is immutable after initialization")
            object.__setattr__(self, name, value)
            return

        try:
            allowed = self._allowed_keys
        except AttributeError:
            object.__setattr__(self, name, value)  # allow setting _allowed_keys during init
            return

        if name not in allowed:
            raise AttributeError(
                f"Cannot set unknown output key: '{name}'\n"
                f"Allowed keys are: {', '.join(allowed)}"
            )
        object.__setattr__(self, name, value)

    def next_empty_key(self) -> str | None:
        """Returns the next empty output slot key, or None if all are filled"""
        for k in self._allowed_keys:
            if getattr(self, k) is None:
                return k
        return None

    def to_dict(self) -> dict:
        """Serialize results to a dict for storage in Manifest"""
        return {k: getattr(self, k) for k in self._allowed_keys}


@dataclass
class ExecutionContext:
    """
    Context object that doesn't hold a file lock. Holds rt and session_id
    in case it needs momentary access to session data (e.g. saving artifacts).
    Helpers safely open the session as needed.
    """
    runtime: "Runtime"
    session_id: str
    task: "Task"
    task_def: "TaskDefinition"
    input_artifacts: dict[str, list["Artifact"]] = field(default_factory=dict)
    results: TaskResults = field(default_factory=lambda: TaskResults([]))
    _temp_dir: tempfile.TemporaryDirectory | None = field(default=None, init=False)

    @property
    def logger(self) -> logging.Logger:
        """Access the runtime logger."""
        return self.runtime.logger

    def get_input_artifact(self, key: str) -> "list[Artifact] | None":
        """Get input Artifact record"""
        return self.input_artifacts.get(key)

    def get_temp_path(self, filename: str) -> Path:
        """Generates an isolated path in the task's scratch directory."""
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix=f"task_{self.task.id[:8]}_")

        return Path(self._temp_dir.name) / filename

    def cleanup(self):
        """Manually trigger the deletion of the scratch directory"""
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _get_inferred_parents(self) -> list[str]:
        """Helper to infer parents artifact_id lineage from Tasks's inputs"""
        return [a.id for artifacts in self.input_artifacts.values() for a in artifacts]

    def _generate_default_name(self, output_key: str | None, data_type: str) -> str:
        """
        Derives a human-readable name from the Task definition's Output label
        """
        label = None
        if output_key and self.task_def and self.task_def.output_model:
            field_info = self.task_def.output_model.model_fields.get(output_key)
            if field_info and hasattr(field_info.default, "label"):
                label = field_info.default.label

        if self.task.name:
            return self.task.name[:12] + "_" + (label or output_key or data_type or "output")
        return (label or output_key or data_type or "output")

    def materialize_file(
        self,
        source_path: Path,
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        metadata: dict | None = None,
        move: bool = True,
        **kwargs,
    ) -> "Artifact":
        """Adopts an existing file (e.g. temp file) in the Session"""
        # Guard and automatic lineage
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file for materialization not found: {source_path}")

        # If not specified, fill outputs positionally
        if output_key is None:
            output_key = self.results.next_empty_key()
            if output_key is None:
                raise ValueError("Cannot assign TaskOutput: all slots are full.")

        if data_type is None:
            field_info = self.task_def.output_model.model_fields.get(output_key, None)
            if field_info is not None and field_info.json_schema_extra is not None:
                dt = field_info.json_schema_extra.get("data_type")
                if isinstance(dt, str):
                    data_type = dt
            if not data_type:
                data_type = "unknown"

        if not name:
            name = self._generate_default_name(output_key, data_type)

        with self.runtime.get_session(self.session_id) as s:
            artifact = s.ingest_artifact(
                path=source_path,
                name=name,
                created_by_task_id=self.task.id,
                data_type=data_type,
                metadata=metadata,
                parent_artifact_ids=self._get_inferred_parents(),
                move=move,
                **kwargs,
            )
            setattr(self.results, output_key, artifact.id)
            self.logger.info("Task ID: %s emitted Artifact: %s to slot: %s",
                             self.task.id, artifact.id, output_key)
            return artifact

    def materialize_content(
        self,
        content: str | bytes,
        name: str | None = None,
        output_key: str | None = None,
        data_type: str | None = None,
        extension: str = "txt",
        metadata: dict | None = None,
        **kwargs,
    ) -> "Artifact":
        """
        Write in-memory content to tempfile of specified type then register it
        as an Artifact. Takes advantage of Session's atomic write and name
        collision logic.
        """
        import tempfile

        ext = f".{extension.lstrip(".")}"

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=ext) as tmp:
            byte_content = content.encode("utf-8") if isinstance(content, str) else content
            tmp.write(byte_content)
            tmp_path = Path(tmp.name)

        try:
            return self.materialize_file(
                source_path=tmp_path,
                name=name,
                output_key=output_key,
                data_type=data_type,
                metadata=metadata,
                move=True,
                **kwargs,
            )
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()  # Clean up temp file on failure
            raise e


def execute_task(rt: 'Runtime', session_id: str, task_id: str):
    """
    Execute a PENDING task within a Session.
    Uses a sandwich pattern to prevent locking/overwriting.
    """
    logger = rt.logger
    logger.info("Preparing Task ID: '%s' in Session ID: '%s'", task_id, session_id)

    # 1. Prepare Task (short lock)
    task_def = None
    with rt.get_session(session_id) as s:
        task = s.manifest.get_task(task_id)
        if not task:
            raise ValueError(f"Task ID: '{task_id}' not found in session manifest.")
        task_def = task_registry.get(task.registry_key)
        if not task_def:
            raise ValueError(f"Task key: '{task.registry_key}' not found in Task Registry.")
        if task.status != "PENDING":
            logger.warning("Task ID: '%s' is %s: skipping execution.)", task_id, task.status)
            return

        try:
            resolved_inputs, input_artifacts = _resolve_task_inputs(s, task_def, task.inputs)
        except Exception as e:
            task.status = "FAILED"
            task.error = f"Input resolution failed: {str(e)}"
            s.manifest.add_task(task)
            return

        # Update State
        task.status = "RUNNING"
        task.started_at = datetime.now(tz=timezone.utc)
        s.manifest.add_task(task)

    # 2. Execute task (long run, no lock)
    logger.info("Running Task ID: '%s' in Session ID: '%s'", task_id, session_id)
    ctx = None  # Initialize to ensure `finally` block can reference it
    output_keys = task_def.output_model.model_fields.keys() if task_def.output_model else []
    results = TaskResults(output_keys=list(output_keys))
    success = False
    error_msg = None

    try:
        # Execute the handler function
        ctx = ExecutionContext(
            runtime=rt,
            session_id=session_id,
            task=task,
            task_def=task_def,
            input_artifacts=input_artifacts,
            results=results,  # collection basket for handler to populate
        )
        task_def.handler(ctx, **resolved_inputs)

        success = True
        logger.info("Success! Task ID: '%s' in Session ID: '%s'", task_id, session_id)

    except Exception as e:
        success = False
        error_msg = str(e)
        logger.error("Failed! Task ID: '%s' in Session ID: '%s'. Reason: %s",
                     task_id, session_id, error_msg, exc_info=True)

    finally:
        # 3. Finalize task (short lock)
        if ctx:
            ctx.cleanup()

        try:
            logger.info("Finalizing Task ID: '%s' in Session ID: '%s'", task_id, session_id)
            with rt.get_session(session_id) as s:
                task = s.manifest.get_task(task_id)
                if not task:
                    logger.error("Task ID: '%s' in Session ID: '%s' vanished during execution!",
                                task_id, session_id)
                    return
                task.completed_at = datetime.now(tz=timezone.utc)
                if success and ctx:
                    task.status = "COMPLETED"
                    task.outputs = ctx.results.to_dict()
                else:
                    task.status = "FAILED"
                    task.error = error_msg or "Unknown error, check logs for details."
                s.manifest.add_task(task)
        except Exception as e:
            # Last resort
            logger.critical("Critical failure finalizing Task ID: '%s' in Session ID: '%s': %s",
                            task_id, session_id, str(e), exc_info=True)
