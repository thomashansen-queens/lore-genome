"""
Disk IO and factory methods for Workflows.
"""

import json
from pathlib import Path
from typing import Any, List, Dict

from lore.core.bindings import Binding, ReferenceBinding, LiteralBinding, UserInputBinding
from lore.core.sessions import Session
from lore.core.tasks.models import Task, TaskStatus
from lore.core.workflows.models import Workflow
from lore.core.utils import auto_increment, slugify


class WorkflowManager:
    """
    Manages the global library of Workflow templates.
    """
    def __init__(self, user_dir: Path | str, read_dirs: list[Path | str] | None = None):
        # 1. Primary directory for user-created workflows
        self.user_dir = Path(user_dir)
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # 2. Additional directories to scan for workflows (user dir + builtins)
        self.read_dirs = [self.user_dir]
        if read_dirs:
            self.read_dirs.extend([Path(d) for d in read_dirs])

    # --- Disk IO ---

    def _generate_path(self, name: str, extension: str) -> Path:
        """Generates a filesystem path for an artifact based on its ID and name."""
        safe_name = slugify(name)
        return self.user_dir / f"{safe_name}.{extension}"

    def list_workflows(self) -> List[Dict[str, str]]:
        """
        Scans the workflows directory and returns a list of available workflow 
        templates with basic metadata.
        """
        workflows = []
        seen_ids = set()

        for dir in self.read_dirs:
            if not dir.exists():
                continue

            is_builtin = dir != self.user_dir

            for file in dir.glob("*.json"):
                try:
                    with open(file) as f:
                        data = json.load(f)
                        wf_id = data.get("id", file.stem)
                        
                        # Avoid user-shadowed duplicates of built-in workflows
                        if wf_id in seen_ids:
                            continue

                        seen_ids.add(wf_id)

                        workflows.append({
                            "id": wf_id,
                            "name": data.get("name", file.stem),
                            "description": data.get("description", ""),
                            "task_count": len(data.get("tasks", [])),
                            "created_at": data.get("created_at", "unknown"),
                            "is_builtin": is_builtin,
                        })
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error decoding JSON from: {file}. Error: {e}")
        return workflows

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        """Loads a JSON template from disk into validated Pydantic model."""
        # 1. Fast path: Look for slugified filename match
        target_filename = f"{slugify(workflow_id)}.json"
        for dir in self.read_dirs:
            target = dir / target_filename
            if target.exists():
                with open(target) as f:
                    data = json.load(f)
                return Workflow(**data)
        # 2. Fallback: Scan the files for a matching "id" field
        for dir in self.read_dirs:
            if not dir.exists():
                continue
            for file in dir.glob("*.json"):
                try:
                    with open(file) as f:
                        data = json.load(f)
                        if data.get("id", file.stem) == workflow_id:
                            return Workflow(**data)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error decoding JSON from: {file}. Error: {e}")
        return None

    def import_workflow(self, json_data: bytes) -> Workflow:
        """Validates and saves a workflow from raw JSON bytes."""
        try:
            workflow = Workflow.model_validate_json(json_data)
        except Exception as e:
            raise ValueError(f"Invalid workflow JSON format: {str(e)}")

        existing_ids = {w["id"] for w in self.list_workflows()}
        if workflow.id in existing_ids:
            workflow.id = auto_increment(workflow.id, existing_ids)
            workflow.name = f"{workflow.name} (Imported)"

        self.save_workflow(workflow)
        return workflow

    def rename_workflow(self, workflow_id: str, new_name: str) -> Workflow:
        """Updates the human-readable name of a Workflow template. ID unchanged."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        old_path = self._generate_path(workflow.id, "json")
        workflow.name = new_name

        new_id = slugify(new_name)
        existing_ids = {w["id"] for w in self.list_workflows()} - {workflow_id}
        if new_id in existing_ids:
            new_id = auto_increment(new_id, existing_ids)
        workflow.id = new_id

        self.save_workflow(workflow)
        # Remove old file if ID changed
        if old_path.exists() and old_path != self._generate_path(workflow.id, "json"):
            old_path.unlink()

        return workflow

    def update_workflow_description(self, workflow_id: str, new_description: str) -> Workflow:
        """Updates the description of a Workflow template."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        workflow.description = new_description
        self.save_workflow(workflow)
        return workflow

    def save_workflow(self, workflow: Workflow) -> Path:
        """Writes a Workflow model to disk as a JSON template in the User dir."""
        target = self._generate_path(workflow.id, "json")
        with open(target, "w") as f:
            f.write(workflow.model_dump_json(indent=2))
        return target

    def clone_workflow(self, workflow_id: str) -> Workflow:
        """Creates a copy of an existing workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        existing_ids = {w["id"] for w in self.list_workflows()}
        new_id = auto_increment(workflow_id, existing_ids)

        cloned_workflow = workflow.model_copy(deep=True, update={
            "id": new_id,
            "name": f"{workflow.name} (Copy)"
        })

        self.save_workflow(cloned_workflow)
        return cloned_workflow

    def delete_workflow(self, workflow_id: str) -> None:
        """Deletes a workflow template from disk."""
        target = self._generate_path(workflow_id, "json")
        if target.exists():
            target.unlink()
        else:
            if self.get_workflow(workflow_id):
                raise ValueError(
                    "Cannot currently delete built-in workflows. You can manually do so by "
                    "removing the JSON file from the lore/builtins/workflows/ directory."
                )

    # --- Factory methods ---

    def extract_from_session(
        self,
        session: Session,
        new_workflow_id: str,
        name: str | None = None,
    ) -> Workflow:
        """
        Dehydrates a live Session into a reusable Workflow template.
        """
        from lore.core.tasks import task_registry

        # 1. First pass: Map Artifacts to the Task that created them
        # This converts hardcoded LiteralBindings (Artifact IDs) into ReferenceBindings
        artifact_to_creator = {}
        for task in session.list_tasks():
            for output_key, artifact_list in task.outputs.items():
                for art_id in (artifact_list or []):
                    artifact_to_creator[art_id] = (task.id, output_key)

        # 2. Second pass: build the Workflow Tasks
        workflow_tasks = []
        for task in session.list_tasks():
            task_def = task_registry.get(task.registry_key)
            if task_def is None:
                raise ValueError(f"Task definition not found for task key: {task.registry_key}")

            dehydrated_inputs = {}

            # 3. Dehydrate each TaskInput into a list of Bindings
            #    (allows for missing TaskDefinition from shared workflows)
            all_keys = set(task_def.input_model.model_fields.keys()) | set(task.inputs.keys())

            for input_key in all_keys:
                _, extra = task_def.field_meta(input_key)
                accepts_artifact = extra.get("is_artifact", False)

                dehydrated_list = []
                seen_refs = set()  # Avoid duplicate bindings between input_slot and output_slot

                bindings = task.inputs.get(input_key, [])
                for b in bindings:
                    # A. Already a UserInputBinding
                    if isinstance(b, UserInputBinding):
                        dehydrated_list.append(b)

                    # B. Already a Reference. Keep, but ensure source ID
                    elif isinstance(b, ReferenceBinding):
                        ref_sig = (b.source_id, b.output_key)
                        if ref_sig not in seen_refs:
                            seen_refs.add(ref_sig)
                            dehydrated_list.append(ReferenceBinding(
                                source_id=b.source_id,
                                output_key=b.output_key,
                            ))

                    # C. LiteralBinding may be to a concrete Artifact; convert to Reference
                    elif isinstance(b, LiteralBinding):
                        # c1. Upgrade an Artifact to a DAG edge binding
                        if accepts_artifact and b.value in artifact_to_creator:
                            creator_task_id, out_key = artifact_to_creator[b.value]
                            ref_sig = (creator_task_id, out_key)

                            if ref_sig not in seen_refs:
                                seen_refs.add(ref_sig)
                                dehydrated_list.append(ReferenceBinding(
                                    source_id=creator_task_id,
                                    output_key=out_key,
                                ))

                        # c2. Upgrade an input to a DAG start node
                        elif accepts_artifact:
                            # Artifact must be provided by user at runtime
                            if session.get_artifact(b.value):
                                dehydrated_list.append(UserInputBinding(input_key=input_key))
                            # Value must be provided by user at runtime
                            else:
                                dehydrated_list.append(LiteralBinding(value=b.value))

                        # D. ValueInput (primitive literal)
                        else:
                            dehydrated_list.append(b)

                dehydrated_inputs[input_key] = dehydrated_list

            workflow_tasks.append(Task(
                id=task.id,
                registry_key=task.registry_key,
                inputs=dehydrated_inputs,
                name=task.name,
                description=task.description or "",
                status=TaskStatus.TEMPLATE,
            ))

        # 4. Check integrity of new Workflow
        from lore.core.topology.traversal import sort_tasks_topologically, DAGValidationError
        try:
            sort_tasks_topologically(workflow_tasks)
        except DAGValidationError as e:
            raise ValueError(f"Cannot extract Workflow. The Session's topology is invalid: {e}")

        if name is None:
            name = f"Workflow from {session.name}"

        return Workflow(id=new_workflow_id, name=name, tasks=workflow_tasks)

    def hydrate_workflow(
        self,
        workflow: Workflow,
        session: "Session",
        runtime_inputs: dict[str, Any] | None = None,
    ) -> None:
        """
        Injects a Workflow template into a live Session.
        If provided, further injects UserInputBindings to Tasks.
        """
        from lore.core.topology.traversal import sort_tasks_topologically
        if runtime_inputs is None:
            runtime_inputs = {}

        # 1. Validate the DAG and get the execution order
        sorted_tasks = sort_tasks_topologically(workflow.tasks)

        # 2. Map of template Task IDs (Workflow) to live Task IDs (Session)
        template_to_live_map = {}

        # 3. Create the Tasks in topological order
        for template_task in sorted_tasks:
            live_task = session.add_task(
                registry_key=template_task.registry_key,
                name=template_task.name or f"{workflow.name} - {template_task.registry_key}",
            )
            template_to_live_map[template_task.id] = live_task.id

            # 4. Translate the Bindings into the Task's inputs
            task_inputs = {}
            for input_key, bindings in template_task.inputs.items():
                translated_bindings = []

                for idx, binding in enumerate(bindings):
                    if isinstance(binding, ReferenceBinding):
                        translated_bindings.append(ReferenceBinding(
                            source_id=template_to_live_map[binding.source_id],
                            output_key=binding.output_key,
                        ))
                    elif isinstance(binding, UserInputBinding):
                        # Magic UI assignment format
                        lookup_key = f"{template_task.id}__{input_key}__{idx}"
                        if lookup_key in runtime_inputs:
                            # Inject user value; Task.update() will coerce to correct Binding type
                            translated_bindings.append(runtime_inputs[lookup_key])
                        else:
                            # No input; leave as UserInputBinding to be filled in later
                            translated_bindings.append(binding)
                    else:
                        # LiteralBinding is passed through as-is
                        translated_bindings.append(binding)

                task_inputs[input_key] = translated_bindings

            # 5. Push the translated inputs to the Task and set its state
            live_task.update(inputs=task_inputs)
            session.mark_dirty()

    # --- Task management ---

    def get_task(self, workflow_id: str, task_id: str) -> Task | None:
        """Retrieve a specific task from a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        return workflow.get_task(task_id)

    def rename_task(self, workflow_id: str, task_id: str, new_name: str) -> Task:
        """Update the name of a specific task within a workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        task = workflow.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found in workflow: {task_id}")

        task.update(name=new_name)

        self.save_workflow(workflow)
        return task

    def update_task(
        self,
        workflow_id: str,
        task_id: str,
        description: str,
        new_inputs: dict[str, list[Binding]],
    ) -> Task:
        """Updates a Workflow Task's inputs and description."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        task = workflow.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found in workflow: {task_id}")

        task.description = description
        task.update(inputs=new_inputs)

        self.save_workflow(workflow)
        return task

    def delete_task(self, workflow_id: str, task_id: str) -> Workflow:
        """Removes a task from the Workflow."""
        workflow = self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {workflow_id}")

        task = workflow.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found in workflow: {task_id}")

        workflow.tasks.remove(task)
        self.save_workflow(workflow)
        return workflow
