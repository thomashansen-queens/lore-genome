"""
Main entry point for the LoRē Genome program. Accessed by 'lore' in the command line.
"""
from functools import update_wrapper
from pathlib import Path
import webbrowser
import click

from lore.core.runtime import build_runtime, Runtime

# Don't need to use invoke, but saving for reference
# def pass_runtime(f):
#     """
#     Decorator to pass a Runtime to Click commands that need it.
#     Ensures a Runtime is created and passed as the first argument.
#     """
#     @click.pass_context
#     def new_func(ctx: click.Context, *args, **kwargs):
#         ctx.ensure_object(dict)
#         rt = ctx.obj.get('rt')
#         if rt is None:
#             opts = ctx.obj.get('global_opts', {})  # user overrides
#             rt = build_runtime(**opts)
#             ctx.obj['rt'] = rt
#         # call the function with the Runtime context
#         return ctx.invoke(f, rt, *args, **kwargs)
#     return update_wrapper(new_func, f)

def pass_runtime(f):
    """
    Decorator to pass a Runtime to Click commands that need it.
    Ensures a Runtime is created and passed as the first argument.
    """
    @click.pass_context
    def new_func(ctx: click.Context, *args, **kwargs):
        ctx.ensure_object(dict)
        rt = ctx.obj.get('rt')
        if rt is None:
            opts = ctx.obj.get('global_opts', {})  # user overrides
            rt = build_runtime(**opts)
            ctx.obj['rt'] = rt
        # call the function with the Runtime context
        return f(ctx.obj['rt'], *args, **kwargs)
    return update_wrapper(new_func, f)

@click.group()
@click.option('--data-root', type=click.Path(path_type=Path), default=None,
              help="Directory for LoRē Genome project data storage.")
@click.option('-v', '--verbose', is_flag=True, default=False,
              help="Very very detailed logging for debugging purposes.")
@click.version_option()
@click.pass_context
def main(ctx, data_root, verbose):
    """LoRē Genome: A pipeline for protein classification using NCBI Datasets."""
    ctx.ensure_object(dict)
    ctx.obj['global_opts'] = {
        'data_root': data_root,
        'verbose': verbose,
    }

@main.command()
@pass_runtime
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=None)
@click.option("--reload", is_flag=True, default=False,
              help="Enable auto-reload for development; code changes are live-reloaded.")
def ui(rt: Runtime, host: str, port: int | None, reload: bool):
    """Launch the LoRē Genome web user interface."""
    # pylint: disable=import-outside-toplevel
    from lore.web.app import run_ui, pick_free_port, make_url
    # make LoRē runtime available to the FastAPI app
    click.echo("LoRē Genome web UI is starting...")
    if port is None:
        port = pick_free_port(host, port)
    webbrowser.open_new_tab(make_url(host, port))
    if reload:
        from lore.web.app import run_ui_reload
        run_ui_reload(rt, host=host, port=port)
    else:
        run_ui(rt, host=host, port=port)

@main.command()
@pass_runtime
@click.argument("task_name")
@click.option("--inputs", "-i", multiple=True, help="Key=Value inputs for the task")
def run(rt: Runtime, task_name: str, inputs: tuple[str]):
    """
    Execute a single task in a new session

    Example: lore run fetch_genome -i accession=NC_000913.3
    """
    task_inputs = {}
    for item in inputs:
        if "=" in item:
            k, v = item.split("=", 1)
            task_inputs[k] = v
        else:
            click.echo(f"Warning: Ignoring invalid input '{item}'. Must be key=value")

    click.echo(f"Initializing session in : {rt.sessions_dir}")

    with rt.create_session() as s:
        click.echo(f"Session ID: {s.id}")
        t = s.add_task(task_name, inputs=task_inputs)
        rt.execute_task(s.id, t.id)
        click.echo(f"Running task: {t.name}")
        # Simulate work for now
        import time  # pylint: disable=import-outside-toplevel
        time.sleep(0.5)
        dummy_file = s.dir / "dummy_output.txt"
        dummy_file.write_text(f"This is a dummy output file for {t.id}.\n")
        s.ingest_artifact(
            path=dummy_file,
            created_by_task_id=t.id,
            metadata={"description": "Dummy output file created by CLI run command."},
        )
        click.echo(f"Task {t.name} completed.")
