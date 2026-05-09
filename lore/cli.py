"""
Main entry point for the LoRe Genome program. Accessed by 'lore' in the command line.
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
        rt = ctx.obj.get("rt")
        if rt is None:
            opts = ctx.obj.get("global_opts", {})  # user overrides
            rt = build_runtime(**opts)
            ctx.obj["rt"] = rt
        # call the function with the Runtime context
        return f(ctx.obj["rt"], *args, **kwargs)
    return update_wrapper(new_func, f)


@click.group()
@click.option("--data-root", type=click.Path(path_type=Path), default=None,
              help="Directory for LoRe Genome project data storage.")
@click.option("-v", "--verbose", is_flag=True, default=False,
              help="Very very detailed logging for debugging purposes.")
@click.version_option()
@click.pass_context
def main(ctx, data_root, verbose):
    """LoRe Genome: A tool for comparitive genomic analysis."""
    ctx.ensure_object(dict)
    ctx.obj["global_opts"] = {
        "data_root": data_root,
        "verbose": verbose,
    }


@main.command()
@pass_runtime
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=None)
@click.option("--reload", is_flag=True, default=False,
              help="Enable auto-reload for development; code changes are live-reloaded.")
def ui(rt: Runtime, host: str, port: int | None, reload: bool):
    """Launch the LoRe Genome web user interface."""
    from lore.web.app import run_ui, pick_free_port, make_url
    # make LoRe runtime available to the FastAPI app
    click.echo("LoRe Genome web UI is starting...")
    if port is None:
        port = pick_free_port(host, port)
    webbrowser.open_new_tab(make_url(host, port))
    if reload:
        from lore.web.app import run_ui_reload
        run_ui_reload(rt, host=host, port=port)
    else:
        run_ui(rt, host=host, port=port)


@main.command(name="execute-task", hidden=True)
@click.option("--session", required=True, help="ID of Session containing the Task.")
@click.option("--task", required=True, help="ID of Task to execute.")
@pass_runtime
def execute_task(rt: Runtime, session: str, task: str):
    """Internal headless worker entrypoint for a single Task execution."""
    rt.logger.info("CLI worker booting up for Task %s in Session %s", task, session)
    from lore.core.execution.worker import run_task_worker

    run_task_worker(rt, session, task)


@main.command(name="execute-session", hidden=True)
@click.option("--session", required=True, help="ID of Session containing the Task.")
@pass_runtime
def execute_session(rt: Runtime, session: str):
    """Internal headless worker entrypoint for a single Task execution."""
    rt.logger.info("CLI orchestrator booting up for Session %s", session)
    from lore.core.execution.orchestrator import SequentialOrchestrator
    SequentialOrchestrator(rt).run_cascade(session)
