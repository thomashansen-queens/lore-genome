"""
Routes for inspecting and managing the Runtime and global state.
"""
import collections
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse

from lore.core.adapters import adapter_registry
from lore.core.tasks import task_registry
from lore.core.utils import fmt_bytes
from lore.web.deps import RT, PageContext, templates


router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.get("/", response_class=RedirectResponse)
def runtime_root():
    """Redirect to the most important Runtime page."""
    return RedirectResponse("/runtime/log")


@router.get("/log", response_class=HTMLResponse)
def view_log(rt: RT, ctx: PageContext = Depends()):
    """View the runtime log."""
    runtime_log = "No log file found."

    # 1. Dynamically find the most recent log file path
    log_file_path = None
    for handler in rt.logger.handlers:
        if hasattr(handler, "baseFilename"):
            log_file_path = getattr(handler, "baseFilename")
            break

    # 2. Efficiently tail the last 1000 lines using deque
    if log_file_path:
        try:
            with open(log_file_path, "r", encoding="utf-8") as f:
                tail_lines = collections.deque(f, maxlen=1000)
                runtime_log = "".join(tail_lines)
        except Exception as e:
            runtime_log = f"Error reading log file: {e}"

    ctx.generate_breadcrumbs({"runtime": "Runtime", "log": "Log"})
    return templates.TemplateResponse(
        "features/runtime/log.html",
        ctx.render(
            log=runtime_log,
        ),
    )


@router.get("/log/poll", response_class=PlainTextResponse)
def poll_log(rt: RT):
    """Raw text poller for the log."""
    log_file_path = None
    for handler in rt.logger.handlers:
        if hasattr(handler, "baseFilename"):
            log_file_path = getattr(handler, "baseFilename")
            break

    if not log_file_path:
        return "No log file found."

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            tail_lines = collections.deque(f, maxlen=1000)
            return "".join(tail_lines)
    except Exception as e:
        return f"Error reading log file: {e}"

# --- Task registry ---

@router.get("/tasks", response_class=HTMLResponse)
def view_task_registry(rt: RT, ctx: PageContext = Depends()):
    """View all registered tasks available to the Runtime."""
    sorted_tasks = sorted(
        task_registry.all.values(),
        key=lambda t: (t.category or "Z", t.name),
    )
    ctx.generate_breadcrumbs({"runtime": "Runtime", "tasks": "Task registry"})
    return templates.TemplateResponse(
        "features/runtime/task_registry.html",
        ctx.render(task_definitions=sorted_tasks),
    )


# --- Adapter registry ---

@router.get("/adapters", response_class=HTMLResponse)
def view_adapter_registry(rt: RT, ctx: PageContext = Depends()):
    """View all registered adapters available to the Runtime."""
    sorted_adapters = sorted(
        adapter_registry._adapters.values(),
        key=lambda a: (a.name, a.version),
    )
    ctx.generate_breadcrumbs({"runtime": "Runtime", "adapters": "Adapter registry"})
    return templates.TemplateResponse(
        "features/runtime/adapter_registry.html",
        ctx.render(adapters=sorted_adapters),
    )


# --- Cache ---

def _build_pickle_fields(info: dict) -> list[dict]:
    """
    Convert raw inspect_cas_item() output into a flat list of display rows.
    Each row: {"label": str, "value": str, "fmt": "plain" | "code" | "pre"}
    Add new fields here as inspect_cas_item() grows — template needs no changes.
    """
    fields = []

    def add(label: str, value: str, fmt: str = "plain"):
        fields.append({"label": label, "value": value, "fmt": fmt})

    add("Type", f"{info['module']}.{info['type']}", "code")

    if "length" in info:
        add("Length", str(info["length"]))

    if "shape" in info:
        add("Shape", " × ".join(str(d) for d in info["shape"]), "code")

    if "element_type" in info:
        add("Element Type", info["element_type"], "code")

    if "columns" in info:
        add("Columns", ", ".join(str(c) for c in info["columns"]), "code")

    if "dtypes" in info:
        add("dtypes", ", ".join(f"{k}: {v}" for k, v in info["dtypes"].items()), "code")

    if "keys" in info:
        value = ", ".join(str(k) for k in info["keys"])
        if info.get("keys_truncated"):
            value += " …"
        add("Keys", value, "code")

    if "zip_namelist" in info:
        add("Zip Contents", "\n".join(info["zip_namelist"]), "pre")

    if info.get("repr_preview"):
        add("repr", info["repr_preview"], "pre")

    return fields


@router.get("/cache", response_class=HTMLResponse)
def view_cache_overview(rt: RT, ctx: PageContext = Depends()):
    """View the cache overview: L1/L2 stats and all CAS entries."""
    ctx.generate_breadcrumbs({"runtime": "Runtime", "cache": "Cache"})

    raw = rt.cache.get_stats()
    stats = {
        "l1_count": raw["l1_count"],
        "l1_usage": f"{fmt_bytes(raw['l1_bytes'])} / {fmt_bytes(raw['l1_max_bytes'])}",
        "l1_pct": round(raw["l1_bytes"] / raw["l1_max_bytes"] * 100, 1) if raw["l1_max_bytes"] else 0,
        "l2_count": raw["l2_count"],
        "l2_usage": f"{fmt_bytes(raw['l2_bytes'])} / {fmt_bytes(raw['l2_max_bytes'])}",
        "l2_pct": round(raw["l2_bytes"] / raw["l2_max_bytes"] * 100, 1) if raw["l2_max_bytes"] else 0,
        "cas_dir": raw["cas_dir"],
    }

    items = rt.cache.list_cas_items()
    for item in items:
        item["size"] = fmt_bytes(item["size_bytes"])
        item["modified"] = datetime.fromtimestamp(item["last_modified"]).strftime("%Y-%m-%d %H:%M:%S")

    return templates.TemplateResponse(
        "features/runtime/cache.html",
        ctx.render(stats=stats, items=items),
    )


@router.get("/cache/view/{key}", response_class=HTMLResponse)
def view_cache_item(key: str, rt: RT, ctx: PageContext = Depends(), inspect: bool = False):
    """Inspect file-level metadata for a single CAS entry. Pass ?inspect=1 to load the pickle."""
    raw = rt.cache.get_cas_item(key)
    if raw is None:
        raise HTTPException(status_code=404, detail="Cache entry not found")

    ctx.generate_breadcrumbs({"runtime": "Runtime", "cache": "Cache", "view": "View"})

    item = {
        "key": raw["key"],
        "path": raw["path"],
        "size": fmt_bytes(raw["size_bytes"]),
        "size_bytes": raw["size_bytes"],
        "modified": datetime.fromtimestamp(raw["last_modified"]).strftime("%Y-%m-%d %H:%M:%S"),
        "accessed": datetime.fromtimestamp(raw["last_accessed"]).strftime("%Y-%m-%d %H:%M:%S"),
        "created": datetime.fromtimestamp(raw["created"]).strftime("%Y-%m-%d %H:%M:%S"),
        "in_ram": raw["in_ram"],
    }

    pickle_info = rt.cache.inspect_cas_item(key) if inspect else None
    pickle_fields = _build_pickle_fields(pickle_info) if pickle_info and not pickle_info.get("error") else None

    return templates.TemplateResponse(
        "features/runtime/cache_detail.html",
        ctx.render(item=item, pickle_info=pickle_info, pickle_fields=pickle_fields),
    )


@router.post("/cache/clear")
def clear_cache(rt: RT, ctx: PageContext = Depends()):
    """Nuke both L1 and L2 caches."""
    rt.cache.clear_cache()
    return ctx.redirect("/runtime/cache", message="Cache cleared.", message_type="success")
