"""
Docstring for lore.settings
"""
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field, create_model, field_validator
from typing import Any, Callable, Type

from lore.core.paths import default_data_root

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

DEFAULT_SETTINGS_FILE = "settings.json"
DEFAULT_SECRETS_FILE = "secrets.json"

# --- Plugin config ---

class ConfigRegistry:
    """Global registry for plugin-specific configuration."""
    def __init__(self):
        self._schemas: dict[str, Type[BaseModel]] = {}

    def register(self, key: str, title: str | None = None) -> Callable:
        """Decorator to compile a DSL config class into a Pydantic schema."""
        def wrapper(cls: type) -> type:
            fields = {}

            # 1. Use dir() so MRO is respected, allowing inherited fields
            for attr_name in dir(cls):
                # Skip dunder methods and callables
                if attr_name.startswith("__") or callable(getattr(cls, attr_name)):
                    continue

                attr_value = getattr(cls, attr_name)

                # 2. Duck-typing checks to avoid circular imports with DSL
                if hasattr(attr_value, "to_field_info") and hasattr(attr_value, "get_type_annotation"):
                    field_type = attr_value.get_type_annotation()
                    field_info = attr_value.to_field_info()
                    fields[attr_name] = (field_type, field_info)

            # 3. Create dynamic Pydantic model
            safe_name = f"{key.replace('.', '_')}_ConfigModel"
            pydantic_model = create_model(safe_name, **fields)

            # 4. Attach UI metadata to the new model
            pydantic_model.model_config["title"] = title or key.replace("_", " ").title()
            pydantic_model.__doc__ = cls.__doc__

            # 5. Save the compiled model to the registry
            self._schemas[key] = pydantic_model

            return cls
        return wrapper

    def __getitem__(self, key: str) -> Type[BaseModel]:
        return self._schemas[key]

    @property
    def all(self) -> dict[str, Type[BaseModel]]:
        """Return all registered configuration models."""
        return self._schemas

config_registry = ConfigRegistry()

# --- Global settings ---

class Settings(BaseModel):
    """Configuration settings for LoRe Genome."""
    # --- Paths ---
    data_root: Path = Field(
        default_factory=default_data_root,
        title="Data Root",
        description="Root directory for all LoRe Genome data (sessions, outputs, etc).",
        json_schema_extra={"widget": "text"},
        examples=[str(default_data_root())],
    )
    cache_root: Path | None = Field(
        default=None,
        title="Cache Root",
        description="Path for temp files. Defaults to {data_root}/cache if empty.",
        json_schema_extra={"widget": "text"},
        examples=["leave blank for {data_root}/cache or specify custom path i.e. $SCRATCH/lore_cache"],
    )
    plugins_dir: Path | None = Field(
        default=None,
        title="Plugins Directory",
        description="Path for plugin files. Defaults to {data_root}/plugins if empty.",
        json_schema_extra={"widget": "text"},
        examples=["leave blank for {data_root}/plugins or specify custom path i.e. $SCRATCH/lore_plugins"],
    )
    workflows_dir: Path | None = Field(
        default=None,
        title="Workflows Directory",
        description="Path for custom user workflows. Defaults to {Data Root}/workflows if left blank.",
        json_schema_extra={"widget": "text"},
        examples=["leave blank for {data_root}/workflows or specify custom path i.e. $SCRATCH/lore_workflows"],
    )

    @property
    def active_cache_root(self) -> Path:
        """
        Dynamically resolves the active cache directory. 
        Uses the explicit cache_root if provided, otherwise falls back to data_root/cache.
        """
        if self.cache_root:
            # Expand environment variables e.g. $SCRATCH
            expanded = os.path.expandvars(str(self.cache_root))
            return Path(expanded).expanduser().resolve()
        return self.data_root / "cache"

    @property
    def active_plugins_dir(self) -> Path:
        """
        Dynamically resolves the active plugins directory. 
        Uses the explicit plugins_dir if provided, otherwise falls back to data_root/plugins.
        """
        if self.plugins_dir:
            # Expand environment variables e.g. $SCRATCH
            expanded = os.path.expandvars(str(self.plugins_dir))
            return Path(expanded).expanduser().resolve()
        return self.data_root / "plugins"

    @property
    def active_workflows_dir(self) -> Path:
        """
        Dynamically resolves the active workflows directory.
        Uses the explicit workflows_dir if provided, otherwise falls back to data_root/workflows.
        """
        if self.workflows_dir:
            # Expand environment variables e.g. $SCRATCH
            expanded = os.path.expandvars(str(self.workflows_dir))
            return Path(expanded).expanduser().resolve()
        return self.data_root / "workflows"

    # --- UI ---
    explore_display_limit: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description=(
            "Max number of rows to display in the Explore view. This only "
            "affects the display; all data is still accessible for queries and "
            "sorting. Increase to see more, but with longer load times."
        ),
        json_schema_extra={
            "widget": "slider",
            "min": 100,
            "max": 10000,
            "step": 100,
        },
        examples=["1000"],
    )

    # --- Engine ---
    verbose: bool = False
    cache_ram_mb: int = Field(
        default=500,
        ge=250,
        le=4096,
        description="Maximum RAM in MB to use for in-memory Task caching.",
        json_schema_extra={"widget": "int"},
        examples=["500"],
    )
    cache_disk_gb: float = Field(
        default=4.0,
        ge=0.5,
        description="Maximum disk space in GB to use for .pkl file Task caching.",
        json_schema_extra={"widget": "float"},
        examples=["4.0"],
    )
    io_max_file_size_mb: int = Field(
        default=100,
        ge=1,
        description="Max file size in MB before forcing stream-only materialization (skip full RAM loads).",
        json_schema_extra={"widget": "float"},
        examples=["100"],
    )

    # --- Network/remote --- (FUTURE: for HPC deployments)
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # --- Plugin configs (junk drawer for now) ---
    plugins: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Raw JSON configuration dictionaries for registered plugins.",
    )

    def get_plugin_config(self, plugin_key: str) -> BaseModel | None:
        """Retrieve a plugin's configuration model instance from the registry."""
        if plugin_key not in config_registry.all:
            return None
        schema = config_registry.all[plugin_key]

        # Get config dict if it has been saved. Then, coerce and set defaults with Pydantic.
        raw_data = self.plugins.get(plugin_key, {})
        return schema.model_validate(raw_data)

    @field_validator("cache_root", mode="before")
    @classmethod
    def clean_empty_paths(cls, v):
        if v in (None, "", "None", "null"):
            return None
        return v

    @field_validator("data_root", "cache_root", mode="after")
    @classmethod
    def validate_dir(cls, v: Path) -> Path:
        """Ensure chosen directory is valid and not the system root"""
        if v is None:
            return v

        # 1. Expand ENV_VARS ($SCRATCH, $HOME)
        expanded_path = os.path.expandvars(str(v))
        v = Path(expanded_path).expanduser().resolve()

        # 2. Safety checks
        if v.parent == v:
            raise ValueError("Directory cannot literally be the system root!")
        if v.exists() and not v.is_dir():
            raise ValueError("Directory must be a valid directory!")
        return v


# --- Persistence functions ---

def load_settings(settings_dir: Path, filename: str = DEFAULT_SETTINGS_FILE) -> Settings:
    """Load settings from a JSON file in the settings directory."""
    path = settings_dir / filename
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Settings.model_validate(data)
    except (json.JSONDecodeError, TypeError):
        # If file is corrupted or invalid, return default settings (failsafe)
        return Settings()


def save_settings(settings_dir: Path, settings: Settings, filename: str = DEFAULT_SETTINGS_FILE) -> Path:
    """Save settings to a JSON file in the settings directory."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / filename
    data = settings.model_dump_json(indent=2)
    path.write_text(data, encoding="utf-8")
    return path
