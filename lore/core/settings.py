"""
Docstring for lore.settings
"""
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field, field_validator

from lore.core.paths import default_data_root

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

DEFAULT_SETTINGS_FILE = "settings.json"
DEFAULT_SECRETS_FILE = "secrets.json"


class Settings(BaseModel):
    """Configuration settings for LoRe Genome."""
    # --- Paths ---
    data_root: Path = Field(
        default_factory=default_data_root,
        description="Root directory for all LoRe Genome data (sessions, outputs, etc).",
        json_schema_extra={"widget": "text"},
        examples=[str(default_data_root())],
    )
    cache_root: Path | None = Field(
        default=None,
        description="Path for temp files. Defaults to {data_root}/cache if empty.",
        json_schema_extra={"widget": "text"},
        examples=["leave blank for {data_root}/cache or specify custom path i.e. $SCRATCH/lore_cache"],
    )
    mmseqs_path: Path = Field(
        default_factory=lambda: Path("mmseqs"),
        description="Path to MMSeqs2 executable. If not in PATH, provide full path. On Windows, path/to/mmseqs.bat",
        json_schema_extra={"widget": "text"},
        examples=["mmseqs or C:/path/to/mmseqs.bat"],
    )
    # --- UI ---
    explore_display_limit: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description=(
            "Max number of rows to display in the Explore view. This only "
            "affects the display; all data is still accessible for queries and "
            "sorting. Increase to see more, but may impact load times."
        ),
        json_schema_extra={"widget": "int"},
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
        description="Maximum disk space in GB to use for in-memory Task caching.",
        json_schema_extra={"widget": "float"},
        examples=["4.0"],
    )
    io_max_file_size_mb: int = Field(
        default=100,
        ge=1,
        description="Max file size in MB before forcing stream-only materialization.",
        json_schema_extra={"widget": "float"},
        examples=["100"],
    )

    # Network/remote (FUTURE: for HPC deployments)
    api_host: str = "127.0.0.1"
    api_port: int = 8000

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


class Secrets(BaseModel):
    """Sensitive configuration for LoRe Genome."""
    ncbi_api_key: str = Field(
        default="",
        description="NCBI API key to increase rate limits.",
        json_schema_extra={"widget": "text"},
        examples=["<a 36-digit hexadecimal string>"],
    )

    @field_validator('ncbi_api_key')
    @classmethod
    def no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("NCBI API key cannot contain spaces.")
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


def load_secrets(secrets_dir: Path, filename: str = DEFAULT_SECRETS_FILE) -> Secrets:
    """Load secrets from a JSON file in the settings directory."""
    path = secrets_dir / filename
    if not path.exists():
        return Secrets()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Secrets.model_validate(data)
    except (json.JSONDecodeError, TypeError):
        # If file is corrupted or invalid, return default secrets (failsafe)
        return Secrets()


def save_secrets(secrets_dir: Path, secrets: Secrets, filename: str = DEFAULT_SECRETS_FILE) -> Path:
    """Save secrets to a JSON file in the settings directory."""
    secrets_dir.mkdir(parents=True, exist_ok=True)
    path = secrets_dir / filename
    data = secrets.model_dump_json(indent=2)
    path.write_text(data, encoding="utf-8")
    try:
        path.chmod(0o600)  # Owner read/write only
    except OSError:  # Windows may not support chmod
        pass
    return path
