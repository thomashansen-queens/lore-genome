"""
Docstring for lore.settings
"""
from dataclasses import asdict, dataclass, field, fields
from os import path
from pathlib import Path
import json
from typing import Any

SETTINGS_FILE = "settings.json"
SECRETS_FILE = "secrets.json"

@dataclass
class Settings:
    """Configuration settings for LoRe Genome."""
    data_root: Path | None = None
    mmseqs_path: str = "mmseqs"  # Default to assuming mmseqs is in PATH
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Settings':
        """Hydrate Settings from a dictionary."""
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        if filtered_data.get("data_root") is not None:
            filtered_data["data_root"] = Path(filtered_data["data_root"])
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Settings to a dictionary."""
        data = asdict(self)
        if data.get("data_root") is not None:
            data["data_root"] = str(data["data_root"])
        return data

@dataclass
class Secrets:
    """Sensitive configuration for LoRe Genome."""
    ncbi_api_key: str = field(default="")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Secrets':
        """Hydrate Secrets from a dictionary (currently no special handling)."""
        normalized_data = {k.lower(): v for k, v in data.items()}
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in normalized_data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Secrets to a dictionary (currently no special handling)."""
        return asdict(self)

# --- Persistence functions ---

def load_settings(settings_dir: Path) -> Settings:
    """Load settings from a JSON file in the settings directory."""
    path = settings_dir / SETTINGS_FILE
    if not path.exists():
        return Settings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Settings.from_dict(data)
    except (json.JSONDecodeError, TypeError):
        # If file is corrupted or invalid, return default settings (failsafe)
        return Settings()

def save_settings(settings_dir: Path, settings: Settings) -> Path:
    """Save settings to a JSON file in the settings directory."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    path = settings_dir / SETTINGS_FILE
    data = settings.to_dict()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path

def load_secrets(secrets_dir: Path) -> Secrets:
    """Load secrets from a JSON file in the settings directory."""
    path = secrets_dir / SECRETS_FILE
    print(f"DEBUG: Looking for secrets at {path.absolute()}")
    print(f"DEBUG: File exists? {path.exists()}")
    if not path.exists():
        return Secrets()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Secrets.from_dict(data)
    except (json.JSONDecodeError, TypeError):
        # If file is corrupted or invalid, return default secrets (failsafe)
        return Secrets()

def save_secrets(secrets_dir: Path, secrets: Secrets) -> Path:
    """Save secrets to a JSON file in the settings directory."""
    secrets_dir.mkdir(parents=True, exist_ok=True)
    path = secrets_dir / SECRETS_FILE
    data = secrets.to_dict()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        path.chmod(0o600)  # Owner read/write only
    except OSError:  # Windows may not support chmod
        pass
    return path
