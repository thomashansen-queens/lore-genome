"""
Default paths for configuration, cache, and data storage.
"""
from pathlib import Path
from platformdirs import user_config_dir

APP_NAME = 'lore-genome'

def default_settings_dir() -> Path:
    """Get the default settings directory for LoRe Genome."""
    return Path(user_config_dir(APP_NAME)).expanduser().resolve()


def default_data_root() -> Path:
    """Get the default data root directory for LoRe Genome."""
    # not the best, but lets user find it easily
    return (Path.home() / APP_NAME).resolve()
