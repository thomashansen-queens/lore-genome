"""
Global configuration for NCBI SRA Toolkit integration.
"""
from contextlib import contextmanager
import logging
from pathlib import Path
import shutil
import tempfile

import lore

logger = logging.getLogger(__name__)


@lore.config(key="sra_tools", title="NCBI SRA Toolkit")
class SraToolsConfig:
    """
    Global settings for NCBI's SRA Toolkit binaries. The software used in these
    tasks can be found here: https://github.com/ncbi/sra-tools .
    """
    prefix = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank if you have added sratoolkit.3.0.0-osxyz/bin to your PATH"],
        label="Path to 'sratoolkit' binaries",
        description="Provide the full path to the the SRA Toolkit binaries (e.g. C:/sratoolkit.3.0.0-win64/bin)",
    )
    cache_dir = lore.ValueInput(
        str | None,
        default=None,
        examples=["Leave blank to use LoRē's default cache directory"],
        label="Directory for temporary files",
        description="Sequence read archive files are often massive (many GB), so you may want to specify a directory with sufficient storage for temporary files.",
    )
    default_threads = lore.ValueInput(
        int,
        default=6,
        min=1, max=32, step=1,
        label="Default CPU Threads",
        description="Number of CPU threads to use for SRA toolkit operations.",
        widget="slider",
    )


def get_sra_binary(config: dict, binary_name: str) -> str:
    """Helper function to get the full path to an SRA Toolkit binary."""
    prefix = config.get("prefix")
    resolved_path = shutil.which(binary_name, path=prefix)
    if not resolved_path:
        if prefix:
            raise RuntimeError(
                f"SRA Toolkit error: '{binary_name}' not found or not executable in the specified "
                f"prefix: '{prefix}'. Please check your global settings."
            )
        else:
            raise RuntimeError(
                f"SRA Toolkit error: '{binary_name}' not found or not executable in system PATH. "
                f"Please install the SRA Toolkit and ensure its binaries are avaialable in PATH "
                f"or specified in the global SRA settings under 'prefix'."
            )

    return resolved_path


@contextmanager
def isolated_vdb_env(config: dict, ctx: lore.ExecutionContext):
    """
    Creates an isolated NCBI VDB (Versatile DataBase) environment. Allows
    LoRē to manage VDB's config without overwriting a user's system-wide
    config. Useful to specify a custom temp directory for VDB files, which
    can be very large.
    Yields an environment dict that can be passed to subprocess calls.
    """
    import os

    custom_temp_dir = config.get("temp_dir")
    if custom_temp_dir:
        base_dir = Path(custom_temp_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        vdb_workspace = Path(tempfile.mkdtemp(dir=base_dir, prefix="lore_vdb_"))
    else:
        vdb_workspace = ctx.get_temp_path("vdb_workspace")
        vdb_workspace.mkdir(parents=True, exist_ok=True)

    vdb_config_file = vdb_workspace / "user-settings.mkfg"

    vdb_config_content = f"""
/repository/user/main/public/root = "{vdb_workspace}"
/repository/user/default-path = "{vdb_workspace}"
    """
    vdb_config_file.write_text(vdb_config_content)

    env = os.environ.copy()
    env["NCBI_SETTINGS"] = str(vdb_config_file)

    try:
        yield env
    finally:
        # ctx.cleanup() will handle removing the temporary directory
        pass
