"""
Little test for the isolated VDB environment context manager.
Prints vdb-config.
"""
import subprocess

import lore

from .config import get_sra_binary, isolated_vdb_env


class VdbConfigOutputs:
    """Output for the vdb-config test"""
    config_text = lore.TaskOutput(
        data_type="text",
        label="VDB Config Output",
        yields="single",
    )


@lore.task(
    "sra.vdb_config",
    inputs=None,
    outputs=VdbConfigOutputs,
    name="SRA Toolkit vdb-config",
    category="SRA Toolkit",
    icon="⚙️",
    preview_mode="full",
)
def vdb_config_handler(ctx: lore.ExecutionContext):
    """
    Test task to ensure vdb-config successfully runs and to inspect the
    configuration of the isolated environment.
    """
    config_model = ctx.get_config("sra_tools")
    sra_config = config_model.model_dump() if config_model else {}
    vdb_config_binary = get_sra_binary(sra_config, "vdb-config")

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        ctx.logger.info("Executing vdb-config in isolated environment...")
        result = subprocess.run(
            [vdb_config_binary],
            env=safe_env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"vdb-config failed: {result.stderr}")

        ctx.logger.info("vdb-config output:")
        ctx.logger.info(result.stdout)
        ctx.materialize_content(result.stdout)
