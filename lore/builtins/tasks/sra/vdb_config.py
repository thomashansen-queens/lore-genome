"""
Little test for the isolated VDB environment context manager.
Prints vdb-config.
"""
import subprocess

import lore

from .config import isolated_vdb_env


# class VdbConfigOutputs:
#     """Output for the vdb-config test"""
#     config_text = lore.TaskOutput(
#         data_type="text", # Triggers the TextAdapter / text.html viewer!
#         label="VDB Config Output",
#         description="The raw stdout from the vdb-config command.",
#         is_primary=True,
#         yields="single",
#         is_artifact=False, # 🚀 It's just a raw python string, not a file!
#     )


@lore.task(
    "sra.vdb_config",
    inputs=None,
    outputs=None,
    name="SRA Toolkit vdb-config",
    category="SRA Toolkit",
    icon="⚙️",
    preview_mode="full",
)
def vdb_config_handler(ctx: lore.ExecutionContext):
    config_model = ctx.get_config("sra_tools")
    sra_config = config_model.model_dump() if config_model else {}

    with isolated_vdb_env(sra_config, ctx) as safe_env:
        ctx.logger.info("Executing vdb-config in isolated environment...")
        result = subprocess.run(["vdb-config"], env=safe_env, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"vdb-config failed: {result.stderr}")

        print(
            "VDB Config Info:\n",
            result.stdout,
            "VDB Config Errors (if any):\n",
            result.stderr or "None",
        )
