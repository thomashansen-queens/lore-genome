"""
Sampling Tasks for LoRe Genome.
"""

from enum import Enum
import json
from typing import Union, List, Optional
import random
import pandas as pd

from lore.core.tasks import ArtifactInput, task_registry, Cardinality, Materialization, ValueInput, TaskOutput
from lore.core.executor import ExecutionContext

class SamplingStrategy(Enum):
    """
    Sampling strategies
    """
    RANDOM = "random"
    STRATIFIED_RELAXED = "stratified_relaxed"  # pull until strata exhausted then fill
    STRATIFIED_STRICT = "stratified_strict"  # pull until filled or strata exhausted
    # SLICE = "slice"  # a randomly placed contiguous block
    # K_MEANS = "k_means"  # is this even possible/worth it?


class SampleInputs:
    """
    Input model for sampling tasks.
    """
    source = ArtifactInput(
        description="Artifact to sample from (JSON list or DataFrame-compatible)",
        label="Source Artifact",
        cardinality=Cardinality.ONE_OR_MORE,
        load_as=Materialization.ID,
    )
    sample_by = ValueInput(
        list[str],
        default=None,
        label="Sample by",
        description="Columns or keys to use for sampling (i.e. for stratification). Comma-separated if multiple.",
        examples=["collection_country, collection_year"],
    )
    sample_size = ValueInput(
        int,
        default=None,
        label="Sample size",
        description="Number of samples to draw. Leave empty (None) for greedy sampling, maximizing the number of pulls.",
        examples=["(leave blank for greedy)"],
    )
    strategy = ValueInput(
        SamplingStrategy,
        default=SamplingStrategy.STRATIFIED_STRICT,
        label="Sampling strategy",
        description="Sampling strategy to use (fill: fill to size, strict: exact proportions)",
    )
    seed = ValueInput(
        int,
        default=42,
        label="Random seed",
        description="Random seed for reproducibility",
        examples=[42],
    )
    partition =ValueInput(
        bool,
        default=False,
        label="Partition remainder",
        description="Save the non-sampled remainder as a separate Artifact",
    )


class SampleOutputs:
    """
    Outputs for sampling tasks.
    """
    sample = TaskOutput(
        data_type="*",  # same as input
        label="Sampled",
        description="Sampled subset of the original data as a new Artifact",
        is_primary=True,
    )
    remainder = TaskOutput(
        data_type="*",
        label="Remainder",
        description="(Optional) Remainder of the data not included in the sample, as a new Artifact",
    )


def stratified_sample(
        pool: pd.DataFrame,
        by: List[str],
        size: Optional[int],
        seed: int = 42,
        strict: bool = True,
) -> List[int]:
    """
    Stratified sampling helper function. Returns indices of selected samples.
    Uses a round-robin approach to pull samples from each stratum until size met.
    If strict, stops when any stratum is exhausted. If not strict, continues
    pulling from remaining strata until size met.
    If size is None, pulls until any stratum is exhausted.
    """
    pool = pool.sort_values(by=by, ascending=True)  # initial sort for reproducibility
    rng = random.Random(seed)
    buckets = []

    # Sort indices into buckets and shuffle
    for _, g in pool.groupby(by=by, dropna=False, sort=False):
        idxs = list(g.index)
        rng.shuffle(idxs)
        buckets.append(idxs)

    picked: list[int] = []
    target_size = size if size is not None else len(pool)

    while len(picked) < target_size and buckets:
        # iterate through buckets, stop early if target_size reached or buckets exhausted
        picked_this_lap = []
        exhausted_mid_lap = False

        for _, bucket in enumerate(buckets):
            if len(picked) + len(picked_this_lap) >= target_size:
                break

            picked_this_lap.append(bucket.pop())
            if len(bucket) == 0:
                exhausted_mid_lap = True

        picked.extend(picked_this_lap)
        buckets = [b for b in buckets if len(b) > 0]  # remove exhausted buckets
        if strict and exhausted_mid_lap:
            # won't make it through the next lap without hit an exhausted bucket
            break

    return picked


def _serialize_records(records: list, extension: str) -> str:
    """Locally serialize a list of records back into text."""
    if extension in ("jsonl", "ndjson"):
        return "\n".join(json.dumps(r) for r in records) + "\n"
    # Default to standard JSON
    return json.dumps(records, indent=2)


@task_registry.register(
    'filter.sample',
    inputs=SampleInputs,
    outputs=SampleOutputs,
    name="Sample data",
    category="Data processing",
    icon="Ω",
)
def sample_handler(
    ctx: ExecutionContext,
    source: list[str],
    strategy: str = SamplingStrategy.STRATIFIED_STRICT.value,  # use strings from enum
    sample_by: Optional[Union[str, List[str]]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    partition: bool = False,
):
    """
    Use sampling to select a representative population from a group.
    Artifact -> DataFrame -> Samples -> JSON Artifact
    By using an Adapter, we are doing "high-dimensional indexing", but the final
    output is untouched records from the original source in their original format.
    """
    # 1. Inheritance
    source_artifacts = ctx.input_artifacts.get("source", [])
    if not source_artifacts:
        raise ValueError("Source Artifact metadata missing from ExecutionContext.")
    if len(set(art.data_type for art in source_artifacts)) > 1:
        raise ValueError("Multiple source Artifacts with different data types found. Dynamic merging not supported.")
    inherited_type = source_artifacts[0].data_type
    extension = source_artifacts[0].extension or "json"

    # 2. Set up Adapter
    from lore.core.adapters import adapter_registry, BaseAdapter
    matches = adapter_registry.get_adapters(source_artifacts[0])
    adapter = matches[0] if matches else BaseAdapter()
    ctx.logger.info("Using Adapter: %s for parsing", adapter.name)

    # 3. Memory-safe dynamic merging
    raw_records = []
    with ctx.runtime.get_session(ctx.session_id) as s:
        for art in source_artifacts:
            raw_text = s.load_artifact_data(art.id)
            records = adapter.parse(raw_text)
            raw_records.extend(records)

    ctx.logger.info("Merged %s records from %s source Artifacts", len(raw_records), len(source_artifacts))
    sampling_pool = adapter.adapt(raw_records)

    df_sampling = pd.DataFrame(sampling_pool)

    # 4. Validation
    sample_cols = []
    if sample_by:
        raw_cols = [s.strip() for s in sample_by.split(",")] if isinstance(sample_by, str) else sample_by

        # Normalize column names from Adapter
        col_map = {str(c).lower().replace(" ", "_"): c for c in df_sampling.columns}

        missing = []
        for raw_col in raw_cols:  # Normalize input to match
            normalized_input = raw_col.lower().replace(" ", "_")
            if normalized_input in col_map:
                sample_cols.append(col_map[normalized_input])
            else:
                missing.append(raw_col)
        if missing:
            raise ValueError(f"Cannot sample by {missing}. Valid options: {df_sampling.columns.tolist()}")

    # 5. Sampling
    sampled_df = pd.DataFrame()
    pulled_indices = []

    if strategy == SamplingStrategy.RANDOM.value:
        if sample_size is None:
            ctx.logger.warning("No sample_size provided for random sampling. Defaulting to 100.")
            sample_size = 100
        # Simple random sample of the whole dataset
        actual_n = min(sample_size, len(df_sampling))
        if actual_n < sample_size:
            ctx.logger.warning(
                "Requested %s > available %s. Using all.",
                sample_size,
                len(df_sampling),
            )
        sampled_df = df_sampling.sample(n=actual_n, random_state=seed)  # that was easy
        pulled_indices = sampled_df.index.tolist()

    elif strategy in [SamplingStrategy.STRATIFIED_STRICT.value, SamplingStrategy.STRATIFIED_RELAXED.value]:
        if not sample_cols:
            raise ValueError("Stratified sampling strategy requires 'sample_by' keys/columns.")

        pulled_indices = stratified_sample(
            pool=df_sampling,
            by=sample_cols,
            size=sample_size,
            seed=seed,
            strict=(strategy == SamplingStrategy.STRATIFIED_STRICT.value),
        )
        sampled_df = df_sampling.loc[pulled_indices]

        ctx.logger.info(
            "Selected %s samples from %s total across %s strata",
            len(sampled_df),
            len(df_sampling),
            sampled_df.groupby(by=sample_cols, dropna=False).ngroups,
        )

    # 6. De-adapt: Map back to original structure
    pulled_set = set(pulled_indices)  # in case of sample with replacement
    final_records = [raw_records[i] for i in pulled_indices]

    if partition:
        remainder_records = [rec for i, rec in enumerate(raw_records) if i not in pulled_set]

    # 7. Materialization (sample)
    ctx.materialize_content(
        output_key="sample",
        content=adapter.serialize(final_records, extension=extension),
        extension=extension,
        data_type=inherited_type,  # could append ".sampled" to be explicit
    )

    if partition:
        ctx.materialize_content(
            output_key="remainder",
            content=adapter.serialize(remainder_records, extension=extension),
            extension=extension,
            data_type=inherited_type,  # could append ".sampled" to be explicit
    )
    ctx.logger.info("Sampling complete!")
