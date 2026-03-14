"""
Sampling Tasks for LoRe Genome.
"""

from enum import Enum
import json
from typing import Union, List, Optional
import random
import pandas as pd

from lore.core.adapters import TableAdapter
import lore.dsl as lore


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
    source = lore.ArtifactInput(
        description="Artifact to sample from (JSON list or DataFrame-compatible)",
        label="Source Artifact",
        select=lore.MULTIPLE,
        load_as=lore.RAW,
    )
    sample_by = lore.ValueInput(
        list[str],
        default=None,
        label="Sample by",
        description="Columns or keys to use for sampling (i.e. for stratification). Comma-separated if multiple.",
        examples=["collection_country, collection_year"],
    )
    sample_size = lore.ValueInput(
        int,
        default=None,
        label="Sample size",
        description="Number of samples to draw. Leave empty (None) for greedy sampling, maximizing the number of pulls.",
        examples=["(leave blank for greedy)"],
    )
    strategy = lore.ValueInput(
        SamplingStrategy,
        default=SamplingStrategy.STRATIFIED_STRICT,
        label="Sampling strategy",
        description="Sampling strategy to use (fill: fill to size, strict: exact proportions)",
    )
    seed = lore.ValueInput(
        int,
        default=42,
        label="Random seed",
        description="Random seed for reproducibility",
        examples=[42],
    )
    partition = lore.ValueInput(
        bool,
        default=False,
        label="Partition remainder",
        description="Save the non-sampled remainder as a separate Artifact",
    )


class SampleOutputs:
    """
    Outputs for sampling tasks.
    """
    sampled_data = lore.TaskOutput(
        data_type="*",  # same as input
        label="Sampled data",
        description="Sampled subset of the original data as a new Artifact",
        is_primary=True,
    )
    remainder = lore.TaskOutput(
        data_type="*",
        label="Remainder",
        description="(Optional) Remainder of the data not included in the sample, as a new Artifact",
        yields=lore.OPTIONAL,
    )


def stratified_sample(
    pool: pd.DataFrame,
    by: List[str],
    size: int | None,
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


@lore.task(
    'filter.sample',
    inputs=SampleInputs,
    outputs=SampleOutputs,
    name="Sample data",
    category="Data processing",
    icon="Ω",
    live_preview=True,
)
def sample_handler(
    ctx: lore.ExecutionContext,
    source: list[dict],
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
    # 1. Get adapter and input artifact metadata
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, TableAdapter):
        raise ValueError(f"The adapter for the input Artifact(s) must be a TableAdapter, but got {type(adapter)}.")

    source_artifacts = ctx.input_artifacts.get("source", [])
    inherited_type = source_artifacts[0].data_type if source_artifacts else "unknown"
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 2. Adapt to DataFrame
    adapted_records = adapter.adapt(source)
    df = pd.DataFrame(adapted_records)

    if df.empty:
        raise ValueError("The adapted DataFrame is empty. Check the input data and adapter schema.")

    # 3. Validation
    sample_cols = []
    if sample_by:
        selected_cols = [s.strip() for s in sample_by.split(",")] if isinstance(sample_by, str) else sample_by

        # Normalize column names from Adapter
        col_map = {str(c).lower().replace(" ", "_"): c for c in df.columns}

        missing = []
        for col in selected_cols:  # Normalize input to match
            normalized_input = col.lower().replace(" ", "_")
            if normalized_input in col_map:
                sample_cols.append(col_map[normalized_input])
            else:
                missing.append(col)
        if missing:
            raise ValueError(f"Cannot sample by {missing}. Valid options: {df.columns.tolist()}")

    # 4. Sampling
    sampled_df = pd.DataFrame()
    pulled_indices = []

    if strategy == SamplingStrategy.RANDOM.value:
        if sample_size is None:
            ctx.logger.warning("No sample_size provided for random sampling. Defaulting to 100.")
            sample_size = 100

        # Simple random sample of the whole dataset
        actual_n = min(sample_size, len(df))
        if actual_n < sample_size:
            ctx.logger.warning("Requested %s > available %s. Using all.", sample_size, len(df))

        sampled_df = df.sample(n=actual_n, random_state=seed)  # that was easy
        pulled_indices = sampled_df.index.tolist()

    elif strategy in [SamplingStrategy.STRATIFIED_STRICT.value, SamplingStrategy.STRATIFIED_RELAXED.value]:
        if not sample_cols:
            raise ValueError("Stratified sampling strategy requires 'sample_by' keys/columns.")

        pulled_indices = stratified_sample(
            pool=df,
            by=sample_cols,
            size=sample_size,
            seed=seed,
            strict=(strategy == SamplingStrategy.STRATIFIED_STRICT.value),
        )
        sampled_df = df.loc[pulled_indices]

        ctx.logger.info(
            "Selected %s samples from %s total across %s strata",
            len(sampled_df),
            len(df),
            sampled_df.groupby(by=sample_cols, dropna=False).ngroups,
        )

    # 6. De-adapt: Map back to original structure
    pulled_set = set(pulled_indices)  # in case of sample with replacement
    final_records = [source[i] for i in pulled_indices]

    if partition:
        remainder_records = [rec for i, rec in enumerate(source) if i not in pulled_set]

    # 7. Materialization (sample)
    ctx.materialize_content(
        output_key="sampled_data",
        content=adapter.serialize(final_records, extension=ext),
        extension=ext,
        data_type=inherited_type,  # could append ".sampled" to be explicit
    )

    if partition:
        ctx.materialize_content(
            output_key="remainder",
            content=adapter.serialize(remainder_records, extension=ext),
            extension=ext,
            data_type=inherited_type,  # could append ".sampled" to be explicit
    )
