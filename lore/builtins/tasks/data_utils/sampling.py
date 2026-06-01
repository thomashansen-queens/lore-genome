"""
Samples data from an input Artifact based on user-specified criteria.
"""

from enum import Enum
from typing import Any, List
import random
import pandas as pd

import lore


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
        accepted_data=lore.TABULAR,
        select=lore.SINGLE,
        load_as=lore.RAW,
    )
    sample_by = lore.ValueInput(
        list[str] | None,
        default=None,
        label="Sample by",
        description="Columns or keys to use for sampling (i.e. for stratification). Comma-separated if multiple.",
        examples=["collection_country, collection_year"],
    )
    sample_size = lore.ValueInput(
        int | None,
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
        data_type=lore.Passthrough("source"),
        label="Sampled data",
        description="Sampled subset of the original data as a new Artifact",
        is_primary=True,
    )
    remainder = lore.TaskOutput(
        data_type=lore.Passthrough("source"),
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


@lore.task(
    "filter.sample",
    inputs=SampleInputs,
    outputs=SampleOutputs,
    name="Sample data",
    category="Data processing",
    icon="Ω",
    live_preview=True,
)
def sample_handler(
    ctx: lore.ExecutionContext,
    source: Any,
    strategy: SamplingStrategy | str = SamplingStrategy.STRATIFIED_STRICT,
    sample_by: str | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    partition: bool = False,
):
    """
    Use sampling to select a representative population from a group.
    """
    strategy = SamplingStrategy(strategy)

    # 2. Artifact metadata
    source_artifacts = ctx.input_artifacts.get("source", [])
    ext = source_artifacts[0].extension if source_artifacts else "json"

    # 3. Prepare adapter and validate
    adapter = ctx.get_input_adapter("source")
    if adapter is None:
        raise ValueError("No adapter found for the input Artifact(s).")
    if not isinstance(adapter, lore.TabularAdapter):
        raise ValueError(
            f"The adapter for the input Artifact(s) must be a TabularAdapter, "
            f"but got {type(adapter)}."
        )

    parsed_records = adapter.parse(source, extension=ext)

    if not parsed_records:
        ctx.logger.warning("Received empty source for sampling. Will propagate empty artifact.")
        ctx.materialize_content("sampled_data", adapter.serialize([], extension=ext), ext)
        if partition:
            ctx.materialize_content("remainder", adapter.serialize([], extension=ext), ext)
        return

    # 4. Adapt to DataFrame
    adapted_records = adapter.adapt(parsed_records, extension=ext)
    df = pd.DataFrame(adapted_records)

    if df.empty:
        raise ValueError(
            "The adapted DataFrame is empty. Check the input data and adapter schema."
        )

    # 5. Validation
    sample_cols = []
    if sample_by:
        # Normalize column names from Adapter
        col_map = {str(c).lower().replace(" ", "_"): c for c in df.columns}

        missing = []
        for col in sample_by:  # Normalize input to match
            normalized_input = col.lower().replace(" ", "_")
            if normalized_input in col_map:
                sample_cols.append(col_map[normalized_input])
            else:
                missing.append(col)
        if missing:
            raise ValueError(f"Cannot sample by {missing}. Valid options: {df.columns.tolist()}")

    # 6. Sampling
    pulled_indices = []

    if strategy == SamplingStrategy.RANDOM:
        actual_n = (
            len(df)
            if (sample_size is None or sample_size <= 0)
            else min(sample_size, len(df))
        )
        if sample_size and actual_n < sample_size:
            ctx.logger.warning("Request %s > available %s. Shuffling all.", sample_size, len(df))

        sampled_df = df.sample(n=actual_n, random_state=seed)  # that was easy
        pulled_indices = sampled_df.index.tolist()

    elif strategy in [SamplingStrategy.STRATIFIED_STRICT, SamplingStrategy.STRATIFIED_RELAXED]:
        if not sample_cols:
            raise ValueError(
                f"Stratified sampling strategy requires 'sample_by' keys/columns. Valid options: "
                f"{df.columns.tolist()}"
            )

        pulled_indices = stratified_sample(
            pool=df,
            by=sample_cols,
            size=sample_size,
            seed=seed,
            strict=(strategy == SamplingStrategy.STRATIFIED_STRICT),
        )
        sampled_df = df.loc[pulled_indices]

        ctx.logger.info(
            "Selected %s samples from %s total across %s strata",
            len(sampled_df),
            len(df),
            sampled_df.groupby(by=sample_cols, dropna=False).ngroups,
        )

    # 7. De-adapt: Map back to original structure (parsed, not adapted)
    pulled_set = set(pulled_indices)  # in case of sample with replacement
    final_records = [parsed_records[i] for i in pulled_indices]

    if not final_records and not partition:
        raise ValueError("Sampling resulted in 0 records. Check your sampling strategy.")

    if partition:
        remainder_records = [rec for i, rec in enumerate(parsed_records) if i not in pulled_set]

    # 8. Materialization (sample)
    ctx.materialize_content(
        output_key="sampled_data",
        content=adapter.serialize(final_records, extension=ext),
        extension=ext,
    )

    if partition:
        ctx.materialize_content(
            output_key="remainder",
            content=adapter.serialize(remainder_records, extension=ext),
            extension=ext,
        )
