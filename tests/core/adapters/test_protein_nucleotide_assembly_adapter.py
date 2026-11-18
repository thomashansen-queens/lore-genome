from lore.builtins.adapters.protein_nucleotide_assembly import (
    ProteinNucleotideAssemblyMappingAdapter,
)
from lore.core.adapters import adapter_registry


def test_ipg_mapping_adapter_preserves_protein_mappings_per_nucleotide():
    registered_adapters = adapter_registry.get_for_type(
        "protein_nucleotide_assembly_mapping", "csv"
    )
    assert any(
        isinstance(adapter, ProteinNucleotideAssemblyMappingAdapter)
        for adapter in registered_adapters
    )

    adapter = ProteinNucleotideAssemblyMappingAdapter()
    content = (
        "representative_protein_accession,protein_accession,"
        "nucleotide_accession,assembly_accession\n"
        "WP_000001.1,WP_000001.1,NC_1,GCF_1\n"
        "WP_000001.1,WP_000002.1,NC_2,GCF_2\n"
        "WP_000001.1,WP_000002.1,NC_3,GCF_3\n"
        "WP_000001.1,WP_000002.1,NC_3,GCF_3\n"
        "WP_000004.1,WP_000004.1,NC_4,GCF_4\n"
    )

    adapted = adapter.adapt(content, config={"ext": "csv"})
    assert [
        {
            "protein_accession": cluster["protein_accession"],
            "cluster_size": cluster["cluster_size"],
        }
        for cluster in adapted
    ] == [
        {"protein_accession": "WP_000001.1", "cluster_size": 3},
        {"protein_accession": "WP_000004.1", "cluster_size": 1},
    ]
    assert [
        [member["protein_accession"] for member in cluster["members"]]
        for cluster in adapted
    ] == [
        ["WP_000001.1", "WP_000002.1", "WP_000002.1"],
        ["WP_000004.1"],
    ]
