import csv
import importlib
from io import StringIO

from lore.builtins.adapters.protein_nucleotide_assembly import (
    ProteinNucleotideAssemblyMappingAdapter,
)
from lore.builtins.tasks.workflow_specific.resolve_protein_clusters import Inputs


class FakeContext:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.materialized_files = {}
        self.logger = self

    def info(self, *args):
        pass

    def warning(self, *args):
        pass

    def get_temp_path(self, name):
        return self.tmp_path / name

    def materialize_file(self, source, output_key):
        self.materialized_files[output_key] = source.read_text(encoding="utf-8")


def test_resolve_protein_clusters_accepts_ipg_mapping_data(tmp_path):
    assert "protein_nucleotide_assembly_mapping" in Inputs.entrez_cluster_table.accepted_data
    ipg_adapter = ProteinNucleotideAssemblyMappingAdapter()
    cluster_provenance = ipg_adapter.adapt(
        "representative_protein_accession,protein_accession,"
        "nucleotide_accession,assembly_accession\n"
        "WP_000001.1,WP_000001.1,NC_1,GCF_1\n"
        "WP_000001.1,WP_000002.1,NC_2,GCF_2\n",
        config={"ext": "csv"},
    )
    task_module = importlib.import_module(
        "lore.builtins.tasks.workflow_specific.resolve_protein_clusters"
    )
    ctx = FakeContext(tmp_path)

    task_module.resolve_protein_clusters(
        ctx,
        cluster_provenance,
        [
            {"cluster_rep": "WP_000001.1", "protein_accession": "WP_000001.1"},
        ],
        [
            {"protein_accession": "WP_000001.1", "protein_sequence": "MA"},
            {"protein_accession": "WP_000002.1", "protein_sequence": "MA"},
        ],
    )

    fasta_records = ctx.materialized_files["protein_match_fasta"].splitlines()
    assert fasta_records == [">WP_000001.1 protein (2 total)", "MA"]
    table_rows = list(csv.DictReader(StringIO(ctx.materialized_files["cluster_table"])))
    assert [row["cluster_size_contribution"] for row in table_rows] == ["2"]
