import csv
import importlib
from io import StringIO

from lore.builtins.adapters.protein_nucleotide_assembly import (
    ProteinNucleotideAssemblyMappingAdapter,
)


class FakeContext:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.materialized_files = {}

    def get_temp_path(self, name):
        return self.tmp_path / name

    def materialize_file(self, source, output_key):
        self.materialized_files[output_key] = source.read_text(encoding="utf-8")


def test_matching_skips_cluster_duplicates_and_sorts_fasta_by_score(tmp_path):
    task_module = importlib.import_module(
        "lore.builtins.tasks.workflow_specific.matching_protein_seq_from_protein_seq"
    )
    ctx = FakeContext(tmp_path)
    cluster_table = ProteinNucleotideAssemblyMappingAdapter().adapt(
        "representative_protein_accession,protein_accession,"
        "nucleotide_accession,assembly_accession\n"
        "P1,P1,N1,A1\n"
        "P1,P2,N2,A2\n"
        "P1,P2,N3,A3\n",
        config={"ext": "csv"},
    )

    task_module.task(
        ctx,
        [
            {"protein_accession": "P1", "protein_sequence": "MKTAYIAK"},
            {"protein_accession": "P2", "protein_sequence": "WWWWWWWW"},
            {"protein_accession": "P3", "protein_sequence": "MKT"},
        ],
        "MKTAYIAK",
        cluster_table=cluster_table,
    )

    fasta_records = ctx.materialized_files["fasta"].strip().splitlines()
    headers = [line for line in fasta_records if line.startswith(">")]
    assert len(headers) == 2
    assert "P1" in headers[0]
    assert "P3" in headers[1]
    assert all("score=" in header for header in headers)
    scores = [float(header.split("score=")[1]) for header in headers]
    assert scores == sorted(scores, reverse=True)

    cluster_rows = list(csv.DictReader(StringIO(ctx.materialized_files["cluster_table"])))
    assert {row["protein_accession"] for row in cluster_rows} == {"P1", "P2", "P3"}
    assert {
        row["nucleotide_accession"] for row in cluster_rows if row["protein_accession"] == "P2"
    } == {"N2", "N3"}


def test_cluster_table_input_remains_adapted():
    task_module = importlib.import_module(
        "lore.builtins.tasks.workflow_specific.matching_protein_seq_from_protein_seq"
    )

    assert task_module.Inputs.cluster_table.materialization.value == "adapted"
