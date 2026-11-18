import csv
import importlib
import io
import logging
from types import SimpleNamespace

from lore.core.adapters.parsers.csv import CsvAdapter


class FakeContext:
    logger = logging.getLogger(__name__)

    def __init__(self, extension="csv", data_type="genome_accessions"):
        self.outputs = []
        self.input_artifacts = {
            "source": [
                SimpleNamespace(
                    extension=extension,
                    data_type=data_type,
                    metadata={},
                )
            ]
        }

    def get_input_adapter(self, input_key):
        assert input_key == "source"
        return CsvAdapter()

    def materialize_content(self, **kwargs):
        self.outputs.append(kwargs)
        return kwargs["output_key"]


def _run_task(
    content,
    use_nucleotide_accession=False,
    extension="csv",
    data_type="genome_accessions",
):
    task_module = importlib.import_module("lore.builtins.tasks.data_utils.deduplicate_accessions")
    ctx = FakeContext(extension=extension, data_type=data_type)
    result = task_module.deduplicate_accessions(
        ctx,
        content,
        "sample_id",
        use_nucleotide_accession,
    )
    outputs = {output["output_key"]: output for output in ctx.outputs}
    return result, outputs


def _csv_rows(content):
    return list(csv.DictReader(io.StringIO(content)))


def test_prefers_gcf_for_matching_assembly_and_keeps_identifiers_separate():
    result, outputs = _run_task(
        "sample_id,assembly_accession,nucleotide_accession,value\n"
        "sample-1,GCA_000001.1,NC_1,gca-first\n"
        "sample-1,GCF_000001.1,NC_2,gcf-wins\n"
        "sample-1,GCF_000001.1,NC_3,duplicate-gcf\n"
        "sample-2,GCA_000001.1,NC_4,other-sample\n"
    )

    assert result == "deduplicated_data"
    assert outputs["deduplicated_data"]["extension"] == "csv"
    assert outputs["deduplicated_data"]["data_type"] == "genome_accessions"
    assert _csv_rows(outputs["deduplicated_data"]["content"]) == [
        {
            "sample_id": "sample-1",
            "assembly_accession": "GCF_000001.1",
            "nucleotide_accession": "NC_2",
            "value": "gcf-wins",
        },
        {
            "sample_id": "sample-2",
            "assembly_accession": "GCA_000001.1",
            "nucleotide_accession": "NC_4",
            "value": "other-sample",
        },
    ]
    assert outputs["removed_data"]["name"] == "removed_data"
    assert outputs["removed_data"]["extension"] == "csv"
    assert outputs["removed_data"]["data_type"] == "csv"
    assert [row["value"] for row in _csv_rows(outputs["removed_data"]["content"])] == [
        "gca-first",
        "duplicate-gcf",
    ]


def test_nucleotide_fallback_deduplicates_and_collects_missing_accessions():
    _, outputs = _run_task(
        "sample_id,assembly_accession,nucleotide_accession,value\n"
        "sample-1,,NC_1,first\n"
        "sample-1,,nc_1,duplicate\n"
        "sample-2,,,missing\n",
        use_nucleotide_accession=True,
    )

    assert [row["value"] for row in _csv_rows(outputs["deduplicated_data"]["content"])] == [
        "first",
    ]
    assert [row["value"] for row in _csv_rows(outputs["removed_data"]["content"])] == [
        "duplicate",
        "missing",
    ]


def test_without_nucleotide_fallback_removes_rows_missing_assembly():
    _, outputs = _run_task(
        "sample_id,assembly_accession,nucleotide_accession,value\n"
        "sample-1,,NC_1,no-assembly\n"
        "sample-2,GCF_000002.1,,has-assembly\n"
    )

    assert [row["value"] for row in _csv_rows(outputs["deduplicated_data"]["content"])] == [
        "has-assembly",
    ]
    assert [row["value"] for row in _csv_rows(outputs["removed_data"]["content"])] == [
        "no-assembly",
    ]


def test_preserves_tsv_format_and_source_data_type():
    _, outputs = _run_task(
        "sample_id\tassembly_accession\tnucleotide_accession\nsample-1\tGCF_000001.1\tNC_1\n",
        extension="tsv",
        data_type="ncbi_ipg",
    )

    assert outputs["deduplicated_data"]["extension"] == "tsv"
    assert outputs["deduplicated_data"]["data_type"] == "ncbi_ipg"
    assert outputs["deduplicated_data"]["content"] == (
        "sample_id\tassembly_accession\tnucleotide_accession\r\nsample-1\tGCF_000001.1\tNC_1\r\n"
    )
