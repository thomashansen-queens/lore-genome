import importlib
import logging

from lore.builtins.tasks.ncbi.efetch_protein import ProteinFormat


def test_ipg_cluster_representative_prefers_wp_on_count_tie():
    task_module = importlib.import_module("lore.builtins.tasks.ncbi.efetch_protein")

    tied_representative = task_module._select_ipg_cluster_representative(
        {"ABC123.1": 2, "WP_000001.1": 2}
    )
    higher_count_representative = task_module._select_ipg_cluster_representative(
        {"ABC123.1": 3, "WP_000001.1": 2}
    )

    assert tied_representative == "WP_000001.1"
    assert higher_count_representative == "ABC123.1"


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeContext:
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.outputs = []

    def get_config(self, name):
        return None

    def materialize_content(self, **kwargs):
        self.outputs.append(kwargs)
        return kwargs["output_key"]


def test_ipg_batches_queries_and_emits_their_deduplicated_clusters(monkeypatch):
    task_module = importlib.import_module("lore.builtins.tasks.ncbi.efetch_protein")
    requests = []

    def fake_entrez_client(**kwargs):
        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, endpoint, data, timeout):
                requests.append(data["id"])
                return FakeResponse(
                    "#IPG\tNucleotide Accession\tProtein\tAssembly\n"
                    "1\tNC_000001\tWP_000001\tGCF_000001\n"
                    "1\tNC_000002\tWP_000001.2\tGCF_000002\n"
                    "1\tNC_000003\tWP_000001.2\tGCF_000003\n"
                    "1\tNC_000003\tWP_000001.2\tGCF_000003\n"
                    "2\tNC_000004\tWP_000002\tGCF_000004\n"
                    "2\tNC_000005\tWP_000002.2\tGCF_000005\n"
                )

        return Client()

    monkeypatch.setattr(task_module, "entrez_client", fake_entrez_client)
    monkeypatch.setattr(task_module, "sleep", lambda _: None)
    ctx = FakeContext()

    result = task_module.efetch_protein(
        ctx,
        ["WP_000000", "WP_000001", "WP_000002"],
        ProteinFormat.IPG,
    )

    assert result == "ipg_records"
    assert requests == ["WP_000000,WP_000001,WP_000002"]
    cluster_output = next(
        output for output in ctx.outputs if output["output_key"] == "ipg_clusters"
    )
    assert cluster_output["extension"] == "csv"
    assert cluster_output["content"] == (
        "representative_protein_accession,protein_accession,nucleotide_accession,assembly_accession\n"
        "WP_000001.2,WP_000001,NC_000001,GCF_000001\n"
        "WP_000001.2,WP_000001.2,NC_000002,GCF_000002\n"
        "WP_000001.2,WP_000001.2,NC_000003,GCF_000003\n"
        "WP_000002,WP_000002,NC_000004,GCF_000004\n"
        "WP_000002,WP_000002.2,NC_000005,GCF_000005\n"
    )


def test_other_formats_keep_batched_queries(monkeypatch):
    task_module = importlib.import_module("lore.builtins.tasks.ncbi.efetch_protein")
    requests = []

    def fake_entrez_client(**kwargs):
        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, endpoint, data, timeout):
                requests.append(data["id"])
                return FakeResponse(">protein\nSEQUENCE")

        return Client()

    monkeypatch.setattr(task_module, "entrez_client", fake_entrez_client)
    monkeypatch.setattr(task_module, "sleep", lambda _: None)
    ctx = FakeContext()

    task_module.efetch_protein(ctx, ["WP_000001", "WP_000002"], ProteinFormat.FASTA)

    assert requests == ["WP_000001,WP_000002"]
    assert [output["output_key"] for output in ctx.outputs] == ["protein_fasta"]
