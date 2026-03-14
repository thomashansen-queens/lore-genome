"""
Tests for the TableAdapter class in lore.core.adapters
"""
from lore.core.adapters import AdapterRegistry, TableAdapter

# --- Adapter data transformation tests ---

class SpaghettiAdapter(TableAdapter):
    """A dummy adapter specifically built to test the extraction engine."""
    @property
    def schema(self):
        return {
            "record_id": "id",                                # 1. Direct key
            "nested_value": "nested.val",                     # 2. Dot notation
            "first_gene": "genes[0]",                         # 3. List indexing
            "first_genus": "nest_list[0][0].genus",           # 4. Mixed dot and list
            "id_as_string": ("id", self.safe_str),            # 5. Tuple with converter function
            "gene_count": lambda r: len(r.get("genes", [])),  # 6. Lambda function
            "crazy_nest": lambda r: r["nest_list"][1][0]["genus"] if len(r.get("nest_list", [])) > 1 else None, # 7. Lambda for extreme nesting
        }


def test_spaghetti_extraction():
    """
    Tests that TableAdapter can successfully navigate complex nested dictionaries 
    using all supported schema declaration methods.
    """
    adapter = SpaghettiAdapter()

    # Record 2 from spaghetti_records in conftest.py
    raw_record = {
        "id": 2,
        "nested": {"val": "B"},
        "genes": ["BRCA1", "TP53"],
        "nest_list": [
            [{"genus": "Canis", "species": "lupus"}],
            [{"genus": "Mus", "species": "musculus"}],
        ],
    }

    # ACT
    adapted = adapter.adapt_record(raw_record)

    # ASSERT
    assert adapted["record_id"] == 2
    assert adapted["nested_value"] == "B"
    assert adapted["first_gene"] == "BRCA1"
    assert adapted["first_genus"] == "Canis"
    assert adapted["crazy_nest"] == "Mus"
    assert adapted["id_as_string"] == "2"
    assert isinstance(adapted["id_as_string"], str)
    assert adapted["gene_count"] == 2


def test_extraction_safe_failures():
    """
    Tests that bad schema paths or missing data safely return None 
    instead of crashing the entire application.
    """
    adapter = SpaghettiAdapter()

    empty_record = {}

    adapted = adapter.adapt_record(empty_record)

    assert adapted["record_id"] is None
    assert adapted["nested_value"] is None
    assert adapted["first_gene"] is None
    assert adapted["first_genus"] is None
    assert adapted["id_as_string"] is None
    # The lambda handles its own safety with .get(), so it returns 0!
    assert adapted["gene_count"] == 0


def test_table_adapter_missing_keys():
    """Test that missing keys safely return None without crashing."""
    adapter = SpaghettiAdapter()
    raw_record = {"id": 2, "nested": {"val": "B"}, "genes": ["BRCA1", "TP53"]} # Missing nest_list

    adapted_record = adapter.adapt_record(raw_record)

    assert adapted_record["record_id"] == 2
    assert adapted_record["nested_value"] == "B"
    assert adapted_record["first_gene"] == "BRCA1"
    assert adapted_record["first_genus"] is None
    assert adapted_record["id_as_string"] == "2"
    assert adapted_record["gene_count"] == 2


# --- Adapter semantic matching tests ---

def test_adapter_registry_resolution(populated_registry: AdapterRegistry):
    """
    Tests that the AdapterRegistry correctly resolves the most specific adapter 
    based on the accepted_types and accepted_formats of registered adapters.
    """
    registry = populated_registry

    # Test that a highly specific adapter is chosen when both type and format match
    ncbi_adapters = registry.get_adapters_by_type("ncbi_genome_reports", "json")

    # Test that a format-specific adapter is chosen when type is generic
    generic_adapters = registry.get_adapters_by_type("some_generic_type", "json")

    # Test that a semantic type-specific adapter is chosen even if format is generic
    fasta_adapters = registry.get_adapters_by_type("protein_fasta", "some_fasta_format")

    # Test that no adapter is found for unsupported types/formats
    unsupported_adapters = registry.get_adapters_by_type("unsupported_type", "unsupported_format")

    assert len(ncbi_adapters) == 2
    assert len(generic_adapters) == 1
    assert len(fasta_adapters) == 1
    assert len(unsupported_adapters) == 0

    assert ncbi_adapters[0] == registry["MockNcbiAdapter"]
    assert generic_adapters[0] == registry["GenericJsonAdapter"]
    assert fasta_adapters[0] == registry["ProteinFastaAdapter"]


