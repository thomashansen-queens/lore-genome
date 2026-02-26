"""
lore/lib/parsing/ncbi.py
Parsing logic for NCBI Datasets artifacts.
"""
import json
from typing import List, Dict, Any
from pathlib import Path
import logging
import pandas as pd

from lore.core.artifacts import Artifact

logger = logging.getLogger(__name__)

def ncbi_genome_json_to_df(json_content: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of NCBI genome report dicts into a flat Pandas DataFrame.
    Uses double underscores ('__') to flatten nested JSON fields.
    """
    if not json_content:
        return pd.DataFrame()

    # json_normalize flattens nested dicts
    # (e.g. assembly_info.release_date -> assembly_info__release_date)
    df = pd.json_normalize(json_content, sep="__")

    # Optional: Clean up common column names for readability
    # (You can expand this mapping as you discover more ugly columns)
    cleanup_map = {
        "assembly_info__biosample__accession": "biosample_acc",
        "assembly_info__assembly_level": "level",
        "assembly_info__assembly_year": "year",
        "organism__tax_id": "tax_id",
        "organism__organism_name": "organism"
    }
    df.rename(columns=cleanup_map, inplace=True)

    return df

def extract_accessions(artifact: Artifact, root_path: Path) -> List[str]:
    """
    Attempts to extract a list of NCBI Accessions (GCF_...) from an artifact.
    Supported formats:
      - ncbi_genome_reports (JSON list of dicts)
      - ncbi_annotation_report (JSONL or JSON array)
      - any .json file with an 'accession' field in list items
    """
    file_path = (root_path / artifact.path).resolve()
    if not file_path.exists():
        logger.warning("Artifact file not found: %s", file_path)
        return []
    # If the artifact is a JSON file, try to parse it
    if file_path.suffix in (".json", ".jsonl"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = []
            if isinstance(data, list):
                # JSON array / JSONL
                items = data
            elif isinstance(data, dict):
                # JSON object with a list under 'reports' or similar
                items = data.get('reports', []) or data.get('assemblies', []) or [data]
            accessions = []
            for item in items:
                # 1. Flattened JSON
                acc = item.get('accession') or item.get('assembly_info__assembly_accession')
                if not acc and isinstance(item.get('assembly_info'), dict):
                    # 2. Nested JSON
                    acc = item['assembly_info'].get('assembly_accession')
                if not acc and isinstance(item.get('annotations'), dict):
                    # 3. Annotation report format
                    acc = item['annotations'].get('assembly_accession')
                if acc and isinstance(acc, str):
                    accessions.append(acc)
            # De-duplicate while preserving order
            seen = set()
            unique_accs = [a for a in accessions if not (a in seen or seen.add(a))]
            logger.info("Extracted %d unique accessions from %s", len(unique_accs), artifact.name or artifact.id)
            return unique_accs
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to parse JSON from artifact %s: %s", artifact.name or artifact.id, e)
            return []
    return []
