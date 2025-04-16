# LoRe Genome

## Overview
LoRe (LOng-REad genomes) is a command-line tool for filtering and retrieving annotated protein sequences using NCBI Datasets API. Originally designed for long-read genome analysis, LoRe is now a general purpose, streamlined interface to get protein information from large, publicly available genomic datasets. It uses the NCBI Datasets API to fetch genome reports, retrieve protein annotations, and summarize the data for further analysis. It is essentially a wrapper around NCBI Datasets with some extra processing steps to provide simple, human-readable output.

## Quick start
1. Install LoRe (see [Install](#Install))
2. Configure your run by editing `config.yaml` (see [Configuration](#Configuration))
3. Run the pipeline: `lore pipeline` (see [Usage](#Usage))
4. Review the output `.csv` files containing the results

## Install
This assumes that you already have git and python installed on your system.
(Optional but recommended: create a virtual environment with python -m venv lore && source lore/bin/activate)
```
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
pip install --find-links wheels ncbi-datasets-pylib
pip install .
lore --help
```

## Configuration
The pipeline gets its instructions from a configuration file (`config.yaml`). It should be updated each time you run the pipeline. Configuration options include:
- **download_dir**: Local directory where results will be stored and data will be cached.
- **ncbi.api_key.cookieAuth**: Paste your [NCBI API](https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us) key here.
- **refseq_only**: Boolean indicating whether to fetch only RefSeq (GCF) records or also include INSDC (GCA).
- **taxons**: The scientific name of the target species.
- **search_terms**: A list of one or more sets of [search terms] to select genome assemblies.
- **genome_limit**: Creates a representative sample of genomes of a given size. This is recommended for practical reasons.
- **sampling_strategy**: When creating the representative sample, which columns to consider. `default` will provide a wide range of dates and locations.
- **assembly_level**: You can limit the completeness of the genome assemblies to consider.
- **mmseqs**: Path to your local install of mmseqs (or, if it is in your namespace, simply `mmseqs`) for clustering
- **cluster_residues**: In cases where you might want to consider only the start/tail X residues when clustering. Leave blank to cluster by whole sequences.
- **min_seq_id**: The minimum sequence identity to cluster proteins by.


## Search terms
LoRe enables advanced searching using NCBI datasets. The search terms look for matching text anywhere in the genome's metadata, which allows you to build various criteria for inclusion (strain, country of origin, isolation site, etc.). If you list multiple search terms, such as when fetching long read genomes

e.g.
```
search_terms:
- Nanopore
- PacBio
```

any genome assemblies that match one or both (OR search) will be included. When multiple search terms are placed on the same line

e.g.
```
search_terms:
- blood, Canada
```

only genomes that contain both terms (AND search) will be included. To include all genomes, leave search_terms blank.

## Usage
**lore pipeline**: The full pipeline uses MMseqs2 to cluster those proteins. This gives information on how prevalent the proteins are across genomes and how conserved the proteins are.

**lore genomes**: The simplest use case of this tool is to fetch and filter genome assemblies. This tool streamlines some parts of NCBI Datasets genome API.

**lore proteins**: Once a suitable set of genome assemblies is identified, you can simply fetch a .FASTA of every unique protein within those genomes. This is the second part of the pipeline.

## How to Cite
If you use LoRe in your research, please cite the associated paper:
> Hansen, T. (2025). *A hypothetical paper here*. [Journal Name], DOI: XXXXX
Until publication, please cite the GitHub repository:
https://github.com/thomashansen-queens/lore-genome

## Contributing
Contributions are welcome! Please submit a pull request, make forks, or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the BSD 3-Clause License. See the LICENSE file for more details.

LoRe makes use of data and tools from the NCBI Datasets service, a product of the U.S. National Library of Medicine. NCBI Datasets is in the public domain (17 U.S.C. § 105). LoRe is an independent project and is not affiliated with or endorsed by NCBI.