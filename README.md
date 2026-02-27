# LoRē Genome

LoRē is a Python toolkit for genome/protein workflows backed by NCBI Datasets, with a web-first interface launched from the CLI.

**Status**: Under active development!

## Quick start
1. Clone the repository
2. Run the bootstrap helper
3. Launch LoRē

```bash
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
python run.py
```

This will generate a `run.bat` (Windows) or `run.sh` (Mac/Linux) that you can 
double-click on like any other program.

## Bootstrap helper <small>(recommended if you're not a computer expert)</small>
The bootstrap helper:
* Creates a .venv virtual environment
* Installs the bundled NCBI wheel
* Installs LoRē
* Prints clear next steps
* It does not download remote code or run arbitrary shell commands!

## Manual setup
If you are familiar with virtual environments and/or prefer to handle things 
yourself:
```bash
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install ./wheels/ncbi_datasets_pylib-1.0.0-py3-none-any.whl
pip install .

lore ui
```

## Workflow
From the UI:
1. Create a session.
2. Configure and run tasks.
3. Explore and download generated artifacts.

## External tools
- Some analysis tasks may require local third-party tools (currently only [MMSeqs2](https://github.com/soedinglab/MMseqs2))
- `ncbi-datasets-pylib` is not installed from PyPI for this project; install it from the bundled wheel.

## How to cite
If you use LoRe in research, cite:

> Hansen, T. (2026). *Identification and characterization of bacterial repeat-in-toxin adhesins using long-read genome analysis*. BioRxiv. https://www.biorxiv.org/content/10.1101/2025.09.30.679566v2

Until publication, also cite the repository:
https://github.com/thomashansen-queens/lore-genome

## Contributing
Pull requests and issue reports are welcome.

## License
This project is licensed under the BSD 3-Clause License. See LICENSE for details.

LoRe uses NCBI Datasets (U.S. National Library of Medicine, public domain under 17 U.S.C. § 105). LoRe is an independent project and is not affiliated with or endorsed by NCBI.
