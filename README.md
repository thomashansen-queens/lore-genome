# LoRē Genome

LoRē is an extensible workflow orchestrator for bioinformatics. It is designed with accessibility to scientists in mind, poriving a browser-based GUI to manage scripts, data, and workflows entirely locally.

**Status**: Under active development!

## Prerequisites
* Python 3.10 or higher
* Note: The installer automatically handles all necessary Python packages. Because data is stored locally on your machine, we recommend having a few gigabytes of free storage space.

## Quick start
1. Open a terminal (Mac: ⌘ + Spacebar -> Terminal, Windows: Start menu -> PowerShell)
2. To use the "bootstrap helper", copy and paste the following lines one at a time, pressing Enter after each

```bash
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
python run.py
```

3. This will generate a `run.bat` (Windows) or `run.sh` (Mac/Linux). Double-click it to launch LoRē.

## Bootstrap helper
The bootstrap helper automates the setup process for non-experts. It creates an isolated `.venv` virtual environment, installs LoRē and its dependencies, and starts the web UI.

## Manual setup
If you are familiar with virtual environments and/or prefer to handle things 
yourself:
```bash
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install LoRē
pip install .

# Launch the UI from terminal
lore ui
```

## Key features
* **Local**: The program and all plugins run on your computer.
* **Interactive**: Tinker with individual bioinformatic tasks and get results in real time. When you are satisfied, you can 'commit' those settings to your Session.
* **Explore**: The built-in browser makes it easy to inspect data (no more digging through huge `.fasta` or `.json` files)
* **Pipelines**: LoRē routes data (Artifacts) from one task directly to the next one
* **Workflows**: Once you have a set of tasks in a session, you can export it to a Workflow template for re-use. You can set independent variables in Workflows to speed up customization of subsequent runs.

## External tools
- Some analysis tasks may require local third-party tools (currently only [MMSeqs2](https://github.com/soedinglab/MMseqs2))

## How to cite
If you use LoRē in research, cite:

> Hansen, T. (2026). *Identification and characterization of bacterial repeat-in-toxin adhesins using long-read genome analysis*. BioRxiv. https://www.biorxiv.org/content/10.1101/2025.09.30.679566v2

Until formal publication, please also cite the repository:
https://github.com/thomashansen-queens/lore-genome

## Contributing
Pull requests and issue reports are welcome.

## License
This project is licensed under the BSD 3-Clause License. See LICENSE for details.

LoRē uses NCBI Datasets (U.S. National Library of Medicine, public domain under 17 U.S.C. § 105). LoRē is an independent project and is not affiliated with or endorsed by NCBI.
