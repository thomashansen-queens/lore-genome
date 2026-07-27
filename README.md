# LoRē Genome

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21615247.svg)](https://doi.org/10.5281/zenodo.21615247)

LoRē is an extensible workflow orchestrator for bioinformatics. It is designed with accessibility to scientists in mind, providing a browser-based GUI to manage scripts, data, and workflows entirely locally.

**Status**: Under active development! Core functionality is assured, but certain features are known to be non-functional at the moment.

## Installation

### Prerequisites
* Python 3.11 or higher
* The installer automatically handles all necessary Python packages (found in requirements.txt)
* Because data is stored locally on your machine, we recommend having a few gigabytes of free storage space.

### Quick start
1. Open a terminal (Mac: ⌘ + Spacebar -> Terminal, Windows: Start menu -> PowerShell)
2. To use the "bootstrap helper", run the following lines one at a time, pressing Enter after each.

```bash
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome
python run.py
```

3. This will generate a `run.bat` (Windows) or `run.sh` (Mac/Linux).
4. Run `./run.sh` (Mac/Linux) or `.\run.bat` (Windows) in your terminal to launch LoRē.

### What is the 'Bootstrap helper'?
The bootstrap helper automates the setup process (useful for non-experts) and is the recommended way to install LoRē. It checks your version of Python, creates an isolated `.venv` virtual environment, installs LoRē and its dependencies, creates a launcher script (`run.bat` or `run.sh`) and starts the web UI.

### Manual setup
If you are familiar with virtual environments and/or prefer to handle things 
yourself:
```bash
# Clone this repo
git clone https://github.com/thomashansen-queens/lore-genome.git
cd lore-genome

# (Recommended) Install in a dedicated environment
python -m venv .venv  # or use conda
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install LoRē
pip install .

# Launch the UI from terminal
lore ui
```

### Updating
If you have already cloned the repository, simply pull the latest changes. The launcher `run` script will automatically upgrade:
```bash
cd lore-genome
git pull
./run.sh  # or (.\run.bat on Windows)
```
For manual setups, activate your virtual environment first, then pull and re-install:
```bash
cd lore-genome
source .venv/bin/activate  # Windows: .venv\Scripts\activate
git pull
pip install .
```

## Using the program

### Walkthroughs
Once installed, please refer to our [getting started](examples/getting_started.md) for an orientation on how to run and use LoRē.

### Key features
* **Local**: The orchestrator runs on your hardware!
* **Interactive**: Tinker with individual bioinformatic tasks and preview results in real time. When you are satisfied, you can 'commit' those settings to your Session.
* **Explore**: The built-in browser makes it easy to inspect data (no more digging through huge `.fasta` or `.json` files)
* **Pipelines**: LoRē routes data (Artifacts) from one task directly to the next one, automatically slicing tables and parsing file types
* **Workflows**: Once you have a set of tasks in a session, you can export it to a Workflow template for re-use. Setting independent variables in Workflows ('user inputs') speeds up customization of subsequent runs.

### External tools
Some plugins may require local third-party tools. The ability to write a thin wrapper plugin around an external tool, giving it a simple GUI and the ability to pipe data in/out is one of the key advantages of LoRē. If a plugin is not available or not working, check the logs to see if there is a missing dependency.

Paths to external tools can be set in the Settings page. You can add them to your system `PATH` or copy-and-paste the direct path to the tool's binary/executable, e.g.:
* Mac/Linux: `/Users/My Name/mmseqs/bin/mmseqs`
* Windows: `C:\Users\My Name\mmseqs\bin\mmseqs.exe`

## Uninstalling
Because LoRē is installed and runs locally from source code, removing it is straightforward:

### 1. Remove the program
If you installed using `python run.py`, the codebase, virtual environment (`.venv`), and launcher scripts are all contained inside the `lore-genome` repository folder. Simply delete the folder! This can be done in your file explorer, or from your terminal:
* Mac/Linux: `rm -rf lore-genome`
* Windows: `Remove-Item -Recurse -Force lore-genome`

If you installed manually (e.g. with pip or conda):
1. Activate the environment you installed it to (if you are unsure, type `pip list`)
2. Uninstall with `pip uninstall lore-genome`
3. Delete the cloned `lore-genome` directory

### 2. Remove cached files
LoRē creates a cache in your Home directory. To clean this up, either find it and delete it in file explorer, or use a console command:
* Mac/Linux: `rm -rf ~/lore-genome`
* Windows: `Remove-Item -Recurse -Force ~$env:USERPROFILE\lore-genome`

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
