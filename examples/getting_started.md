# Getting started with LoRē

This walkthrough will help you install, start, and get oriented with the user-
interface.

## 1. Launch the web interface
LoRē runs as a web app that runs on your computer. You start it in a terminal,
but interact with it in a web browser. That means if you close your terminal,
you close the program, but you can open and close as many tabs as you like.

Either run the launch script created by the `run.py` installer
(Linux/Mac: `run.sh`, Windows: `run.bat`) or activate the virtual environment
you installed to and type `lore ui` into the terminal.

* **Walkthrough:** Double-click the run script. A console will open, and a tab in
your web browser will open.

## 2. Select a workflow
A workflow is a series of "tasks" that are run sequentially. The program comes
pre-loaded with a handful of workflows, each with a title and brief description.

* **Walkthrough:** Click on the 'RTX adhesin identification' workflow.

## 3. Configure the workflow
At least one input to a task should be designated a "User Input" in a workflow.
These are the independent variables of a workflow and they will appear in the
sidebar to the left.

* **Walkthrough:**
  1. In the **Taxa** field, type a species name: `Vibrio cholerae`
  2. In the **New Session name** field, give your session a name: `Walkthrough test`

## 4. Edit the workflow
That's it! The workflow has been configured. You can inspect the tasks by
clicking on one (represented as rectangular nodes in the "graph view" of your
Session). To edit the task, click on the `Edit` button below the task name.

* **Walkthrough:**
  1. Click on the 'Find genome assemblies' task
  2. Click 'Edit'

## 5. Edit and run a task
This screen has two parts: the task configuration (left) and a preview output
(right). Some tasks have live previews, some will generate a preview only by
clicking on the `Preview` button at the bottom of the configuration panel.

* **Walkthrough:**
  1. Set the **Released after** date to 02/01/2026
  2. Set the **Released before** date to 03/31/2026
  3. Click on **Preview**. This will make an API call
to NCBI. You should see that 493 Vibrio cholerae genome assemblies were released
in the months of Feburary and March, 2026 for this configuration
  4. Next, click on 'Commit & Run' to confirm the configuration

## 6. View an artifact
Artifacts are pieces of data. The LoRē framework magically knows how to pass
information from an artifact into tasks. Clicking on an artifact will open the
'Explore' view. This is a simple in-browser tool that allows you to inspect
often messy data in a familiar tabular interface.

* **Walkthrough:**
  1. Click the 'Vibrio_cholerae' artifact of type 'ncbi.genome_reports' to open the explore view
  2. Try clicking on the 'Adapters' button at the top of the screen then select 'TextAdapter' to see the raw data returned by NCBI's server
  3. Switch back to the dedicated 'NcbiGenomeReportsAdapter' and type `pacbio` into the filter bar. You should see that one Vibrio cholerae genome assembly was sequenced using both PacBio and Illumina from our dataset.

## 7. Run the workflow
To run a workflow, click on the "Run all Tasks in order" button. This will automatically skip completed tasks and run the workflow one task at a time. This
runs in the background, so feel free to explore data as it is generated.

* **Walkthrough:**
  1. Click on the **Walkthrough test** navigation button at the top of the screen to go back to the overview of your session
  2. Click on "Run all tasks in order" and you should see the tasks in the graph turn green one-by-one and new artifacts appear as the workflow runs

## 8. Configure settings and dependencies
Some tasks require third-party programs to run. For example, the MMseqs2 task
requires you to get a copy of MMseqs2 and make it available to LoRē. This is
handled app-wide in the 'Settings', which can be found in the top-right hand corner of the screen. Here, you can configure core LoRē settings, as well as any
app-wide settings defined by the task plugins.

* **Walkthrough:**
  1. The workflow likely failed on the MMseqs2 task (unless your install was to a conda environment that already had mmseqs2 installed)
  2. To install MMseqs2, go to https://github.com/soedinglab/mmseqs2 and download a release for your operating system.
      * <small>Note: If you are using conda, in your terminal, type `conda install -c bioconda mmseqs2` and skip to step 7.</small>
  3. Extract it to a local directpry (e.g. `C:\Tools\` or `~/Applications`)
  4. If you are on Windows, double-click on `mmseqs.bat` to finish the installation
  5. In LoRē, click on **Settings**, then on **MMseqs2 suite** in the sidebar
  6. Copy and paste the system path to *mmseqs.exe* (Windows) or *mmseqs* (Linux/Mac). This will be something like `C:\Tools\mmseqs\bin\mmseqs.exe` or `~/Applications/mmseqs/bin/mmseqs`
  7. Click **Save MMseqs2 suite settings**. Now, LoRē knows where to find this program.
  8. Return to the main session screen, and click on "Run all tasks in order" again

## 9. Add a new Task
Workflows can be modified. You may want to add new tasks to modify or expand the
analysis of a workflow. The 'Task catalogue' in the sidebar is where you can
find a list of all task plugins currently installed.

* **Walkthrough:**
  1. Click on the artifact of data type *clustered_summary*
  2. Right-click on **Task catalogue** and select **Open in new tab** (for this walkthrough, we will be copying values from one tab into the other)
  3. Choose the "Genomic Neighbourhood" task
  4. Select the only available genome annotations (should be a 3.2 MB file)
  5. Which protein accessions would we like to see? Switch back to the **Clustered summary** tab and select a cell of the 'Cluster members' column containing two accessions. For *glutamate synthase large subunit* (symbol *gltB*) there are two. Select that cell, then press Ctrl+C to copy them: `WP_000227751.1,WP_182293089.1`
  6. Paste them into the manual entry box for **Protein accessions**. You should see a live preview of this syntenic neighbourhood!
  7. What's going on in that upstream region? Not all of the genes match up in this syntenic neighbourhood! Click on one of the genes to get more information.
  8. In one of the pop-ups, you will see that one of these assemblies (`GCF_054906185.1`) has a *IS200/IS605-like element IS1004 family transposase* inserted in this locus
  9. Check the **Write report** option (unchecked by default)
  10. Click on **Commit & Run**
  11. Look at the **Outputs** in the bottom right. This task generated two artifacts: a *neighbourhood_view** and a **neighbourhood_report**

## 10. Getting output
LoRē is simple by design. Each 'Session' is a directory on your hard drive. An Artifact is one or more file (e.g. paired-end FastQ reads are two files that count as one artifact). You can find all of the outputs in your file explorer (e.g. `C:\Users\thomas\lore-genome\sessions`) or you can use the interface to save copies to somewhere more convenient. From the Session overview, you can click on the triple-dot button ⋮ then select *Download*, or in the Explore view, click on *Export > Download Original*.

* **Walkthrough:**
  1. Open the **neighbourhood_view** artifact
  2. Click on *Export* then *Download Original*.
     * <small> The *Save as .png* is not currently functional! Coming soon</small>
  3. This SVG can be opened in Inkscape/Illustrator/your favourite SVG viewer
