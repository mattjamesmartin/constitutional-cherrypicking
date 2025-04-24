# Constitutional Cherry-Picking Replication Materials

This repository contains the replication materials for the manuscript Constitutional Cherry-picking: How Drafters Leverage Public Input in Constitution-Making. The files include:

- Processed transcripts of constitutional deliberations from Chile (2021–22) and Cuba (2018–19), annotated with metadata.
- N-gram search tool used to identify and classify references to public input.
- Categorical tagging scheme and annotated results for six rhetorical uses of public input: Citation, Justification, Rejection, Legitimacy, Agenda-Setting, and Critique.
- Scripts for data extraction, analysis, and visualization, including semantic similarity scoring and metadata integration.
- This README file with step-by-step instructions for reproducing the analysis.

These materials enable full replication of the paper’s findings and support further exploration of how public consultation is invoked during constitutional negotiations. All code is written in Python and designed for compatibility with Jupyter Notebooks.

## Quick Start

1. Install [Anaconda](https://www.anaconda.com/download)
2. Open a new terminal window and run:
   conda create -n pubcon python=3.9.21 pip jupyter  
   conda activate pubcon
3. Clone or download this repository.
4. Install required packages:
   pip install -r required_packages.txt
5. Launch Jupyter Notebook:
   jupyter notebook
6. Open `installer.ipynb` and select **"Run All Cells"** from the "Run" tab.

---

## First Steps

### Download Anaconda

- Download and install [Anaconda](https://www.anaconda.com/download)
- Accept the default installation settings and choose to initialize Conda when prompted.

### Create Conda Environment

After installation, open a new terminal window and create the environment:

   `conda create -n pubcon python=3.9.21 pip jupyter` 
   `conda activate pubcon`

---

## Installation

### Obtain the Repository

The recommended method is to use GitHub Desktop to clone the repository. The directory includes:

- `cython/`: Prebuilt shared objects (`angular_distance.so`) for Mac Intel, Mac ARM, and Linux Intel. You may need to build your own version for other systems.
- `installer.ipynb`: Jupyter notebook for installing data and NLP model resources.
- `es_core_news_lg-3.7.0/`: spaCy model for Spanish text segmentation (compatible with spaCy v3.7.x).
- `processing/`: Python code for initializing the CCP model and processing Chilean data. This code is user-configurable.
- `required_packages.txt`: Required Python packages for your Conda environment.
- `README.md`: This file.

### Install Required Packages

Make sure you’re in the same directory as `required_packages.txt`. You can set your working directory using the `cd` command in the terminal. For example:

   `cd "/Users/janedoe/Downloads/pubcon-main"`
   
Then run:

   `pip install -r required_packages.txt`

---

### Open Jupyter Notebook

With the `pubcon` environment activated, run:

   `jupyter notebook`

This will open a new browser tab with the Jupyter interface.

---

## Run the Installer

Using Jupyter, navigate to the repository folder and open `installer.ipynb`. Select **"Run All Cells"** from the "Run" menu.

This will populate your top-level directory with:

- `data/`: Chilean data sources and benchmark data
- `model/`: Serialized objects from text and topic processing
- `use_ml_3/`: Multilingual sentence-level encoder

Depending on your machine and internet connection, this may take several minutes.

---

## Platform-Specific Step: Install `angular_distance.so`

The `processing/` directory includes `angular_distance.so` compiled for Mac Intel by default.

If you're on a different architecture, replace it with the correct file from the `cython/` subdirectories.

To build your own shared object file, set your directory to the `cython/` folder and run:

   `conda install cython`
   `python setup.py build_ext --inplace`

Ensure this is run within the activated `pubcon` environment.

Rename `angular_distance.cpython-39-darwin.so` to `angular_distance.so` and move the file to `analysis/_library` and `processing/`. You can replace the old .so files.

---

## Processing: Run `pipeline.py`

`pipeline.py` processes data sources located in `../data/` and is configured through the `config` data structure in the `main()` function.

### Steps:

1. Switch to the `processing/` directory in your terminal.
2. In `pipeline.py`, set the `run` flag to `True` for any data source you want to process.
3. Run `pipeline.py`. It will process all selected data sources based on your configuration.

   `python pipeline.py`

The configuration mirrors the `data/` directory layout, so no changes are required beyond updating the `run` flags.

---

## Run the 

Using Jupyter, navigate to the `analysis/` folder and open `pubcon.ipynb`. 

Run the first cell complete initialization. Then you are good to go. Run other cells as needed.

Have fun!
