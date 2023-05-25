# SLAP Platform - Data Analysis and Virtual Library
Code to accompany the paper 

>J. Götz, M. K. Jackl, C. Jindakun, A. N. Marziale, J. André, D. J. Gosling, C. Springer, M. Palmieri, M. Reck, A. Luneau, C. E. Brocklehurst, J. W. Bode, High-throughput synthesis and data generation for the prediction of molecular properties and synthesizability, *under review*.

The code in this repository serves two purposes:
1) Provide an accessible frontend for the virtual library
2) Reproduce the data analysis and virtual library generation from the paper

If you only want to use the virtual library, follow the instructions under installation, then skip ahead to 
Usage/Virtual Library Frontend.


## Installation

Install the required packages through conda. An `environment.yaml` file is provided to facilitate installation.
```bash
conda env create -f environment.yaml
```

---
## Usage 

The code is organized into Jupyter notebooks.
- Data Processing: `featurize_slap_reaction.ipynb` and `generate_ml_datasets.ipynb`
- Data splitting: `canonical_0D_data_split.ipynb`, `canonical_1D_data_split.ipynb`, and `canonical_2D_data_split.ipynb`
- Virtual Library Enumeration: `vl_building_block_filtering.ipynb` and `vl_enumeration.ipynb`
- Virtual Library Frontend: `vl_frontend.ipynb`

to run the Jupyter server, follow the installation instructions above, then activate the environment and run:
```bash
conda activate slap-platform
cd <path/to/slap-platform>
jupyter-notebook
```

### Data
Data are read from `PROJECT_ROOT/data/`. Download the supplementary data from
[Zenodo](https://doi.org/10.5281/zenodo.7950707) and place it in the `data` directory.
You will need Data S1 to run the data processing notebooks,
Data S3 to run the data splitting notebooks,
and Data S4 (extract the tar archive!) to run the virtual library frontend.

### Virtual Library Frontend
The virtual library frontend allows querying the virtual library for a single SMILES string.
To use it, download Data S4 from [Zenodo](https://doi.org/10.5281/zenodo.7950707) and extract it into the `data` directory.
Then, run the `vl_frontend.ipynb` notebook (start the Jupyter server as described in `Usage`).
