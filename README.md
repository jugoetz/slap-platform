# SLAP Platform - Data Analysis and Virtual Library
Code to accompany the paper ... [INSERT CITATION]

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

to run the Jupyter server, follow the installation instructions above, then activate the environment and run:
```bash
conda activate slap-platform
jupyter-notebook
```

### Data
Data are read from `PROJECT_ROOT/data/`. Download the supplementary data from
[Zenodo](https://zenodo.org/) [CHANGE URL] and place it in the `data` directory.
