# The repeated-stimulus confound in electroencephalography.

This archive contains the source code used to perform the experiments and analyses described in *The repeated-stimulus confound in electroencephalography (2025)*.

## Executing the experiments and analyses

### 1. Installing dependencies

Before the code can be executed, the required dependencies must be installed. This can be achieved by installing Python (version 3.12), and then running ```pip install -r requirements.txt``` from the root directory of the archive.

Note that under Debian 12.11 bookworm/stable you may need to ```apt install r-base``` before doing ```pip install -r requirements.txt```.

### 2. Configuration

Next, to execute experiments in parallel on multi-GPU systems, modify the variable ```devices``` in ```src/main.py``` with the ids of the GPUs to be used.

### 3. Running the code

Lastly, the code can be executed by running ```python src/main.py``` from the root directory of the archive. This will then execute all experiments, and analyses. Model predictions and other data outputs will populate the ```outputs``` directory, while any figures referenced in the text will appear in the ```figures``` directory.

Note that you may need to do ```source .env``` followed by ```set -a``` before doing ```python src/main.py```.

## Modifying the experiments

The search space used for hyperparameter selection can be modified by changing the corresponding values in ```config/experiments/paired_category_decoding.yaml``` or ```config/experiments/paired_pseudocategory_decoding.yaml``` for the category-decoding and pseudocategory-decoding experiments respectively.
