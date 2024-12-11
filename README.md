# ProCyon: A multimodal foundation model for protein phenotypes
![ProCyon logo](assets/procyon_logo_large.png)

ProCyon is an open-source model for predicting protein phenotypes across scales.
This repository provides the official implementation of the model as described in our [paper](https://github.com/mims-harvard/ProCyon).
Our associated HuggingFace collection containing model weights and datasets can be found at the following links:

- Dataset: [ProCyon-Instruct](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct)
- Full model: [ProCyon-Full](https://huggingface.co/mims-harvard/ProCyon-Full)
- Benchmarking model: [ProCyon-Split](https://huggingface.co/mims-harvard/ProCyon-Split)
- Binding prediction model: [ProCyon-Bind](https://huggingface.co/mims-harvard/ProCyon-Bind)

## Installation
We recommend installing with [uv](https://docs.astral.sh/uv/), but install can also be done via `pip` alone. The `procyon` package used to interact with pre-trained models or train new models can be installed via
```
cd /path/to/ProCyon

# OPTIONAL: create virtual environment
python3 -m venv ./procyon_venv
source ./procyon_venv/bin/activate

# RECOMMENDED: use uv to install
python3 -m pip install uv
pyton3 -m uv sync --extra build
pyton3 -m uv sync --extra build --extra compile
python3 -m uv pip install -e .

# OR if omitting uv
python3 pip install -e .
```
We encourage installation within a virtual environment.

In addition to the package code, ProCyon also requires pre-trained weights for associated
models (e.g. Llama-3, ESM2) as well as access to the ProCyon-Instruct dataset. These dependencies
will all be stored in a single directory, which we denote `DATA_DIR`.

```
DATA_DIR=/path/to/data
mkdir $DATA_DIR
cd $DATA_DIR

# Clone ProCyon-Instruct dataset from HuggingFace
git clone git@hf.co:datasets/mims-harvard/ProCyon-Instruct

# Clone model weights for associated Llama models from HuggingFace
# Llama-3-8b for ProCyon-Full
cd model_weights/llama-3-8b
git clone git@hf.co:meta-llama/Meta-Llama-3-8B

# Llama-2-7b for ProCyon-Split
cd ../llama-2-7b-hf
git clone git@hf.co:meta-llama/Llama-2-7b-hf

# Add a `.env` file which the `procyon` package will use to find the `DATA_DIR`
cd /path/to/ProCyon
echo "DATA_DIR=\"$DATA_DIR\"" > .env
echo "HOME_DIR=\"$(pwd)\"" > .env
```

## Examples
For the core capabilities of ProCyon models, please see the provided demo notebooks.
- [Phenotype generation](https://github.com/mims-harvard/ProCyon/blob/main/examples/phenotype_generation.ipynb)
- [Retrieval](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb)

## Coming soon!
- Additional notebooks with analysis examples
- Reproduction code from the manuscript
- Full training documentation and tutorial
