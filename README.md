# ProCyon: A multimodal foundation model for protein phenotypes
![ProCyon logo](assets/procyon_logo_large.png)

ProCyon is an open-source model for predicting protein phenotypes across scales.
This repository provides the official implementation of the model as described in our [overview page](https://github.com/mims-harvard/ProCyon) and our [paper](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1).
Our associated HuggingFace collection containing model weights and datasets can be found at the following links:

- Dataset: [ProCyon-Instruct](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct)
- Full model: [ProCyon-Full](https://huggingface.co/mims-harvard/ProCyon-Full)
- Benchmarking model: [ProCyon-Split](https://huggingface.co/mims-harvard/ProCyon-Split)
- Binding prediction model: [ProCyon-Bind](https://huggingface.co/mims-harvard/ProCyon-Bind)

## Installation
Requirements:
- CUDA toolkit, particularly `nvcc`
- Sign up for Huggingface permissions for LLaMA-3 at [this link](https://huggingface.co/meta-llama/Meta-Llama-3-8B). You'll need this to use ProCyon-Full and ProCyon-Bind.

### Quick start
We recommend installing with [uv](https://docs.astral.sh/uv/) but install can also be done via `pip` alone. The `procyon` package used to interact with pre-trained models or train new models can be installed via
```
cd /path/to/ProCyon

uv sync --extra build
uv sync --extra build --extra compile
uv pip install -e .
source .venv/bin/activate

# OR if omitting uv
python3 -m pip install -e .
```
Installation with `uv` should take less than 10 minutes, depending on the
speed of your internet connection for downloading packages.

ProCyon also requires pre-trained weights for associated
models (e.g. Llama-3, ESM2) as well as access to the ProCyon-Instruct dataset.
You'll need to request access to the LLaMA-3 model through the model page [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
These dependencies
will all be stored in a single directory, which we denote `DATA_DIR`.
```
# Decide where you'd like to keep the ProCyon-Instruct repo and clone
cd /desired/path/for/data
git clone git@hf.co:datasets/mims-harvard/ProCyon-Instruct

# Clone model weights for associated Llama models from HuggingFace
cd /path/to/llama3/ # Llama-3-8b for ProCyon-Full
# Ensure you've signed up for LLaMA-3 access
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B
echo "LLAMA3_PATH=/path/to/llama3/Meta-Llama-3-8B" >> .env

cd ../llama-2-7b-hf # Llama-2-7b for ProCyon-Split
git clone git@hf.co:meta-llama/Llama-2-7b-hf

# Add a `.env` file which the `procyon` package will use to find the `DATA_DIR`
DATA_DIR=/desired/path/for/data/ProCyon-Instruct
cd /path/to/ProCyon

echo "DATA_DIR=\"$DATA_DIR\"" > .env
echo "HOME_DIR=\"$(pwd)\"" >> .env
```

**Version note**: We are aware of a bug where having `transformers>4.31.0` changes generated model outputs. Please ensure your `transformers` version is set to 4.31.0 (as in environment requirements) for inference of ProCyon.

## Examples
### Core capabilities
For performing retrieval and phenotype generation with ProCyon models, please see the following demo
notebooks. Both examples should run in less than 5 minutes depending on the
speed of your GPU.
- [Phenotype generation](https://github.com/mims-harvard/ProCyon/blob/main/examples/phenotype_generation.ipynb)
- [Retrieval](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb)

### Additional capabilities
We also provide notebooks and scripts for exploring other capabilities of ProCyon models:
- [Drug-binding domain prediction](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/drugdomain.ipynb)
- [Protein-peptide binding prediction](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/prot_pep.ipynb)
- [Pleiotropic protein retrieval](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/composition_retrieval.ipynb)
- [Bulk phenotype generation](https://github.com/mims-harvard/ProCyon/blob/main/scripts/caption_bulk.py) and [QA filtering](https://github.com/mims-harvard/ProCyon/blob/main/scripts/qa_filter_captions.py)

Additionally, we provide all scripts and notebooks for reproducing the analyses in our manuscript figures [here](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/README.md).

### Benchmarking
We developed an evaluation framework for systematic comparison of ProCyon models against other baselines and models. Please see the
[example configs and scripts](https://github.com/mims-harvard/ProCyon/blob/main/examples/evaluation)
or the [evaluation README](https://github.com/mims-harvard/ProCyon/blob/main/procyon/evaluate/README.md) for instructions.

### Training
For details on training a ProCyon model and example scripts, please see the [training README](https://github.com/mims-harvard/ProCyon/tree/main/examples/training/README.md).

## Citation
```
@article {Queen2024.12.10.627665,
  author = {Queen, Owen and Huang, Yepeng and Calef, Robert and Giunchiglia, Valentina and Chen, Tianlong and Dasoulas, George and Tai, LeAnn and Ektefaie, Yasha and Noori, Ayush and Brown, Joseph and Cobley, Tom and Hrovatin, Karin and Hartvigsen, Tom and Theis, Fabian and Pentelute, Bradley L. and Khurana, Vikram and Kellis, Manolis and Zitnik, Marinka},
  title = {ProCyon: A multimodal foundation model for protein phenotypes},
  elocation-id = {2024.12.10.627665},
  year = {2024},
  doi = {10.1101/2024.12.10.627665},
  URL = {https://www.biorxiv.org/content/early/2024/12/15/2024.12.10.627665},
  eprint = {https://www.biorxiv.org/content/early/2024/12/15/2024.12.10.627665.full.pdf},
  journal = {bioRxiv}
}
```
