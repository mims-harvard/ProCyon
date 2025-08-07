# Reproducibility README

This directory contains code to reproduce all experiments in the [ProCyon preprint](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1).
All experiments are listed below; if a panel is not listed, that means that this panel was qualitative and does not have reproducibility code included.

Before attempting to reproduce, please follow all installation instructions [here](https://github.com/mims-harvard/ProCyon?tab=readme-ov-file#installation), and ensure the correct versions of each library are installed.
This includes downloading datasets, model checkpoints, etc. as many notebooks and scripts make use of these resources, including files from the dataset available on [Huggingface](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct).

## Training
All details about training are found in [`ProCyon/examples/training`](https://github.com/mims-harvard/ProCyon/tree/main/examples/training).

## Figure 2 (ProCyon accurately retrieves proteins from flexible phenotypes)
**Panels b,c**: These results were generated using our evalaution framework. Please see the [README](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate) for more details on how to run it.

**Panel e**: Please see [`composition_retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/composition_retrieval.ipynb).

**Panel f**: Please see [`protein_retrieval_multiple_sources.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/protein_retrieval_multiple_sources.ipynb).

**Panel g**: Due to copyrights of the DSM used for this analysis, we are not allowed to share this notebook publicly. Please reach out to us if you are interested in reproducing this analysis.

**Panel h**: Please see [`sting_retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/sting_retrieval.ipynb).

## Figure 3 (ProCyon generates accurate responses and phenotype descriptions from multimodal prompts)

**Panels b,c**: These results were generated using our evalaution framework. Please see the [README](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate) for more details on how to run it.

**Panels e,g,h**: These results were generated through our evaluation framework and LLM-as-a-Judge pipeline. Please see more details on generating comparisons against external LLMs [here](https://github.com/mims-harvard/ProCyon/tree/main/examples/paper_analyses/external_llm_eval), which includes scripts for the following:
 - generating phenotype generation and QA prompts for external LLMs
 - generating prompts for LLM-as-a-Judge ranking
 - parsing and analyzing LLM-as-a-Judge results
 - scoring generated phenotypes against reference texts

## Figure 4 (ProCyon models domains, peptides, and small molecules beyond proteins)

**Panels a,b**: Please see [`ProCyon/examples/paper_analyses/drugdomain.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/drugdomain.ipynb).

**Panel c**: Reproducing this experiment requires a few steps:
  1. Download the ProCyon-Bind model, which has been finetuned for protein-peptide prediction. Training details for ProCyon-Bind are in [`examples/training`](https://github.com/mims-harvard/ProCyon/tree/main/examples/training), but it suffices to use the downloaded model weights for the experiment.
  2. [OPTIONAL] Once downloaded, you can run [`protpep_qa_score.py`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/protpep_qa_scores.py) to generate the file `ace2_preds.pickle`, which contains the predictions for the QA formulation of the protein-peptide binding prediction problem. You can optionally use the provided file in `ProCyon-Instruct/experimental_data/ProteinPeptideBinding/ace2_preds.pickle`.
  3. All remaining information and analyses is in [prot_pep.pynb](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/prot_pep.ipynb).

**Panel d**: Reproducibility code is included in the retrieval example in [`ProCyon/examples/retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb).

## Figure 5
**Panel d**: Please see [`ProCyon/examples/paper_analyses/fig5_function_retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/fig5_function_retrieval.ipynb).

**Panel e** All reproduction code is contained in the following notebooks: [`pd_control_lists.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/pd_control_lists.ipynb), [`pd_uncharacterized.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/pd_uncharacterized.ipynb).

## Figure 6

Coming soon!


## Extended data figures

**ED Fig 1**: Please see [`embedding_comparison.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/embedding_comparison.ipynb).

**ED Fig 2**:  These results are included in the notebook [`ProCyon/examples/phenotype_generation.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb).

**ED Fig 3**: Please see [`bertscore_by_qa_filter.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/bertscore_by_qa_filter.ipynb).

**ED Fig 6**: Coming soon!

**ED Fig 7**: Coming soon!

**ED Fig 8**: Coming soon!

**ED Fig 9**: Generation for these proteins can be ran by running two scripts:
1. `ProCyon/scripts/caption_bulk.py`: Run the following command:
```
python3 scripts/caption_bulk.py \
  --ckpt /MYPATH/ProCyon-Full \
  --save_path ./edfig9_phenotypes.csv \
  --uniprot_id_file /MYPATH/ProCyon-Instruct/experimental_data/PD_uncharacterized/gene_lists/pdu_caption_proteins.csv \
  --prompt_dataset go \
  --prompt_relation process
```
2. `ProCyon/scripts/qa_filter_captions.py`: Run the following command:
```
python3 scripts/qa_filter_captions.py \
  --ckpt /MYPATH/ProCyon-Full \
  --caption_fpath ./edfig9_phenotypes.csv \
  --save_path ./tmp_caption_qafilter.csv \
  --prompt_dataset go \
  --prompt_relation process
```
