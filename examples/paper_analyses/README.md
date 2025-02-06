# Reproducibility README

This directory contains code to reproduce all experiments in the [ProCyon preprint](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1).
All experiments are listed below; if a panel is not listed, that means that this panel was qualitative and does not have reproducibility code included.

Before attempting to reproduce, please follow all installation instructions [here](https://github.com/mims-harvard/ProCyon?tab=readme-ov-file#installation), and ensure the correct versions of each library are installed.
This includes downloading datasets, model checkpoints, etc. as many notebooks and scripts make use of these resources, including files from the dataset available on [Huggingface](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct).

## Figure 2

**Panel f,g**: These results were generated using our evalaution framework. Please see the [README](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate) for more details on how to run it.

## Figure 3

**Panel b**: Please see [`protein_retrieval_multiple_sources.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/protein_retrieval_multiple_sources.ipynb).

**Panel c**: Due to copyrights of the DSM used for this analysis, we are not allowed to share this notebook publicly. Please reach out to us if you are interested in reproducing this analysis; contact info on main README.

**Panel d**: Please see [`embedding_comparison.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/embedding_comparison.ipynb).

**Panel e,f**: Please see [`composition_retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/composition_retrieval.ipynb).

**Panel g**: Please see [`sting_retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/sting_retrieval.ipynb).

## Figure 4

**Panel a**: Reproducibility for this example is shown in the phenotype generation demo notebook located in [`ProCyon/examples/phenotype_generation.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb).

**Panel b,c,d**: These results were generated through our evaluation framework and LLM-as-a-Judge pipeline. Please see more details [here](https://github.com/mims-harvard/ProCyon/tree/main/examples/paper_analyses/external_llm_eval).

## Figure 5

**Panel a,b**: Please see [`ProCyon/examples/paper_analyses/drugdomain.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/drugdomain.ipynb).

**Panel c**: Coming soon!

**Panel d**: Coming soon!

**Panel e**: Coming soon!

**Panel f**: Reproducibility code is included in the retrieval example in [`ProCyon/examples/retrieval.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb).

## Figure 6
All code to reproduce Figure 6 is contained in the following notebooks: [`pd_control_lists.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/pd_control_lists.ipynb), [`pd_uncharacterized.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/pd_uncharacterized.ipynb).

## Extended data figures

**ED Fig 2,3,4**: Please see the evaluation framework [README](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate).

**ED Fig 6**: Coming soon!

**ED Fig 7**: These results are included in the notebook [`ProCyon/examples/phenotype_generation.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/retrieval.ipynb); this is a more in-depth view of the results shown in Figure 4a.

**ED Fig 8**: Please see [`bertscore_by_qa_filter.ipynb`](https://github.com/mims-harvard/ProCyon/blob/main/examples/paper_analyses/bertscore_by_qa_filter.ipynb).

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
