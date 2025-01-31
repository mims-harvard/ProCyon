# Reproducibility README

This directory contains code to reproduce all experiments in the [ProCyon preprint](https://www.biorxiv.org/content/10.1101/2024.12.10.627665v1).
All experiments are listed below; if a panel is not listed, that means that this panel was qualitative and does not have reproducibility code included.

Before attempting to reproduce, please follow all installation instructions here, and ensure the correct versions of each library are installed.
This includes downloading datasets, model checkpoints, etc. as many notebooks and scripts make use of these resources, including files from the dataset available on [Huggingface](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct).

## Figure 2

**Panel f**: Coming soon!

**Panel g**: Coming soon!

## Figure 3

**Panel a**: Coming soon!

**Panel b**: Coming soon!

**Panel c**: Coming soon!

**Panel d**: Coming soon!

**Panel e**: Coming soon!

**Panel f**: Coming soon!

**Panel g**: Coming soon!

## Figure 4

**Panel a**: Reproducibility for this example is shown in the phenotype generation demo notebook located in `ProCyon/examples/phenotype_generation.ipynb`.

**Panel b**: Coming soon!

**Panel c**: Coming soon!

**Panel d**: Coming soon!

## Figure 5

**Panel a,b**: Notebook `ProCyon/examples/paper_analyses/drugdomain.ipynb`.

**Panel c**: Coming soon!

**Panel d**: Coming soon!

**Panel e**: Coming soon!

**Panel f**: Reproducibility code is included in the retrieval example in `ProCyon/examples/retrieval.ipynb`.

## Figure 6
All code to reproduce Figure 6 is contained in the `pd_uncharacterized.ipynb` notebook.

## Extended data figures

**ED Fig 2**: Coming soon!

**ED Fig 3**: Coming soon!

**ED Fig 4**: Coming soon!

**ED Fig 6**: Coming soon!

**ED Fig 7**: These results are included in the notebook `ProCyon/examples/phenotype_generation.ipynb`; this is a more in-depth view of the results shown in Figure 4a.

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
