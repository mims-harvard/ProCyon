# Scripts for ProCyon Model

These scripts assist users with running queries on and evaluating ProCyon.

`run_eval_framework.py`: 
------------------------
Entrypoint for running the evaluation framework to benchmark ProCyon models against other baselines and models across datasets and tasks. For detailed instructions, please see the [evaluation framework README](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate).

`caption_bulk.py`:
------------------
Generates captions for a list of UniProt IDs. Accesses the interal ProCyon database, so UniProt IDs must be contained within their (see ProCyon-Instruct for details). 

Usage:
```
usage: caption_bulk.py [-h] --ckpt CKPT [--save_path SAVE_PATH] [--chunk_idx CHUNK_IDX] [--num_chunks NUM_CHUNKS] [--max_len MAX_LEN]
                       [--beam_size BEAM_SIZE] [--diversity_penalty DIVERSITY_PENALTY] --uniprot_id_file UNIPROT_ID_FILE
                       [--prompt_dataset PROMPT_DATASET] [--prompt_relation PROMPT_RELATION]

options:
  -h, --help            show this help message and exit
  --ckpt CKPT           Path to ProCyon checkpoint
  --save_path SAVE_PATH
                        CSV file name to save captions to
  --chunk_idx CHUNK_IDX
                        Used for chunking the dataframe into separate pieces
  --num_chunks NUM_CHUNKS
                        Used for chunking the dataframe into separate pieces
  --max_len MAX_LEN     Maximum length of generated text
  --beam_size BEAM_SIZE
                        Beam size if using beam search
  --diversity_penalty DIVERSITY_PENALTY
                        Diversity penalty for diverse beam search
  --uniprot_id_file UNIPROT_ID_FILE
                        CSV with uniprot id's to process. UniProt IDs must be contained in a column titled 'uniprot_id'
  --prompt_dataset PROMPT_DATASET
                        Dataset prompt to use for the model. Generates in the style of this dataset. See dataset instructions for more
                        information.
  --prompt_relation PROMPT_RELATION
                        Relation for the dataset. Some datasets have more than one, such as GO process, component, and function.
(ProCyon) [oqueen@boslogin07 scripts]$ 

```
