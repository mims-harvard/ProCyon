# External LLM evaluation
This directory contains the scripts necessary to recreate our experiments comparing ProCyon model phenotype generation
to vanilla LLMs. These results are largely those presented in Figures 4C and 4D in our manuscript. Note that
automated reference-based metrics, as shown in the lower facet of Figure 4B, can be generated for ProCyon models using
the evaluation framework described [here](https://github.com/mims-harvard/ProCyon/tree/main/procyon/evaluate).

Our workflow for comparing to external LLMs is based on an [LLM-as-a-Judge approach](https://arxiv.org/abs/2306.05685) which
aims to approximate the nuances of human evaluation while being more scalable due to using an LLM to rank outputs instead of
a human. Our approach is based on pairwise comparisons: the LLM judge is presented with two phenotype texts and is asked to
select which better captures a provided set of reference annotations, or to output a tie if both seem equally good or bad.

The overall workflow consists of the following steps:
1. Create set of proteins for evaluation - since ProCyon-Instruct is a large dataset covering many proteins, selecting a subset of proteins for evaluation can greatly reduce the cost of external LLM calls
2. Generate ProCyon phenotype descriptions for the selected subset
3. Generate external LLM phenotype descriptions for the selected subset
4. Generate LLM-as-a-Judge prompts using both sets of descriptions
5. Generate responses from the judge LLM
6. Parse final results

We provide scripts and notebooks for each step of this process.

## Protein subset selection
To ensure that our evaluation subset captures proteins with varying levels of annotation, i.e. includes both well-studied and
under-studied proteins, we perform subsampling stratified by [UniProt annotation scores](https://www.uniprot.org/help/annotation_score).

The notebook used for our subsampling is `select_llm_samples.ipynb`, and should recreate the exact subset of proteins used for
our analyses, but the precomputed protein subsets as well as the UniProt annotation scores used can also be found in the
[ProCyon-Instruct dataset](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct/tree/main/experimental_data/llm_judge_eval/selected_caption_samples).

## Generate ProCyon phenotype descriptions
Generating phenotype descriptions from a ProCyon model can be accomplished using the evaluation framework. If using a protein subset as
described above, the dataset config should be modified to specify the desired subset as a TSV file. For example:
```
> head dataset_config.yml
it_datasets:
  testing:
     - aaseq_type: protein
       text_type: reactome
       relations: [all]
       tasks: [caption]
       splits: [all]
       split_method: random_reactome_centric
       dataset_args:
         aaseq_subset_tsv_path: /path/to/reactome_subset.tsv
> head /path/to/reactome_subset.tsv
protein_id      seq_id  num_relations   annotation_score        num_pubs
Q8TD47  13344   48      2       2
Q6P5R6  12992   37      2       9
A5YM69  915     8       2       3
A8MVX0  914     8       2       2
A1IGU5  922     8       2       1
Q53LP3  15114   7       2       9
Q4QY38  3736    5       2       2
Q30KP9  3737    5       2       1
Q30KQ7  3721    5       2       1
```
where `/path/to/reactome_subset.tsv` follows the TSV format output by `select_llm_samples.ipynb` but can be any TSV with a `protein_id` column.

After running the evaluation run, the generated phenotypes per dataset will be contained in `{eval_output_dir}/caption/ProCyon/{dataset_name}/full_captions.tsv.gz`.

## Generate external LLM phenotype descriptions
The exact workflow for generating phenotype descriptions with an external LLM will vary depending on the external LLM used, so we only provide
a script for generating prompts that match those used for ProCyon models, which can then be used to query your LLM of choice.

The script `generate_llm_prompts.py` takes in the same dataset config used by the evaluation framework and outputs one CSV per dataset containing
the prompt and the associated protein's UniProt ID. Note that this script can also be used to generate QA prompts by including `qa` in the list of
tasks in the corresponding dataset entry. If generating a QA prompt, the output CSV will also contain a column giving the expected answer.

The script can be run as follows:
```
/path/to/ProCyon/examples/paper_analyses/llm_judge_phenotype_eval/generate_llm_prompts.py \
  --it_data_config_yml dataset_config.yml \
  --encoding aaseq
```
with output written to the current directory. The `--encoding` argument specifies how to represent proteins in text. The possible options are:
  - `aaseq` - use full amino acid sequence
  - `gene_name` - use HGNC gene name as provided by UniProt. In the case of multiple names, provide all of them.
  - `protein_name` - use long-form protein name provided by UniProt
  - `uniprot` - use UniProt ID

Given the resulting CSV of prompts, one can then prompt the desired LLM using your workflow of choice.

## Generate LLM-as-a-Judge prompts
Given phenotype descriptions from a ProCyon model and an external LLM, we can then generate the prompts for the LLM judge to rank the phenotype
descriptions. For each protein, we generate prompt that consists of a preamble, the reference phenotype annotations for that protein and the
corresponding knowledge domain, and the two phenotype descriptions. The judge is then asked to output either a single winner or a tie.

To combat positionality bias, in which the judge shows a systematic bias to the description provided either first or second, we actually generate
two prompts per protein, one for each possible ordering of the two prompts. In the parsing, we remove examples where the judge shows inconsistent
ranking across the two permutations.

We generate the judge prompts using the `generate_judge_prompts.py` script as follows:
```
/path/to/ProCyon/examples/paper_analyses/llm_judge_phenotype_eval/generate_judge_prompts.py \
  --procyon_phenotypes_path  /path/to/procyon_eval/caption/ProCyon/{dataset_name}/full_captions.tsv.gz  \
  --llm_phenotypes_path /path/to/llm_phenotypes/{dataset_name}.captions.csv \
  --output_path judge_prompts.csv
```
where the external LLM phenotype descriptions (`{dataset_name}.captions.csv` above) should be in a CSV with a `protein_id` column containing the
UniProt ID of the protein and a `response` column containing the external LLM's output. Note that the prompt template is also contained within
`generate_judge_prompts.py` and can be modified if desired.

The output file (`judge_prompts.csv` in the example above) is a CSV with four columns:
- `Protein ID` - UniProt protein ID
- `Prompt` - the prompt for the judge
- `prompt_a` - name of the model in the "A" (i.e. first) description position
- `prompt_b` - name of the model in the "B" (i.e. second) description position

Given the judge prompts, one can then use these to query the judge model of choice. In our analysis, we used Claude-3.5-Sonnet. This model was
selected as a frontier LLM that could be used to evaluate GPT-4o phenotypes without worrying about any self-bias that may be introduced by using
GPT-4o as a judge of its own outputs.

## Parsing LLM-as-a-Judge results
We provide example judge prompts and responses within the ProCyon-Instruct dataset [here](https://huggingface.co/datasets/mims-harvard/ProCyon-Instruct/tree/main/experimental_data/llm_judge_eval/judge_responses), which allow for recreating the results shown in Figure 4D. Our particular outputs are CSV
files containing the judge response as well as a `finish_reason` column that describes whether the response was truncated
due to max length constraints when querying the judge LLM.

For an example of how to parse LLM judge results, please refer to the notebook `parse_llm_judge_results.ipynb`, which contains the code
used to parse responses into final decisions across both permutations, including tiebreaking as described in our manuscript.