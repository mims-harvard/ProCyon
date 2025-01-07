SPLIT_MAPS = {
    "protein_go": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_five_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": None,
    },
    "domain_go": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_five_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": None,
    },
    "domain_pfam": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": None,
    },
    "protein_disgenet": {
        "pt_ft": None,
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_reactome": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": None,
    },
    "protein_protein": None,  # Shouldn't get called anyways
    "protein_omim": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_drugbank": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_drugbank:moa": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_drugbank:indication": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_gtop": None,
    "protein_ec": {
        "pt_ft": "eval_pt_ft",
        "few_shot": "eval_two_shot",
        "zero_shot": "eval_zero_shot",
        "zero_shot_easy": "eval_zero_shot_easy",
    },
    "protein_uniprot": None,
}
