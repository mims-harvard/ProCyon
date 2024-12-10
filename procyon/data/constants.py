import numpy as np

# Below dictionary holds all description names that we'll use in each dataset
ENTITY_DESCRIPTION_NAMES = {
    "go": [
        "description_name_type_def",
        "description_name_type_def_abstract_1",
        "description_name_type_def_abstract_2",
        "description_name_type_def_abstract_3",
    ],
    "pfam": ["description_pfam", "description_interpro"],
    "disgenet": [
        "description_air",
        "description_aot",
        "description_chv",
        "description_csp",
        "description_fma",
        "description_go",
        "description_hl7v3.0",
        "description_hpo",
        "description_lnc",
        "description_mcm",
        "description_medlineplus",
        "description_msh",
        "description_nci",
        "description_pdq",
        "description_spn",
        "description_uwda",
        "description_primekg_mondo",
        "description_primekg_orphanet",
    ],
    "reactome": [
        "description_name_description",
    ],
    "protein": [None],  # Shouldn't get called anyways
    "omim": [
        "description_omim",
        "description_mondo",
        "description_umls",
        "description_orphanet",
        "description_mayo",
    ],
    "drugbank": [
        "indication",
        "moa",
        "description_name_description",
        "description_name_description_moa_pharmacodynamics",
        "description_name_description_moa_pharmacodynamics_indication_toxicity",
    ],
    "gtop": [
        "description_name_overview",
        "description_name_comments",
        "description_name_introduction",
    ],
    "ec": [],  # TODO
    "uniprot": ["function"],
}

ONTOLOGY_RAG_SUBSETS = {
    "reactome": "description_name_description",
    "go": "description_name_type_def",
}

ONTOLOGY_RAG_LEVEL_GROUPS = {
    "reactome": ([5, 7], ["High", "Mid", "Low"]),
    "go": ([5, 7], ["High", "Mid", "Low"]),
}

QA_SUBSETS = {
    1: {
        "go": [
            "description_name_type_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [  # FIX
            "description_air",
            "description_aot",
            "description_chv",
            "description_csp",
            "description_fma",
            "description_go",
            "description_hl7v3.0",
            "description_hpo",
            "description_lnc",
            "description_mcm",
            "description_medlineplus",
            "description_msh",
            "description_nci",
            "description_pdq",
            "description_spn",
            "description_uwda",
            "description_primekg_mondo",
            "description_primekg_orphanet",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
            "description_mondo",
            "description_umls",
            "description_orphanet",
            "description_mayo",
        ],
        "drugbank": [
            "indication",
            "moa",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": [
            "description_name_overview",
            "description_name_comments",
            "description_name_introduction",
        ],
        "ec": [
            "description_explorenz",
        ],  # TODO
        "uniprot": ["function"],
    },
    5: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "omim_def_curated",
            "omim_clinical_curated",
            "omim_molecular_curated",
            "omim_title_curated",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["target_family_overview", "target_family_comments"],
        "ec": [
            "description_explorenz",  # ONLY THING THAT CHANGED COMPARED TO ABOVE
        ],
        "uniprot": ["function"],
    },
    "ProtLLM": {
        "go": [
            "description_name_type_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
            "description_mondo",
            "description_umls",
            "description_orphanet",
            "description_mayo",
        ],
        "drugbank": [
            "indication",
            "moa",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": [
            "description_name_overview",
            "description_name_comments",
            "description_name_introduction",
        ],
        "ec": [
            "description_explorenz",
        ],  # TODO
        "uniprot": ["function"],
    },
    "ProtLLM_name": {
        "go": [
            "go_name",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
            "description_mondo",
            "description_umls",
            "description_orphanet",
            "description_mayo",
        ],
        "drugbank": [
            "indication",
            "moa",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": [
            "description_name_overview",
            "description_name_comments",
            "description_name_introduction",
        ],
        "ec": [
            "explorenz_accepted_name",
        ],  # TODO
        "uniprot": ["function"],
    },
}


RETRIEVAL_SUBSETS = {
    1: {
        "go": [
            "description_name_type_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [  # FIX
            "description_air",
            "description_aot",
            "description_chv",
            "description_csp",
            "description_fma",
            "description_go",
            "description_hl7v3.0",
            "description_hpo",
            "description_lnc",
            "description_mcm",
            "description_medlineplus",
            "description_msh",
            "description_nci",
            "description_pdq",
            "description_spn",
            "description_uwda",
            "description_primekg_mondo",
            "description_primekg_orphanet",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
            "description_mondo",
            "description_umls",
            "description_orphanet",
            "description_mayo",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": [
            "description_name_overview",
            "description_name_comments",
            "description_name_introduction",
        ],
        "ec": [
            "description_explorenz",
        ],
        "uniprot": ["function"],
    },
    2: {
        "go": [
            "description_name_type_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [  # FIX
            "description_all_collapse",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
            "description_mondo",
            "description_umls",
            "description_orphanet",
            "description_mayo",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": [
            "description_name_overview",
            "description_name_comments",
            "description_name_introduction",
        ],
        "ec": [
            "description_explorenz",
        ],
        "uniprot": ["function"],
    },
    5: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "omim_def_curated",
            "omim_clinical_curated",
            "omim_molecular_curated",
            "omim_title_curated",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["target_family_overview", "target_family_comments"],
        "ec": [
            "description_explorenz",  # ONLY THING THAT CHANGED COMPARED TO ABOVE
        ],
        "uniprot": ["function"],
    },
}

# NOTE: Caption SUBSETS must only contain one column per dataset - for consistency

CAPTION_SUBSETS = {
    1: {
        "go": [
            "description_name_type_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "allDescriptions",
        ],
        "reactome": [
            "description_name_description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["description_name_overview", "description_name_comments"],
        "ec": [],  # TODO
        "uniprot": ["function"],
    },
    2: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "allDescriptions",  # FIX
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["description_name_overview", "description_name_comments"],
        "ec": [
            "description_explorenz",
        ],
        "uniprot": ["function"],
    },
    3: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["description_name_overview", "description_name_comments"],
        "ec": [
            "description_explorenz",
        ],
        "uniprot": ["function"],
    },
    4: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "description_omim",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["description_name_overview", "description_name_comments"],
        "ec": [
            "description_explorenz",  # ONLY THING THAT CHANGED COMPARED TO ABOVE
        ],
        "uniprot": ["function"],
    },
    5: {
        "go": [
            "go_def",
        ],
        "pfam": ["description_pfam", "description_interpro"],
        "disgenet": [
            "description_all_collapse",
        ],
        "reactome": [
            "description",
        ],
        "protein": [None],  # Shouldn't get called anyways
        "omim": [
            "omim_def_curated",
            "omim_clinical_curated",
            "omim_molecular_curated",
            "omim_title_curated",
        ],
        "drugbank": [
            "moa",
            "indication",
        ],
        "drugbank:moa": [
            "moa",
        ],
        "drugbank:indication": [
            "indication",
        ],
        "gtop": ["target_family_overview", "target_family_comments"],
        "ec": [
            "description_explorenz",  # ONLY THING THAT CHANGED COMPARED TO ABOVE
        ],
        "uniprot": ["function"],
    },
}

# Split names:
SPLIT_NAMES = {"protein_go": ["eval_pt_ft", "eval_five_shot", "eval_zero_shot"]}

PERSONALITY_PROMPTS = {
    "junior": "You are pursuing a Bachelor's degree interested in pursuing a biomedical research career interested in starting a PhD, MD/PhD, or research-oriented MD program. You have basic medical science knowledge, but do not have sufficient background in any specialized biological field yet.",
    "mid": "You are pursuing a MD, MD/PhD, PhD, or ScD in biomedical sciences. You are learning deep, specialized medical science knowledge; you can use specialized expert words, but not too much.",
    "senior": "You are a full-time research scientist with a doctoral degree, at least five years of postdoctoral experience, and knowledge of cutting-edge biomedical technologies. You have mastered in depth, specialized medical science knowledge, try using advanced terminology.",
}

EXPERTISE_LEVEL = ["junior", "mid", "senior"]
REPHRASE_ENTITY_LEVEL = ["rephrasing", "summarisation"]
REPHRASE_TASK_DEF_LEVEL = ["rephrasing", "summarisation", "simplification"]

ENTITY_REPHRASING_FILES = {
    "go": {
        "go_def": "go_def_filtered_rephrased.pkl",
        "description_name_type_def": "go_def_filtered_rephrased.pkl",
    },
    "pfam": {
        "description_pfam": "description_pfam_filtered_rephrased.pkl",
        "description_interpro": "description_interpro_filtered_rephrased.pkl",
    },
    "disgenet": {
        "description_name": "disgenet_description_filtered_rephrased.pkl",
        "description_all_collapse": "disgenet_description_filtered_rephrased.pkl",
        "description_air": "disgenet_description_filtered_rephrased.pkl",
        "description_aot": "disgenet_description_filtered_rephrased.pkl",
        "description_chv": "disgenet_description_filtered_rephrased.pkl",
        "description_csp": "disgenet_description_filtered_rephrased.pkl",
        "description_fma": "disgenet_description_filtered_rephrased.pkl",
        "description_go": "disgenet_description_filtered_rephrased.pkl",
        "description_hl7v3.0": "disgenet_description_filtered_rephrased.pkl",
        "description_hpo": "disgenet_description_filtered_rephrased.pkl",
        "description_lnc": "disgenet_description_filtered_rephrased.pkl",
        "description_mcm": "disgenet_description_filtered_rephrased.pkl",
        "description_medlineplus": "disgenet_description_filtered_rephrased.pkl",
        "description_msh": "disgenet_description_filtered_rephrased.pkl",
        "description_nci": "disgenet_description_filtered_rephrased.pkl",
        "description_pdq": "disgenet_description_filtered_rephrased.pkl",
        "description_spn": "disgenet_description_filtered_rephrased.pkl",
        "description_uwda": "disgenet_description_filtered_rephrased.pkl",
        "description_primekg_mondo": "disgenet_description_filtered_rephrased.pkl",
        "description_primekg_orphanet": "disgenet_description_filtered_rephrased.pkl",
    },
    "reactome": {
        "description_name_description": "reactome_description_filtered_rephrased.pkl",
        "description": "reactome_description_filtered_rephrased.pkl",
    },
    "protein": None,  # Shouldn't get called anyways
    "omim": None,
    "drugbank": {
        "moa": "drugbank_moa_filtered_rephrased.pkl",
        "indication": "drugbank_indication_filtered_rephrased.pkl",
    },
    "drugbank:moa": {"moa": "drugbank_moa_filtered_rephrased.pkl"},
    "drugbank:indication": {"indication": "drugbank_indication_filtered_rephrased.pkl"},
    "gtop": {
        # "target_family_overview",
        # "target_family_comments"
        "description_name_overview": "gtop_target_family_overview_filtered_rephrased.pkl",
        "description_name_comments": "gtop_target_family_comments_filtered_rephrased.pkl",
        "description_name_introduction": "gtop_target_family_introduction_filtered_rephrased.pkl",
        "target_family_overview": "gtop_target_family_overview_filtered_rephrased.pkl",
        "target_family_comments": "gtop_target_family_comments_filtered_rephrased.pkl",
        "target_family_introduction": "gtop_target_family_introduction_filtered_rephrased.pkl",
    },
    "ec": {  # None,
        "description_explorenz": "ec_description_explorenz_filtered_rephrased.pkl",
        "description_name_explorenz": "ec_description_explorenz_filtered_rephrased.pkl",
    },
    "uniprot": {
        "function": "uniprot_function_filtered_rephrased.pkl",
    },
}

ENTITY_REPHRASING_COLUMN_NAMES = {
    "go": {
        "description_name_type_def": "description_name_type_def",
        "go_def": "go_def",
    },
    "pfam": {
        "description_pfam": "description_pfam",
        "description_interpro": "description_interpro",
    },
    "disgenet": {
        "description_name": "description_name",
        "description_all_collapse": "description",
        "description_air": "description_name",
        "description_aot": "description_name",
        "description_chv": "description_name",
        "description_csp": "description_name",
        "description_fma": "description_name",
        "description_go": "description_name",
        "description_hl7v3.0": "description_name",
        "description_hpo": "description_name",
        "description_lnc": "description_name",
        "description_mcm": "description_name",
        "description_medlineplus": "description_name",
        "description_msh": "description_name",
        "description_nci": "description_name",
        "description_pdq": "description_name",
        "description_spn": "description_name",
        "description_uwda": "description_name",
        "description_primekg_mondo": "description_name",
        "description_primekg_orphanet": "description_name",
    },
    "reactome": {
        "description_name_description": "description_name_description",
        "description": "reactome_description",
    },
    "protein": None,  # Shouldn't get called anyways
    "omim": None,
    "drugbank": {"moa": "moa", "indication": "indication"},
    "drugbank:moa": {"moa": "moa"},
    "drugbank:indication": {"indication": "indication"},
    "gtop": {
        "description_name_overview": "gtop_target_family_overview",
        "description_name_comments": "gtop_target_family_comments",
        "description_name_introduction": "gtop_target_family_introduction",
        "target_family_overview": "gtop_target_family_overview",
        "target_family_comments": "gtop_target_family_comments",
        "target_family_introduction": "gtop_target_family_introduction",
    },
    "ec": {  # None,
        "description_explorenz": "description_explorenz",
        "description_name_explorenz": "description_name_explorenz",
    },
    "uniprot": {
        "function": "function",
    },
}

DATASET_ID = {
    "go": 0,
    "pfam": 1,
    "disgenet": 2,
    "reactome": 3,
    "protein": 4,
    "omim": 5,
    "drugbank": 6,
    "drugbank:moa": 6,
    "drugbank:indication": 6,
    "gtop": 7,
    "ec": 8,
    "uniprot": 9,
    "peptide": 10,
}

CAPTION_TRAIN_WEIGHTS = {
    0: {
        "protein_go": 0.5,
        "domain_go": 0.5,
        "domain_pfam": 2.0,
        "protein_disgenet": 2.0,
        "protein_reactome": 1.0,
        "protein_omim": 2.0,
        "protein_drugbank": 2.0,
        "protein_drugbank:moa": 2.0,
        "protein_drugbank:indication": 2.0,
        "protein_gtop": 2.0,
        "protein_ec": 2.0,
        "protein_uniprot": 2.0,
    }
}
