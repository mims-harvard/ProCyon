import os
import math
from typing import Dict, Optional, Tuple

import argparse
from huggingface_hub import login as hf_login
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from scipy import stats
from tqdm import trange

from procyon.data.data_utils import DATA_DIR
from procyon.data.inference_utils import (
    create_input_retrieval,
    get_proteins_from_embedding,
)
from procyon.evaluate.framework.utils import move_inputs_to_device
from procyon.model.model_unified import UnifiedProCyon

CKPT_NAME = os.path.expanduser(os.getenv("CHECKPOINT_PATH"))


def load_model_onto_device() -> Tuple[UnifiedProCyon, torch.device, Dict]:
    # Load the pre-trained ProCyon model
    logger.info("Loading pretrained model")
    # Replace with the path where you downloaded a pre-trained ProCyon model (e.g. ProCyon-Full)
    data_args = torch.load(os.path.join(CKPT_NAME, "data_args.pt"))

    model, _ = UnifiedProCyon.from_pretrained(checkpoint_dir=CKPT_NAME)
    logger.info("Done loading pretrained model")

    logger.info("Applying pretrained model to device")
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")
    # identify available devices on the machine
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Total memory allocated by PyTorch: {torch.cuda.memory_allocated()}")

    model.bfloat16()  # Quantize the model to a smaller precision
    _ = model.eval()

    logger.info("Done applying pretrain model to device")

    return model, device, data_args


def run_retrieval(args):
    """
    This function demonstrates how to use the pre-trained ProCyon model ProCyon-Full to perform protein retrieval
    for a given drug and disease.
    """

    logger.info("Logging into huggingface hub")
    hf_login(token=os.getenv("HF_TOKEN"))
    logger.info("Done logging into huggingface hub")

    if args.inference_bool:
        # load the pre-trained ProCyon model
        model, device, data_args = load_model_onto_device()

    # Load the pre-calculated protein target embeddings
    logger.info("Load protein target embeddings")
    all_protein_embeddings, all_protein_ids = torch.load(
        os.path.join(CKPT_NAME, "protein_target_embeddings.pkl")
    )
    all_protein_embeddings = all_protein_embeddings.float()
    logger.info(
        f"shape of precalculated embeddings matrix: {all_protein_embeddings.shape}"
    )

    logger.info("Loading DrugBank info")
    # Load DrugBank info, namely the mapping from DrugBank IDs to mechanism
    # of action descriptions and ProCyon-Instruct numeric IDs.
    drugbank_info = pd.read_pickle(
        os.path.join(
            DATA_DIR,
            "integrated_data",
            "v1",
            "drugbank",
            "drugbank_info_filtered_composed.pkl",
        )
    )
    db_map = {row["drugbank_id"]: row["moa"] for _, row in drugbank_info.iterrows()}
    db_idx_map = {
        row["drugbank_id"]: row["index"] for _, row in drugbank_info.iterrows()
    }
    logger.info("Done loading DrugBank info")

    logger.info("entering task description and prompt")
    db_idx_map = {
        row["drugbank_id"]: row["index"] for _, row in drugbank_info.iterrows()
    }

    # read the task description from a file
    with open(args.task_desc_infile, "r") as f:
        task_desc = f.read()
    task_desc = task_desc.replace("\n", " ")

    # Next we set up the specific prompt contexts provided for retrieval using
    # bupropion and either depression or smoking cessation.

    # DrugBank ID for bupropion
    db_id = "DB01156"
    drug_desc = db_map[db_id]
    depression_desc = """The pathophysiology of depression is not completely understood, but current theories center around monoaminergic systems, the circadian rhythm, immunological dysfunction, HPA-axis dysfunction and structural or functional abnormalities of emotional circuits.
    
    Derived from the effectiveness of monoaminergic drugs in treating depression, the monoamine theory posits that insufficient activity of monoamine neurotransmitters is the primary cause of depression. Evidence for the monoamine theory comes from multiple areas. First, acute depletion of tryptophan—a necessary precursor of serotonin and a monoamine—can cause depression in those in remission or relatives of people who are depressed, suggesting that decreased serotonergic neurotransmission is important in depression.[63] Second, the correlation between depression risk and polymorphisms in the 5-HTTLPR gene, which codes for serotonin receptors, suggests a link. Third, decreased size of the locus coeruleus, decreased activity of tyrosine hydroxylase, increased density of alpha-2 adrenergic receptor, and evidence from rat models suggest decreased adrenergic neurotransmission in depression.[64] Furthermore, decreased levels of homovanillic acid, altered response to dextroamphetamine, responses of depressive symptoms to dopamine receptor agonists, decreased dopamine receptor D1 binding in the striatum,[65] and polymorphism of dopamine receptor genes implicate dopamine, another monoamine, in depression.[66][67] Lastly, increased activity of monoamine oxidase, which degrades monoamines, has been associated with depression.[68] However, the monoamine theory is inconsistent with observations that serotonin depletion does not cause depression in healthy persons, that antidepressants instantly increase levels of monoamines but take weeks to work, and the existence of atypical antidepressants which can be effective despite not targeting this pathway.[69]
    
    One proposed explanation for the therapeutic lag, and further support for the deficiency of monoamines, is a desensitization of self-inhibition in raphe nuclei by the increased serotonin mediated by antidepressants.[70] However, disinhibition of the dorsal raphe has been proposed to occur as a result of decreased serotonergic activity in tryptophan depletion, resulting in a depressed state mediated by increased serotonin. Further countering the monoamine hypothesis is the fact that rats with lesions of the dorsal raphe are not more depressive than controls, the finding of increased jugular 5-HIAA in people who are depressed that normalized with selective serotonin reuptake inhibitor (SSRI) treatment, and the preference for carbohydrates in people who are depressed.[71] Already limited, the monoamine hypothesis has been further oversimplified when presented to the general public.[72] A 2022 review found no consistent evidence supporting the serotonin hypothesis, linking serotonin levels and depression.[73]
    
    HPA-axis abnormalities have been suggested in depression given the association of CRHR1 with depression and the increased frequency of dexamethasone test non-suppression in people who are depressed. However, this abnormality is not adequate as a diagnosis tool, because its sensitivity is only 44%.[74] These stress-related abnormalities are thought to be the cause of hippocampal volume reductions seen in people who are depressed.[75] Furthermore, a meta-analysis yielded decreased dexamethasone suppression, and increased response to psychological stressors.[76] Further abnormal results have been obscured with the cortisol awakening response, with increased response being associated with depression.[77]
    
    There is also a connection between the gut microbiome and the central nervous system, otherwise known as the Gut-Brain axis, which is a two-way communication system between the brain and the gut. Experiments have shown that microbiota in the gut can play an important role in depression as people with MDD often have gut-brain dysfunction. One analysis showed that those with MDD have different bacteria living in their guts. Bacteria Bacteroidetes and Firmicutes were most affected in people with MDD, and they are also impacted in people with Irritable Bowel Syndrome.[78] Another study showed that people with IBS have a higher chance of developing depression, which shows the two are connected.[79] There is even evidence suggesting that altering the microbes in the gut can have regulatory effects on developing depression. [78]
    
    Theories unifying neuroimaging findings have been proposed. The first model proposed is the limbic-cortical model, which involves hyperactivity of the ventral paralimbic regions and hypoactivity of frontal regulatory regions in emotional processing.[80] Another model, the cortico-striatal model, suggests that abnormalities of the prefrontal cortex in regulating striatal and subcortical structures result in depression.[81] Another model proposes hyperactivity of salience structures in identifying negative stimuli, and hypoactivity of cortical regulatory structures resulting in a negative emotional bias and depression, consistent with emotional bias studies.[82]
    
    Immune Pathogenesis Theories on Depression
    The newer field of psychoneuroimmunology, the study between the immune system and the nervous system and emotional state, suggests that cytokines may impact depression.
    
    Immune system abnormalities have been observed, including increased levels of cytokines -cells produced by immune cells that affect inflammation- involved in generating sickness behavior, creating a pro-inflammatory profile in MDD.[83][84][85] Some people with depression have increased levels of pro-inflammatory cytokines and some have decreased levels of anti-inflammatory cytokines.[86] Research suggests that treatments can reduce pro-inflammatory cell production, like the experimental treatment of ketamine with treatment-resistant depression.[87] With this, in MDD, people will more likely have a Th-1 dominant immune profile, which is a pro-inflammatory profile. This suggests that there are components of the immune system affecting the pathology of MDD. [88]
    
    Another way cytokines can affect depression is in the kynurenine pathway, and when this is overactivated, it can cause depression. This can be due to too much microglial activation and too little astrocytic activity. When microglia get activated, they release pro-inflammatory cytokines that cause an increase in the production of COX2. This, in turn, causes the production of PGE2, which is a prostaglandin, and this catalyzes the production of indolamine, IDO. IDO causes tryptophan to get converted into kynurenine and kynurenine becomes quinolinic acid.[89] Quinolinic acid is an agonist for NMDA receptors, so it activates the pathway. Studies have shown that the post-mortem brains of patients with MDD have higher levels of quinolinic acid than people who did not have MDD. With this, researchers have also seen that the concentration of quinolinic acid correlates to the severity of depressive symptoms."""
    depression_prompt = "Disease: {} Drug: {}".format(depression_desc, drug_desc)

    smoking_desc = """Smoking cessation: Nicotine is an amine found in tobacco and tobacco products. It is the addictive agent which confers a much lower risk than other elements of tobacco, but it is not completely benign. When tobacco smoke is inhaled, nicotine rapidly enters the bloodstream through the pulmonary circulation. Inhaled nicotine escapes the first pass intestinal and liver metabolism. Nicotine readily crosses the blood-brain barrier which then promptly diffuses into the brain tissue. The process is said to take only 2 to 8 seconds from the time of inhalation. Nicotine is a selective binder to nicotinic cholinergic receptors (nAChRs) in the brain and other tissues. The half-life of nicotine in the human body is estimated to be around 2 hours from the time of consumption.
    
    Brain imaging studies have demonstrated that nicotine acutely increases activity in the prefrontal cortex, thalamus, and visual system consistent with activation of corticobasal ganglia and thalamic brain circuits. Nicotine which stimulates nAChRs produces the release of neurotransmitters, predominantly dopamine but also norepinephrine, acetylcholine, serotonin, GABA, glutamate, and endorphins. These neurotransmitters cause the various responses and behaviors after nicotine intake. When there is repeated exposure to nicotine, tolerance develops to some of the physiological effects of nicotine. Nicotine is a sympathomimetic drug that causes the release of catecholamines and increases heart rate, cardiac contractility, constricts cutaneous and coronary blood vessels and increases blood pressure.
    
    Nicotine undergoes metabolism in the liver, primarily by the liver enzyme CYP2A6, and converts nicotine to cotinine. Cotinine is a metabolite that can be used as a marker for exposure to nicotine. There are broad individual and racial variations in the rate of nicotine metabolism due to genetic polymorphism in CYP2A6. Thus the metabolism of nicotine is faster in Caucasians than Asians and Africans. Sex hormones also significantly affect CYP2A6 activity, and females metabolize nicotine faster than males.[10][11]"""
    smoking_prompt = "Disease: {} Drug: {}".format(smoking_desc, drug_desc)

    logger.info("Done entering task description and prompt")

    logger.info("Now performing protein retrieval for example 1")

    input_simple = create_input_retrieval(
        input_description=depression_prompt,
        data_args=data_args,
        drug_input_idx=db_idx_map[db_id],
        task_definition=task_desc,
        instruction_source_dataset="drugbank",
        instruction_source_relation="drug_target",  # "all" - disgenet, omim, uniprot, reactome
        aaseq_type="protein",
        icl_example_number=1,  # 0, 1, 2
    )

    input_simple = move_inputs_to_device(input_simple, device=device)
    with torch.no_grad():
        model_out = model(
            inputs=input_simple,
            retrieval=True,
            aaseq_type="protein",
        )
    df_dep = get_proteins_from_embedding(all_protein_embeddings, model_out, top_k=None)
    logger.info(f"top results: {df_dep.head(10).to_dict(orient='records')}")

    logger.info("Done performaing protein retrieval for example 1")

    logger.info("DONE WITH ALL WORK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_desc_infile",
        type=str,
        help="Description of the task.",
    )
    parser.add_argument(
        "--inference_bool",
        action="store_false",
        help="OPTIONAL; choose this if you do not intend to do inference or load the model",
    )
    args = parser.parse_args()

    run_retrieval(args)
