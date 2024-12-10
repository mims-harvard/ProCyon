import os
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
)

import torch
import tqdm
import pandas as pd

import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)

from procyon.model.model_unified import (
    DEFAULT_PRETRAINED_WEIGHTS_DIR,
    UnifiedProCyon,
)

from procyon.data.data_utils import DATA_DIR

from procyon.model.biotranslator_tencoder import HFTextEncoder
# from procyon.training.args_bio_translator import ModelArgs

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.retrieval import (
    AbstractRetrievalModel,
    get_retrieval_target_proteins_loader,
    get_retrieval_target_set,
)
from procyon.evaluate.framework.utils import (
    compare_and_warn_model_args,
    move_inputs_to_device,
)

from Bio import SeqIO

import torch.nn as nn
import collections
import numpy as np
import torchvision.transforms as transforms
from esm.data import Alphabet

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Optional
from dataclasses import dataclass, field

from peft import get_peft_model, LoraConfig, PeftModel

from procyon.evaluate.framework.retrieval import (
    AbstractRetrievalModel,
    get_retrieval_target_proteins_loader,
    get_retrieval_target_set,
)
from procyon.evaluate.framework.qa import AbstractQAModel
from procyon.evaluate.general_eval import prepare_inputs

from procyon.training.training_args_IT import ModelArgs
from procyon.model.protllm import ProtLlmForBinaryCls, Trainer4ProtLlm

from functools import partial
import math

@dataclass
class MyModelArguments:
    model_name_or_path: Optional[str] = field(default="")
    esm_model_name: Optional[str] = field(default="ESM-2-650M")
    esm_model_file_path: Optional[str] = field(default="")
    protein_model_checkpoint: Optional[str] = field(default="")
    esm_tok_arch_name: Optional[str] = field(default="ESM-1b")
    protein_model_name: Optional[str] = field(default="protst")
    prot_output_size: Optional[int] = field(default=512)
    # sft arguments
    learn_protst: Optional[bool] = field(default=False)
    sft_with_lora: Optional[bool] = field(default=False)
    llm_name_or_path: Optional[str] = field(default="")
    # pretrain lora arguments
    lora_r: Optional[int] = field(default=32)
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_bias: Optional[str] = field(default="none")
    sft_lora_r: Optional[int] = field(default=32)
    sft_lora_alpha: Optional[int] = field(default=64)
    sft_target_modules: Optional[str] = field(default="down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj")
    pretrain_target_modules: Optional[str] = field(default="down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj")

@dataclass
class MyTrainingArguments(transformers.TrainingArguments):
    main_training_task: Optional[str] = field(default="pretrain")
    exclude_test_prot: Optional[bool] = field(default=False)
    lr_ratio: Optional[float] = field(default=1.0)
    device: str = field(default='cuda')


@dataclass
class MyEvaluationArguments:
    task: Optional[str] = field(default="ppi", metadata={"help": "ppi, mf, cc, bp, ec"})
    data_path: Optional[str] = field(default="")
    eval_seed: Optional[int] = field(default=42)
    resume_from_sft_checkpoint: Optional[str] = field(default="")
    n_demo: Optional[int] = field(default=0)
    n_labels: Optional[int] = field(default=1)
    batch_size: Optional[int] = field(default=4)
    save_every_n_samples: Optional[int] = field(default=200000)
    eval_split: Optional[str] = field(default="test")

def f1_max(pred, target):
    """
    F1 score with the optimal threshold.
    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.
    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / (is_start.cumsum(0) + 1e-10) # TODO: to prevent division by zero
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def area_under_prc(pred, target):
    """
    Area under precision-recall curve (PRC).
    Parameters:
        pred (Tensor): predictions of shape :math:`(n,)`
        target (Tensor): binary targets of shape :math:`(n,)`
    """
    order = pred.argsort(descending=True)
    target = target[order]
    precision = target.cumsum(0) / torch.arange(1, len(target) + 1, device=target.device)
    auprc = precision[target == 1].sum() / ((target == 1).sum() + 1e-10)
    return auprc

def create_internal_protid_map_to_protllm_map(protein_cache):
    # Map the id's of protllm to our id's to make use of their internal protein cache
    our_ids = pd.read_pickle("/n/holystore01/LABS/mzitnik_lab/Lab/PLM/integrated_data/v1/protein/protein_info_filtered.pkl")

    gene_name_to_id = {("uniprot-{}".format(row["entry"])):row["index"] for i, row in our_ids.iterrows()}

    # Get their name to id:
    protllm_map = protein_cache["nid2index"]

    internal_to_protllm = {}
    number_not_map = 0
    #df_dict = {"internal": [], "protllm": [], "entry": []}
    for k in gene_name_to_id.keys():
        internal_id = gene_name_to_id[k]
        try:
            protllm_id = protllm_map[k]
        except:
            print(f"WARNING, COULDN'T MAP {k}")
            number_not_map += 1
            continue
        internal_to_protllm[internal_id] = protllm_id

        # df_dict["internal"].append(internal_id)
        # df_dict["protllm"].append(protllm_id)
        # df_dict["entry"].append(k)

    #pd.DataFrame(df_dict).to_csv("protllm_matches.csv", index=False)

    #import ipdb; ipdb.set_trace()

    return internal_to_protllm

@dataclass
class ProtLlmExample:
    input_ids: List[int] = field(default_factory=lambda :[])
    labels: List[int] = field(default_factory=lambda :[])
    prot_masks: List[bool] = field(default_factory=lambda :[])
    prot_input_ids_batch: List[List[int]] = field(default_factory=lambda :[])
    prot_residue_mask_batch: List[List[bool]] = field(default_factory=lambda :[])

    def extend(self, item):
        self.input_ids.extend(item.input_ids)
        self.labels.extend(item.labels)
        self.prot_masks.extend(item.prot_masks)
        self.prot_input_ids_batch.extend(item.prot_input_ids_batch)
        self.prot_residue_mask_batch.extend(item.prot_residue_mask_batch)

    def prepend_bos(self, bos_token_id, update_labels=True):
        self.input_ids = [bos_token_id] + self.input_ids
        self.prot_masks = [False] + self.prot_masks
        if update_labels:
            self.labels = [-100] + self.labels

def ProtLLM_general_collate_fn(inputs, pad_to_multiple_of=1, pad_token_id=None, return_pt=True, model_max_length=10000000):
    _numpify_inputs = []
    for a in inputs:
        if isinstance(a, list):
            a = np.array(a)
        elif isinstance(a, np.ndarray):
            pass
        elif isinstance(a, str):
            return inputs
        else:
            raise ValueError
        _numpify_inputs.append(a)
    inputs = _numpify_inputs
    max_len = max(a.shape[-1] for a in inputs)
    if max_len % pad_to_multiple_of != 0:
        max_len += 8 - (max_len % pad_to_multiple_of)
    ret = np.empty(shape=(len(inputs), max_len), dtype=inputs[0].dtype)
    ret.fill(pad_token_id)
    for i, a, in enumerate(inputs):
        ret[i, :a.shape[-1]] = a

    if ret.shape[-1] > model_max_length:
        ret = ret[:, :model_max_length]
        print(f"[W] batch length exceed model max length, shape: f{ret.shape}", flush=True)

    if return_pt: ret = torch.from_numpy(ret)
    return ret



class ProtLLMCollatorFunction:
    '''
    Adapted from:
    '''
    def __init__(
            self,
            tok=None,
            prot_tok=None,
            max_len=1024,
            prot_max_len=512,
            split="train",
            prepend_bos=True,
            ec_prompt = False,
        ) -> None:

        self.tok = tok
        self.prot_tok = prot_tok
        self.max_len = max_len
        self.prot_max_len = prot_max_len
        self.prepend_bos = prepend_bos
        self.ec_prompt = ec_prompt

        self._load_ds_prompts()

    def __len__(self):
        return len(self.ds)

    def _load_ds_prompts(self):
        prompt_templates = {
        "prot_bos": ["<PROT>"],
        "prot_eos": ["</PROT>"],
        }
        self.prompt_input_ids = {}
        for key, texts in prompt_templates.items():
            id_lists = []
            for text in texts:
                ids = self.tok(text, add_special_tokens=False)["input_ids"]
                id_lists.append(ids)
            self.prompt_input_ids[key] = id_lists
        self.prot_token_len = len(self.prompt_input_ids["prot_bos"][0]) + 1 + len(self.prompt_input_ids["prot_eos"][0])

        self.label2tok_ids = [
            self.tok("No", add_special_tokens=False)["input_ids"],
            self.tok("Yes", add_special_tokens=False)["input_ids"],
        ]
        assert all(len(tok_ids) == 1 for tok_ids in self.label2tok_ids)

    def __call__(self, example) -> ProtLlmExample:
        text = example["text"]
        protein = example["protein"]
        label = example["label"]

        # Using prompt template in appendix of ProtLLM
        # Add requisite text wrapping around the example - GO terminology, etc. (done in call)
        if self.ec_prompt:
            text = "Does the protein catalyze " + text + "?"
        else:
            text = "Does the protein belong to " + text + "?"

        assert label == 0 or label == 1
        input_ids = []
        labels = []
        prot_masks = []
        prot_input_ids_batch = []
        prot_residue_mask_batch = []

        def _append_text(ids):
            assert isinstance(ids, list)
            input_ids.extend(ids)
            prot_masks.extend([False] * len(ids))

        def _append_prot_without_bos_eos(_prot):
            input_ids.append(0)
            prot_masks.append(True)
            prot_input_ids = self.prot_tok.encode(_prot)
            prot_residue_mask = [True] * len(prot_input_ids)
            if self.prot_tok.prepend_bos:
                prot_input_ids = [self.prot_tok.cls_idx] + prot_input_ids
                prot_residue_mask = [False] + prot_residue_mask
            if self.prot_tok.append_eos:
                prot_input_ids = prot_input_ids + [self.prot_tok.eos_idx]
                prot_residue_mask = prot_residue_mask + [False]
            prot_input_ids_batch.append(prot_input_ids)
            prot_residue_mask_batch.append(prot_residue_mask)

        def _append_prot(_prot):
            _append_text(self.prompt_input_ids["prot_bos"][0])
            _append_prot_without_bos_eos(_prot)
            _append_text(self.prompt_input_ids["prot_eos"][0])

        _append_prot(protein)
        _append_text(self.tok(text, add_special_tokens=False)["input_ids"])
        labels = [label]

        ret = ProtLlmExample(input_ids, labels, prot_masks, prot_input_ids_batch, prot_residue_mask_batch)
        if self.prepend_bos:
            ret.prepend_bos(self.tok.bos_token_id, update_labels=False)
        assert len(ret.labels) == 1
        return ret

    @staticmethod
    def collator(features, lm_pad_token_id=None, prot_pad_token_id=None, **unused):
        all_input_ids = []
        all_labels = []
        all_attn_masks = []
        all_prot_masks = []
        all_prot_input_ids = []
        all_prot_attn_masks = []
        all_residue_masks = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_attn_masks.append([True] * len(feature.input_ids))
            all_labels.append(feature.labels)
            all_prot_masks.append(feature.prot_masks)
            all_prot_input_ids.extend(feature.prot_input_ids_batch)
            all_prot_attn_masks.extend([[True] * len(prot_input_ids) for prot_input_ids in feature.prot_input_ids_batch])
            all_residue_masks.extend(feature.prot_residue_mask_batch)

        _collate_fn = partial(ProtLLM_general_collate_fn, pad_to_multiple_of=1, return_pt=True)
        batch = {
            "input_ids": _collate_fn(all_input_ids, pad_token_id=lm_pad_token_id),
            "labels": _collate_fn(all_labels, pad_token_id=-100),
            "attention_mask": _collate_fn(all_attn_masks, pad_token_id=False),
            "prot_emb_mask": _collate_fn(all_prot_masks, pad_token_id=False),
            "prot_input_ids": _collate_fn(all_prot_input_ids, pad_token_id=prot_pad_token_id),
            "prot_attention_mask": _collate_fn(all_prot_attn_masks, pad_token_id=False),
            "residue_mask": _collate_fn(all_residue_masks, pad_token_id=False).float(),
            "return_loss": True,
        }
        return batch


PROTEIN_RETRIEVAL_TEMPLATE = "<PROT>"



class ProtLLMQA(nn.Module):
    def __init__(self,
                device,
                eval_dataset = None,
                ec_prompt = False,
                model_name_or_path = "",
                esm_model_name = "ESM-2-650M",
                esm_model_file_path = "",
                protein_model_checkpoint = "",
                esm_tok_arch_name = "ESM-1b",
                protein_model_name = "protst",
                prot_output_size = 512,
                learn_protst = False,
                sft_with_lora = False,
                llm_name_or_path = "",
                lora_r = 32,
                lora_alpha = 64,
                lora_dropout = 0.1,
                lora_bias = "none",
                sft_lora_r = 32,
                sft_lora_alpha = 64,
                sft_target_modules = "down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj",
                pretrain_target_modules = "down_proj,up_proj,q_proj,v_proj,k_proj,o_proj,gate_proj",
                bf16 = True,
                logging_steps = 1000,
                report_to = 'none',
                resume_from_checkpoint = "",
                resume_from_sft_checkpoint = "",
                save_every_n_samples = 20000,
                ):
        super().__init__()
        self.model_config = MyModelArguments(
           model_name_or_path,
           esm_model_name,
           esm_model_file_path,
           protein_model_checkpoint,
           esm_tok_arch_name,
           protein_model_name,
           prot_output_size,
           learn_protst,
           sft_with_lora,
           llm_name_or_path,
           lora_r,
           lora_alpha,
           lora_dropout,
           lora_bias,
           sft_lora_r,
           sft_lora_alpha,
           sft_target_modules,
           pretrain_target_modules
        )

        self.ec_prompt = ec_prompt

        device = torch.device(device)
        self.training_args = MyTrainingArguments(
           bf16=bf16,
           logging_steps=logging_steps,
           report_to=report_to,
           resume_from_checkpoint=resume_from_checkpoint,
           device = device,
           output_dir='output_dir'
        )
        if not hasattr(self.training_args, 'distributed_state'):
           self.training_args.distributed_state = None

        self.eval_args = MyEvaluationArguments(
           resume_from_sft_checkpoint = resume_from_sft_checkpoint,
           save_every_n_samples = save_every_n_samples
        )

        self.model = ProtLlmForBinaryCls(self.model_config, device=device)
        self.model.init_prot_model(self.model_config, device=device)

        #protein_cache = torch.load(os.path.join(resume_from_checkpoint, "../pretrain", "protein_cache.pt"))
        #self.internal_to_protllm_idmap = create_internal_protid_map_to_protllm_map(protein_cache)

        peft_config = LoraConfig(
            inference_mode=False, r=self.model_config.lora_r,
            lora_alpha=self.model_config.lora_alpha,
            lora_dropout=self.model_config.lora_dropout,
            bias=self.model_config.lora_bias,
            target_modules=self.model_config.pretrain_target_modules.split(","))
        self.model = get_peft_model(self.model, peft_config)

        #import ipdb; ipdb.set_trace()

        self.model = PeftModel.from_pretrained(
            self.model,
            resume_from_checkpoint,
            config = peft_config
        )

        # QUESTION: does the above set the PEFT modules?

        # Replicate _load_from_checkpoint from trainer:
        prot2llm_linear_file = os.path.join(self.training_args.resume_from_checkpoint, "prot2llm_linear.bin")
        state_dict = torch.load(prot2llm_linear_file, map_location=torch.device("cpu")).state_dict()
        load_result = self.model.prot2llm_linear.load_state_dict(state_dict, strict=True)

        #import ipdb; ipdb.set_trace()

        self.data_prep = ProtLLMCollatorFunction(
            tok = self.model.llm_tok,
            prot_tok = self.model.prot_tok, # TODO
            ec_prompt = self.ec_prompt,
        )

        # init cls head and post_init_prot_model
        label2tok_ids = self.data_prep.label2tok_ids
        self.model.post_init_prot_model(self.model_config, device=self.training_args.device, learn_protst=False)
        self.model.init_cls_head(label2tok_ids=label2tok_ids)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Set yes/no tokens:
        self.yes_token = 1 #self.data_prep.label2tok_ids[1]
        self.no_token = 0 #self.data_prep.label2tok_ids[0]

        # data_collator = partial(eval_dataset.collator, lm_pad_token_id=self.model.llm_tok.eos_token_id, prot_pad_token_id=self.model.prot_tok.padding_idx)
        # trainer = Trainer4ProtLlm(
        #     model=self.model,
        #     args=self.training_args,
        #     eval_dataset=eval_dataset,
        #     tokenizer=self.model.llm_tok,
        #     data_collator=data_collator,
        # )
        # trainer._load_from_checkpoint(self.training_args.resume_from_checkpoint)
        # self.model = trainer.model.merge_and_unload()

        # peft_config = LoraConfig(
        #     inference_mode=False, r=self.model_config.sft_lora_r,
        #     lora_alpha=self.model_config.sft_lora_alpha,
        #     lora_dropout=self.model_config.lora_dropout,
        #     bias=self.model_config.lora_bias,
        #     target_modules=self.model_config.sft_target_modules.split(","))
        # self.model = get_peft_model(self.model, peft_config)
        # self.model.base_model.post_init_prot_model(self.model_config, device=self.training_args.device, learn_protst=False)
        # self.model.base_model.init_cls_head(label2tok_ids=eval_dataset.label2tok_ids)

        # self.trainer = Trainer4ProtLlm(
        #     model=self.model,
        #     args=self.training_args,
        #     eval_dataset=eval_dataset,
        #     tokenizer=self.model.llm_tok,
        #     data_collator=data_collator,
        # )
        # self.trainer._load_from_checkpoint(self.eval_args.resume_from_sft_checkpoint)

        # self.trainer._move_model_to_device(self.trainer.model, self.training_args.device)

        # self.trainer.model.eval()

class ProtLLMQAEval(AbstractQAModel):
    def __init__(self,
                model_config: Dict,
                eval_args: EvalArgs,
                model_args: ModelArgs,
                device: torch.device
            ):
        # super().__init__(model_config, eval_args, device)
        # print(model_config)
        self.model = ProtLLMQA(**model_config)

        self.yes_token = self.model.yes_token
        self.no_token = self.model.no_token

        self.num_samples = eval_args.qa_num_samples
        self.rng = np.random.default_rng(seed=eval_args.seed)

        self.PROTEIN_SEQS = [str(seq.seq) for seq in SeqIO.parse(os.path.join(DATA_DIR, f"integrated_data/v1/protein/protein_sequences.fa"), "fasta")]
        self.DOMAIN_SEQS = [str(seq.seq) for seq in SeqIO.parse(os.path.join(DATA_DIR, f"integrated_data/v1/domain/domain_sequences.fa"), "fasta")]

    @torch.no_grad()
    def get_predictions(
        self,
        data_loader: DataLoader,
        aaseq_type: str = 'protein',
    ) -> Dict[str, torch.Tensor]:

        results_dict = {"pred": [], "y": []}

        samples_to_hit = None
        if self.num_samples is not None:
            if self.num_samples < len(data_loader): # Keep None if we don't need to downsample
                samples_to_hit = self.rng.choice(
                    np.arange(len(data_loader)),
                    size=self.num_samples,
                    replace=False,
                )
                samples_to_hit = set(samples_to_hit) # Faster lookup

        for i, model_inputs in enumerate(tqdm(data_loader)):
            if samples_to_hit is not None:
                if not (i in samples_to_hit):
                    continue

            # 1. convert sequences to tokens for forward pass
            seq_idx = model_inputs["data"]["seq_idx"]
            if aaseq_type == 'protein':
                seqs = [self.PROTEIN_SEQS[i] for i in seq_idx]
            elif aaseq_type == 'domain':
                seqs = [self.DOMAIN_SEQS[i] for i in seq_idx]

            # 3. Expand to full examples
            input_seqs = [seqs[inds[-1]] for inds in model_inputs["input"]["seq"]]
            input_texts = [model_inputs["data"]["text"][inds[-1]] for inds in model_inputs["input"]["text"]]
            #labels_str = [inst.split()[-1] for inst in model_inputs["instructions"]]
            labels_str = model_inputs["target"]["text"]
            labels = torch.LongTensor([(self.model.yes_token if l.lower() == 'yes' else self.model.no_token) for l in labels_str])

            assert len(input_seqs) == len(input_texts)

            num_examples = len(input_seqs)
            # Detect ec:
            ec_option = False

            example_list = []
            for i in range(num_examples):
                # 4. Wrap all inputs to the dictionary (batched):
                example = {"text": input_texts[i], "protein": input_seqs[i], "label": labels[i].item()}
                example_list.append(self.model.data_prep(example))
                #   protein = raw protein sequence (gets passed to tokenizer)
                #   text = raw text (string) also gets passed to tokenizer - needs wrapped query text though

            # 5. Pass list to collator function
            batch = self.model.data_prep.collator(
                example_list,
                lm_pad_token_id = self.model.model.llm_tok.eos_token_id,
                prot_pad_token_id = self.model.model.prot_tok.padding_idx
            )

            del batch["labels"]
            batch.update({"return_dict": True})

            # Move all elements of dictionary to device
            batch = prepare_inputs(batch)

            # Model fwd pass:
            pred_logits = self.model.model(**batch).logits.detach().cpu()
            # Extract predictions (yes/no)
            pred = pred_logits.argmax(dim=1)

            results_dict["pred"].append(pred)
            results_dict["y"].append(labels)

            #import ipdb; ipdb.set_trace()

        results_dict["pred"] = torch.cat(results_dict["pred"])
        results_dict["y"] = torch.cat(results_dict["y"])

        return results_dict

def run_qa_eval(
    model: ProtLLMQAEval,
    data_loader: DataLoader,
    eval_args: EvalArgs,
    dataset_eval_args: Dict,
    model_name: str,
    dataset_key: str,
    output_dir: str,
) -> Dict:
    n_correct = 0
    n_example = 0
    n_answer0 = 0
    n_chunk = 0
    all_preds = []
    all_labels = []
    eval_args = model.model.eval_args

    for step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        labels = batch["labels"]
        del batch["labels"]
        batch = model.model.trainer._prepare_inputs(batch)
        batch.update({"return_dict": True})
        with torch.no_grad():
            logits = model.model.trainer.model(**batch).logits.detach().cpu()
        pred = logits.argmax(dim=1)

        n_correct += (pred.reshape(-1).cpu() == labels.reshape(-1)).sum().item()
        n_example += pred.shape[0]
        n_answer0 += (pred.reshape(-1) == 0).sum().item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_preds.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

        if n_example >= model.model.eval_args.save_every_n_samples * (n_chunk + 1):
            all_preds = torch.cat(all_preds, dim=0).view(-1)
            all_labels = torch.cat(all_labels, dim=0).view(-1)
            torch.save(all_preds, os.path.join(model.model.training_args.output_dir, f"all_preds_{n_chunk}.pt"))
            torch.save(all_labels, os.path.join(model.model.training_args.output_dir, f"all_labels_{n_chunk}.pt"))
            st_idx = math.ceil(n_prev / model.model.eval_args.n_labels) * model.model.eval_args.n_labels - n_prev
            all_labels = all_labels[st_idx:]
            all_labels = all_labels[:all_labels.shape[0] // model.model.eval_args.n_labels * model.model.eval_args.n_labels]
            all_preds = all_preds[st_idx:]
            all_preds = all_preds[:all_preds.shape[0] // model.model.eval_args.n_labels * model.model.eval_args.n_labels]
            auprc = area_under_prc(all_preds, all_labels)
            f1_score = f1_max(all_preds.reshape(-1, model.model.eval_args.n_labels), all_labels.reshape(-1, model.model.eval_args.n_labels))
            print(f'Chunk {n_chunk}: f1: {round(f1_score.item(), 4)}, auprc: {round(auprc.item(), 4)}', flush=True)
            all_preds = []
            all_labels = []
            n_chunk += 1
            n_prev = n_example

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0).view(-1)
        all_labels = torch.cat(all_labels, dim=0).view(-1)
        torch.save(all_preds, os.path.join(model.model.training_args.output_dir, f"all_preds_{n_chunk}.pt"))
        torch.save(all_labels, os.path.join(model.model.training_args.output_dir, f"all_labels_{n_chunk}.pt"))
        n_chunk += 1
        all_preds = []
        all_labels = []

    for i in range(n_chunk):
      all_preds.append(torch.load(os.path.join(model.model.training_args.output_dir, f"all_preds_{i}.pt")))
      all_labels.append(torch.load(os.path.join(model.model.training_args.output_dir, f"all_labels_{i}.pt")))

    all_labels = torch.cat(all_labels, dim=0).view(-1)
    all_preds = torch.cat(all_preds, dim=0).view(-1)
    all_labels = all_labels[:all_labels.shape[0] // eval_args.n_labels * eval_args.n_labels]
    all_preds = all_preds[:all_preds.shape[0] // eval_args.n_labels * eval_args.n_labels]
    auprc = area_under_prc(all_preds, all_labels)
    f1_score = f1_max(all_preds.reshape(-1, eval_args.n_labels), all_labels.reshape(-1, eval_args.n_labels))
    ret_dict = {'AUPRC': auprc, 'F1': f1_score, 'acc': n_correct/n_example}
    print(ret_dict)
    return ret_dict