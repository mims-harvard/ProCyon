import os
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
)

import torch
import torch.nn.functional as F
import torchdrug.data as td_data

from torch.utils.data import DataLoader
from torchdrug.utils import cuda
from tqdm import tqdm

from procyon.data.dataset import (
    AASeqDataset,
    AASeqTextUnifiedDataset,
)
from procyon.data.data_utils import DATA_DIR

from procyon.training.training_args_IT import ModelArgs

from procyon.evaluate.framework.args import EvalArgs
from procyon.evaluate.framework.retrieval import (
    AbstractRetrievalModel,
    get_retrieval_target_proteins_loader,
    get_retrieval_target_set,
)

from procyon.evaluate.framework.baseline_models.protst import PretrainESM, PubMedBERT

class ProtSTRetrievalEval(AbstractRetrievalModel):
    def __init__(
        self,
        model_config: Dict,
        eval_args: EvalArgs,
        model_args: ModelArgs,
        device: torch.device
    ):

        #############################
        # LOADING PLM (protein model), and BLM (text model)
        # Make sure everything that needs to be downloaded separately exists first.
        model_weights_path = os.path.join(
            DATA_DIR,
            "model_weights",
        )
        pubmedbert_path = os.path.join(
            model_weights_path,
            "pubmedbert-abs",
            "BiomedNLP-BiomedBERT-base-uncased-abstract",
        )
        esm_path = os.path.join(
            model_weights_path,
            "esm-1b",
        )
        protst_path = os.path.join(
            model_weights_path,
            "ProtST",
            "protst_esm1b.pth",
        )
        if not os.path.exists(pubmedbert_path):
            raise Exception(
                  f"PubMedBERT-abs weights not found at {pubmedbert_path} , you can download them by running:\n"
                  f"  cd {os.path.dirname(pubmedbert_path)}\n"
                  "   git clone https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
            )
        if not os.path.exists(protst_path):
            raise Exception(
                  f"ProtST weights not found at {protst_path} , you can download them from "
                  "https://github.com/DeepGraphLearning/ProtST/blob/main/README.md#pre-trained-model-zoo"
            )

        # Max prompt length of 128 used by ProtST for text-to-protein retrieval, but we
        # may want to bump this up as it's quite small.
        max_len = model_config.get("max_prompt_len", 128)
        if max_len > 512:
            raise ValueError(f"ProtST: max_prompt_len={max_len} is greater than max context length of 512")

        # Initialize the two models (Pretrained ESM, and PubMedBERT)

        # Note that there's currently a confluence of incompatibilities. torchdrug==0.2.1 breaks importing
        # the ProtST checkpoints shared by the authors, but torchdrug==0.2.0 is incompatible with pytorch >= 2.0
        # Unfortunately the current best fix is to use torchdrug==0.2.0 and comment out torchdrug/patch.py:71-73
        #
        # The following error is a sign that you need to perform the steps above.
        #      68 if '_buffers' not in self.__dict__:
        #      69     raise AttributeError(
        #      70         "cannot assign buffer before Module.__init__() call")
        # ---> 71 elif not isinstance(name, torch._six.string_classes):
        #      72     raise TypeError("buffer name should be a string. "
        #      73                     "Got {}".format(torch.typename(name)))
        #      74 elif '.' in name:
        # AttributeError: module 'torch' has no attribute '_six'
        protein_model = PretrainESM(path=esm_path ,model='ESM-1b')
        text_model = PubMedBERT(
            model='PubMedBERT-abs',
            path=pubmedbert_path,
            output_dim=512,
            readout='mean',
        )

        # Load the weights from the pretrained ProtST
        model_state = torch.load(protst_path, map_location=torch.device("cpu"))["model"]

        protein_model_state, text_model_state = {}, {}

        for k in model_state.keys():
            if k.startswith("protein_model."):
                protein_model_state[k[14:]] = model_state[k]

            if k.startswith("text_model."):
                text_model_state[k[11:]] = model_state[k]

        protein_model.load_state_dict(protein_model_state, strict=False)
        text_model.load_state_dict(text_model_state, strict=False)

        for _, p in protein_model.named_parameters():
            p.requires_grad = False
        for _, p in text_model.named_parameters():
            p.requires_grad = False

        protein_model = protein_model.to(device)
        protein_model.eval()
        text_model = text_model.to(device)
        text_model.eval()

        self.protein_model = protein_model
        self.text_model = text_model
        self.device = device
        self.batch_size = eval_args.batch_size
        self.use_cached_target_embeddings = eval_args.retrieval_use_cached_target_embeddings
        self.checkpoint_dir = os.path.dirname(protst_path)
        self.max_len = max_len

    @torch.no_grad()
    def _get_text_embedding(self, prompts):
        # Code below for getting prompt embeddings is directly from ProtST for consistency.
        # If we wanted to, we could change some odd choices like no EOS token, processing each
        # prompt one at a time, etc.
        prompt_feature = []
        for prompt in prompts:
            prompt_token = self.text_model.tokenizer.encode(
                prompt,
                max_length=self.max_len,
                truncation=True,
                add_special_tokens=False,
            )
            prompt_token = [self.text_model.cls_idx] + prompt_token
            prompt_token = torch.tensor(prompt_token, dtype=torch.long, device=self.text_model.device).unsqueeze(0)

            attention_mask = prompt_token != self.text_model.pad_idx
            model_output = self.text_model(None, input_ids=prompt_token, attention_mask=attention_mask)
            prompt_feature.append(model_output["text_feature"])

        prompt_feature = torch.cat(prompt_feature, dim=0)
        return prompt_feature.detach().cpu()

    def _extract_text(self, model_input):
        inds_full = model_input['input']['text']
        inds = [inds_full[i][-1] for i in range(len(inds_full))]

        return [model_input['data']['text'][i] for i in inds]

    def _get_query_embeddings(
        self,
        query_loader: DataLoader,
        query_order: List,
    ) -> torch.Tensor:
        if isinstance(query_loader.dataset, AASeqDataset):
           raise ValueError(f"ProtST only supports text->protein retrieval, received PPI dataset")
        elif not isinstance(query_loader.dataset, AASeqTextUnifiedDataset):
           raise ValueError(f"unexpected dataset type: {type(query_loader.dataset)}")

        query_embeddings = []
        query_ids = []
        for model_inputs in tqdm(query_loader):
            # Where the query ID is stored depending on whether the query
            # is text or sequence (i.e. PPI)
            query_ids += [x[-1] for x in model_inputs["reference_indices"]["input"]["text"]]

            text_inputs = self._extract_text(model_inputs)
            query_embeddings.append(self._get_text_embedding(text_inputs))

        query_idxs = {query_id:idx for idx, query_id in enumerate(query_ids)}
        rearrange_idxs = [query_idxs[query_id] for query_id in query_order]
        query_embeddings = torch.cat(query_embeddings, dim=0)[rearrange_idxs]
        return query_embeddings

    @torch.no_grad()
    def _calculate_target_embeddings(
        self,
        target_loader: DataLoader,
        collate_fn: Callable,
    ) -> Tuple[torch.Tensor, List]:
        target_protein_embeddings = []
        target_ids = []
        # This is a loader that just provides a list of integer IDs
        # for each protein.
        # Collect all IDs, get FASTA sequences, convert to torchdrug dataset
        for protein_ids in target_loader:
            target_ids += protein_ids.tolist()
        seqs = [collate_fn.aaseq_sequences[x] for x in target_ids]

        ds = td_data.ProteinDataset()
        ds.load_sequence(seqs, targets={}, atom_feature=None, bond_feature=None)
        dataloader = td_data.DataLoader(ds, shuffle=False, batch_size=self.batch_size)

        for batch in tqdm(dataloader):
            if self.device.type == "cuda":
                batch = cuda(batch, device=self.device)
            graph = batch["graph"]
            output = self.protein_model(graph, graph.residue_feature.float())
            protein_feature = output["graph_feature"]
            target_protein_embeddings.append(protein_feature.detach().clone().cpu())

        target_protein_embeddings = torch.cat(target_protein_embeddings, dim=0)
        return target_protein_embeddings, target_ids

    def _get_cached_target_embeddings(
        self,
        collate_fn: Callable,
        aaseq_type: str,
    ):
        print("ProtST: loading cached target embeddings")
        target_embeddings_path = os.path.join(self.checkpoint_dir, f"{aaseq_type}_target_embeddings.pkl")
        if not os.path.exists(target_embeddings_path):
            print(
                f"ProtST: retrieval_use_cached_target_embeddings is set to True but cached "
                f"embeddings not found, calculating and writing to: {target_embeddings_path}"
            )
            all_targets = get_retrieval_target_set(
                        None,
                        {},
                        EvalArgs(retrieval_eval_all_aaseqs=True),
                        aaseq_type=aaseq_type,
            )
            target_loader = get_retrieval_target_proteins_loader(
                all_targets,
                self.batch_size,
            )

            target_protein_embeddings, target_ids = self._calculate_target_embeddings(target_loader, collate_fn)
            with open(target_embeddings_path, "wb") as fh:
                torch.save((target_protein_embeddings, target_ids), fh)
        else:
            target_protein_embeddings, target_ids = torch.load(target_embeddings_path)
        return target_protein_embeddings, target_ids

    @torch.no_grad()
    def _get_target_embeddings(
        self,
        target_loader: DataLoader,
        target_order: List,
        collate_fn: Callable,
        aaseq_type: str,
    ) -> torch.Tensor:
        if self.use_cached_target_embeddings:
            target_protein_embeddings, target_ids = self._get_cached_target_embeddings(collate_fn, aaseq_type)
        else:
            target_protein_embeddings, target_ids = self._calculate_target_embeddings(
                target_loader,
                collate_fn,
            )

        # Rearrange to expected order and/or subset down to just the targets of interest.
        target_idxs = {target_id:idx for idx, target_id in enumerate(target_ids)}
        rearrange_idxs = [target_idxs[target_id] for target_id in target_order]
        target_protein_embeddings = target_protein_embeddings[rearrange_idxs]
        return target_protein_embeddings

    @torch.no_grad()
    def get_predictions(
        self,
        query_loader: DataLoader,
        target_loader: DataLoader,
        query_order: List,
        target_order: List,
    ) -> torch.Tensor:
        # Get embeddings of queries (could be text or proteins).
        query_embeddings = self._get_query_embeddings(query_loader, query_order)

        # Get embeddings of set of target proteins.
        # NOTE: Somewhat hacky that we're using the collator from the query loader,
        # can we move this out to a separate object at some point?
        # I think this may also mean we're storing all the raw protein sequences
        # once for each instantiated collator fxn.
        target_embeddings = self._get_target_embeddings(
            target_loader,
            target_order,
            query_loader.collate_fn,
            query_loader.dataset.aaseq_type,
        )

        # Normalize embeddings and calculate cosine similarities.
        query_embs_normalized = F.normalize(query_embeddings)
        target_embs_normalized = F.normalize(target_embeddings)
        # Dims like: num_queries X num_targets
        sims = query_embs_normalized @ target_embs_normalized.T

        return sims.detach().cpu().to(torch.float64)
