import os, sys, logging, random, json, pickle, math
from typing import Any, Tuple, Callable, Union, Dict, Optional, Iterable, List
from collections.abc import Mapping

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import RAdam
from transformers.optimization import AdamW, Adafactor
from transformers.deepspeed import deepspeed_init
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import get_parameter_names, IterableDatasetShard
#from transformers.training_args import ShardedDDPOption
from transformers.trainer_callback import TrainerState

import deepspeed
from deepspeed import comm as dist

from esm.data import Alphabet

from procyon.model.model_unified import UnifiedProCyon, deepspeed_init_with_checkpoint
from procyon.model.model_utils import detect_zero_stages
from procyon.data.data_collator import ProteinMLMCollator
from procyon.training.training_args import TrainArgs, DataArgs, ModelArgs

from procyon.training.wandb_logger import WandbLogger
from dataclasses import asdict, is_dataclass
from procyon.data.data_utils import DATA_DIR
from procyon.data.dataset import ProteinEvalDataset, ProteinDataset
from procyon.evaluate.general_eval import protein_retrieval_eval_from_embeddings, get_testing_predictions
from procyon.data.data_utils import get_relation_fname
from procyon.data.samplers import DistributedSamplerResume, SequentialDistributedSampler

import time
import wandb

import pynvml
import pathlib

## Sklearn Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    f1_score,
    # fbeta_score,
)

from procyon.training.train_utils import (
    get_mlm_loss,
    get_kepler_loss,
    get_cl_metrics,
    get_qa_metrics,
    get_qa_metrics_from_preds,
    get_scheduler,
    unwrap_model,
    get_retrieval_scores_inbatch,
    decompose_dataset_name,
    barrier
)

from procyon.data.constants import DATASET_ID, CAPTION_TRAIN_WEIGHTS

from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state

SAVE_TRAINING_STATE_FNAME = 'training_state.json'
SAVE_CONFIG_FNAME = 'txplm_config.json'
SAVE_TRAINING_ARGS_FNAME = "training_args.pt"
SAVE_MODEL_ARGS_FNAME = "model_args.pt"
SAVE_DATA_ARGS_FNAME = "data_args.pt"

QA_METRICS = ["acc", "f1", "ppl", "loss"]
RETRIEVAL_METRICS = ["loss", "auprc", "auroc"] # Can add fmax later
CAPTION_METRICS = ["ppl", "loss"]

def set_prot_lora_group(model: nn.Module, index):
    for mn, m in model.named_modules():
        if hasattr(m, 'set_prot_lora_group'):
            m.set_prot_lora_group(index)

def set_text_lora_group(model: nn.Module, index):
    for mn, m in model.named_modules():
        if hasattr(m, 'set_text_lora_group'):
            m.set_text_lora_group(index)

class TxPLMTrainerIT(Trainer):
    """
    Trainer adapted from OntoProtein. Referenced transformers.Trainer and DRAGON.
    Notes:
        - We'll not be using Apex or sagemaker or TPU, so all related code was removed.
        - Custom training, evaluation and wandb logging are added.

    Args:
        model: Torch module containing our model
        args:
        train_datasets: Tuple of training datasets in the following order:
            MLM, Protein-GO, Protein-Protein, Pfam
        val_dataset: Tuple of validation datasets in the exact same order as above
    """

    def __init__(
        self,
        model: nn.Module,
        args: TrainArgs,
        data_args: DataArgs, # to allow saving with checkpoints
        model_args: ModelArgs, # to allow saving with checkpoints
        train_datasets: Tuple[Any],
        val_datasets: Tuple[Any],
        protein_mlm_collator: Any,
        qa_collator: Any,
        retrieval_collator: Any,
        caption_collator: Any,
        output_dir: str,
        device: torch.device,
        logger: logging.Logger,
        wandb: WandbLogger,
    ):
        #args.place_model_on_device = False # Can set this in training args if needed
        super().__init__(model=model, args=args)

        # TODO: Fix below dataset assignments, could use just one dataset
        (
            self.train_protein_mlm_datasets,
            self.train_qa_dataset,
            self.train_retrieval_dataset,
            self.train_caption_dataset,
        ) = train_datasets

        (
            self.val_protein_mlm_dataset,
            self.val_qa_dataset,
            self.val_retrieval_dataset,
            self.val_caption_dataset,
        ) = val_datasets

        # TODO: merge into a single param as above
        self.protein_mlm_collator = protein_mlm_collator
        self.qa_collator = qa_collator
        self.retrieval_collator = retrieval_collator
        self.caption_collator = caption_collator

        self.output_dir = output_dir
        self.device = device
        self.logger = logger
        self.wandb = wandb

        self.data_args = data_args
        self.model_args = model_args

        assert self.args.use_deepspeed


        # Set local_rank and global_rank automatically:
        self.args.local_rank = int(os.environ["LOCAL_RANK"])
        self.args.global_rank = int(os.environ["RANK"])

        if args.watch_gradient: # Log gradients if defined
            # Try only watching on one rank globally, not sure if this will work?
            self.wandb.watch(self.model, log_freq = args.gradient_log_frequency)

        #self.relation_filename = get_relation_fname(go_split_method = data_args.go_split_method, shot_level = data_args.val_split_type, split = 'val')

        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.args.local_rank)

        # Get caption_loss_reweight version:
        self.caption_loss_rescale = None
        if self.args.caption_loss_rescale_version is not None:
            self.caption_loss_rescale = CAPTION_TRAIN_WEIGHTS[self.args.caption_loss_rescale_version]

    def train(self):

        # Setup state in the first steps, will be overwritten by checkpoint load if needed
        self.state = TrainerState()
        self.state.is_hyper_param_search = False
        # last_improvement: global_step of most recent measured checkpoint that improved the validation performance metric (for early stopping)
        self.state.last_improvement = 0

        if self.args.resume_from_checkpoint:
            #deepspeed.init_distributed()
            barrier()
            training_progress = self._load_from_checkpoint_deepspeed(self.args.resume_from_checkpoint)
            if self.args.resume_training_progress:
                self.training_progress = training_progress
            barrier()
        else:
            #deepspeed.init_distributed() # Must call to initialize distributed environment (for model parallel)
            model_engine, optimizer, _, _ = deepspeed.initialize(
                self.args,
                self.model,
            )
            self.model_engine = model_engine
            self.optimizer = optimizer

        if (self.args.resume_from_checkpoint is None) or (not self.args.resume_training_progress):
            # This is already set in conditional above if neither of these conditions hit
            # Keep track of epochs, steps for each task:
            # Set to 0
            self.training_progress = {
                "epoch": {
                    "mlm": 0,
                    "qa": 0,
                    "retrieval": 0,
                    "caption": 0,
                },
                "step": {
                    "mlm": 0,
                    "qa": 0,
                    "retrieval": 0,
                    "caption": 0,
                }
            }

        # Init deepspeed first:
        if self.args.use_deepspeed:

            # Set data parallel and model parallel world sizes:
            # See here for how to determine dp world size since we're using a pipeline module: https://github.com/microsoft/DeepSpeed/blob/c37fe9cbfb8bc10c8dd6ccd8cac9b34ded218990/deepspeed/runtime/pipe/module.py#L174C18-L174C51
            if self.model_args.model_splitting:
                self.dp_world_size = dist.get_world_size() // self.model_args.n_model_pieces
                self.mp_world_size = self.model_args.n_model_pieces
            else:
                self.dp_world_size = dist.get_world_size()
                self.mp_world_size = 1

                # Can also use: torch.distributed.get_world_size() to get world size

            # PipelineModule asserts that n_model_pieces evenly divides into the world size

            # dp_rank = data parallel rank
            # mp_rank = model parallel rank
            # https://github.com/microsoft/DeepSpeed/blob/060a8e185a33bdd45cacc826f5e309a7e3675f8a/deepspeed/runtime/engine.py#L2612
            # See above reference for how to extract these variables from the engine
            #self.dp_rank = dist.get_rank(group = self.model_engine.optimizer.dp_process_group)
            self.dp_rank = int(os.environ["RANK"])
            #self.mp_rank = 0 if self.model.mpu is None else self.model.mpu.get_model_parallel_rank()
            # MP rank doesn't matter for trainer, so don't get it

        else:
            self.create_optimizer()

            if torch.distributed.is_initialized() and train_args.world_size != torch.distributed.get_world_size():
                self.dp_world_size = torch.distributed.get_world_size()
            self.mp_world_size = 1 # Cannot do model parallelism without deepspeed

        # get dataloaders
        (
            train_protein_mlm_loader,
            train_qa_loader,
            train_retrieval_loader,
            train_caption_loader,
            val_protein_mlm_loader,
            val_qa_loader,
            val_retrieval_loader,
            val_retrieval_protein_loader,
            val_caption_loader,
        ) = self._get_dataloaders()

        #if self.args.resume_from_checkpoint:
        # Set the states to current training state:
        # Set the distributed samplers to resume from given epoch and step (this is why we need DistributedSamplerResume])
        if train_protein_mlm_loader is not None and isinstance(train_protein_mlm_loader.sampler, DistributedSamplerResume):
            train_protein_mlm_loader.sampler.set_epoch(self.training_progress["epoch"]["mlm"],self.training_progress["step"]["mlm"])
        if train_qa_loader is not None and isinstance(train_qa_loader.sampler, DistributedSamplerResume):
            train_qa_loader.sampler.set_epoch(self.training_progress["epoch"]["qa"],self.training_progress["step"]["qa"])
        if train_retrieval_loader is not None and isinstance(train_retrieval_loader.sampler, DistributedSamplerResume):
            train_retrieval_loader.sampler.set_epoch(self.training_progress["epoch"]["retrieval"],self.training_progress["step"]["retrieval"])
        if train_caption_loader is not None and isinstance(train_caption_loader.sampler, DistributedSamplerResume):
            train_caption_loader.sampler.set_epoch(self.training_progress["epoch"]["caption"],self.training_progress["step"]["caption"])
        # else:
        #     # Set distributed samplers from beginning
        #     if train_protein_mlm_loader is not None and isinstance(train_protein_mlm_loader.sampler, DistributedSamplerResume):
        #         train_protein_mlm_loader.sampler.set_epoch(0,0)
        #     if train_qa_loader is not None and isinstance(train_qa_loader.sampler, DistributedSamplerResume):
        #         train_qa_loader.sampler.set_epoch(0,0)
        #     if train_retrieval_loader is not None and isinstance(train_retrieval_loader.sampler, DistributedSamplerResume):
        #         train_retrieval_loader.sampler.set_epoch(0,0)
        #     if train_caption_loader is not None and isinstance(train_caption_loader.sampler, DistributedSamplerResume):
        #         train_caption_loader.sampler.set_epoch(0,0)

        # calculate total batch size
        total_protein_mlm_batch_size = (
            self.args.protein_mlm_batch_size
            * self.args.gradient_accumulation_steps
            * self.dp_world_size
        )
        total_qa_batch_size = (
            self.args.qa_batch_size
            * self.args.gradient_accumulation_steps
            * self.dp_world_size
        )
        total_retrieval_batch_size = (
            self.args.retrieval_batch_size
            * self.args.gradient_accumulation_steps
            * self.dp_world_size
        )
        total_caption_batch_size = (
            self.args.caption_batch_size
            * self.args.gradient_accumulation_steps
            * self.dp_world_size
        )

        # if self.args.local_rank not in {0, 1}:
        #     self.logger.setLevel(logging.ERROR)

        self.logger.info(
        f"""global_step: {self.state.global_step}
        DP world_size: {self.dp_world_size}
        MP world_size: {self.mp_world_size}
        local_rank: {self.args.local_rank}
        global_rank: {self.args.global_rank}
        dp_rank: {self.dp_rank}
        # devices: {torch.cuda.device_count()}"""
        )

        # if delay_optimizer_creation:
        #     self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)

        # get number of steps per epoch.  different from num_protein_mlm_update_steps_per_epoch because here we consider raw steps, but not update steps after considering gradient accumulation.
        # Need to multiply by world size because this rank only sees the sampler output by its given rank - previous runs were inflating the size
        protein_mlm_steps_per_epoch = len(train_protein_mlm_loader) * self.dp_world_size if train_protein_mlm_loader else -1
        qa_steps_per_epoch = len(train_qa_loader) * self.dp_world_size if train_qa_loader else -1
        retrieval_steps_per_epoch = len(train_retrieval_loader) * self.dp_world_size if train_retrieval_loader else -1
        caption_steps_per_epoch = len(train_caption_loader) * self.dp_world_size if train_caption_loader else -1

        # log info
        if self.args.max_steps == -1: # Default argument via HF TrainingArguments
            # Set steps by epochs:
            num_epochs = self.args.num_train_epochs

            # Steps total gets
            mlm_steps_total = num_epochs * len(train_protein_mlm_loader) if train_protein_mlm_loader else 0
            qa_steps_total = self.args.qa_epoch_multiplier * num_epochs * len(train_qa_loader) if train_qa_loader else 0
            retrieval_steps_total = self.args.retrieval_epoch_multiplier * num_epochs * len(train_retrieval_loader) if train_retrieval_loader else 0
            caption_steps_total = self.args.caption_epoch_multiplier * num_epochs * len(train_caption_loader) if train_caption_loader else 0

            # Steps are entirely based on the epochs and number of steps via batch sizes
            max_steps = max(mlm_steps_total, qa_steps_total, retrieval_steps_total, caption_steps_total)

            (
                mlm_skip_frequency,
                qa_skip_frequency,
                retrieval_skip_frequency,
                caption_skip_frequency
            ) = [None] * 4

            # Get remainder steps for each task:
            if (max_steps - mlm_steps_total):
                if (mlm_steps_total < (max_steps // 2)) and (mlm_steps_total > 0):
                    mlm_skip_frequency = -(max_steps // mlm_steps_total)
                else:
                    mlm_skip_frequency = math.ceil(max_steps / (max_steps - mlm_steps_total))
            if ((max_steps - qa_steps_total) > 0):
                if (qa_steps_total < (max_steps // 2)) and (qa_steps_total > 0):
                    qa_skip_frequency = -(max_steps // qa_steps_total)
                else:
                    qa_skip_frequency = math.ceil(max_steps / (max_steps - qa_steps_total))
            if ((max_steps - retrieval_steps_total) > 0):
                if (retrieval_steps_total < (max_steps // 2)) and (retrieval_steps_total > 0):
                    retrieval_skip_frequency = -(max_steps // retrieval_steps_total)
                else:
                    retrieval_skip_frequency = math.ceil(max_steps / (max_steps - retrieval_steps_total))
            if ((max_steps - caption_steps_total) > 0):
                if (caption_steps_total < (max_steps // 2)) and (caption_steps_total > 0):
                    caption_skip_frequency = -(max_steps // caption_steps_total)
                else:
                    caption_skip_frequency = math.ceil(max_steps / (max_steps - caption_steps_total))

            # mlm_skip_frequency = math.ceil(max_steps / (max_steps - mlm_steps_total)) if (max_steps - mlm_steps_total) > 0 else None
            # qa_skip_frequency = math.ceil(max_steps / (max_steps - qa_steps_total)) if (max_steps - qa_steps_total) > 0 else None
            # retrieval_skip_frequency = math.ceil(max_steps / (max_steps - retrieval_steps_total)) if (max_steps - retrieval_steps_total) > 0 else None
            # caption_skip_frequency = math.ceil(max_steps / (max_steps - caption_steps_total)) if (max_steps - caption_steps_total) > 0 else None

        else:
            max_steps = self.args.max_steps

            (
                mlm_skip_frequency,
                qa_skip_frequency,
                retrieval_skip_frequency,
                caption_skip_frequency
            ) = [None] * 4

            # Set total number of epochs:
            mlm_steps_total = (
                max_steps / protein_mlm_steps_per_epoch
                if protein_mlm_steps_per_epoch > 0 else 0
            )
            qa_steps_total = (
                max_steps / qa_steps_per_epoch
                if qa_steps_per_epoch > 0 else 0
            )
            retrieval_steps_total = (
                max_steps / retrieval_steps_per_epoch
                if retrieval_steps_per_epoch > 0 else 0
            )
            caption_steps_total = (
                max_steps / caption_steps_per_epoch
                if caption_steps_per_epoch > 0 else 0
            )

        steps_trained = self.state.global_step // self.dp_world_size  # Set based on global steps, divide by world size bc we track global steps according to world size, while max_steps are not

        # cur_protein_mlm_epoch = 0
        # cur_qa_epochs = {key: 0 for key in train_qa_loaders.keys()} if train_qa_loaders else 0
        # cur_retrieval_epochs = {key: 0 for key in train_retrieval_loaders.keys()} if train_retrieval_loaders else 0
        # cur_caption_epochs = {key: 0 for key in train_caption_loaders.keys()} if train_caption_loaders else 0

        time_cache = []

        # Log right before stepping into training steps:
        if self.args.global_rank == 0:
            self.logger.info(f"Total protein MLM batch size: {total_protein_mlm_batch_size}\nTotal QA batch size: {total_qa_batch_size}\nTotal retrieval batch size: {total_retrieval_batch_size}\nTotal caption batch size: {total_caption_batch_size}\n")
            self.logger.info(
                f"Number of protein MLM epochs: {mlm_steps_total}\n" +
                (f"Number of QA steps: {qa_steps_total}\n") +
                (f"Number of retrieval steps: {retrieval_steps_total}\n") +
                (f"Number of caption steps: {caption_steps_total}\n")
            )
            self.logger.info(f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}")
            self.logger.info(f"Total optimization steps = {max_steps}")

        # NOTE: iteration of steps and epochs is deterministic, i.e., we are not sampling and therefore can assume that when one hits, so do the others

        # Set the iters for each sampler beforehand:
        train_protein_mlm_iter = iter(train_protein_mlm_loader) if train_protein_mlm_loader is not None else None
        train_qa_iter = iter(train_qa_loader) if train_qa_loader is not None else None
        train_retrieval_iter = iter(train_retrieval_loader) if train_retrieval_loader is not None else None
        train_caption_iter = iter(train_caption_loader) if train_caption_loader is not None else None

        barrier()

        self.loss_recorder = []

        cache_dir = pathlib.Path("gradient_cache")
        cache_dir.mkdir(exist_ok = True)
        whole_grad_conflict = {}
        valid_step = {}
        update_times = 0
        # Actual training steps:
        for step in range(steps_trained, max_steps): # Step counter does not account for dp_world_size

            self.logger.info(f"Logger Step {step}, RANK={self.args.global_rank}")
            self.state.global_step =  step * self.dp_world_size # This is where the global step is updated, so it's NOT +=
            # self.dp_world_size *
            # Use world size * step because DistributedSampler is iterating in world_size parallel times along each dataset

            (
                train_protein_mlm_inputs,
                train_qa_inputs,
                train_retrieval_inputs,
                train_caption_inputs,
            ) = (None, None, None, None)

            # update epoch and iterator
            if train_protein_mlm_loader:
                if (mlm_skip_frequency is None):
                    pass_condition = True
                else:
                    pass_condition = ((step % mlm_skip_frequency != 0) if mlm_skip_frequency > 0 else (step % mlm_skip_frequency == 0))

                if pass_condition:

                    if self.training_progress["step"]["mlm"] >= protein_mlm_steps_per_epoch:
                        # Indicates that we've reached an epoch for this task
                        self.training_progress["step"]["mlm"] = 0 # Reset back to 0
                        self.training_progress["epoch"]["mlm"] += 1
                        if isinstance(train_protein_mlm_loader.sampler, DistributedSamplerResume):
                            train_protein_mlm_loader.sampler.set_epoch(self.training_progress["epoch"]["mlm"],0)

                    self.training_progress["step"]["mlm"] += self.dp_world_size
                    train_protein_mlm_inputs = next(train_protein_mlm_iter)

                    if self.args.global_rank in {-1, 0}:
                        self.wandb.log(self.state.global_step, {"protein_mlm epoch": step / protein_mlm_steps_per_epoch if protein_mlm_steps_per_epoch else 0})

            if train_qa_loader:
                if (qa_skip_frequency is None):
                    pass_condition = True
                else:
                    pass_condition = ((step % qa_skip_frequency != 0) if qa_skip_frequency > 0 else (step % (-1 * qa_skip_frequency) == 0))

                #pass_condition = True if (qa_skip_frequency is None) else (step % qa_skip_frequency != 0)
                if pass_condition:
                    if self.training_progress["step"]["qa"] >= qa_steps_per_epoch:
                        # Indicates that we've reached an epoch for this task
                        self.training_progress["step"]["qa"] = 0 # Reset back to 0
                        self.training_progress["epoch"]["qa"] += 1
                        if isinstance(train_qa_loader.sampler, DistributedSamplerResume):
                            train_qa_loader.sampler.set_epoch(self.training_progress["epoch"]["qa"],0)
                            train_qa_iter = iter(train_qa_loader) # Reset the iter

                    # Sample:
                    train_qa_inputs = next(train_qa_iter)
                    self.training_progress["step"]["qa"] += self.dp_world_size

                    if self.args.global_rank in {-1, 0}:
                        self.wandb.log(self.state.global_step, {"qa epoch": (self.training_progress["step"]["qa"] / qa_steps_per_epoch) + self.training_progress["epoch"]["qa"] if qa_steps_per_epoch else 0})

            if train_retrieval_loader:
                if (retrieval_skip_frequency is None):
                    pass_condition = True
                else:
                    pass_condition = ((step % retrieval_skip_frequency != 0) if retrieval_skip_frequency > 0 else (step % (-1 * retrieval_skip_frequency) == 0))

                #pass_condition = True if (retrieval_skip_frequency is None) else (step % retrieval_skip_frequency != 0)
                if pass_condition:

                    if self.training_progress["step"]["retrieval"] >= retrieval_steps_per_epoch:
                        # Indicates that we've reached an epoch for this task
                        self.training_progress["step"]["retrieval"] = 0 # Reset back to 0
                        self.training_progress["epoch"]["retrieval"] += 1
                        if isinstance(train_retrieval_loader.sampler, DistributedSamplerResume):
                            train_retrieval_loader.sampler.set_epoch(self.training_progress["epoch"]["retrieval"],0)
                            train_retrieval_iter = iter(train_retrieval_loader) # Reset the iter

                    # Sample:
                    train_retrieval_inputs = next(train_retrieval_iter)
                    self.training_progress["step"]["retrieval"] += self.dp_world_size

                    if self.args.global_rank in {-1, 0}:
                        self.wandb.log(self.state.global_step, {"retrieval epoch": (self.training_progress["step"]["retrieval"] / retrieval_steps_per_epoch) + self.training_progress["epoch"]["retrieval"] if retrieval_steps_per_epoch else 0})

            if train_caption_loader:
                if (caption_skip_frequency is None):
                    pass_condition = True
                else:
                    pass_condition = ((step % caption_skip_frequency != 0) if caption_skip_frequency > 0 else (step % (-1 * caption_skip_frequency) == 0))
                #pass_condition = True if  else (step % caption_skip_frequency != 0)
                if pass_condition:

                    if self.training_progress["step"]["caption"] >= caption_steps_per_epoch:
                        # Indicates that we've reached an epoch for this task
                        self.training_progress["step"]["caption"] = 0 # Reset back to 0
                        self.training_progress["epoch"]["caption"] += 1
                        if isinstance(train_caption_loader.sampler, DistributedSamplerResume):
                            # Must reset distributed sampler to re-align seed shuffled indices: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                            train_caption_loader.sampler.set_epoch(self.training_progress["epoch"]["caption"],0)
                            train_caption_iter = iter(train_caption_loader) # Reset the iter

                    # Sample:
                    train_caption_inputs = next(train_caption_iter)
                    self.training_progress["step"]["caption"] += self.dp_world_size

                    if self.args.global_rank in {-1, 0}:
                        self.wandb.log(self.state.global_step, {"caption epoch": (self.training_progress["step"]["caption"] / caption_steps_per_epoch) + self.training_progress["epoch"]["caption"] if caption_steps_per_epoch else 0})

            s_time = time.time()
            if step < 0:
                barrier()
                if self.args.use_deepspeed:
                    loss, all_loss = self.training_step(
                        self.model_engine,
                        train_protein_mlm_inputs,
                        train_qa_inputs,
                        train_retrieval_inputs,
                        train_caption_inputs,
                        step = step,
                    )
                else:
                    loss, all_loss = self.training_step(
                        self.model,
                        train_protein_mlm_inputs,
                        train_qa_inputs,
                        train_retrieval_inputs,
                        train_caption_inputs,
                        step = step,
                    )
                barrier()
                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if not self.args.use_deepspeed:
                    self.optimizer.step()
                    self.model.zero_grad()
                else:
                    self.model_engine.step()
                    self.model_engine.zero_grad()

            else:
                if self.model_args.lora_specific_style == 'specific':
                    all_loss = self.lora_specific_training(
                        step, train_protein_mlm_inputs,
                        train_qa_inputs,
                        train_retrieval_inputs,
                        train_caption_inputs
                    )
                elif (self.model_args.lora_specific_style == 'single_lora') \
                    or (self.model_args.lora_specific_style == 'space_specific'):
                    barrier()
                    if self.args.use_deepspeed:
                        loss, all_loss = self.training_step(
                            self.model_engine,
                            train_protein_mlm_inputs,
                            train_qa_inputs,
                            train_retrieval_inputs,
                            train_caption_inputs,
                            step = step,
                        )
                    else:
                        loss, all_loss = self.training_step(
                            self.model,
                            train_protein_mlm_inputs,
                            train_qa_inputs,
                            train_retrieval_inputs,
                            train_caption_inputs,
                            step = step,
                        )
                    barrier()
                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if not self.args.use_deepspeed:
                        self.optimizer.step()
                        self.model.zero_grad()
                    else:
                        self.model_engine.step()
                        self.model_engine.zero_grad()
                elif self.model_args.lora_specific_style == 'qa_retrieval_share':
                    set_prot_lora_group(self.model, 0)
                    set_text_lora_group(self.model, 0)
                    all_loss = dict()
                    barrier()
                    if self.args.use_deepspeed:
                        loss, step_loss = self.training_step(
                            self.model_engine,
                            None,
                            train_qa_inputs,
                            None,
                            None,
                            step = step,
                        )
                    else:
                        loss, step_loss = self.training_step(
                            self.model,
                            None,
                            train_qa_inputs,
                            None,
                            None,
                            step = step,
                        )
                    all_loss.update(step_loss)
                    barrier()
                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if not self.args.use_deepspeed:
                        self.optimizer.step()
                        self.model.zero_grad()
                    else:
                        self.model_engine.step()
                        self.model_engine.zero_grad()


                    set_prot_lora_group(self.model, 1)
                    set_text_lora_group(self.model, 1)
                    barrier()
                    if self.args.use_deepspeed:
                        loss, step_loss = self.training_step(
                            self.model_engine,
                            None,
                            train_qa_inputs,
                            train_retrieval_inputs,
                            None,
                            step = step,
                        )
                    else:
                        loss, step_loss = self.training_step(
                            self.model,
                            None,
                            train_qa_inputs,
                            train_retrieval_inputs,
                            None,
                            step = step,
                        )
                    all_loss.update(step_loss)
                    barrier()
                    # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                    if not self.args.use_deepspeed:
                        self.optimizer.step()
                        self.model.zero_grad()
                    else:
                        self.model_engine.step()
                        self.model_engine.zero_grad()


            barrier()

            print(f"LR={self.args.local_rank}, DP={self.dp_rank}: {time.time() - s_time}")
            time_cache.append(time.time() - s_time)
            # record loss  # TODO: Investigate whether this is still compatible
            if self.args.global_rank in {-1, 0}:
                all_loss["global_step"] = step
                # all_loss["learning_rate"] = self._get_learning_rate()
                all_loss = dict(all_loss)

                all_loss['time'] = time_cache[-1]

                meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                all_loss['gpu_memory(GB)'] = meminfo.used/(1024**3)
                self.loss_recorder.append(all_loss)
                # print(all_loss)

                # self.save_training_info(time_cache)

            # if self.args.global_rank in {-1, 0}:


            evaluate_for_save = False
            if (step % self.args.save_steps == 0) and (step > 0):
                current_checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.state.global_step}")
                # NOTE: All ranks must call save checkpoint in deepspeed, so do not rank filter here
                self._save_checkpoint_deepspeed(current_checkpoint_dir)
                #if not self.args.use_deepspeed:
                # if self.args.local_rank in {-1, 0}:
                #     # This condition SAVES
                #     os.makedirs(current_checkpoint_dir, exist_ok=True)
                #     evaluate_for_save = True

            # Evaluate during training
            # Always evaluate if saving so we can save validation results to the checkpoint directory
            # Added (step > 0) below to make sure we don't save the model on the first checkpoint
            if ((((step % self.args.eval_steps == 0) or \
                        (
                            step % self.args.initial_eval_steps == 0 and \
                            step <= self.args.initial_eval_steps_limit
                        )) and (step > 0)
                ) or \
                (evaluate_for_save)) and (self.args.eval_on_the_fly):

                self.logger.info("Begin eval")

                if self.args.eval_on_the_fly:
                    # Add above zero check to make sure we don't save at very first step
                    # This condition EVALUATES

                    self.model.eval()
                    # Evaluate:
                    # Measure eval performance (TODO: change this to val perf) & save results to checkpoint dir
                    self._run_eval(
                        val_protein_mlm_loader = val_protein_mlm_loader,
                        val_qa_loader = val_qa_loader,
                        val_retrieval_loader = val_retrieval_loader,
                        val_retrieval_protein_loader = val_retrieval_protein_loader,
                        val_caption_loader = val_caption_loader,
                    )
                    self.model.train()
                #     # TODO: Save results to checkpoint directory if below condition hits
                #     if evaluate_for_save:
                #         pass

        self.logger.info("Training completed.\n")
        # Tianlong's saving procedures: (03-23)
        # if self.args.global_rank in {-1, 0}:
        #     eval_name = self.save_training_info(time_cache)

        # self.model.eval()
        # qa_metrics_dict, retrieval_metrics_dict, caption_metrics_dict = self._run_eval(
        #             val_protein_mlm_loader = val_protein_mlm_loader,
        #             val_qa_loader = val_qa_loader,
        #             val_retrieval_loader = val_retrieval_loader,
        #             val_retrieval_protein_loader = val_retrieval_protein_loader,
        #             val_caption_loader = val_caption_loader,
        #         )
        #     # print(val_qa_loader, val_retrieval_loader, val_retrieval_protein_loader, val_caption_loader)
        #     # print(qa_metrics_dict, retrieval_metrics_dict, caption_metrics_dict)
        # if self.args.global_rank in {-1, 0}:
        #     eval_results = {}
        #     for dkey in qa_metrics_dict.keys():
        #         for m in QA_METRICS:
        #             eval_results[f"QA-{dkey}-{m}"] = f"{np.mean(qa_metrics_dict[dkey][m]):.6f}"
        #             print(f"QA, {m}, {dkey}: {np.mean(qa_metrics_dict[dkey][m]):.6f}")

        #     for dkey in retrieval_metrics_dict.keys():
        #         for m in RETRIEVAL_METRICS:
        #             eval_results[f"Retrieval-{dkey}-{m}"] = f"{np.mean(retrieval_metrics_dict[dkey][m]):.6f}"
        #             print(f"Retrieval, {m}, {dkey}: {np.mean(retrieval_metrics_dict[dkey][m]):.6f}")

        #     for dkey in caption_metrics_dict.keys():
        #         for m in CAPTION_METRICS:
        #             eval_results[f"Caption-{dkey}-{m}"] = f"{np.mean(caption_metrics_dict[dkey][m]):.6f}"
        #             print(f"Caption, {m}, {dkey}: {np.mean(caption_metrics_dict[dkey][m]):.6f}")

        # if self.args.global_rank in {-1, 0}:
        #     with open(cache_dir.joinpath(f"eval_results_{eval_name}.json"), 'w') as fout:
        #         json.dump(eval_results, fout, indent=2)

        wandb.finish()

        self.is_in_train = False
        print(f"{self.args.local_rank}: {np.mean(time_cache)}")

    def lora_specific_training(self,
                               step,
                               train_protein_mlm_inputs,
                               train_qa_inputs,
                               train_retrieval_inputs,
                               train_caption_inputs):
        # task qa
        gather_loss = dict()
        set_prot_lora_group(self.model, 0)
        set_text_lora_group(self.model, 0)
        barrier()
        if self.args.use_deepspeed:
            loss, all_loss = self.training_step(
                self.model_engine,
                None,
                train_qa_inputs,
                None,
                None,
                step = step,
            )
        else:
            loss, all_loss = self.training_step(
                self.model,
                None,
                train_qa_inputs,
                None,
                None,
                step = step,
            )
        barrier()
        if len(all_loss) > 0:
            gather_loss.update(all_loss)
            if not self.args.use_deepspeed:
                self.optimizer.step()
                self.model.zero_grad()
            else:
                self.model_engine.step()
                self.model_engine.zero_grad()

        # task retrieval
        set_prot_lora_group(self.model, 1)
        set_text_lora_group(self.model, 1)
        barrier()
        if self.args.use_deepspeed:
            loss, all_loss = self.training_step(
                self.model_engine,
                None,
                None,
                train_retrieval_inputs,
                None,
                step = step,
            )
        else:
            loss, all_loss = self.training_step(
                self.model,
                None,
                None,
                train_retrieval_inputs,
                None,
                step = step,
            )
        barrier()

        if len(all_loss) > 0:
            gather_loss.update(all_loss)
            if not self.args.use_deepspeed:
                self.optimizer.step()
                self.model.zero_grad()
            else:
                self.model_engine.step()
                self.model_engine.zero_grad()

        # MLM
        set_prot_lora_group(self.model, 2)
        set_text_lora_group(self.model, 2)
        barrier()
        if self.args.use_deepspeed:
            loss, all_loss = self.training_step(
                self.model_engine,
                train_protein_mlm_inputs,
                None,
                None,
                None,
                step = step,
            )
        else:
            loss, all_loss = self.training_step(
                self.model,
                train_protein_mlm_inputs,
                None,
                None,
                None,
                step = step,
            )
        barrier()

        if len(all_loss) > 0:
            gather_loss.update(all_loss)
            if not self.args.use_deepspeed:
                self.optimizer.step()
                self.model.zero_grad()
            else:
                self.model_engine.step()
                self.model_engine.zero_grad()

        set_prot_lora_group(self.model, 3)
        set_text_lora_group(self.model, 3)
        barrier()
        if self.args.use_deepspeed:
            loss, all_loss = self.training_step(
                self.model_engine,
                train_protein_mlm_inputs,
                train_qa_inputs,
                train_retrieval_inputs,
                train_caption_inputs,
                step = step,
            )
        else:
            loss, all_loss = self.training_step(
                self.model,
                train_protein_mlm_inputs,
                train_qa_inputs,
                train_retrieval_inputs,
                train_caption_inputs,
                step = step,
            )
        barrier()

        if len(all_loss) > 0:
            gather_loss.update(all_loss)
            if not self.args.use_deepspeed:
                self.optimizer.step()
                self.model.zero_grad()
            else:
                self.model_engine.step()
                self.model_engine.zero_grad()
        return gather_loss


    def save_training_info(self, time_cache):
        print(f"{self.args.global_rank}: {np.mean(time_cache)}")

        calc_recorder = self.loss_recorder
        avg_loss = {}
        for i_loss in calc_recorder:
            for key in i_loss:
                if type(i_loss[key]) is dict:
                    for sub_key in i_loss[key]:
                        all_key = f'{key}_{sub_key}'
                        if all_key not in avg_loss:
                            avg_loss[all_key] = []
                        avg_loss[all_key].append(i_loss[key][sub_key])
                elif key != 'global_step':
                    if key not in avg_loss:
                        avg_loss[key] = []
                    avg_loss[key].append(i_loss[key])
        with open(self.args.deepspeed_config) as fin:
            deep_conf = json.load(fin)
        max_len = 0
        for key in avg_loss:
            avg_loss[key].append(np.mean(avg_loss[key][-200:]))
            max_len = max(max_len, len(avg_loss[key]))
        # print(avg_loss)
        # align data with -1
        for key in avg_loss:
            for _ in range(max_len - len(avg_loss[key])):
                avg_loss[key].insert(0, -1)

        df = pd.DataFrame(avg_loss)
        # batch_size = f"{self.args.protein_mlm_batch_size}_{self.args.qa_batch_size}_{self.args.retrieval_batch_size}_{self.args.caption_batch_size}"
        # weight_decay = deep_conf['optimizer']['params']['weight_decay']
        prot_train_mode = self.model_args.freeze_protein_encoder
        text_train_mode = self.model_args.freeze_text_encoder
        # train_mode = f"{prot_train_mode}_{text_train_mode}"
        file_name_1 = f"{self.model_args.text_encoder_fname}-{self.model_args.freeze_text_encoder}-{self.model_args.protein_encoder_num_params}-{self.model_args.freeze_protein_encoder}-{self.model_args.protein_lora_parameters}_{self.model_args.protein_task_spc_lora}"
        file_name = f"{file_name_1}_lr{deep_conf['optimizer']['params']['lr']}.csv"
        csv_dir = pathlib.Path('tmp')
        csv_dir.mkdir(exist_ok=True)
        df.to_csv(csv_dir.joinpath(file_name))
        return file_name_1

    # TODO: make type annotations more specific?
    def training_step(
        self,
        model: nn.Module,
        protein_mlm_inputs: Dict,
        qa_inputs: Dict,
        retrieval_inputs: Dict,
        caption_inputs: Dict,
        step = None,
    ) -> Tuple[torch.Tensor]:
        model.train()

        # Prepare inputs (i.e. to GPU)
        if protein_mlm_inputs:
            protein_mlm_inputs = self._prepare_inputs(protein_mlm_inputs)
        if qa_inputs:
            qa_inputs = (qa_inputs[0], self._prepare_inputs(qa_inputs[1]))
        if retrieval_inputs:
            retrieval_inputs = (retrieval_inputs[0], self._prepare_inputs(retrieval_inputs[1]))
        if caption_inputs:
            caption_inputs = (caption_inputs[0], self._prepare_inputs(caption_inputs[1]))

        with self.compute_loss_context_manager():
            loss, all_loss = self.compute_loss(
                model,
                protein_mlm_inputs,
                qa_inputs,
                retrieval_inputs,
                caption_inputs,
                step = step,
            )

        return loss, all_loss

    # TODO: If there is a bug about _prepare_input, create our own method
    # def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    #     """
    #     Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    #     """
    #     if isinstance(data, Mapping):
    #         return type(data)({k: self._prepare_input(v) for k, v in data.items()})
    #     elif isinstance(data, (tuple, list)):
    #         return type(data)(self._prepare_input(v) for v in data)
    #     elif isinstance(data, torch.Tensor):
    #         kwargs = dict(device=self.args.device)
    #         if self.deepspeed and data.dtype != torch.int64:
    #             # NLP models inputs are int64 and those get adjusted to the right dtype of the
    #             # embedding. Other models such as wav2vec2's inputs are already float and thus
    #             # may need special handling to match the dtypes of the model
    #             kwargs.update(dict(dtype=self.args.hf_deepspeed_config.dtype()))
    #         return data.to(**kwargs)
    #     return data

    def _get_learning_rate(self):
        last_lr = self.lr_scheduler.get_last_lr()
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    def compute_loss(
        self,
        model: nn.Module,
        protein_mlm_inputs: Dict,
        qa_inputs: Dict,
        retrieval_inputs: Dict,
        caption_inputs: Dict,
        step = None,
    ) -> Tuple[torch.Tensor]:
        total_loss = None if self.args.use_deepspeed else 0
        #total_loss = []
        all_loss = dict()
        # NOTE: For current input scenario, used as an extra scaler for the CL losses
        if protein_mlm_inputs:
            # set_prot_lora_group(self.model, 3)
            # set_text_lora_group(self.model, 3)
            assert (
                "data" in protein_mlm_inputs.keys()
                and "labels" in protein_mlm_inputs.keys()
            )
            protein_mlm_loss = self.compute_mlm_loss(
                model, protein_mlm_inputs, # DATASET_KEY SHOULD BE LEFT NONE FOR NOW
            )

            if not self.args.use_deepspeed:
                protein_mlm_loss.backward()
                total_loss += protein_mlm_loss.item()
            else:
                self.model_engine.backward(protein_mlm_loss)

            all_loss["mlm"] = protein_mlm_loss.item()

        if qa_inputs:
            # set_prot_lora_group(self.model, 0)
            # set_text_lora_group(self.model, 0)
            all_loss["qa"] = dict()
            # Sample dataset:
            key, qa_model_input = qa_inputs
            #for key, qa_inputs in qa_inputs_dict.items():
            assert "data" in qa_model_input.keys()
            qa_loss = self.compute_lm_loss(
                model, qa_model_input, task_type = 'qa', step = step, dataset_key = key,
            )

            if not self.args.use_deepspeed: # This is very inefficient, but is for legacy purposes and debugging
                qa_loss.backward()
                total_loss += qa_loss.item()
            else:
                self.model_engine.backward(qa_loss)

            #total_loss.append(qa_loss.item())
            all_loss["qa"] = qa_loss.item()

        if retrieval_inputs:
            # set_prot_lora_group(self.model, 1)
            # set_text_lora_group(self.model, 1)
            # Missing format assertions here, could replace later

            all_loss["retrieval"] = dict()
            key, retrieval_model_input = retrieval_inputs

            retrieval_loss = self.compute_retrieval_loss(
                model,
                retrieval_model_input,
                task_type = 'retrieval',
                dataset_key = key,
            )

            if not self.args.use_deepspeed:
                retrieval_loss.backward()
                total_loss += retrieval_loss.item()
            else:
                self.model_engine.backward(retrieval_loss)

            all_loss["retrieval"][key] = retrieval_loss.item()

        if caption_inputs:
            # set_prot_lora_group(self.model, 2)
            # set_text_lora_group(self.model, 2)
            all_loss["caption"] = dict()
            key, caption_model_input = caption_inputs
            caption_loss = self.compute_lm_loss(
                model, caption_model_input, task_type = 'caption', dataset_key = key,
            )

            if self.dp_world_size > 1:
                # TODO: Investigate if the loss behavior in calculating metrics is correct (I (Tom) am pretty sure mean() does not average across devices)
                caption_loss = caption_loss.mean()  # mean() to average on multi-gpu parallel training

            if not self.args.use_deepspeed:
                caption_loss.backward()
                total_loss += caption_loss.item()
            else:
                self.model_engine.backward(caption_loss)

            all_loss["caption"][key] = caption_loss.item()

        # DONT USE UNLESS WE DEBUG FIRST
        # if self.args.local_rank in {-1, 0}:
        #     self.log_gradient_norm("model_gradient_norm")

        return total_loss, all_loss

    def compute_mlm_loss(
        self,
        model: nn.Module,
        protein_mlm_inputs: Dict,
        dataset_key = None,
        ):
        input_ids = protein_mlm_inputs["data"]
        labels = protein_mlm_inputs["labels"]

        output = model( # CAN CHANGE HERE
            inputs=protein_mlm_inputs,
            return_mlm=True,
        )
        logits = output['mlm']

        # NOTE: the operation of splitting a long sequence to a batch of chunks is done in the model
        # Ensure the protein encoder returns a dict with key 'logits' when `return_mlm=True`
        # if self.args.world_size <= 1:
        #     output = model( # CAN CHANGE HERE
        #         inputs=protein_mlm_inputs,
        #         return_mlm=True,
        #     )
        #     logits = output['mlm']
        # else:
        #     output = model.module( # CAN CHANGE HERE
        #         inputs=protein_mlm_inputs,
        #         return_mlm=True,
        #     )
        #     logits = output['mlm']

        outputs = get_mlm_loss(logits, labels)
        mask_count = (labels != -100).sum().item()

        # TODO: add support for token-level accuracy
        if dataset_key is None:
            self.logger.info(
                "batch_train_mlm_loss: {:.4f}, batch_train_mlm_accuracy: {:.4f}, batch_train_mlm_perplexity: {:.4f}".format(
                    outputs["loss"].item(), outputs["accuracy"], outputs["perplexity"]
                )
            )
        else:
            self.logger.info(
                "{}_batch_train_mlm_loss: {:.4f}, {}_batch_train_mlm_accuracy: {:.4f}, {}_batch_train_mlm_perplexity: {:.4f}".format(
                    dataset_key, outputs["loss"].item(), dataset_key, outputs["accuracy"], dataset_key, outputs["perplexity"]
                )
            )

        self.wandb.log(self.state.global_step,
            {
                ("batch_train_mlm_loss" if dataset_key is None else f"{dataset_key}_batch_train_mlm_loss"): outputs["loss"].item(),
                ("batch_train_mlm_accuracy" if dataset_key is None else f"{dataset_key}_batch_train_mlm_accuracy"): outputs["accuracy"],
                ("batch_train_mlm_perplexity" if dataset_key is None else f"{dataset_key}_batch_train_mlm_perplexity"): outputs["perplexity"],
            }
        )

        return outputs["loss"] * self.args.mlm_loss_weight

    def compute_lm_loss(self,
            model: nn.Module,
            inputs,
            task_type,
            step = None,
            dataset_key = None,
        ):

        # Logger outputs:
        prefix = "batch_train" if dataset_key is None else f"{dataset_key}_batch_train"
        # Other metrics:
        task_prefix = f"{prefix}_{task_type}"
        # self.logger.info(f"Fwd pass {task_prefix}, RANK={self.args.global_rank}")

        aaseq_type, text_type, relation = decompose_dataset_name(dataset_key)

        crop_off = (task_type == 'caption')

        out = model(inputs, return_mlm = False, retrieval = False, get_full_labels = True, aaseq_type = aaseq_type, crop_off = crop_off)

        #self.logger.info(f"Fwd pass DONE {task_prefix}, RANK={self.args.global_rank}")

        loss = out['outputs'].loss # Get loss from labels already computed by Huggingface

        if task_type == 'qa':
            #auroc, auprc = get_qa_metrics(outputs.logits if model.causal_qa else out['qa_out'], out['full_labels'], yes_token = model.yes_token, no_token = model.no_token, causal_qa = model.causal_qa)
            acc, f1 = get_qa_metrics(out, yes_token = model.yes_token, no_token = model.no_token, answer_token = model.answer_idx)#padding_token = model.tokenizer.pad_token_id)
            self.logger.info(f'{task_prefix}_acc {acc:.4f} {self.args.local_rank} {self.dp_rank}')
            self.logger.info(f'{task_prefix}_f1 {f1:.4f} {self.args.local_rank} {self.dp_rank}')
            self.wandb.log(self.state.global_step,
                {
                    (f'{task_prefix}_acc'): acc,
                    (f'{task_prefix}_f1'): f1,
                }
            )
            loss_weight = self.args.qa_loss_weight
        elif task_type == 'caption':
            # TODO: Need to implement caption metrics - train first
            #rouge_L = get_caption_metrics(out, answer_token = model.answer_idx)
            if self.caption_loss_rescale is not None:
                # Product of both weights - easy because there's only one dataset per batch (micro-batch, i.e., one device)
                loss_weight = self.args.caption_loss_weight * self.caption_loss_rescale["{}_{}".format(aaseq_type, text_type)]
            else:
                loss_weight = self.args.caption_loss_weight

            print("TEST WEIGHT: {} = {}".format(dataset_key, loss_weight))

        # Loss for captioning is negative log-likelihood averaged over predicted tokens
        # (i.e. cross-entropy), perplexity is just exponentiated NLL.
        # Log perplexity for both captioning and QA
        ppl = np.exp(loss.item())
        self.logger.info(f'{task_prefix}_ppl {ppl:.4f}')
        self.wandb.log(self.state.global_step,
            {
                (f'{task_prefix}_ppl'): ppl,
            }
        )

        # wandb logging:
        self.wandb.log(self.state.global_step,
            {
                (f"{task_prefix}_loss"): loss.item(),
            }
        )
        self.logger.info(f"{task_prefix}_loss {loss.item()}")

        return loss * loss_weight

    def compute_retrieval_loss(self, model: nn.Module, inputs, task_type = "retrieval", dataset_key: str = None):
        # outputs = model(inputs, return_mlm = False, retrieval = True)
        aaseq_type, text_type, relation = decompose_dataset_name(dataset_key)

        if self.model_args.filter_negatives_by_id_contrastive: # This batch size counting method should work for retrieval always
            dset_id = torch.LongTensor([DATASET_ID[text_type] for _ in range(len(inputs["input"]["text"]))])
            inputs["dataset_id"] = dset_id

        outputs = model(inputs, return_mlm = False, retrieval = True, aaseq_type = aaseq_type)
        # Loss computed by internal module
        loss = outputs['contrastive_loss']

        # Logger outputs:
        # TODO: Need a better metric
        if dataset_key is None:
            self.logger.info(f"batch_train_{task_type}_loss {loss.mean()}")
        else:
            self.logger.info(f"{dataset_key}_batch_train_{task_type}_loss {loss.mean()}")

        # Compute sims from contrastive output:
        pos_scores, neg_scores = get_retrieval_scores_inbatch(outputs['contrastive_out'])

        # Compute metrics:
        pos_count, neg_count, auroc, auprc = get_cl_metrics(pos_scores, neg_scores)

        if dataset_key is None:
            self.logger.info(f'batch_train_{task_type}_auroc {auroc:.4f}')
            self.logger.info(f'batch_train_{task_type}_auprc {auprc:.4f}')
        else:
            self.logger.info(f'{dataset_key}_batch_train_{task_type}_auroc {auroc:.4f} {self.args.local_rank} {self.dp_rank}')
            self.logger.info(f'{dataset_key}_batch_train_{task_type}_auprc {auprc:.4f} {self.args.local_rank} {self.dp_rank}')

        # wandb logging:
        self.wandb.log(self.state.global_step,
            {
                (f"batch_train_{task_type}_loss" if dataset_key is None else f'{dataset_key}_batch_train_{task_type}_loss'): loss.item(),
                (f'batch_train_{task_type}_auroc' if dataset_key is None else f'{dataset_key}_batch_train_{task_type}_auroc'): auroc,
                (f'batch_train_{task_type}_auprc' if dataset_key is None else f'{dataset_key}_batch_train_{task_type}_auprc'): auprc,
            }
        )

        return loss * self.args.retrieval_loss_weight

    @torch.no_grad()
    def validation_pass(
            self,
            model: nn.Module,
            inputs,
            task_type,
            dataset_key = None,
            aaseq_embeddings = None,
        ):

        inputs = self._prepare_inputs(inputs)

        aaseq_type, text_type, relation = decompose_dataset_name(dataset_key)

        rdict = None

        if task_type == 'qa':
            out = model(inputs, return_mlm = False, retrieval = False, get_full_labels = True, aaseq_type = aaseq_type, crop_off = False)

            loss = out['outputs'].loss.detach().clone().item() # Get loss from labels already computed by Huggingface
            ppl = np.exp(loss)

            if task_type == 'qa':
                #auroc, auprc = get_qa_metrics(outputs.logits if model.causal_qa else out['qa_out'], out['full_labels'], yes_token = model.yes_token, no_token = model.no_token, causal_qa = model.causal_qa)
                acc, f1 = get_qa_metrics(out, yes_token = model.yes_token, no_token = model.no_token, answer_token = model.answer_idx)#padding_token = model.tokenizer.pad_token_id)

            rdict = {
                "loss": loss,
                "acc": acc,
                "f1": f1,
                "ppl": ppl
            }

        elif task_type == 'retrieval':

            out = model(inputs, return_mlm = False, retrieval = True, get_full_labels = False, aaseq_type = aaseq_type, crop_off = False)

            # Can move a lot of this to external functions, but putting here for now...
            loss = out["contrastive_loss"].detach().clone().item()

            tz = out["contrastive_out"]["positive"]["text"].detach().clone()
            labels = inputs["data"]["seq_idx"][inputs["target"]["seq"]["positive"]].detach().clone()
            text_idxs = inputs["data"]["text_idx"][inputs["input"]["text"]].detach().clone()

            # Float cast before normalizing 1) because it's more stable, 2) it's required for softmax below
            #pz_norm = F.normalize(aaseq_embeddings.float(), dim = -1)
            tz_norm = F.normalize(tz.float(), dim = -1)

            #["data"]["seq_idx"]: gives sequence index with respect to all proteins in database

            rdict = {
                "loss": loss,
                "text_z": tz_norm,
                "labels": labels,
                "text_id": text_idxs,
                "seq_id": labels
            }

            # rdict = {
            #     "loss": loss,
            #     "auprc": auprc,
            #     "auroc": roc_auc
            # }

        elif task_type == 'caption':

            out = model(inputs, return_mlm = False, retrieval = False, get_full_labels = True, aaseq_type = aaseq_type, crop_off = True)

            loss = out['outputs'].loss.detach().clone().item() # Get loss from labels already computed by Huggingface
            ppl = np.exp(loss)

            rdict = {
                "loss": loss,
                "ppl": ppl
            }

        else:
            raise NotImplementedError

        return rdict

    def report_validation_metrics(
            self,
            metrics_dict,
            task_type,
        ):
        '''
        Reports metrics based on the metrics dict and loss received
            - Report to: 1) wandb, 2) logger
        '''

        metric_name_list = []
        if task_type == 'qa':
            metric_name_list = QA_METRICS
        elif task_type == 'retrieval':
            metric_name_list = RETRIEVAL_METRICS
        elif task_type == 'caption':
            metric_name_list = CAPTION_METRICS

        for dkey in metrics_dict.keys():
            # Log loss:
            # loss_log_name = 'eval_{}_{}_loss'.format(dkey, task_type)
            # self.wandb.log(self.state.global_step,
            #     {
            #         loss_log_name: np.mean(loss_list),
            #     }
            # )

            # self.logger.info("[R={}] {}: {:.6f}".format(self.dp_rank, loss_log_name, np.mean(metrics_dict[dkey]['loss'])))

            # Log all other metrics:
            for m in metric_name_list:
                log_name = f"eval_{dkey}_{task_type}_{m}"
                self.wandb.log(self.state.global_step,
                    {
                        log_name: np.mean(metrics_dict[dkey][m])
                    }
                )

                self.logger.info("[R={}] {}: {:.6f}".format(self.dp_rank, log_name, np.mean(metrics_dict[dkey][m])))

        inputs = self._prepare_inputs(inputs)

        aaseq_type, text_type, relation = decompose_dataset_name(dataset_key)

        rdict = None

        if task_type == 'qa':
            out = model(inputs, return_mlm = False, retrieval = False, get_full_labels = True, aaseq_type = aaseq_type, crop_off = False)

            loss = out['outputs'].loss.detach().clone().item() # Get loss from labels already computed by Huggingface
            ppl = np.exp(loss)

            if task_type == 'qa':
                #auroc, auprc = get_qa_metrics(outputs.logits if model.causal_qa else out['qa_out'], out['full_labels'], yes_token = model.yes_token, no_token = model.no_token, causal_qa = model.causal_qa)
                acc, f1 = get_qa_metrics(out, yes_token = model.yes_token, no_token = model.no_token, answer_token = model.answer_idx)#padding_token = model.tokenizer.pad_token_id)

            rdict = {
                "loss": loss,
                "acc": acc,
                "f1": f1,
                "ppl": ppl
            }

        elif task_type == 'retrieval':

            out = model(inputs, return_mlm = False, retrieval = True, get_full_labels = False, aaseq_type = aaseq_type, crop_off = False)

            # Can move a lot of this to external functions, but putting here for now...
            loss = out["contrastive_loss"].detach().clone().item()

            tz = out["contrastive_out"]["positive"]["text"].detach().clone()
            labels = inputs["data"]["seq_idx"][inputs["target"]["seq"]["positive"]].detach().clone()

            # Float cast before normalizing 1) because it's more stable, 2) it's required for softmax below
            #pz_norm = F.normalize(aaseq_embeddings.float(), dim = -1)
            tz_norm = F.normalize(tz.float(), dim = -1)

            #["data"]["seq_idx"]: gives sequence index with respect to all proteins in database

            rdict = {
                "loss": loss,
                "text_z": tz_norm,
                "labels": labels
            }

            # rdict = {
            #     "loss": loss,
            #     "auprc": auprc,
            #     "auroc": roc_auc
            # }

        elif task_type == 'caption':

            out = model(inputs, return_mlm = False, retrieval = False, get_full_labels = True, aaseq_type = aaseq_type, crop_off = True)

            loss = out['outputs'].loss.detach().clone().item() # Get loss from labels already computed by Huggingface
            ppl = np.exp(loss)

            rdict = {
                "loss": loss,
                "ppl": ppl
            }

        else:
            raise NotImplementedError

        return rdict

    def report_validation_metrics(
            self,
            metrics_dict,
            task_type,
        ):
        '''
        Reports metrics based on the metrics dict and loss received
            - Report to: 1) wandb, 2) logger
        '''

        metric_name_list = []
        if task_type == 'qa':
            metric_name_list = QA_METRICS
        elif task_type == 'retrieval':
            metric_name_list = RETRIEVAL_METRICS
        elif task_type == 'caption':
            metric_name_list = CAPTION_METRICS

        for dkey in metrics_dict.keys():
            # Log loss:
            # loss_log_name = 'eval_{}_{}_loss'.format(dkey, task_type)
            # self.wandb.log(self.state.global_step,
            #     {
            #         loss_log_name: np.mean(loss_list),
            #     }
            # )

            # self.logger.info("[R={}] {}: {:.6f}".format(self.dp_rank, loss_log_name, np.mean(metrics_dict[dkey]['loss'])))

            # Log all other metrics:
            for m in metric_name_list:
                log_name = f"eval_{dkey}_{task_type}_{m}"
                self.wandb.log(self.state.global_step,
                    {
                        log_name: np.mean(metrics_dict[dkey][m])
                    }
                )

                self.logger.info("[R={}] {}: {:.6f}".format(self.dp_rank, log_name, np.mean(metrics_dict[dkey][m])))

        if (val_caption_loader is not None):
            barrier()

            caption_dkeys = val_caption_loader.dataset.dataset_keys

            caption_metrics_dict = {dk: {s:[] for s in CAPTION_METRICS} for dk in caption_dkeys}

            # Same rough logic as QA:

            for i, (dkey, model_input) in enumerate(val_caption_loader):
                rdict = self.validation_pass(self.model, inputs = model_input, task_type = 'caption', dataset_key = dkey)

                #self.logger.info(f"Caption Eval Step {i}, dkey = {dkey}, loss = {rdict['loss']:.6f}")

                for cm in CAPTION_METRICS:
                    caption_metrics_dict[dkey][cm].append(rdict[cm])

            # Report to wandb and logger:
            self.report_validation_metrics(caption_metrics_dict, task_type = "caption")

    @torch.no_grad()
    def _run_eval(self,
            val_protein_mlm_loader,
            val_qa_loader,
            val_retrieval_loader,
            val_retrieval_protein_loader,
            val_caption_loader,
        ):
        qa_metrics_dict, retrieval_metrics_dict, caption_metrics_dict = {}, {}, {}
        if (val_qa_loader is not None):
            barrier()
            set_prot_lora_group(self.model, 0)
            set_text_lora_group(self.model, 0)
            qa_dkeys = val_qa_loader.dataset.dataset_keys
            qa_metrics_dict = {dk: {s:[] for s in QA_METRICS} for dk in qa_dkeys}

            for i, (dkey, model_input) in enumerate(val_qa_loader):
                rdict = self.validation_pass(self.model, inputs = model_input, task_type = 'qa', dataset_key = dkey)

                self.logger.info(f"QA Eval Step {i}, dkey = {dkey}, loss = {rdict['loss']:.6f}")

                #qa_loss_list.append(rdict["loss"])

                for qam in QA_METRICS:
                    qa_metrics_dict[dkey][qam].append(rdict[qam])
                # if i > 200:
                #     break

            # Report to wandb and logger:
            self.report_validation_metrics(qa_metrics_dict, task_type = "qa")

        if (val_retrieval_loader is not None):
            set_prot_lora_group(self.model, 1)
            set_text_lora_group(self.model, 1)
            barrier()
            retrieval_dkeys = val_retrieval_loader.dataset.dataset_keys
            # print(retrieval_dkeys, "val_retrieval_loader")
            retrieval_out_dict = {dk: {s:[] for s in ["loss", "text_z", "labels"]} for dk in retrieval_dkeys}

            # Run all protein sequences first, then all_gather protein representations:
            pz = []
            for model_input in val_retrieval_protein_loader:
                seq_input = torch.LongTensor(model_input["indices"]) if self.model.config.use_aaseq_embeddings else model_input["data"]
                out = self.model.forward_sequences(seq_input = self._prepare_inputs(seq_input))
                pz.append(out["shared"].detach().clone())

            pz_tensor = torch.cat(pz, dim=0)

            # All gather prep:
            pz_all = [torch.empty_like(pz_tensor)] * self.dp_world_size

            # DOES NOT gather with backprop, but more stable than nn version:
            barrier() # all_gather calls barrier anyways, but do this just to make sure
            torch.distributed.all_gather(pz_all, pz_tensor) # In-place op

            pz_all = torch.cat(pz_all, dim = 0) # Concat works because SequentialDistributedSampler ensures contiguous chunks ordered by rank

            # Then run batches of text through, cache predictions and calculate:

            for i, (dkey, model_input) in enumerate(val_retrieval_loader):
                rdict = self.validation_pass(self.model, inputs = model_input,
                    task_type = 'retrieval', dataset_key = dkey, aaseq_embeddings = pz_all)

                for ram in ["loss", "text_z", "labels"]:
                    retrieval_out_dict[dkey][ram].append(rdict[ram])
                # if i > 200:
                #     break

            retrieval_metrics_dict = {} #{dk: {s:[] for s in RETRIEVAL_METRICS} for dk in retrieval_dkeys}

            # Normalize all protein embeddings:
            pz_norm = F.normalize(pz_all.float(), dim=-1)

            for dkey in retrieval_out_dict.keys():
                # if len(retrieval_out_dict[dkey]["text_z"]) <= 0:
                #     continue
                tz_norm = torch.cat(retrieval_out_dict[dkey]["text_z"], dim=0) # Already normalized

                predictions = tz_norm @ pz_norm.t() # Cosine similarity

                preds_np = predictions.softmax(dim=-1).cpu().numpy() # Softmax over predictions to turn into multiclass

                all_labels = torch.cat(retrieval_out_dict[dkey]["labels"]) # Should be one-dimensional
                # all_seq_ids = torch.cat(retrieval_out_dict[dkey]["seq_id"])
                # all_text_ids = torch.cat(retrieval_out_dict[dkey]["text_id"])
                labels_one_hot = F.one_hot(all_labels, num_classes = preds_np.shape[1]).cpu().numpy()

                # self.logger.info(f'[R={self.dp_rank}] All_labels: {all_labels}')
                # self.logger.info(f'[R={self.dp_rank}] one_hot: {labels_one_hot}')

                # Calc metrics:
                roc_auc = roc_auc_score(labels_one_hot.flatten(), preds_np.flatten(), average='macro', multi_class='ovr')
                auprc = average_precision_score(labels_one_hot.flatten(), preds_np.flatten(), average='macro')

                # WILL NEED TO CHANGE THIS EXPLICITLY IF YOU WANT MORE METRICS
                retrieval_metrics_dict[dkey] = {
                    "loss": retrieval_out_dict[dkey]["loss"],
                    "auprc": [auprc],
                    "auroc": [roc_auc],
                }

            self.report_validation_metrics(retrieval_metrics_dict, task_type="retrieval")

            # Call metrics in inner loop after getting scores from text encoder - macro averaging

        if (val_caption_loader is not None):
            barrier()
            set_prot_lora_group(self.model, 2)
            set_text_lora_group(self.model, 2)
            caption_dkeys = val_caption_loader.dataset.dataset_keys
            print(caption_dkeys, "val_caption_loader")
            caption_metrics_dict = {dk: {s:[] for s in CAPTION_METRICS} for dk in caption_dkeys}

            # Same rough logic as QA:

            for i, (dkey, model_input) in enumerate(val_caption_loader):
                rdict = self.validation_pass(self.model, inputs = model_input, task_type = 'caption', dataset_key = dkey)

                #self.logger.info(f"Caption Eval Step {i}, dkey = {dkey}, loss = {rdict['loss']:.6f}")

                for cm in CAPTION_METRICS:
                    caption_metrics_dict[dkey][cm].append(rdict[cm])

                # if i > 200:
                #     break
            # Report to wandb and logger:
            self.report_validation_metrics(caption_metrics_dict, task_type = "caption")
        return qa_metrics_dict, retrieval_metrics_dict, caption_metrics_dict

    @torch.no_grad()
    def _run_eval_OLD(self,
            val_protein_mlm_loader,
            val_qa_loaders,
            val_retrieval_loaders,
            val_retrieval_protein_loaders,
            val_caption_loaders,
            step = None,
        ):
        # TODO: fix this to work with IT ##################################
        """Perform evaluation on the validation set"""

        self.logger.info("Evaluation:")

        # Workflow:
        # test_preds = get_testing_predictions()
        # metrics = protein_retrieval_eval_from_embeddings()
        # Report metrics via logger

        if self.data_args.use_qa:
            batch_size = self.args.qa_batch_size
            max_num_pos_samples = self.args.eval_max_number
            for dkey, dataset in val_qa_loaders.items():
                print('Evaluation:', dkey)
                test_preds, reference_indices = get_testing_predictions(
                    model = self.model,
                    #dataloader = val_qa_loader,
                    dataloader = dataset,
                    protein_dataloader = None,
                    batch_size = batch_size,
                    task_type = 'qa',
                    max_num_pos_samples = max_num_pos_samples, # TODO: get
                    old_eval = False,
                )

                acc, f1 = get_qa_metrics_from_preds(
                    pred_toks = test_preds[0],
                    y_toks = test_preds[1],
                    yes_token = self.model.yes_token,
                    no_token = self.model.no_token,
                )

                self.logger.info(f"{dkey}_eval_qa_acc {acc:.4f}")
                self.logger.info(f"{dkey}_eval_qa_f1 {f1:.4f}")

                self.wandb.log(self.state.global_step,
                    {
                        f'{dkey}_eval_qa_acc': acc,
                        f'{dkey}_eval_qa_f1': f1,
                    }
                )

        # Retrieval evaluation:
        if self.data_args.use_retrieval:
            batch_size = None
            max_num_pos_samples = None # Keep None for now - can expand if we have a dataset that needs it
            for dkey, ret_loader in val_retrieval_loaders.items():
                prot_ret_loader = val_retrieval_protein_loaders[dkey]
                test_preds, reference_indices = get_testing_predictions(
                    model = self.model,
                    #dataloader = val_retrieval_loader,
                    dataloader = ret_loader,
                    #protein_dataloader = val_retrieval_protein_loader,
                    protein_dataloader = prot_ret_loader,
                    batch_size = batch_size,
                    task_type = 'retrieval',
                    max_num_pos_samples = max_num_pos_samples,
                    old_eval = False,
                )

                # import ipdb; ipdb.set_trace()

                # relation_fname = get_relation_fname(
                #     aaseq_type = None,
                #     text_type = dkey,
                #     text_split_method = None,
                #     shot_level = None,
                #     split = 'val',
                # )
                relation_df = ret_loader.dataset.full_aaseq_text_relations_eval

                metric_dict = protein_retrieval_eval_from_embeddings(
                    text_embeds = test_preds[0],
                    prot_embeds = test_preds[1],
                    relation_file = relation_df,
                    protein_file = None,
                    text_file = None,
                    text_alignment_relations = reference_indices[0],
                    prot_alignment_relations = reference_indices[1].numpy(),
                    max_sep_topk = self.args.eval_retrieval_k
                )

                self.logger.info(f'{dkey}_eval_retrieval_fmax {metric_dict["Fmax"]:.4f}')
                self.logger.info(f'{dkey}_eval_retrieval_aurpc {metric_dict["AUPRC"]:.4f}')
                self.logger.info(f'{dkey}_eval_retrieval_auroc {metric_dict["AUROC"]:.4f}')
                self.logger.info(f'{dkey}_eval_retrieval_top-k-acc {metric_dict["TOPK_ACC"]:.4f}')

                self.wandb.log(self.state.global_step,
                    {
                        f'{dkey}_eval_retrieval_fmax': metric_dict["Fmax"],
                        f'{dkey}_eval_retrieval_auprc': metric_dict["AUPRC"],
                        f'{dkey}_eval_retrieval_auroc': metric_dict["AUROC"],
                        f'{dkey}_eval_retrieval_top-k-acc': metric_dict["TOPK_ACC"],
                    }
                )

        if self.data_args.use_caption:
            pass # No implementation yet - need evaluation metric for captioning


    def _early_stopping(self, eval_results, step, current_checkpoint_dir):
        # TODO: Fix this to work with IT
        # TODO: use chosen training text column here (rather than overall)?
        # TODO: investigate why overall value != value for description_combined when only one column is used
        checkpoint_metric = eval_results["auprc_protein_go_overall"]

        if checkpoint_metric - (self.state.best_metric if self.state.best_metric else 0) > self.args.early_stopping_delta:
            # Improvement detected
            self.state.best_model_checkpoint = step
            self.state.best_metric = checkpoint_metric
            # Update soft link to point to the best model checkpoint
            symlink_path = os.path.join(self.output_dir, "best_model_checkpoint")
            if os.path.exists(symlink_path):
                os.unlink(symlink_path)
            os.symlink(current_checkpoint_dir, symlink_path)

        else:
            # No improvement, check for early stopping
            steps_since_last_improvement = step - self.state.best_model_checkpoint
            if steps_since_last_improvement > self.args.early_stopping_patience:
                self.logger.info(
                    f"Stopping training because there was no improvement > {self.args.early_stopping_delta} in the last {self.args.early_stopping_patience} steps"
                )
                self.logger.info(
                    f"Best model checkpoint: {self.state.best_model_checkpoint}, with metric value: {self.state.best_metric}"
                )
                return True

        return False


    def _save_checkpoint_deepspeed(self, checkpoint_dir):

        # Save dataset progess based on local_rank:----
        # if self.qa_tracker:
        #     fname = os.path.join(checkpoint_dir, f"tracker_qa_rank={self.dp_rank}.pt")
        #     local_progress = self.qa_tracker.save_state()
        #     torch.save(local_progress, fname)
        # if self.retrieval_tracker:
        #     fname = os.path.join(checkpoint_dir, f"tracker_retrieval_rank={self.dp_rank}.pt")
        #     local_progress = self.retrieval_tracker.save_state()
        #     torch.save(local_progress, fname)
        # if self.caption_tracker:
        #     fname = os.path.join(checkpoint_dir, f"tracker_caption_rank={selfdp_rank}.pt")
        #     local_progress = self.caption_tracker.save_state()
        #     torch.save(local_progress, fname)

        self.model_engine.save_checkpoint(checkpoint_dir, exclude_frozen_parameters=False)

        if self.args.global_rank in {-1, 0}: # If in main local rank, also save configs
            step_entry = {
                'global_step': self.state.global_step + self.dp_world_size, # Add because we need to start from one ahead
            }
            save_progress = self.training_progress.copy()
            save_progress.update(step_entry)
            # Save training state, which for now will just contain the step
            with open(os.path.join(checkpoint_dir, SAVE_TRAINING_STATE_FNAME), 'w') as f:
                json.dump(save_progress, f)
            #torch.save(os.path.join(checkpoint_dir, SAVE_TRAINING_STATE_FNAME))

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(checkpoint_dir, SAVE_TRAINING_ARGS_FNAME))
            torch.save(self.model_args, os.path.join(checkpoint_dir, SAVE_MODEL_ARGS_FNAME))
            torch.save(self.data_args, os.path.join(checkpoint_dir, SAVE_DATA_ARGS_FNAME))

        barrier()

    def _load_from_checkpoint_deepspeed(self, resume_from_checkpoint):

        # Load via deepspeed:
        #load_path, _ = self.model_engine.load_checkpoint(resume_from_checkpoint, load_module_strict=False)

        # Loading function to control for differences in ZeRO stage
        source_zero_stage, target_zero_stage = detect_zero_stages(self.args)
        self.model_engine, self.optimizer = deepspeed_init_with_checkpoint(
            train_args = self.args,
            model_args = self.model_args,
            data_args = self.data_args,
            model = self.model,
            source_zero_stage = source_zero_stage,
            target_zero_stage = target_zero_stage,
            logger = self.logger,
        )

        self.logger.info("WARNING: Loading checkpoint without strict loading rule, certain new modules might not be loaded correctly if the model args don't match between checkpoint and loading process config")
        #self.logger.info(f"Load load_path on rank={self.args.local_rank}: {load_path}")

        # Load training state:
        training_state = json.load(open(os.path.join(resume_from_checkpoint, SAVE_TRAINING_STATE_FNAME)))
        if self.args.resume_training_progress:
            self.state.global_step = training_state['global_step']

        del training_state['global_step'] # Do this to avoid carrying over step in training_state, will be merged back at save time

        return training_state

    def _get_dataloaders(self):
        (
            train_protein_mlm_loader,
            train_qa_loader,
            train_retrieval_loader,
            train_caption_loader,
            val_protein_mlm_loader,
            val_qa_loader,
            val_retrieval_loader,
            val_retrieval_protein_loader,
            val_caption_loader,
        ) = [None] * 9

        # get samplers
        (
            train_protein_mlm_sampler,
            train_qa_sampler,
            train_retrieval_sampler,
            train_caption_sampler,
            val_protein_mlm_sampler,
            val_qa_sampler,
            val_retrieval_sampler,
            val_caption_sampler,
        ) = self._get_samplers()

        # `batch_size` and `drop_last` are fed into `sampler` to create a `batch_sampler` (`shuffle` is useless when `sampler` exists). The output (list) of `batch_sampler` is then fed into `collate_fn` to create a batch.
        if self.train_protein_mlm_datasets:
            train_protein_mlm_loader = DataLoader(
                self.train_protein_mlm_dataset,
                batch_size=self.args.protein_mlm_batch_size,
                # shuffle=True,
                collate_fn=self.protein_mlm_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                drop_last=True,
                sampler=train_protein_mlm_sampler,
            )
            val_protein_mlm_loader = DataLoader(
                self.val_protein_mlm_dataset,
                batch_size=self.args.protein_mlm_batch_size,
                # shuffle=False,
                collate_fn=self.protein_mlm_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                sampler=val_protein_mlm_sampler,
            )
        # TODO: Make below take in general datasets
        if self.train_qa_dataset:

            train_qa_loader = DataLoader(
                self.train_qa_dataset,
                batch_size=1,
                # shuffle=True,
                collate_fn=self.qa_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                drop_last=True,
                sampler=train_qa_sampler,
            )
            val_qa_loader = DataLoader(
                self.val_qa_dataset,
                batch_size=1,
                # shuffle=False,
                collate_fn=self.qa_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                sampler=val_qa_sampler,
            )
        if self.train_retrieval_dataset:

            train_retrieval_loader = DataLoader(
                self.train_retrieval_dataset,
                batch_size=1,
                # shuffle=True,
                collate_fn=self.retrieval_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                drop_last=True,
                sampler=train_retrieval_sampler,
            )

            val_retrieval_loader = DataLoader(
                self.val_retrieval_dataset,
                batch_size=1,
                # shuffle=True,
                collate_fn=self.retrieval_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                drop_last=True,
                sampler=val_retrieval_sampler,
            )

            # Get whole protein dataloader:
            protein_eval_dataset = ProteinDataset(all_data=True)

            protein_eval_collator = ProteinMLMCollator( # Repurpose MLM collator without MLM for evaluation
                       data_dir = self.data_args.data_dir,
                       is_protein_tokenized = False,
                       protein_tokenizer = Alphabet.from_architecture(self.model_args.protein_tokenizer_name), # CHECK THIS
                       max_protein_len = self.model_args.max_protein_len,
                       mlm = False
                   )

            protein_eval_sampler = SequentialDistributedSampler(
                protein_eval_dataset,
                num_replicas = self.dp_world_size,
                rank = self.dp_rank,
            )

            val_retrieval_protein_loader = DataLoader(
                protein_eval_dataset,
                batch_size = self.args.eval_batch_size,
                collate_fn = protein_eval_collator,
                num_workers = self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                sampler = protein_eval_sampler,
                drop_last = False,
            )

            # TODO: Fill for validation
            # for key, dataset in self.val_retrieval_datasets.items():
            #     val_retrieval_loaders[key] = DataLoader(
            #         dataset,
            #         batch_size=self.args.retrieval_batch_size,
            #         # shuffle=False,
            #         drop_last = False,
            #         collate_fn=self.retrieval_collators[key],
            #         num_workers=self.args.num_workers,
            #         pin_memory=(self.args.world_size<=1),
            #         sampler=val_retrieval_samplers[key],
            #     )

            #     protein_dataset = ProteinEvalDataset(
            #        dataset.unique_aaseq,
            #     )
            #     protein_eval_collator = None
            #     if not self.model_args.use_aaseq_embeddings:
            #        protein_eval_collator = ProteinMLMCollator( # Repurpose MLM collator without MLM for evaluation
            #            data_dir = self.data_args.data_dir,
            #            is_protein_tokenized = self.model_args.is_protein_tokenized,
            #            protein_tokenizer = Alphabet.from_architecture(self.model_args.protein_tokenizer_name),
            #            max_protein_len = self.model_args.max_protein_len,
            #            mlm = False
            #        )
            #     val_retrieval_protein_loaders[key] = DataLoader(
            #        protein_dataset,
            #        batch_size = self.args.retrieval_batch_size,
            #        num_workers = self.args.num_workers,
            #        collate_fn = protein_eval_collator,
            #        drop_last = False,
            #        shuffle = False,
            #        pin_memory = False
            #     )

        if self.train_caption_dataset:

            train_caption_loader = DataLoader(
                self.train_caption_dataset,
                batch_size=1,
                collate_fn=self.caption_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                drop_last=True,
                sampler=train_caption_sampler,
            )
            val_caption_loader = DataLoader(
                self.val_caption_dataset,
                batch_size=1,
                collate_fn=self.caption_collator,
                num_workers=self.args.num_workers,
                pin_memory=(self.dp_world_size<=1),
                sampler=val_caption_sampler,
            )

        return (
            train_protein_mlm_loader,
            train_qa_loader,
            train_retrieval_loader,
            train_caption_loader,
            val_protein_mlm_loader,
            val_qa_loader,
            val_retrieval_loader,
            val_retrieval_protein_loader,
            val_caption_loader,
        )

    def _get_samplers(self):

        samplers_dict = {}
        for dataset, sampler_key in zip([
            {'mlm':self.train_protein_mlm_datasets},
            self.train_qa_dataset,
            self.train_retrieval_dataset,
            self.train_caption_dataset,
        ], ['mlm', 'qa', 'retrieval', 'caption',]):
            #for key, dataset in datasets.items():
            if not dataset:  # if dataset is None
                samplers_dict[sampler_key] = None
                continue

            generator = None
            if self.dp_world_size <= 1:
                generator = torch.Generator()
                generator.manual_seed(
                    int(torch.empty((), dtype=torch.int64).random_().item())
                )
                samplers_dict[sampler_key] = RandomSampler(dataset, generator=generator)
            else:
                samplers_dict[sampler_key] = DistributedSamplerResume(
                    dataset,
                    num_replicas=self.dp_world_size,
                    rank=self.dp_rank,  # property of TrainingArguments class
                    seed=self.args.seed,
                )

        sampler_val_dict = {}
        for dataset, sampler_key in zip([
            {'mlm':self.val_protein_mlm_dataset},
            self.val_qa_dataset,
            self.val_retrieval_dataset,
            self.val_caption_dataset,
        ], ['mlm', 'qa', 'retrieval', 'caption',]):
            if not dataset:  # if dataset is None
                sampler_val_dict[sampler_key] = None
                continue

            #sampler_val_dict[sampler_key] = SequentialSampler(dataset) # Don't do distributed sampling for validation for now

            sampler_val_dict[sampler_key] = DistributedSampler(
                dataset,
                num_replicas = self.dp_world_size,
                rank = self.dp_rank,
                seed = self.args.seed,
                shuffle = False,
            )

            #TODO: Distributed sampling for validation
            # if self.args.world_size <= 1:
            #     samplers[key] = SequentialSampler(dataset)
            # else:
            #     samplers[key] = SequentialDistributedSampler(
            #         dataset,
            #         # num_replicas=self.args.world_size,  # TODO: investigate need or not?
            #         # rank=self.args.process_index,  # property of TrainingArguments class
            #     )

        train_mlm_sampler = samplers_dict['mlm']
        train_qa_sampler = samplers_dict['qa']
        train_retrieval_sampler = samplers_dict['retrieval']
        train_caption_sampler = samplers_dict['caption']

        val_mlm_sampler = sampler_val_dict['mlm']
        val_qa_sampler = sampler_val_dict['qa']
        val_retrieval_sampler = sampler_val_dict['retrieval']
        val_caption_sampler = sampler_val_dict['caption']

        return train_mlm_sampler, train_qa_sampler, train_retrieval_sampler, train_caption_sampler, val_mlm_sampler, val_qa_sampler, val_retrieval_sampler, val_caption_sampler

    def create_optimizer(self):
        """
        Rewrites the create_optimizer method of Trainer class to enable independent learning rates for different parts of the model.

        TODO: Change to work with projection layers
        """
        assert self.optimizer is None

        protein_encoder_decay_params, protein_encoder_no_decay_params, text_encoder_decay_params, text_encoder_no_decay_params, embedding_params, projection_params, contrastive_params = self.model.get_grouped_parameter_names()

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in protein_encoder_no_decay_params
                ],
                "weight_decay": 0.0,
                "lr": self.args.protein_encoder_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in protein_encoder_decay_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.protein_encoder_lr,
            },
            {
                'params': [p for n, p in self.model.named_parameters() if n in text_encoder_no_decay_params],
                'weight_decay': 0.0,
                'lr': self.args.text_encoder_lr
            },
            {
                'params': [p for n, p in self.model.named_parameters() if n in text_encoder_decay_params],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.text_encoder_lr
            },
            {
                'params': [
                    p
                    for n, p in self.model.named_parameters()
                    if n in embedding_params
                ],
                'weight_decay': self.args.weight_decay,
                'lr': self.args.embedding_lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in projection_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.projection_lr,
            },
            { # Contrastive learning head parameters:
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n in contrastive_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.contrastive_lr,
            },
        ]

        OPTIMIZER_CLASSES = {"radam": RAdam, "adamw": AdamW, "adafactor": Adafactor}

        # TODO: add support for Adafactor and RAdam
        assert self.args.optimizer_type == "adamw"
        optimizer_class = OPTIMIZER_CLASSES[self.args.optimizer_type]
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }

        self.optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)

    def create_scheduler(self, num_training_steps, optimizer = None):
        """
        Rewrites the create_scheduler method of Trainer class to enable schedulers for different parts of the model.
        """
        if self.lr_scheduler is None:
            # TODO: Investigate whether this is really needed
            if self.args.deepspeed:
                num_training_steps = (
                    num_training_steps // self.args.gradient_accumulation_steps
                    + int(
                        num_training_steps % self.args.gradient_accumulation_steps > 0
                    )
                )

            # TODO: Separate out warmup step args for different parts of the model
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_protein_encoder_warmup_steps=self.args.warmup_steps,
                num_text_encoder_warmup_steps=self.args.warmup_steps,
                num_embedding_warmup_steps=self.args.warmup_steps,
                num_decoder_warmup_steps=self.args.warmup_steps,
                num_contrastive_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

    # FUNCTION DOESN'T WORK
    # def log_gradient_norm(self, logname):

    #     for p in self.model.parameters():
    #         param_norm = p.grad.detach().data.norm(2)
    #         total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5

    #     self.wandb.log(self.state.global_step, {
    #         logname: total_norm
    #     })