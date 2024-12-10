import os, sys, logging, random, json, pickle
from typing import Any, Tuple, Callable, Union, Dict, Optional, Iterable, List
from collections.abc import Mapping
import shutil

# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4"

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
from transformers.trainer_pt_utils import get_parameter_names, IterableDatasetShard, SequentialDistributedSampler
from transformers.training_args import ShardedDDPOption
from transformers.trainer_callback import TrainerState

from procyon.model.model import TxPLM
from scripts.run_general_eval import run_general_eval

from procyon.training.training_args import TrainArgs, DataArgs, ModelArgs
from downstream.on_the_fly_eval.unsupervised.eval_retrieval import run_eval_retrieval

from procyon.training.wandb_logger import WandbLogger
from dataclasses import asdict, is_dataclass
from procyon.data.data_utils import DATA_DIR
import torch.distributed as dist


from procyon.training.train_utils import (
    get_mlm_loss,
    get_kepler_loss,
    get_cl_metrics,
    get_scheduler,
    report_train_smoothed_metrics,
    unwrap_model,
)

from procyon.model.model import SAVE_TRAINING_STATE_FNAME, SAVE_CONFIG_FNAME
SAVE_TRAINING_ARGS_FNAME = "training_args.pt"
SAVE_MODEL_ARGS_FNAME = "model_args.pt"
SAVE_DATA_ARGS_FNAME = "data_args.pt"

class TxPLMTrainer(Trainer):
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
        text_cl_collator: Any,
        protein_go_collator: Any,
        protein_protein_collator: Any,
        # pfam_collator: Any,
        domain_go_collator: Any,
        domain_pfam_collator: Any,
        output_dir: str,
        device: torch.device,
        logger: logging.Logger,
        wandb: WandbLogger,
    ):
        super().__init__(model=model, args=args)
        (
            self.train_protein_mlm_dataset,
            self.train_text_cl_dataset,
            self.train_protein_go_dataset,
            self.train_protein_protein_dataset,
            # self.train_pfam_dataset,
            self.train_domain_go_dataset,
            self.train_domain_pfam_dataset,
        ) = train_datasets

        (
            self.val_protein_mlm_dataset,
            self.val_text_cl_dataset,
            self.val_protein_go_dataset,
            self.val_protein_protein_dataset,
            # self.val_pfam_dataset,
            self.val_domain_go_dataset,
            self.val_domain_pfam_dataset,
        ) = val_datasets

        # TODO: merge into a single param as above
        self.protein_mlm_collator = protein_mlm_collator
        self.text_cl_collator = text_cl_collator
        self.protein_go_collator = protein_go_collator
        self.protein_protein_collator = protein_protein_collator
        # self.pfam_collator = pfam_collator
        self.domain_go_collator = domain_go_collator
        self.domain_pfam_collator = domain_pfam_collator

        self.output_dir = output_dir
        self.device = device
        self.logger = logger
        self.wandb = wandb

        self.data_args = data_args
        self.model_args = model_args

        self.last_ephemeral_checkpoint = None 


    def train(
        self,
    ):
        # get dataloaders
        (
            train_protein_mlm_loader,
            train_text_cl_loader,
            train_protein_go_loader,
            train_protein_protein_loader,
            # train_pfam_loader,
            train_domain_go_loader,
            train_domain_pfam_loader,
            val_protein_mlm_loader,
            val_text_cl_loader,
            val_protein_go_loader,
            val_protein_protein_loader,
            # val_pfam_loader,
            val_domain_go_loader,
            val_domain_pfam_loader,
        ) = self._get_dataloaders()

        # calculate total batch size
        total_protein_mlm_batch_size = (
            self.args.protein_mlm_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        total_text_cl_batch_size = (
            self.args.text_cl_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        total_protein_go_batch_size = (
            self.args.protein_go_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        total_protein_protein_batch_size = (
            self.args.protein_protein_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        # total_pfam_batch_size = (
        #     self.args.pfam_batch_size
        #     * self.args.gradient_accumulation_steps
        #     * self.args.world_size
        # )
        total_domain_go_batch_size = (
            self.args.domain_go_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )
        total_domain_pfam_batch_size = (
            self.args.domain_pfam_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )

        # calculate epochs each type of data will be trained for
        num_protein_mlm_update_steps_per_epoch = (
            max(
                len(train_protein_mlm_loader) // self.args.gradient_accumulation_steps,
                1,
            )
            if train_protein_mlm_loader
            else 0
        )
        num_text_cl_update_steps_per_epoch = (
            max(
                len(train_text_cl_loader) // self.args.gradient_accumulation_steps,
                1,
            )
            if train_text_cl_loader
            else 0
        )
        num_protein_go_update_steps_per_epoch = (
            max(
                len(train_protein_go_loader) // self.args.gradient_accumulation_steps, 1
            )
            if train_protein_go_loader
            else 0
        )
        num_protein_protein_update_steps_per_epoch = (
            max(
                len(train_protein_protein_loader)
                // self.args.gradient_accumulation_steps,
                1,
            )
            if train_protein_protein_loader
            else 0
        )
        # num_pfam_update_steps_per_epoch = (
        #     max(len(train_pfam_loader) // self.args.gradient_accumulation_steps, 1)
        #     if train_pfam_loader
        #     else 0
        # )
        num_domain_go_update_steps_per_epoch = (
            max(
                len(train_domain_go_loader) // self.args.gradient_accumulation_steps, 1
            )
            if train_domain_go_loader
            else 0
        )
        num_domain_pfam_update_steps_per_epoch = (
            max(
                len(train_domain_pfam_loader) // self.args.gradient_accumulation_steps, 1
            )
            if train_domain_pfam_loader
            else 0
        )

        assert self.args.max_steps > 0
        num_protein_mlm_epochs = (
            self.args.max_steps / num_protein_mlm_update_steps_per_epoch
            if num_protein_mlm_update_steps_per_epoch
            else 0
        )
        num_text_cl_epochs = (
            self.args.max_steps / num_text_cl_update_steps_per_epoch
            if num_text_cl_update_steps_per_epoch
            else 0
        )
        num_protein_go_epochs = (
            self.args.max_steps / num_protein_go_update_steps_per_epoch
            if num_protein_go_update_steps_per_epoch
            else 0
        )
        num_protein_protein_epochs = (
            self.args.max_steps / num_protein_protein_update_steps_per_epoch
            if num_protein_protein_update_steps_per_epoch
            else 0
        )
        # num_pfam_epochs = (
        #     self.args.max_steps / num_pfam_update_steps_per_epoch
        #     if num_pfam_update_steps_per_epoch
        #     else 0
        # )
        num_domain_go_epochs = (
            self.args.max_steps / num_domain_go_update_steps_per_epoch
            if num_domain_go_update_steps_per_epoch
            else 0
        )
        num_domain_pfam_epochs = (
            self.args.max_steps / num_domain_pfam_update_steps_per_epoch
            if num_domain_pfam_update_steps_per_epoch
            else 0
        )

        # TODO: enable FSDP
        delay_optimizer_creation = (
            self.sharded_ddp is not None and self.sharded_ddp != ShardedDDPOption.SIMPLE
        )
        if self.args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self,
                num_training_steps=self.args.max_steps,
                resume_from_checkpoint=self.args.resume_from_checkpoint,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer()
            self.create_scheduler(num_training_steps=self.args.max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = False
        # last_improvement: global_step of most recent measured checkpoint that improved the validation performance metric (for early stopping)
        self.state.last_improvement = 0
        
        print('global_step:', self.state.global_step)
        print('world_size:', self.args.world_size)
        print('local_rank:', self.args.local_rank)
        print('#devices:', torch.cuda.device_count())

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=self.args.max_steps)

        # Handled in _load_from_checkpoint
        # TODO: change to use transformers code
        # Check if saved optimizer or scheduler states exist
        # self._load_optimizer_and_scheduler(self.args.resume_from_checkpoint)

        # TODO: also handle lr_scheduler??
        # TODO: may not work if we actually wrap model
        if self.args.resume_from_checkpoint:
            self._load_from_checkpoint(self.args.resume_from_checkpoint, model=self.model_wrapped, optimizer=self.optimizer)

        # wrap model with either DDP, Deepspeed, ShardedDDP or FSDP
        model = self._wrap_model(self.model_wrapped)
        if model is not self.model:
            self.model_wrapped = model

        # TODO: save scheduler state instead (less hacky)
        # Update LR scheduler to current step
        if self.args.resume_from_checkpoint:
            self.logger.info("Updating LR scheduler to current step")
            for i in range(self.state.global_step):
                self.lr_scheduler.step()

        # log info
        self.logger.info(
            # f"Number of protein MLM samples: {len(self.train_protein_mlm_dataset) if self.train_protein_mlm_dataset else 0}\nNumber of protein GO samples: {len(self.train_protein_go_dataset) if self.train_protein_go_dataset else 0}\nNumber of protein-protein samples: {len(self.train_protein_protein_dataset) if self.train_protein_protein_dataset else 0}\nNumber of Pfam samples: {len(self.train_pfam_dataset) if self.train_pfam_dataset else 0}\n"
            f"Number of protein MLM samples: {len(self.train_protein_mlm_dataset) if self.train_protein_mlm_dataset else 0}\nNumber of text CL samples: {len(self.train_text_cl_dataset) if self.train_text_cl_dataset else 0}\nNumber of protein GO samples: {len(self.train_protein_go_dataset) if self.train_protein_go_dataset else 0}\nNumber of protein-protein samples: {len(self.train_protein_protein_dataset) if self.train_protein_protein_dataset else 0}\nNumber of domain GO samples: {len(self.train_domain_go_dataset) if self.train_domain_go_dataset else 0}\nNumber of domain Pfam samples: {len(self.train_domain_pfam_dataset) if self.train_domain_pfam_dataset else 0}\n"
        )

        self.logger.info(
            # f"Total protein MLM batch size: {total_protein_mlm_batch_size}\nTotal protein GO batch size: {total_protein_go_batch_size}\nTotal protein-protein batch size: {total_protein_protein_batch_size}\nTotal Pfam batch size: {total_pfam_batch_size}\n"
            f"Total protein MLM batch size: {total_protein_mlm_batch_size}\nTotal text CL batch size: {total_text_cl_batch_size}\nTotal protein GO batch size: {total_protein_go_batch_size}\nTotal protein-protein batch size: {total_protein_protein_batch_size}\nTotal domain GO batch size: {total_domain_go_batch_size}\nTotal domain Pfam batch size: {total_domain_pfam_batch_size}\n"
        )
        

        self.logger.info(
            # f"Number of protein MLM epochs: {num_protein_mlm_epochs}\nNumber of protein GO epochs: {num_protein_go_epochs}\nNumber of protein-protein epochs: {num_protein_protein_epochs}\nNumber of Pfam epochs: {num_pfam_epochs}\n"
            f"Number of protein MLM epochs: {num_protein_mlm_epochs}\nNumber of text CL epochs: {num_text_cl_epochs}\nNumber of protein GO epochs: {num_protein_go_epochs}\nNumber of protein-protein epochs: {num_protein_protein_epochs}\nNumber of domain GO epochs: {num_domain_go_epochs}\nNumber of domain Pfam epochs: {num_domain_pfam_epochs}\n"
        )

        self.logger.info(
            f"Gradient accumulation steps: {self.args.gradient_accumulation_steps}"
        )
        self.logger.info(f"Total optimization steps = {self.args.max_steps}")

        steps_trained = self.state.global_step  # TODO: Make it an argument

        if self.args.resume_from_checkpoint is not None:
            # TODO
            pass

        # TODO: Investigate whether self.state and callback_handler stuff are needed: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/trainer.py#L1676

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = 0
        self.loss_recorder = []
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        # Skip the first steps_trained steps to get the random state of the dataloader at the right point: https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/trainer.py#L1722
        # TODO: The following code only works for epoch-based training, not for step-based training.  Revise this.
        # if not self.args.ignore_data_skip:
        #     for _ in range(steps_trained):  # only meaningful when resuming training from a checkpoint
        #         is_random_sampler = hasattr(train_protein_mlm_loader, "sampler") and isinstance(
        #             train_protein_mlm_loader.sampler, RandomSampler
        #         )
        #         if is_torch_less_than_1_11 or not is_random_sampler:
        #             # We just need to begin an iteration to create the randomization of the sampler.
        #             # That was before PyTorch 1.11 however...
        #             for _ in train_protein_mlm_loader:
        #                 break
        #         else:
        #             # Otherwise we need to call the whooooole sampler cause there is some random operation added
        #             # AT THE VERY END!
        #             _ = list(train_protein_mlm_loader.sampler)

        # TODO: fix this wrt loading from checkpoint
        if train_protein_mlm_loader is not None and isinstance(
            train_protein_mlm_loader.sampler, DistributedSampler
        ):
            train_protein_mlm_loader.sampler.set_epoch(0)
        if train_text_cl_loader is not None and isinstance(
            train_text_cl_loader.sampler, DistributedSampler
        ):
            train_text_cl_loader.sampler.set_epoch(0)
        if train_protein_go_loader is not None and isinstance(
            train_protein_go_loader.sampler, DistributedSampler
        ):
            train_protein_go_loader.sampler.set_epoch(0)
        if train_protein_protein_loader is not None and isinstance(
            train_protein_protein_loader.sampler, DistributedSampler
        ):
            train_protein_protein_loader.sampler.set_epoch(0)
            val_protein_protein_loader.sampler.set_epoch(0)
        # if train_pfam_loader is not None and isinstance(
        #     train_pfam_loader.sampler, DistributedSampler
        # ):
        #     train_pfam_loader.sampler.set_epoch(0)
        #     val_pfam_loader.sampler.set_epoch(0)
        if train_domain_go_loader is not None and isinstance(
            train_domain_go_loader.sampler, DistributedSampler
        ):
            train_domain_go_loader.sampler.set_epoch(0)
            # val_domain_go_loader.sampler.set_epoch(0)
        if train_domain_pfam_loader is not None and isinstance(
            train_domain_pfam_loader.sampler, DistributedSampler
        ):
            train_domain_pfam_loader.sampler.set_epoch(0)
            # TODO: set epochs for val samplers?
            # val_domain_pfam_loader.sampler.set_epoch(0)

        # get iter of loaders
        train_protein_mlm_iter = (
            iter(train_protein_mlm_loader) if train_protein_mlm_loader else None
        )
        train_text_cl_iter = (
            iter(train_text_cl_loader) if train_text_cl_loader else None
        )
        train_protein_go_iter = (
            iter(train_protein_go_loader) if train_protein_go_loader else None
        )
        train_protein_protein_iter = (
            iter(train_protein_protein_loader) if train_protein_protein_loader else None
        )
        # train_pfam_iter = iter(train_pfam_loader) if train_pfam_loader else None
        train_domain_go_iter = (
            iter(train_domain_go_loader) if train_domain_go_loader else None
        )
        train_domain_pfam_iter = (
            iter(train_domain_pfam_loader) if train_domain_pfam_loader else None
        )

        # get number of steps per epoch.  different from num_protein_mlm_update_steps_per_epoch because here we consider raw steps, but not update steps after considering gradient accumulation.
        num_protein_mlm_steps_per_epoch = (
            max(len(train_protein_mlm_loader), 1) if train_protein_mlm_loader else -1
        )
        num_text_cl_steps_per_epoch = (
            max(len(train_text_cl_loader), 1) if train_text_cl_loader else -1
        )
        num_protein_go_steps_per_epoch = (
            max(len(train_protein_go_loader), 1) if train_protein_go_loader else -1
        )
        num_protein_protein_steps_per_epoch = (
            max(len(train_protein_protein_loader), 1)
            if train_protein_protein_loader
            else -1
        )
        # num_pfam_steps_per_epoch = (
            # max(len(train_pfam_loader), 1) if train_pfam_loader else -1
        # )
        num_domain_go_steps_per_epoch = (
            max(len(train_domain_go_loader), 1) if train_domain_go_loader else -1
        )
        num_domain_pfam_steps_per_epoch = (
            max(len(train_domain_pfam_loader), 1) if train_domain_pfam_loader else -1
        )

        cur_protein_mlm_epoch = 0
        cur_text_cl_epoch = 0
        cur_protein_go_epoch = 0
        cur_protein_protein_epoch = 0
        # cur_pfam_epoch = 0
        cur_domain_go_epoch = 0
        cur_domain_pfam_epoch = 0

        # initialize train metrics tracker
        train_batch_smoothed_tracker = dict()
        train_batch_smoothed_tracker["mlm"] = dict()
        train_batch_smoothed_tracker["text_cl"] = dict()
        train_batch_smoothed_tracker["protein_go_relations"] = dict()
        train_batch_smoothed_tracker["protein_protein_relations"] = dict()
        # train_batch_smoothed_tracker["pfam_pfam_relations"] = dict()
        # train_batch_smoothed_tracker["pfam_protein_relations"] = dict()
        # train_batch_smoothed_tracker["pfam_go_relations"] = dict()
        train_batch_smoothed_tracker["domain_go_relations"] = dict()
        train_batch_smoothed_tracker["domain_pfam_relations"] = dict()

        for step in range(steps_trained, self.args.max_steps + 1):
            self.logger.info(f"Step {step}")
            self.state.global_step = step

            # update epoch and iterator
            if (
                num_protein_mlm_steps_per_epoch != -1
                and (step + 1) % num_protein_mlm_steps_per_epoch == 0
            ):
                cur_protein_mlm_epoch += 1
                if isinstance(train_protein_mlm_loader.sampler, DistributedSampler):
                    train_protein_mlm_loader.sampler.set_epoch(cur_protein_mlm_epoch)
                train_protein_mlm_iter = iter(train_protein_mlm_loader)
            if (
                num_text_cl_steps_per_epoch != -1
                and (step + 1) % num_text_cl_steps_per_epoch == 0
            ):
                cur_text_cl_epoch += 1
                if isinstance(train_text_cl_loader.sampler, DistributedSampler):
                    train_text_cl_loader.sampler.set_epoch(cur_text_cl_epoch)
                train_text_cl_iter = iter(train_text_cl_loader)
            if (
                num_protein_go_steps_per_epoch != -1
                and (step + 1) % num_protein_go_steps_per_epoch == 0
            ):
                cur_protein_go_epoch += 1
                if isinstance(train_protein_go_loader.sampler, DistributedSampler):
                    train_protein_go_loader.sampler.set_epoch(cur_protein_go_epoch)
                train_protein_go_iter = iter(train_protein_go_loader)
            if (
                num_protein_protein_steps_per_epoch != -1
                and (step + 1) % num_protein_protein_steps_per_epoch == 0
            ):
                cur_protein_protein_epoch += 1
                if isinstance(train_protein_protein_loader.sampler, DistributedSampler):
                    train_protein_protein_loader.sampler.set_epoch(
                        cur_protein_protein_epoch
                    )
                train_protein_protein_iter = iter(train_protein_protein_loader)
            # if (
            #     num_pfam_steps_per_epoch != -1
            #     and (step + 1) % num_pfam_steps_per_epoch == 0
            # ):
            #     cur_pfam_epoch += 1
            #     if isinstance(train_pfam_loader.sampler, DistributedSampler):
            #         train_pfam_loader.sampler.set_epoch(cur_pfam_epoch)
            #         val_pfam_loader.sampler.set_epoch(cur_pfam_epoch)
            #     train_pfam_iter = iter(train_pfam_loader)
            if (
                num_domain_go_steps_per_epoch != -1
                and (step + 1) % num_domain_go_steps_per_epoch == 0
            ):
                cur_domain_go_epoch += 1
                if isinstance(train_domain_go_loader.sampler, DistributedSampler):
                    train_domain_go_loader.sampler.set_epoch(cur_domain_go_epoch)
                train_domain_go_iter = iter(train_domain_go_loader)
            if (
                num_domain_pfam_steps_per_epoch != -1
                and (step + 1) % num_domain_pfam_steps_per_epoch == 0
            ):
                cur_domain_pfam_epoch += 1
                if isinstance(train_domain_pfam_loader.sampler, DistributedSampler):
                    train_domain_pfam_loader.sampler.set_epoch(cur_domain_pfam_epoch)
                train_domain_pfam_iter = iter(train_domain_pfam_loader)

            (
                train_protein_mlm_inputs,
                train_text_cl_inputs,
                train_protein_go_inputs,
                train_protein_protein_inputs,
                # train_pfam_inputs,
                train_domain_go_inputs,
                train_domain_pfam_inputs,
            ) = (None, None, None, None, None, None)

            # TODO: Undo (saving first batch from each dataset to improve debug iter time)
            if not self.args.overfit_first_batch:
                if train_protein_mlm_iter:
                    train_protein_mlm_inputs = next(train_protein_mlm_iter)
                if train_text_cl_iter:
                    train_text_cl_inputs = next(train_text_cl_iter)
                if train_protein_go_iter:
                    train_protein_go_inputs = next(train_protein_go_iter)
                if train_protein_protein_iter:
                    train_protein_protein_inputs = next(train_protein_protein_iter)
                # if train_pfam_iter:
                #     train_pfam_inputs = next(train_pfam_iter)
                if train_domain_go_iter:
                    train_domain_go_inputs = next(train_domain_go_iter)
                if train_domain_pfam_iter:
                    train_domain_pfam_inputs = next(train_domain_pfam_iter)
            else:
                # Load cached first_iter_inputs
                with open(f'{DATA_DIR}/cached_data/first_iter_inputs.pkl', 'rb') as f:
                    first_iter_inputs = pickle.load(f)
                # (train_protein_mlm_inputs, train_protein_go_inputs, train_protein_protein_inputs, train_pfam_inputs) = first_iter_inputs
                (_, train_text_cl_inputs, train_protein_go_inputs, _, train_domain_go_inputs, train_domain_pfam_inputs) = first_iter_inputs

            # NOTE: uncomment these to save first iter inputs (with args.overfit_first_batch = False)
            # first_iter_inputs = (
            #     train_protein_mlm_inputs, 
            #     train_text_cl_inputs,
            #     train_protein_go_inputs, 
            #     train_protein_protein_inputs, 
            #     # train_pfam_inputs,
            #     train_domain_go_inputs,
            #     train_domain_pfam_inputs,
            # )
            # with open(f'{DATA_DIR}/cached_data/first_iter_inputs.pkl', 'wb') as f:
            #     pickle.dump(first_iter_inputs, f, pickle.HIGHEST_PROTOCOL)

            # when using gradient accumulation, we do backward pass only at the last step of each accumulation, so in most cases we avoid DDP synchronization.
            if (
                ((step + 1) % self.args.gradient_accumulation_steps != 0)
                and self.args.local_rank != -1
                and self.args._no_sync_in_gradient_accumulation
            ):
                # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                with model.no_sync():
                    loss, all_loss = self.training_step(
                        model,
                        train_protein_mlm_inputs,
                        train_text_cl_inputs,
                        train_protein_go_inputs,
                        train_protein_protein_inputs,
                        # train_pfam_inputs,
                        train_domain_go_inputs,
                        train_domain_pfam_inputs,
                        train_batch_smoothed_tracker,
                    )
                    tr_loss += loss
            else:
                loss, all_loss = self.training_step(
                    model,
                    train_protein_mlm_inputs,
                    train_text_cl_inputs,
                    train_protein_go_inputs,
                    train_protein_protein_inputs,
                    # train_pfam_inputs,
                    train_domain_go_inputs,
                    train_domain_pfam_inputs,
                    train_batch_smoothed_tracker,
                )
                tr_loss += loss

            # record loss  # TODO: Investigate whether this is still compatible
            if self.args.local_rank in {-1, 0}:
                all_loss["global_step"] = step
                all_loss["learning_rate"] = self._get_learning_rate()
                all_loss = dict(all_loss)
                # print(all_loss)
                self.loss_recorder.append(all_loss)
                
                self.wandb.log(step, {
                    "protein encoder learning rate": all_loss["learning_rate"][0], 
                    "text encoder learning rate": all_loss["learning_rate"][2],
                    "embedding learning rate": all_loss["learning_rate"][-2],
                    "decoder learning rate": all_loss["learning_rate"][-1],
                })

            # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
            if self.deepspeed:
                self.deepspeed.step()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if (
                    self.args.max_grad_norm is not None
                    and self.args.max_grad_norm > 0
                    and not self.deepspeed
                ):
                    # deepspeed does its own clipping
                    # the value of self.do_grad_scaling is always the same as self.use_cuda_amp
                    if self.do_grad_scaling:
                        self.scaler.unscale_(self.optimizer)
                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm(self.args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling full precision
                        nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.args.max_grad_norm,
                        )

                # Optimizer step
                optimizer_was_run = True
                if self.deepspeed:
                    pass  # called outside the loop
                elif self.do_grad_scaling:
                    raise NotImplementedError("TODO: handle loading from checkpoint")
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()

                if optimizer_was_run and not self.deepspeed:
                    self.lr_scheduler.step()

                model.zero_grad()

                # Log number of epochs of each dataset seen so far
                self.wandb.log(step, {
                    "protein_mlm epoch": step / num_protein_mlm_update_steps_per_epoch if num_protein_mlm_update_steps_per_epoch else 0,
                    "text_cl epoch": step / num_text_cl_update_steps_per_epoch if num_text_cl_update_steps_per_epoch else 0,
                    "protein_go epoch": step / num_protein_go_update_steps_per_epoch if num_protein_go_update_steps_per_epoch else 0,
                    "protein_protein epoch": step / num_protein_protein_update_steps_per_epoch if num_protein_protein_update_steps_per_epoch else 0,
                    # "pfam epoch": step / num_pfam_update_steps_per_epoch if num_pfam_update_steps_per_epoch else 0,
                    "domain_go epoch": step / num_domain_go_update_steps_per_epoch if num_domain_go_update_steps_per_epoch else 0,
                    "domain_pfam epoch": step / num_domain_pfam_update_steps_per_epoch if num_domain_pfam_update_steps_per_epoch else 0,
                })

            # Barrier before evaluation (ensure processes are in sync)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if self.args.local_rank in {-1, 0}:
                do_eval = (step % self.args.eval_steps == 0) or (step % self.args.initial_eval_steps == 0 and step <= self.args.initial_eval_steps_limit)
                # evaluate during training (first evaluation performed right after the first step of training, for the sake of baseline)
                if (step % self.args.checkpoint_steps == 0) or do_eval:
                    current_checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{self.state.global_step}")
                    os.makedirs(current_checkpoint_dir, exist_ok=True)
                    self._save_checkpoint(current_checkpoint_dir)

                    # Delete previous ephemeral checkpoint (if any)
                    if self.last_ephemeral_checkpoint is not None:
                        shutil.rmtree(self.last_ephemeral_checkpoint)

                    # This checkpoint should be deleted next time, unless we're doing eval (eval checkpoints are persistent)
                    self.last_ephemeral_checkpoint = None if do_eval else current_checkpoint_dir

                    # Measure eval performance (TODO: change this to val perf) & save results to checkpoint dir
                    if do_eval and (step > 0 or self.args.eval_on_first_step):
                        # TODO: support eval/early stop when aaseq embeddings are not used!
                        if self.model.config.use_aaseq_embeddings:
                            eval_results = self._run_eval(current_checkpoint_dir)

                            # EARLY STOPPING 
                            stop_training = self._early_stopping(eval_results, step, current_checkpoint_dir)

                            if stop_training:
                                break

            # Barrier after evaluation ensures other processes do not start next step and time out while waiting for the process doing evaluation
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                
            # report training smoothed metrics
            if step % self.args.log_interval == 0:
                report_train_smoothed_metrics(
                    train_batch_smoothed_tracker,
                    step,
                    self.args.eval_steps,
                    self.logger,
                    self.wandb,
                )

                # reinitialize `train_batch_smoothed_tracker`
                train_batch_smoothed_tracker = dict()
                train_batch_smoothed_tracker["mlm"] = dict()
                train_batch_smoothed_tracker["text_cl"] = dict()
                train_batch_smoothed_tracker["protein_go_relations"] = dict()
                train_batch_smoothed_tracker["protein_protein_relations"] = dict()
                # train_batch_smoothed_tracker["pfam_pfam_relations"] = dict()
                # train_batch_smoothed_tracker["pfam_protein_relations"] = dict()
                # train_batch_smoothed_tracker["pfam_go_relations"] = dict()
                train_batch_smoothed_tracker["domain_go_relations"] = dict()
                train_batch_smoothed_tracker["domain_pfam_relations"] = dict()

        self.logger.info("Training completed.\n")
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss
        train_loss = self._total_loss_scalar / self.state.global_step

        self.is_in_train = False
    
    # TODO: make type annotations more specific?
    def training_step(
        self,
        model: nn.Module,
        protein_mlm_inputs: Dict,
        text_cl_inputs: Dict,
        protein_go_inputs: Dict,
        protein_protein_inputs: Dict,
        # pfam_inputs: Dict,
        domain_go_inputs: Dict,
        domain_pfam_inputs: Dict,
        train_batch_smoothed_tracker: Dict
    ) -> Tuple[torch.Tensor]:
        model.train()

        if text_cl_inputs:
            text_cl_inputs = self._prepare_inputs(text_cl_inputs)
        if protein_mlm_inputs:
            protein_mlm_inputs = self._prepare_inputs(protein_mlm_inputs)
        if protein_go_inputs:
            protein_go_inputs = self._prepare_inputs(protein_go_inputs)
        if protein_protein_inputs:
            protein_protein_inputs = self._prepare_inputs(protein_protein_inputs)
        # if pfam_inputs:
        #     pfam_inputs = self._prepare_inputs(pfam_inputs)
        if domain_go_inputs:
            domain_go_inputs = self._prepare_inputs(domain_go_inputs)
        if domain_pfam_inputs:
            domain_pfam_inputs = self._prepare_inputs(domain_pfam_inputs)

        with self.compute_loss_context_manager():
            loss, all_loss = self.compute_loss(
                model,
                protein_mlm_inputs,
                text_cl_inputs,
                protein_go_inputs,
                protein_protein_inputs,
                # pfam_inputs,
                domain_go_inputs,
                domain_pfam_inputs,
                train_batch_smoothed_tracker,
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
        if self.deepspeed:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                # last_lr = self.lr_scheduler.get_last_lr()[0]
                last_lr = self.lr_scheduler.get_last_lr()
            except AssertionError as e:
                if "need to call step" in str(e):
                    self.logger.warning(
                        "tried to get lr value before scheduler/optimizer started stepping, returning lr=0"
                    )
                    last_lr = 0
                else:
                    raise
        else:
            # last_lr = self.lr_scheduler.get_last_lr()[0]
            last_lr = self.lr_scheduler.get_last_lr()
            if torch.is_tensor(last_lr):
                last_lr = last_lr.item()
        return last_lr

    def compute_loss(
        self,
        model: nn.Module,
        protein_mlm_inputs: Dict,
        text_cl_inputs: Dict,
        protein_go_inputs: Dict,
        protein_protein_inputs: Dict,
        # pfam_inputs: Dict,
        domain_go_inputs: Dict,
        domain_pfam_inputs: Dict,
        train_batch_smoothed_tracker: Dict
    ) -> Tuple[torch.Tensor]:
        total_loss = 0
        all_loss = dict()
        # NOTE: For current input scenario, used as an extra scaler for the CL losses
        NUM_RELATION_TYPES = 4
        if not protein_go_inputs:
            NUM_RELATION_TYPES -= 1
        if not protein_protein_inputs:
            NUM_RELATION_TYPES -= 1
        # if not pfam_inputs:
        #     NUM_RELATION_TYPES -= 3
        if not domain_go_inputs:
            NUM_RELATION_TYPES -= 1
        if not domain_pfam_inputs:
            NUM_RELATION_TYPES -= 1


        # Temp 

        

        if protein_mlm_inputs:
            assert (
                "input_ids" in protein_mlm_inputs.keys()
                and "labels" in protein_mlm_inputs.keys()
            )
            protein_mlm_loss = self.compute_mlm_loss(
                model, protein_mlm_inputs, train_batch_smoothed_tracker
            )

            if self.args.world_size > 1:
                # TODO: Investigate if the loss behavior in calculating metrics is correct (I (Tom) am pretty sure mean() does not average across devices)
                protein_mlm_loss = protein_mlm_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                protein_mlm_loss = protein_mlm_loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(protein_mlm_loss).backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                protein_mlm_loss = self.deepspeed.backward(protein_mlm_loss)
            else:
                protein_mlm_loss.backward()
            
            total_loss += protein_mlm_loss.item()
            all_loss["mlm"] = protein_mlm_loss.item()

        if text_cl_inputs:
            assert (
                "input_ids" in text_cl_inputs.keys()
                and "attn_masks" in text_cl_inputs.keys()
            )
            text_cl_loss = self.compute_text_cl_loss(
                model, text_cl_inputs, train_batch_smoothed_tracker
            )

            if self.args.world_size > 1:
                # TODO: Investigate if the loss behavior in calculating metrics is correct (I (Tom) am pretty sure mean() does not average across devices)
                text_cl_loss = text_cl_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                text_cl_loss = text_cl_loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(text_cl_loss).backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                text_cl_loss = self.deepspeed.backward(text_cl_loss)
            else:
                text_cl_loss.backward()
            
            total_loss += text_cl_loss.item()
            all_loss["text_cl"] = text_cl_loss.item()

        if protein_go_inputs:
            assert sorted(list(protein_go_inputs.keys())) == sorted(['toks', 'indices', 'attn_masks', 'relations'])
            assert sorted(list(protein_go_inputs['relations'].keys())) == sorted(['positive_relations', 'negative_relations'])
            
            if self.args.model_type == 'lm':
                (
                    protein_go_loss,
                    protein_go_loss_dict,
                ) = self.compute_kepler_loss(
                    model,
                    protein_go_inputs,
                    NUM_RELATION_TYPES,
                    train_batch_smoothed_tracker,
                    relation_type = 'protein_go_relations',
                )  # loss has already been /4 in compute_kepler_loss
            # elif self.args.model_type == 'linkpred':
            #     protein_go_loss, protein_go_loss_dict = self.compute_bce_loss(
            #         model, protein_go_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker,
            #     )
            
            if protein_go_loss is not None:
                if self.args.world_size > 1:
                    # TODO: Investigate if the loss behavior in calculating metrics is correct
                    protein_go_loss = protein_go_loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    protein_go_loss = protein_go_loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(protein_go_loss).backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    protein_go_loss = self.deepspeed.backward(protein_go_loss)
                else:
                    protein_go_loss.backward()
                
                total_loss += protein_go_loss.item()
                all_loss["protein_go_kepler"] = protein_go_loss_dict[
                    "protein_go_relations"
                ]

        if protein_protein_inputs:
            assert sorted(list(protein_protein_inputs.keys())) == sorted(['node_toks', 'node_indices', 'protein_protein_relations'])
            assert sorted(list(protein_protein_inputs['protein_protein_relations'].keys())) == sorted(['positive_relations', 'negative_relations'])
            
            if self.args.model_type == 'lm':
                (
                    protein_protein_loss,
                    protein_protein_loss_dict,
                ) = self.compute_kepler_loss(
                    model,
                    protein_protein_inputs,
                    NUM_RELATION_TYPES,
                    train_batch_smoothed_tracker,
                    relation_type = 'protein_protein_relations',
                )  # loss has already been /5 in compute_kepler_loss
            # elif self.args.model_type == 'linkpred':
            #     protein_protein_loss, protein_protein_loss_dict = self.compute_bce_loss(
            #         model, protein_protein_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker,
            #     )
            
            if protein_protein_loss is not None:
                
                if self.args.world_size > 1:
                    # TODO: Investigate if the loss behavior in calculating metrics is correct
                    protein_protein_loss = protein_protein_loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    protein_protein_loss = protein_protein_loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(protein_protein_loss).backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    protein_protein_loss = self.deepspeed.backward(protein_protein_loss)
                else:
                    protein_protein_loss.backward()
                    
                total_loss += protein_protein_loss.item()
                all_loss["protein_protein_kepler"] = protein_protein_loss_dict[
                    "protein_protein_relations"
                ]

        # if pfam_inputs:
        #     assert sorted(list(pfam_inputs.keys())) == sorted(['node_toks', 'node_indices', 'pfam', "pfam_go_relations" , "pfam_pfam_relations" , "pfam_protein_relations"])
        #     if self.args.model_type == 'lm':
        #         pfam_loss, pfam_loss_dict = self.compute_kepler_loss(
        #             model, pfam_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker, head_is_pfam = True,
        #         )  # loss has already been /5 in compute_kepler_loss
        #     elif self.args.model_type == 'linkpred':
        #         pfam_loss, pfam_loss_dict = self.compute_bce_loss(
        #             model, pfam_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker, head_is_pfam = True,
        #         )

        #     if pfam_loss is not None:
                
        #         if self.args.world_size > 1:
        #             # TODO: Investigate if the loss behavior in calculating metrics is correct
        #             pfam_loss = pfam_loss.mean()  # mean() to average on multi-gpu parallel training

        #         if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
        #             # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
        #             pfam_loss = pfam_loss / self.args.gradient_accumulation_steps

        #         if self.do_grad_scaling:
        #             self.scaler.scale(pfam_loss).backward()
        #         elif self.deepspeed:
        #             # loss gets scaled under gradient_accumulation_steps in deepspeed
        #             pfam_loss = self.deepspeed.backward(pfam_loss)
        #         else:
        #             pfam_loss.backward()
                
        #         total_loss += pfam_loss.item()
        #         all_loss["pfam_pfam_kepler"] = pfam_loss_dict["pfam_pfam_relations"]
        #         all_loss["pfam_protein_kepler"] = pfam_loss_dict["pfam_protein_relations"]
        #         all_loss["pfam_go_kepler"] = pfam_loss_dict["pfam_go_relations"]

        if domain_go_inputs:
            assert sorted(list(domain_go_inputs.keys())) == sorted(['toks', 'indices', 'attn_masks', 'relations'])
            assert sorted(list(domain_go_inputs['relations'].keys())) == sorted(['positive_relations', 'negative_relations'])
            
            if self.args.model_type == 'lm':
                (
                    domain_go_loss,
                    domain_go_loss_dict,
                ) = self.compute_kepler_loss(
                    model,
                    domain_go_inputs,
                    NUM_RELATION_TYPES,
                    train_batch_smoothed_tracker,
                    relation_type = 'domain_go_relations',
                )  # loss has already been /5 in compute_kepler_loss
            # elif self.args.model_type == 'linkpred':
            #     domain_go_loss, domain_go_loss_dict = self.compute_bce_loss(
            #         model, domain_go_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker
            #     )
            
            if domain_go_loss is not None:
                if self.args.world_size > 1:
                    # TODO: Investigate if the loss behavior in calculating metrics is correct
                    domain_go_loss = domain_go_loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    domain_go_loss = domain_go_loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(domain_go_loss).backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    domain_go_loss = self.deepspeed.backward(domain_go_loss)
                else:
                    domain_go_loss.backward()
                
                total_loss += domain_go_loss.item()
                all_loss["domain_go_kepler"] = domain_go_loss_dict[
                    "domain_go_relations"
                ]
                
        if domain_pfam_inputs:
            assert sorted(list(domain_pfam_inputs.keys())) == sorted(['toks', 'indices', 'attn_masks', 'relations'])
            assert sorted(list(domain_pfam_inputs['relations'].keys())) == sorted(['positive_relations', 'negative_relations'])
            
            if self.args.model_type == 'lm':
                (
                    domain_pfam_loss,
                    domain_pfam_loss_dict,
                ) = self.compute_kepler_loss(
                    model,
                    domain_pfam_inputs,
                    NUM_RELATION_TYPES,
                    train_batch_smoothed_tracker,
                    relation_type = 'domain_pfam_relations',
                )  # loss has already been /5 in compute_kepler_loss
            # elif self.args.model_type == 'linkpred':
            #     domain_pfam_loss, domain_pfam_loss_dict = self.compute_bce_loss(
            #         model, domain_pfam_inputs, NUM_RELATION_TYPES, train_batch_smoothed_tracker,
            #     )
            
            if domain_pfam_loss is not None:
                if self.args.world_size > 1:
                    # TODO: Investigate if the loss behavior in calculating metrics is correct
                    domain_pfam_loss = domain_pfam_loss.mean()  # mean() to average on multi-gpu parallel training

                if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                    # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                    domain_pfam_loss = domain_pfam_loss / self.args.gradient_accumulation_steps

                if self.do_grad_scaling:
                    self.scaler.scale(domain_pfam_loss).backward()
                elif self.deepspeed:
                    # loss gets scaled under gradient_accumulation_steps in deepspeed
                    domain_pfam_loss = self.deepspeed.backward(domain_pfam_loss)
                else:
                    domain_pfam_loss.backward()
                
                total_loss += domain_pfam_loss.item()
                all_loss["domain_pfam_kepler"] = domain_pfam_loss_dict[
                    "domain_pfam_relations"
                ]
        
        return total_loss, all_loss

    def compute_mlm_loss(
        self, 
        model: nn.Module, 
        protein_mlm_inputs: Dict, 
        train_batch_smoothed_tracker: Dict,
    ):
        input_ids = protein_mlm_inputs["input_ids"]
        labels = protein_mlm_inputs["labels"]

        # NOTE: the operation of splitting a long sequence to a batch of chunks is done in the model
        # Ensure the protein encoder returns a dict with key 'logits' when `return_mlm=True`
        if self.args.world_size <= 1:
            output = model( # CAN CHANGE HERE
                inputs=protein_mlm_inputs,
                return_mlm=True,
            )
            logits = output['mlm']
        else:
            output = model.module( # CAN CHANGE HERE
                inputs=protein_mlm_inputs,
                return_mlm=True,
            )
            logits = output['mlm']

        outputs = get_mlm_loss(logits, labels)
        mask_count = (labels != -100).sum().item()

        # TODO: add support for token-level accuracy
        self.logger.info(
            "batch_train_mlm_loss: {:.4f}, batch_train_mlm_accuracy: {:.4f}, batch_train_mlm_perplexity: {:.4f}".format(
                outputs["loss"].item(), outputs["accuracy"], outputs["perplexity"]
            )
        )

        self.wandb.log(self.state.global_step, 
            {
                "batch_train_mlm_loss": outputs["loss"].item(),
                "batch_train_mlm_accuracy": outputs["accuracy"],
                "batch_train_mlm_perplexity": outputs["perplexity"],
            }
        )

        train_batch_smoothed_tracker["mlm"].setdefault("loss", []).append(
            outputs["loss"].item()
        )
        train_batch_smoothed_tracker["mlm"].setdefault("accuracy", []).append(
            outputs["accuracy"]
        )
        train_batch_smoothed_tracker["mlm"].setdefault("perplexity", []).append(
            outputs["perplexity"]
        )
        train_batch_smoothed_tracker["mlm"].setdefault("count", []).append(mask_count)

        return outputs["loss"] * self.args.mlm_loss_weight


    def compute_text_cl_loss(
        self, 
        model: nn.Module, 
        text_cl_inputs: Dict, 
        train_batch_smoothed_tracker: Dict,
    ):
        input_ids = text_cl_inputs["input_ids"]
        attention_mask = text_cl_inputs["attn_masks"]

        text_encoder = unwrap_model(model).text_encoder

        # Perform Text-CL forward pass
        outputs = text_encoder.cl_forward(input_ids, attention_mask)

        # TODO: add support for token-level accuracy
        self.logger.info( "batch_text_cl_loss: {:.4f}".format(outputs["loss"].item()))
        if outputs["loss"].item() == torch.nan:
            raise ValueError('NaN loss detected')

        self.wandb.log(self.state.global_step, 
            {
                "batch_text_cl_loss": outputs["loss"].item(),
            }
        )

        train_batch_smoothed_tracker["text_cl"].setdefault("loss", []).append(
            outputs["loss"].item()
        )
        train_batch_smoothed_tracker["text_cl"].setdefault("count", []).append(len(input_ids))


        return outputs["loss"] * self.args.text_cl_loss_weight

    def compute_kepler_loss(
        self, model, inputs, num_all_relation_types, train_batch_smoothed_tracker, relation_type: str,
    ):
        scores_dict = model(  # CAN CHANGE HERE
            inputs=inputs,
            seq_type = relation_type.split('_')[0], 
            text_type = relation_type.split('_')[1], # Extract directly from relation_type
        )

        all_kepler_loss_dict = dict()
        all_kepler_losses = []
        
        # for _, scores_dict in outputs.items():
        if scores_dict is None:
            positive_loss = torch.tensor(0)
            negative_loss = torch.tensor(0)
            loss = torch.tensor(0)
            pos_count, neg_count = 0, 0
            auroc, auprc = 0, 0

        else:
            positive_loss = - get_kepler_loss(
                scores_dict["positive_scores"], 
                self.args.kepler_margin,
                is_neg = False
            )
            negative_loss = - get_kepler_loss(
                scores_dict["negative_scores"], 
                self.args.kepler_margin,
                is_neg = True
            )
            loss = (
                (positive_loss + negative_loss)
                * (1 - self.args.mlm_loss_weight - self.args.text_cl_loss_weight)
                / num_all_relation_types
            )
            
            try:
                pos_count, neg_count, auroc, auprc = get_cl_metrics(
                    scores_dict["positive_scores"].flatten().detach().cpu().numpy(),
                    scores_dict["negative_scores"].flatten().detach().cpu().numpy(),
                )
            except (RuntimeError, ValueError) as e:
                self.logger.warning("Positive loss: ")
                self.logger.warning(positive_loss)
                self.logger.warning("Negative loss: ")
                self.logger.warning(negative_loss)
                self.logger.warning("Positive num nan: ")
                self.logger.warning(scores_dict["positive_scores"].flatten().detach().cpu().isnan().sum())
                self.logger.warning("Positive scores: ")
                self.logger.warning(scores_dict["positive_scores"].flatten().detach().cpu().numpy())
                self.logger.warning("Negative num nan: ")
                self.logger.warning(scores_dict["negative_scores"].flatten().detach().cpu().isnan().sum())
                self.logger.warning("Negative scores: ")
                self.logger.warning(scores_dict["negative_scores"].flatten().detach().cpu().numpy())
                raise e

        # FIXME: Potential issue re. parallelization (multiplet machines)
        if scores_dict is not None:
            all_kepler_losses.append(loss.unsqueeze(0))
        all_kepler_loss_dict[relation_type] = loss.item()

        self.logger.info(
            f"batch_train_{relation_type}_positive_loss: {positive_loss.item():.4f}, batch_train_{relation_type}_negative_loss: {negative_loss.item():.4f}, batch_train_{relation_type}_loss: {loss.item():.4f}, batch_train_{relation_type}_auroc: {auroc:.4f}, batch_train_{relation_type}_auprc: {auprc:.4f}"
        )
        self.wandb.log(self.state.global_step, 
            {
                f"batch_train_{relation_type}_positive_loss": positive_loss.item(),
                f"batch_train_{relation_type}_negative_loss": negative_loss.item(),
                f"batch_train_{relation_type}_loss": loss.item(),
                f"batch_train_{relation_type}_auroc": auroc,
                f"batch_train_{relation_type}_auprc": auprc,
            }
        )

        train_batch_smoothed_tracker[relation_type].setdefault("loss", []).append(
            loss.item()
        )
        train_batch_smoothed_tracker[relation_type].setdefault(
            "positive_loss", []
        ).append(positive_loss.item())
        train_batch_smoothed_tracker[relation_type].setdefault(
            "negative_loss", []
        ).append(negative_loss.item())
        train_batch_smoothed_tracker[relation_type].setdefault(
            "positive_count", []
        ).append(pos_count)
        train_batch_smoothed_tracker[relation_type].setdefault(
            "negative_count", []
        ).append(neg_count)
        train_batch_smoothed_tracker[relation_type].setdefault("count", []).append(
            neg_count + pos_count
        )
        train_batch_smoothed_tracker[relation_type].setdefault("auroc", []).append(
            auroc
        )
        train_batch_smoothed_tracker[relation_type].setdefault("auprc", []).append(
            auprc
        )

        #ipdb.set_trace()
        all_kepler_loss = torch.cat(all_kepler_losses).sum()
        return all_kepler_loss, all_kepler_loss_dict

    def compute_bce_loss(
        self, model, inputs, num_all_relation_types, train_batch_smoothed_tracker, relation_type: str,
    ):
        outputs = model(  # CAN CHANGE HERE
            inputs=inputs,
        )

        criterion = nn.BCEWithLogitsLoss()

        all_loss_dict = dict()
        all_losses = []
        non_empty = False
        # CHECK FOR EMPTY
        # NOTE: Keys of `scores_dict` are like 'protein_protein_relations', 'protein_go_relations', 'pfam_go_relations', 'pfam_pfam_relations', 'pfam_protein_relations', etc.
        for _, scores_dict in outputs.items():
            positive_loss = torch.tensor(0)
            negative_loss = torch.tensor(0)
            if scores_dict is None:
                loss = torch.tensor(0)
                pos_count, neg_count = 0, 0
                auroc, auprc = 0, 0

            else:
                all_scores = torch.cat([scores_dict["positive_scores"], scores_dict['negative_scores']])
                target = torch.cat([torch.ones_like(scores_dict["positive_scores"]), torch.zeros_like(scores_dict['negative_scores'])]).to(all_scores.device)
                loss = criterion(all_scores, target)

                pos_count, neg_count, auroc, auprc = get_cl_metrics(
                    scores_dict["positive_scores"].detach().clone().cpu().numpy(),
                    scores_dict["negative_scores"].detach().clone().cpu().numpy(),
                )
                print('Done eval')

            # FIXME: Potential issue re. parallelization (multiplet machines)
            if scores_dict is not None:
                all_losses.append(loss.unsqueeze(0))
                non_empty = True

            all_loss_dict[relation_type] = loss.item()

            self.logger.info(
                f"batch_train_{relation_type}_positive_loss: {positive_loss.item():.4f}, batch_train_{relation_type}_negative_loss: {negative_loss.item():.4f}, batch_train_{relation_type}_loss: {loss.item():.4f}, batch_train_{relation_type}_auroc: {auroc:.4f}, batch_train_{relation_type}_auprc: {auprc:.4f}"
            )
            self.wandb.log(self.state.global_step,
                {
                    f"batch_train_{relation_type}_positive_loss": positive_loss.item(),
                    f"batch_train_{relation_type}_negative_loss": negative_loss.item(),
                    f"batch_train_{relation_type}_loss": loss.item(),
                    f"batch_train_{relation_type}_auroc": auroc,
                    f"batch_train_{relation_type}_auprc": auprc,
                }
            )

            train_batch_smoothed_tracker[relation_type].setdefault("loss", []).append(
                loss.item()
            )
            train_batch_smoothed_tracker[relation_type].setdefault(
                "positive_loss", []
            ).append(positive_loss.item())
            train_batch_smoothed_tracker[relation_type].setdefault(
                "negative_loss", []
            ).append(negative_loss.item())
            train_batch_smoothed_tracker[relation_type].setdefault(
                "positive_count", []
            ).append(pos_count)
            train_batch_smoothed_tracker[relation_type].setdefault(
                "negative_count", []
            ).append(neg_count)
            train_batch_smoothed_tracker[relation_type].setdefault("count", []).append(
                neg_count + pos_count
            )
            train_batch_smoothed_tracker[relation_type].setdefault("auroc", []).append(
                auroc
            )
            train_batch_smoothed_tracker[relation_type].setdefault("auprc", []).append(
                auprc
            )

        #ipdb.set_trace()
        if non_empty:
            all_loss = torch.cat(all_losses).sum()
        else:
            all_loss = None
        return all_loss, all_loss_dict


    @torch.no_grad()
    def _run_eval(self, current_checkpoint_dir):
        """Perform evaluation on the validation set"""
        self.model.eval()
        # TODO: add arg to enable / disable eval_retrieval        
        # TODO: switch to val set (not eval)
        # TODO: unify eval with training (use variants rather than col names), and match with training value

        results = {}

        for go_eval_type in ['val', 'eval']:

            if go_eval_type == 'val' and 'v1' not in self.data_args.go_split_method:
                # no val set exists
                continue

            args_dict = {
                'model_dir': current_checkpoint_dir,
                'batch_size': self.args.eval_text_batch_size,
                'max_text_len': self.model_args.max_text_len,
                'go_text_variant_type': 'standard',
                'pfam_text_variant_type': 'standard',
                'protein_embed_path': self.model_args.protein_seq_embeddings_path,
                'domain_embed_path': self.model_args.domain_embeddings_path,
                'mouse_ortholog_embed_path': self.model_args.mouse_ortholog_embeddings_path,
                'frozen_text': False,
                'frozen_aaseq': True,
                'go_eval_type': go_eval_type,
                'go_split_method': self.data_args.go_split_method,
            }   

            # FIXME: once data exists
            if go_eval_type == 'eval' and not self.model_args.use_text_embeddings:
                args_dict['reactome_text_variant_type'] = 'standard'
                args_dict['drugbank_text_variant_type'] = 'standard'
            
            df = run_general_eval(self.logger, args_dict)
            print(df)

            # FIXME: improve hacky solution (val set is only relevant for protein_go, domain_go currently)
            def include(row):
                if go_eval_type == 'val':
                    return ('protein_go' in row) or ('domain_go' in row)
                else:
                    return True
            
            eval_results_dict = {f'{go_eval_type}_{col}_{row}': df.at[row, col] for row in df.index for col in df.columns if include(row)}

            # TODO: move to eval script
            eval_results_dict[f'{go_eval_type}_auprc_protein_go_overall'] = np.mean([
                eval_results_dict[f"{go_eval_type}_auprc_protein_go_pt_ft_overall_standard_256"],
                eval_results_dict[f"{go_eval_type}_auprc_protein_go_five_shot_overall_standard_256"],
                eval_results_dict[f"{go_eval_type}_auprc_protein_go_zero_shot_overall_standard_256"],
            ])

            self.wandb.log(self.state.global_step, eval_results_dict)
            results.update(eval_results_dict)
        
        self.model.train()
        return results

    
    def _early_stopping(self, eval_results, step, current_checkpoint_dir):
        # TODO: use chosen training text column here (rather than overall)?
        # TODO: investigate why overall value != value for description_combined when only one column is used
        if 'val_auprc_protein_go_overall' in eval_results:
            checkpoint_metric = eval_results["val_auprc_protein_go_overall"]
        else:
            checkpoint_metric = eval_results["eval_auprc_protein_go_overall"]

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


    def _save_checkpoint(self, checkpoint_dir):
        # FIXME
        # If we are executing this function, we are the process zero, so we don't check for that.
        
        self.logger.info(f"Saving model checkpoint to {checkpoint_dir}")

        model_state_dict, model_config = self.model.save_pretrained()

        training_state = {
            'step': self.state.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            }
    
        torch.save(training_state, os.path.join(checkpoint_dir, SAVE_TRAINING_STATE_FNAME))

        # workaround to save DictClass from save method
        if is_dataclass(model_config):
            config_dict = asdict(model_config)
        else:
            # From DictClass defined in model.py
            config_dict = model_config.as_dict()

        with open(os.path.join(checkpoint_dir, SAVE_CONFIG_FNAME), 'w') as f:
            json.dump(config_dict, f)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(checkpoint_dir, SAVE_TRAINING_ARGS_FNAME))
        torch.save(self.model_args, os.path.join(checkpoint_dir, SAVE_MODEL_ARGS_FNAME))
        torch.save(self.data_args, os.path.join(checkpoint_dir, SAVE_DATA_ARGS_FNAME))
        
        if self.deepspeed:
            self.deepspeed.save_checkpoint(checkpoint_dir)

        # save loss traces. NOTE: Omitted since we have wandb
        # with open(
        #     os.path.join(checkpoint_dir, "loss_trace.json"), "w", encoding="utf-8"
        # ) as handle:
        #     handle.write(json.dumps(self.loss_recorder, indent=2, ensure_ascii=False))


    def _load_from_checkpoint(self, resume_from_checkpoint, model, optimizer):
        # TODO: also handle lr_scheduler??

        self.logger.info(f"Loading model checkpoint from {resume_from_checkpoint}")

        training_state = torch.load(os.path.join(resume_from_checkpoint, SAVE_TRAINING_STATE_FNAME))

        # Load pretrained weights into model
        TxPLM.from_pretrained(checkpoint_dir=resume_from_checkpoint, model=model)

        # Load pretrained weights into model
        optimizer.load_state_dict(training_state['optimizer_state_dict'])

        self.state.global_step = training_state['step']


    def _get_dataloaders(self):
        (
            train_protein_mlm_loader,
            train_text_cl_loader,
            train_protein_go_loader,
            train_protein_protein_loader,
            # train_pfam_loader,
            train_domain_go_loader,
            train_domain_pfam_loader,
            val_protein_mlm_loader,
            val_text_cl_loader,
            val_protein_go_loader,
            val_protein_protein_loader,
            # val_pfam_loader,
            val_domain_go_loader,
            val_domain_pfam_loader,
        ) = [None] * 12

        # get samplers
        (
            train_protein_mlm_sampler,
            train_text_cl_sampler,
            train_protein_go_sampler,
            train_protein_protein_sampler,
            # train_pfam_sampler,
            train_domain_go_sampler,
            train_domain_pfam_sampler,
            val_protein_mlm_sampler,
            val_text_cl_sampler,
            val_protein_go_sampler,
            val_protein_protein_sampler,
            # val_pfam_sampler,
            val_domain_go_sampler,
            val_domain_pfam_sampler,
        ) = self._get_samplers()

        # `batch_size` and `drop_last` are fed into `sampler` to create a `batch_sampler` (`shuffle` is useless when `sampler` exists). The output (list) of `batch_sampler` is then fed into `collate_fn` to create a batch.
        if self.train_protein_mlm_dataset:
            train_protein_mlm_loader = DataLoader(
                self.train_protein_mlm_dataset,
                batch_size=self.args.protein_mlm_batch_size,
                # shuffle=True,
                collate_fn=self.protein_mlm_collator,
                num_workers=self.args.protein_mlm_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_protein_mlm_sampler,
            )
            val_protein_mlm_loader = DataLoader(
                self.val_protein_mlm_dataset,
                batch_size=self.args.protein_mlm_batch_size,
                # shuffle=False,
                collate_fn=self.protein_mlm_collator,
                num_workers=self.args.protein_mlm_num_workers,
                pin_memory=(self.args.world_size<=1),
                sampler=val_protein_mlm_sampler,
            )
        if self.train_text_cl_dataset:
            train_text_cl_loader = DataLoader(
                self.train_text_cl_dataset,
                batch_size=self.args.text_cl_batch_size,
                # shuffle=True,
                collate_fn=self.text_cl_collator,
                num_workers=self.args.text_cl_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_text_cl_sampler,
            )
            val_text_cl_loader = DataLoader(
                self.val_text_cl_dataset,
                batch_size=self.args.text_cl_batch_size,
                # shuffle=False,
                collate_fn=self.text_cl_collator,
                num_workers=self.args.text_cl_num_workers,
                pin_memory=(self.args.world_size<=1),
                sampler=val_text_cl_sampler,
            )
        if self.train_protein_go_dataset:
            train_protein_go_loader = DataLoader(
                self.train_protein_go_dataset,
                batch_size=self.args.protein_go_batch_size,
                # shuffle=True,
                collate_fn=self.protein_go_collator,
                num_workers=self.args.protein_go_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_protein_go_sampler,
            )
            val_protein_go_loader = DataLoader(
                self.val_protein_go_dataset,
                batch_size=self.args.protein_go_batch_size,
                # shuffle=False,
                collate_fn=self.protein_go_collator,
                num_workers=self.args.protein_go_num_workers,
                pin_memory=(self.args.world_size<=1),
                sampler=val_protein_go_sampler,
            )
        if self.train_protein_protein_dataset:
            train_protein_protein_loader = DataLoader(
                self.train_protein_protein_dataset,
                batch_size=self.args.protein_protein_batch_size,
                collate_fn=self.protein_protein_collator,
                num_workers=self.args.protein_protein_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_protein_protein_sampler,
            )
            val_protein_protein_loader = DataLoader(
                self.val_protein_protein_dataset,
                batch_size=self.args.protein_protein_batch_size,
                collate_fn=self.protein_protein_collator,
                num_workers=self.args.protein_protein_num_workers,
                pin_memory=(self.args.world_size<=1),
                sampler=val_protein_protein_sampler,
            )
        # if self.train_pfam_dataset:
        #     train_pfam_loader = DataLoader(
        #         self.train_pfam_dataset,
        #         batch_size=self.args.pfam_batch_size,
        #         collate_fn=self.pfam_collator,
        #         num_workers=self.args.pfam_num_workers,
        #         pin_memory=(self.args.world_size<=1),
        #         drop_last=True,
        #         sampler=train_pfam_sampler,
        #     )
        #     val_pfam_loader = DataLoader(
        #         self.val_pfam_dataset,
        #         batch_size=self.args.pfam_batch_size,
        #         collate_fn=self.pfam_collator,
        #         num_workers=self.args.pfam_num_workers,
        #         pin_memory=(self.args.world_size<=1),
        #         sampler=val_pfam_sampler,
        #     )
        if self.train_domain_go_dataset:
            train_domain_go_loader = DataLoader(
                self.train_domain_go_dataset,
                batch_size=self.args.domain_go_batch_size,
                collate_fn=self.domain_go_collator,
                num_workers=self.args.domain_go_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_domain_go_sampler,
            )
            val_domain_go_loader = DataLoader(
                self.val_domain_go_dataset,
                batch_size=self.args.domain_go_batch_size,
                collate_fn=self.domain_go_collator,
                num_workers=self.args.domain_go_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=val_domain_go_sampler,
            )
        if self.train_domain_pfam_dataset:
            train_domain_pfam_loader = DataLoader(
                self.train_domain_pfam_dataset,
                batch_size=self.args.domain_pfam_batch_size,
                collate_fn=self.domain_pfam_collator,
                num_workers=self.args.domain_pfam_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=train_domain_pfam_sampler,
            )
            val_domain_pfam_loader = DataLoader(
                self.val_domain_pfam_dataset,
                batch_size=self.args.domain_pfam_batch_size,
                collate_fn=self.domain_pfam_collator,
                num_workers=self.args.domain_pfam_num_workers,
                pin_memory=(self.args.world_size<=1),
                drop_last=True,
                sampler=val_domain_pfam_sampler,
            )

        return (
            train_protein_mlm_loader,
            train_text_cl_loader,
            train_protein_go_loader,
            train_protein_protein_loader,
            # train_pfam_loader,
            train_domain_go_loader,
            train_domain_pfam_loader,
            val_protein_mlm_loader,
            val_text_cl_loader,
            val_protein_go_loader,
            val_protein_protein_loader,
            # val_pfam_loader,
            val_domain_go_loader,
            val_domain_pfam_loader,
        )

    def _get_samplers(self):
        samplers = []
        for dataset in [
            self.train_protein_mlm_dataset,
            self.train_text_cl_dataset,
            self.train_protein_go_dataset,
            self.train_protein_protein_dataset,
            # self.train_pfam_dataset,
            self.train_domain_go_dataset,
            self.train_domain_pfam_dataset,
        ]:
            if not dataset:  # if dataset is None
                samplers.append(None)
                continue

            generator = None
            if self.args.world_size <= 1:
                generator = torch.Generator()
                generator.manual_seed(
                    int(torch.empty((), dtype=torch.int64).random_().item())
                )

            if self.args.world_size <= 1:
                samplers.append(RandomSampler(dataset, generator=generator))
            else:
                samplers.append(
                    DistributedSampler(
                        dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,  # property of TrainingArguments class
                        seed=self.args.seed,
                    )
                )

        for dataset in [
            self.val_protein_mlm_dataset,
            self.val_text_cl_dataset,
            self.val_protein_go_dataset,
            self.val_protein_protein_dataset,
            # self.val_pfam_dataset,
            self.val_domain_go_dataset,
            self.val_domain_pfam_dataset,
        ]:
            if not dataset:  # if dataset is None
                samplers.append(None)
                continue

            if self.args.world_size <= 1:
                samplers.append(SequentialSampler(dataset))
            else:
                samplers.append(
                    SequentialDistributedSampler(
                        dataset,
                        # num_replicas=self.args.world_size,  # TODO: investigate need or not?
                        # rank=self.args.process_index,  # property of TrainingArguments class
                    )
                )

        return samplers

    def create_optimizer(self):
        """
        Rewrites the create_optimizer method of Trainer class to enable independent learning rates for different parts of the model.
        """
        assert self.optimizer is None

        protein_encoder_decay_params, protein_encoder_no_decay_params, text_encoder_decay_params, text_encoder_no_decay_params, embedding_params, decoder_params = self.model.get_grouped_parameter_names()

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
                    if n in decoder_params
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_lr,
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

        self.optimizer = optimizer_class(
            optimizer_grouped_parameters, **optimizer_kwargs
        )

    def create_scheduler(self, num_training_steps):
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

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_protein_encoder_warmup_steps=self.args.warmup_steps,  # TODO: Separate out warmup step args for different parts of the model
                num_text_encoder_warmup_steps=self.args.warmup_steps,
                num_embedding_warmup_steps=self.args.warmup_steps,
                num_decoder_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

