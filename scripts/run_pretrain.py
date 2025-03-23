import os, sys
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from time import sleep
import json

import torch
import deepspeed
from deepspeed import comm as dist
from transformers import HfArgumentParser

from procyon.training.training_args_IT import (
    TrainArgs,
    DataArgs,
    ModelArgs,
    get_hparams,
    postprocess_args,
)
from procyon.model.model_unified import UnifiedProCyon
from procyon.model.model_utils import detect_zero_stages
from procyon.training.trainIT import ProCyonTrainer
from procyon.training.train_utils import (
    get_root_logger,
    set_seed,
    get_IT_datasets,
    get_all_datasets,
    get_data_collators_IT,
    get_data_collators_IT_new,
    get_datasets_and_collators_from_config,
    barrier
)
from procyon.training.wandb_logger import WandbLogger

import sys
sys.setrecursionlimit(10000)

def main(train_args: TrainArgs, data_args: DataArgs, model_args: ModelArgs, output_dir, logger, wandb: WandbLogger):

    torch.hub.set_dir(data_args.data_dir + 'model_weights/')
   
    resume = train_args.resume_from_checkpoint is not None
    logger.info(
        "Process rank: %s, device: %s, world_size: %s, distributed training: %s",
        train_args.local_rank,
        # device,
        train_args.device,
        train_args.world_size,
        bool(train_args.local_rank != -1),
    )

    if train_args.local_rank in {-1, 0}:
        logger.info(f"Data parameters: {data_args}")
        logger.info(f"Training parameters: {train_args}")
        logger.info(f"Model parameters: {model_args}")

    set_seed(train_args.seed)

    train_protein_dataset, val_protein_dataset = get_IT_datasets(data_args, task_type = 'mlm')
    print("loading IT datasets from config")
    train_datasets, val_datasets, collators = get_datasets_and_collators_from_config(
        data_args = data_args, 
        model_args = model_args, 
        train_args = train_args
    )

    # Collators:
    protein_mlm_collator, qa_collator, retrieval_collator, caption_collator = collators

    torch.cuda.empty_cache()

    barrier()

    pretrained_weights_dir = data_args.data_dir + 'model_weights/'
    if train_args.resume_from_checkpoint is not None:
        model, _ = UnifiedProCyon.from_pretrained( # Must load model
            pretrained_weights_dir = pretrained_weights_dir,
            checkpoint_dir = train_args.resume_from_checkpoint,
            config = model_args
        )

        torch.cuda.empty_cache()

    else:
        model = UnifiedProCyon(pretrained_weights_dir = pretrained_weights_dir, config = model_args)

    barrier()

    print("After model load")

    trainer = ProCyonTrainer(
        model=model,
        args=train_args,
        data_args=data_args,
        model_args=model_args,
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        protein_mlm_collator=protein_mlm_collator,
        qa_collator=qa_collator,
        retrieval_collator=retrieval_collator,
        caption_collator=caption_collator,
        output_dir = output_dir,
        device=train_args.device,
        logger=logger,
        wandb=wandb
    )

    trainer.train()

if __name__ == '__main__':

    import warnings
    warnings.simplefilter("ignore", UserWarning) # Specific to ignore warning from checkpointing

    print("THIS SCRIPT RUNS DEEPSPEED TRAINING")

    torch.distributed.init_process_group(backend="nccl")

    # Parse arguments:
    parser = HfArgumentParser((TrainArgs, DataArgs, ModelArgs))

    # If resuming, we only load data_args and train_args, model_args are fixed
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()

    if train_args.from_yaml is not None:
        train_args, data_args, model_args = parser.parse_yaml_file(train_args.from_yaml)
    if train_args.from_json is not None:
        train_args, data_args, model_args = parser.parse_json_file(train_args.from_json)

    train_args, data_args, model_args = postprocess_args(train_args, data_args, model_args)

    if train_args.resume_from_checkpoint is not None:
        data_args_from_ckpt, model_args_from_ckpt, train_args_from_ckpt = UnifiedProCyon.get_checkpoint_configs(resume_from_checkpoint = train_args.resume_from_checkpoint)
        if train_args.resume_data_args:
            data_args = data_args_from_ckpt
        if train_args.resume_model_args:
            model_args = model_args_from_ckpt
        if train_args.resume_train_args:
            train_args = train_args_from_ckpt
        train_args, data_args, model_args = postprocess_args(train_args, data_args, model_args)
        hparams = get_hparams((train_args, data_args, model_args))
    else:
        hparams = get_hparams((train_args, data_args, model_args))

    train_args.fp16 = train_args.fp16 and (torch.__version__ >= '1.6.0')

    # Change these for WANBD configuration
    WANDB_PROJECT_NAME = 'pretrain'
    WANDB_PROJECT_ENTITY = 'procyon'

    if train_args.distributed_wandb_logging:

        GR = int(os.environ["RANK"])

        # Get ID to resume based on global rank:
        resume_arg = "allow" 
        my_resume_id = None
        if train_args.resume_wandb_id_config is not None:
            resume_id_file = json.load(open(train_args.resume_wandb_id_config, "r"))
            #resume_arg = "allow"
            my_resume_id = resume_id_file["gr={}".format(GR)]
            resume_arg = "must"
        
        rname_prefix = train_args.run_name if train_args.run_name != train_args.output_dir else wandb.run.name
        run_name = f"{rname_prefix}_gr={GR}"
        print('WANDB --- Group: {}, Run Name: {}'.format(train_args.group_name, run_name))
        barrier()
        wandb = WandbLogger(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_PROJECT_ENTITY,
            config=hparams,
            dir=train_args.output_dir,
            mode='offline' if train_args.debug else 'online',
            resume=resume_arg,
            id=my_resume_id,
            group = train_args.group_name if (train_args.group_name is not None) else None
        )
        wandb.run.name = run_name # Overwrite run name compared to below with group rank
    else:
        if train_args.local_rank in {-1, 0}:  # only on main process
            print('Run name:', train_args.run_name)
            wandb = WandbLogger(
                project=WANDB_PROJECT_NAME,
                entity=WANDB_PROJECT_ENTITY,
                config=hparams,
                dir=train_args.output_dir,
                mode='offline' if train_args.debug else 'online',
                resume='allow' if train_args.resume_wandb_id is None else 'must',
                id=train_args.resume_wandb_id,
                #group = train_args.group_name if train_args.group_name is not None
            )
        else:
            # TODO: Fix this hack used for DDP as a work-around
            @dataclass
            class Dummy():
                name: str = train_args.run_name
            class Wandb():
                def __init__(self):
                    self.run = Dummy()
                    self.run.id = None
                def log(self, *args, **kwargs):
                    pass
                def watch(self, *args, **kwargs):
                    pass
            wandb = Wandb()

        wandb.run.name = train_args.run_name if train_args.run_name != train_args.output_dir else wandb.run.name
        if wandb.run.name is not None and train_args.resume_wandb_id is not None:
            # wandb will reuse existing run name, so append "-resume"
            wandb.run.name = wandb.run.name + '-resume'
        elif wandb.run.name is not None and train_args.run_name_suffix is not None:
            wandb.run.name = wandb.run.name + '__' + train_args.run_name_suffix

    try:
        output_dir = os.environ.get("OUTPUTDIR").format(output_dir = train_args.output_dir, run_name = train_args.run_name)
        print('OUTPUT DIR SET STATICALLY: {}'.format(output_dir))
    except:
        cur_time = datetime.now().strftime('%Y-%m-%d_%H:%M')
        output_dir = f'{train_args.output_dir}/{cur_time}_{wandb.run.name}'

    barrier()

    print('Run ID WANDB:', wandb.run.id)

    # NOTE: assuming in DDP, makedirs will happen at the same time for all processes
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = get_root_logger(f'{output_dir}/log_{train_args.local_rank}.txt')

    main(train_args, data_args, model_args, output_dir, logger, wandb)
