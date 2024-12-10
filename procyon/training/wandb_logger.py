import wandb
from typing import Set

try:
    WANDB_API_KEY = os.getenv('WANDB_API_KEY')
except: 
    WANDB_API_KEY = None

class WandbLogger:
    """
    Wrapper class for wandb logging.
    Provides same api as using wandb 
    """

    def __init__(self, **kwargs):
        if WANDB_API_KEY is not None:
            wandb.login(key=WANDB_API_KEY)
        wandb.require("service")
        wandb.init(**kwargs, settings=wandb.Settings(_service_wait=10000, start_method="fork"))
        wandb.define_metric("global_step")

        #wandb.watch()
        
        self.all_metrics: Set[str] = set()
        self.run = wandb.run
        self.all_metrics.add("global_step")
    
    def set_run_name(self, run_name: str):
        wandb.run.name = run_name

    def log(self, global_step: int, metrics: dict):
        for key in metrics.keys():
            if key not in self.all_metrics:
                wandb.define_metric(key, step_metric="global_step")
                self.all_metrics.add(key)
        
        metrics['global_step'] = global_step
        wandb.log(metrics)

    def watch(self, model, log = 'gradients', log_freq = 500):
        wandb.watch(model, log = log, log_freq = log_freq)
