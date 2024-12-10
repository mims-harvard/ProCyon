import numpy as np
from multiprocessing import Lock

class DatasetTracker:
    '''
    Tracks dataset usage during iterations
        - Samples from each dataset based on how many times we've seen it
    '''
    def __init__(self, 
            dataloaders_dict = None, 
            saved_state = None,
            world_size = 1,
        ):

        if dataloaders_dict is not None:
            self.dataset_keys = list(dataloaders_dict.keys()) # Aligned by index to dataset_sizes

            # len(dataloader) gets the number of batches after drop_last=True, batch size integrated or whatever 
            # Divide by world size bc we know that indices are evenly distributed across processes
            # Values in this tracker will be different per process, so we need to account for this in setting it up
            self.fixed_dataset_sizes = [len(dataloaders_dict[d]) // world_size for d in self.dataset_keys] # Aligned by index to dataset_keys, dataset_upper_bounds
            self.total_batches = sum(self.fixed_dataset_sizes)

            self.dataset_progress = np.array(self.fixed_dataset_sizes)
            self.dataset_progress_upwards = np.zeros_like(self.dataset_progress, dtype = int)
            self.epoch_progress = self.total_batches
            self.world_size = world_size
        else:
            self.fixed_dataset_sizes = saved_state['fixed_dataset_sizes']
            self.total_batches = saved_state['total_batches']

            self.dataset_progress = saved_state['dataset_progress']
            self.dataset_progress_upwards = saved_state['dataset_progress_upwards']
            self.epoch_progress = saved_state['epoch_progress']

            if world_size != saved_state["world_size"]:
                # Need to transition internal state if the world size is new
                self.transition_world_size(old_world_size = saved_state["world_size"], new_world_size = world_size)

    def sample(self):
        isample = np.random.choice(np.arange(len(self.dataset_keys)), p = self.dataset_progress / self.epoch_progress)
        self.dataset_progress[isample] -= 1
        self.dataset_progress_upwards += 1
        self.epoch_progress -= 1

        return self.dataset_keys[isample]

    def reset_on_epoch(self):
        # Resets the progress counters
        self.epoch_progress = self.total_batches
        self.dataset_progress = np.array(self.fixed_dataset_sizes, dtype = int)

    def get_progress(self, key, over_epoch = False):
        key_index = self.dataset_keys.index(key)
        nstep_by_key = self.dataset_progress_upwards[key_index]
        if over_epoch:
            return nstep_by_key / self.fixed_dataset_sizes[key_index]
        else:
            return int(nstep_by_key)

    def transition_world_size(self, old_world_size, new_world_size):
        # Idea: old values are bounded by world size, so multiply by that and then divide by new world size
        self.fixed_dataset_sizes = self.fixed_dataset_sizes * old_world_size // new_world_size
        self.total_batches = self.fixed_dataset_sizes

        self.dataset_progress_upwards = np.floor(self.dataset_progress_upwards * old_world_size / new_world_size)
        self.dataset_progress =  self.fixed_dataset_sizes - self.dataset_progress_upwards
        self.epoch_progress = self.dataset_progress_upwards.sum()


    def save_state(self):
        return {
            'fixed_dataset_sizes': self.fixed_dataset_sizes,
            'total_batches': self.total_batches,
            'dataset_progress': self.dataset_progress,
            'dataset_progress_upwards': self.dataset_progress_upwards,
            'epoch_progress': self.epoch_progress,
            "world_size": self.world_size,
        }
