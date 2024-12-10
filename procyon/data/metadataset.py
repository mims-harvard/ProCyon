import numpy as np

from torch.utils.data import Dataset


class MetaDataset(Dataset):
    '''
    Meta-dataset that handles rebalancing of multiple datasets during training
    - Has transformations to ensure that correct datasets are indexed
    - HIGHLY RECOMMENDED that the dataloader is shuffle=True, or else you'll iterate through datasets sequentially (VERY bad curriculum for training)
    '''

    def __init__(self,
            dataset_dict: dict,
            batch_size: int,
            shuffle = False,
            seed = 123456,
        ):

        self.datasets = dataset_dict
        self.batch_size = batch_size
        self.dataset_keys = list(self.datasets.keys()) # Aligned by index to dataset_sizes

        self.shuffle = shuffle
        self.seed = seed
        if self.shuffle:
            self.seed_by_key = {dkey:(self.seed+i) for i, dkey in enumerate(self.dataset_keys)}

        self._init_index()

        # Upper bounds the sizes of each dataset
        # Lays out each dataset in a cumulative sum such that we can scan and retrieve the correct index

    def _init_index(self):
        self.fixed_dataset_sizes = [len(self.datasets[d]) for d in self.dataset_keys]# Aligned by index to dataset_keys, dataset_upper_bounds

        self.total_number = sum(self.fixed_dataset_sizes) # Total number of samples

        sub_indices = []
        for j, dkey in enumerate(self.dataset_keys): # Yes, this is inefficient, but the inner operations are very lightweight
            N = self.fixed_dataset_sizes[j]
            input_list = np.arange(N)

            if self.shuffle: # Shuffle if needed
                rng = np.random.default_rng(seed = self.seed_by_key[dkey])
                rng.shuffle(input_list)

            input_list = input_list.tolist()
            for i in range(0, len(input_list) - self.batch_size, self.batch_size):
                sub_indices.append( (dkey, input_list[i:(i + self.batch_size)]) )
            else:
                if (N % self.batch_size) > 0:
                    # Get remainder size:
                    N = len(input_list)
                    start_i = N - (N % self.batch_size)
                    leftover = input_list[start_i:]
                    to_fill_n = self.batch_size - len(leftover)
                    L = leftover + input_list[:to_fill_n] # Get leftover indices from the beginning of the list, could shuffle
                    sub_indices.append((dkey, L))

        self.sub_index = sub_indices
        self.meta_index = np.arange(len(self.sub_index)).tolist()

    def __len__(self):
        return len(self.meta_index)

    def __getitem__(self,i):
        dataset_i = self.meta_index[i] # Should match 1-1 - i.e., meta_index[i] == i

        dkey, samples_i = self.sub_index[dataset_i]

        # Get list of outputs from dataset:
        list_dataset_outputs = [self.datasets[dkey][i] for i in samples_i]

        return dkey, list_dataset_outputs

class MetaCollator:
    '''
    Meta-collator that calls the required collator at the given iteration based on output from metadataset
    '''
    def __init__(self, collator_dict):
        self.collators = collator_dict

    def __call__(self, batch_input):

        assert len(batch_input) == 1, "Batch size in upstream collator is greater than 1"

        dataset_name, inputs = batch_input[0] # Index [0] because it's of len 1

        return dataset_name, self.collators[dataset_name](inputs)
