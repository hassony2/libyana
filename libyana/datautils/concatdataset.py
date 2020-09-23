from torch.utils.data import Dataset


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        lengths = [len(dataset) for dataset in datasets]
        # Map idx to dataset idx
        dataset_idxs = [[dat_idx] * dat_len for dat_idx, dat_len in enumerate(lengths)]
        self.dataset_idxs = [val for vals in dataset_idxs for val in vals]
        # Map idx to idx of sample in dataset
        data_idxs = [list(range(length)) for length in lengths]
        self.data_idxs = [val for vals in data_idxs for val in vals]
        self.tot_len = sum(lengths)

    def __getitem__(self, idx):
        dataset = self.datasets[self.dataset_idxs[idx]]
        data_idx = self.data_idxs[idx]
        return dataset[data_idx]

    def __len__(self):
        return self.tot_len
