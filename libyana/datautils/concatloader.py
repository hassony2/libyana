import itertools
from torch.utils.data import Subset


import itertools


class ConcatDataloader:
    def __init__(self, dataloaders):
        self.loaders = dataloaders

    def __iter__(self):
        self.iters = [iter(loader) for loader in self.loaders]
        self.idx_cycle = itertools.cycle(list(range(len(self.loaders))))
        return self

    def __next__(self):
        loader_idx = next(self.idx_cycle)
        loader = self.iters[loader_idx]
        batch = next(loader)
        return batch

    def __len__(self):
        return min(len(loader) for loader in self.loaders) * len(self.loaders)
