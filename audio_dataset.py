import torch
from typing import List
import os


class DistributedTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, prefix: str, fields: List[str], real_batch_size: int = 64, virtual_batch_size: int = 1):
        super().__init__()
        self._dir_path = dir_path
        self._prefix = prefix
        self._fields = fields
        self._virtual_batch_size = virtual_batch_size
        self._real_batch_size = real_batch_size
        self._bs_ratio = real_batch_size // virtual_batch_size
        self._init_calc()

    def _init_calc(self):
        max_idx = 0
        for p in os.listdir(self._dir_path):
            if not p.startswith(self._prefix):
                continue
            stripped_p = p[len(self._prefix) : p.rfind('.')]
            data_idx = p[:p.rfind('.')].split('_')
            
            try:            
                data_idx = int(data_idx)
            except Exception as e:
                continue

            max_idx = max(max_idx, data_idx)
        
        self._num_batches = max_idx + 1

    def _get_idx_path(self, idx: int):
        idx_basename = f"{self._prefix}_{idx}.pt"
        result = os.path.join(self._dir_path, idx_basename)
        return result
    
    def __len__(self):
        return self._num_batches * self._bs_ratio
    
    def __getitem__(self, idx):
        if idx >= self._num_batches * self._bs_ratio:
            raise StopIteration()

        idx_file = idx // self._bs_ratio
        idx_in = idx % self._bs_ratio
        with open(self._get_idx_path(idx_file), 'rb') as f:
            data = torch.load(f)

        if self._virtual_batch_size == 1:
            return [data[field][idx_in] for field in self._fields]
            
        return [data[field][idx_in * self._virtual_batch_size : (idx_in + 1) * self._virtual_batch_size]
                for field in self._fields]
    

class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.num_batches = len(dataset)
        super().__init__(self, batch_size=dataset._virtual_batch_size, shuffle=False, **kwargs)

    def __iter__(self):
        # Ignore self.shuffle
        batch_indices = torch.arange(len(self.dataset))
        for batch_idx in batch_indices:
            # Yield the batch data
            yield self.dataset[batch_idx]