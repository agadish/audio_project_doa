import torch
import torch.nn.functional as F
from typing import List, Dict
import os
from config import ANGLE_RES, NUM_CLASSES


class DictTorchPartedDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, prefix: str, fields: List[str], real_batch_size: int = 64, virtual_batch_size: int = 1, device: str = 'cpu',
                 limit_num_files: int = 0, pin_memory: bool = True):
        super().__init__()
        self._dir_path = dir_path
        self._prefix = prefix
        self._fields = fields
        self._virtual_batch_size = virtual_batch_size
        self._real_batch_size = real_batch_size
        self._bs_ratio = real_batch_size // virtual_batch_size
        self._device = device
        self._limit_num_files = limit_num_files
        self._init_calc()

    def _init_calc(self):
        max_idx = 0
        for p in os.listdir(self._dir_path):
            if not p.startswith(self._prefix):
                continue
            stripped_p = p[len(self._prefix) : p.rfind('.')]
            data_idx = stripped_p.split('_')[-1]
            
            try:            
                data_idx = int(data_idx)
            except Exception as e:
                continue

            max_idx = max(max_idx, data_idx)

        if self._limit_num_files:
            self._num_batches = self._limit_num_files
        else:        
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

        # Process target - NOT NEEDED ANYMORE
        # self._process_data(data)

        return [data[field][idx_in * self._virtual_batch_size : (idx_in + 1) * self._virtual_batch_size].to(self._device)
                for field in self._fields]
    
    def _process_data(self, data: Dict[str, torch.tensor]):
        raise ValueError('Dont need it')
        target = data['target'] // ANGLE_RES
        target = F.one_hot(target, num_classes=NUM_CLASSES)
        target = target.permute([0, 3, 1, 2]).float()
        data['target'] = target


class PinDictTorchPartedDataset(torch.utils.data.TensorDataset):
    def __init__(self, dir_path: str, prefix: str, fields: List[str], real_batch_size: int = 64, virtual_batch_size: int = 1, device: str = 'cpu',
                 limit_num_files: int = 0):
        self._dir_path = dir_path
        self._prefix = prefix
        self._fields = fields
        self._virtual_batch_size = virtual_batch_size
        self._real_batch_size = real_batch_size
        self._bs_ratio = real_batch_size // virtual_batch_size
        self._device = device
        self._limit_num_files = limit_num_files
        self._init_calc()
        self._prefetch_data()
        super().__init__(*self._data)

    def _prefetch_data(self):
        data = {}
        for i in range(self._num_batches):
            with open(self._get_idx_path(i), 'rb') as f:
              current_data = torch.load(f)
              for k in self._fields:
                data.setdefault(k, []).append(current_data[k])
        
        data = [torch.cat(data[f]) for f in self._fields]
        self._data = data

    def _init_calc(self):
        max_idx = 0
        for p in os.listdir(self._dir_path):
            if not p.startswith(self._prefix):
                continue
            stripped_p = p[len(self._prefix) : p.rfind('.')]
            data_idx = stripped_p.split('_')[-1]
            
            try:            
                data_idx = int(data_idx)
            except Exception as e:
                continue

            max_idx = max(max_idx, data_idx)

        if self._limit_num_files:
            self._num_batches = self._limit_num_files
        else:        
            self._num_batches = max_idx + 1

    def _get_idx_path(self, idx: int):
        idx_basename = f"{self._prefix}_{idx}.pt"
        result = os.path.join(self._dir_path, idx_basename)
        return result
    
    def __len__(self):
        return self._num_batches * self._bs_ratio
