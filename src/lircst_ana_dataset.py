# Custom Torch Dataset for our phantom / sinogram data

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import astra
import torch
import hashlib

class LircstAnaDataset(Dataset):
    # Our data for our analytical simulation is laid out as such:
    # /data
    #   /<phantom-id>
    #       /meta.npy (contains metadata about the phantom, applies to all slices
    #       /phan-<slice-idx>.npy (contains the slice Ground Truth) (2x128x128)
    #       /sino-<slice-idx>.npy (contains the sinogram of the slice) (128x200x100)

    def __init__(self, data_dir: str, transform_phan: transforms=None, transform_sino: transforms=None):
        self.data_dir: str = data_dir
        self.transform_phan: transforms = transform_phan
        self.transform_sino: transforms = transform_sino
        # Get all the phantom_ids
        self.phantom_ids: list[str] = os.listdir(data_dir)
        self.phantom_ids.sort()
        # Iterate over all phantom_id directories and get all slice indices
        self.idxs: list[tuple[str, int]] = []
        for phantom_id in self.phantom_ids:
            phantom_dir = os.path.join(data_dir, phantom_id)
            # P.S Get sinogram count - as it asserts that the phan also exists
            slice_idxs = [int(f.split('-')[1].split('.')[0]) for f in os.listdir(phantom_dir) if f.startswith('sino-')]
            for idx in slice_idxs:
                self.idxs.append((phantom_id, idx))

        
    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int):
        # Return a tuple of the phantom slice, the sinogram, and the phantom_id (in case we need to look up the metadata)
        phantom_id, slice_idx = self.idxs[idx]
        phantom_dir = os.path.join(self.data_dir, phantom_id)
        phan: torch.Tensor = torch.from_numpy(np.load(os.path.join(phantom_dir, f'phan-{slice_idx}.npy'))).float()
        sino: torch.Tensor = torch.from_numpy(np.load(os.path.join(phantom_dir, f'sino-{slice_idx}.npy'))).float()

        if self.transform_phan:
            phan = self.transform_phan(phan)
        if self.transform_sino:
            sino = self.transform_sino(sino)

        return phan, sino, phantom_id

    def get_phan_metadata(self, phantom_id: str) -> dict:
        # Load the metadata for the given phantom_id
        phantom_dir = os.path.join(self.data_dir, phantom_id)
        meta_path = os.path.join(phantom_dir, 'meta.npy')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Metadata file not found for phantom_id {phantom_id}')
        return np.load(meta_path, allow_pickle=True).item()
    
