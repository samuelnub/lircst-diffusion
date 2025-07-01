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

    phan0_mean: float | None = None
    phan0_std: float | None = None
    phan0_min: float | None = None
    phan0_max: float | None = None

    phan1_mean: float | None = None
    phan1_std: float | None = None
    phan1_min: float | None = None
    phan1_max: float | None = None

    sino_mean: float | None = None
    sino_std: float | None = None
    sino_min: float | None = None
    sino_max: float | None = None

    def __init__(self, data_dir: str, transform_phan: transforms=None, transform_sino: transforms=None, scale: bool=False):
        self.data_dir: str = data_dir
        self.transform_phan: transforms = transform_phan
        self.transform_sino: transforms = transform_sino
        self.scale: bool = scale
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

        if False:
            # Load the mean and std of the dataset
            self.load_save_stats()
        
        
    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int):
        # Return a tuple of the phantom slice, the sinogram, and the phantom_id (in case we need to look up the metadata)
        phantom_id, slice_idx = self.idxs[idx]
        phantom_dir = os.path.join(self.data_dir, phantom_id)
        phan: torch.Tensor = torch.from_numpy(np.load(os.path.join(phantom_dir, f'phan-{slice_idx}.npy'))).float()
        sino: torch.Tensor = torch.from_numpy(np.load(os.path.join(phantom_dir, f'sino-{slice_idx}.npy'))).float()

        if self.scale:
            # Normalize the phantom and sinogram data to [-1, 1]
            # This is done by first converting to float32,
            # Dividing by half of the max value of the tensor, and then subtracting 1
            # The scatter map is normalised slightly differently
            # We scale non-zeroes to [0, 1] and zeroes to -1

            nonzero_phan0 = torch.nonzero(phan[0])

            nonzero_min_phan0 = torch.min(phan[0][nonzero_phan0])
            nonzero_max_phan0 = torch.max(phan[0][nonzero_phan0])

            min_phan1 = torch.min(phan[1])
            max_phan1 = torch.max(phan[1])

            min_sino = torch.min(sino)
            max_sino = torch.max(sino)

            if False: # TODO not nonzero_max_phan0 == nonzero_min_phan0:
                phan[0][nonzero_phan0] = (phan[0][nonzero_phan0] - nonzero_min_phan0) / (nonzero_max_phan0 - nonzero_min_phan0)
                phan[0][torch.where(phan[0] == 0.0)] = -1.0
            else:
                min_phan0 = torch.min(phan[0])
                max_phan0 = torch.max(phan[0])
                phan[0] = ((phan[0] - min_phan0) / (max_phan0 - min_phan0)) * 2 - 1

            phan[1] = ((phan[1] - min_phan1) / (max_phan1 - min_phan1)) * 2 - 1
            sino = ((sino - min_sino) / (max_sino - min_sino)) * 2 - 1

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
    
    def load_save_stats(self):
        # Try to load the mean and std from a file
        mean_std_path = os.path.join(self.data_dir, 'dataset_meta.npy')
        # Compute an md5 hash of our directories to check if the mean and std are still valid
        md5 = hashlib.md5(b"".join(self.phantom_ids)).hexdigest()
        if os.path.exists(mean_std_path):
            # If the file exists, load it
            mean_std = np.load(mean_std_path, allow_pickle=True).item()
            
            if mean_std['md5'] != md5:
                # If the md5 hash does not match, recompute the mean and std
                print(f"MD5 hash mismatch for {mean_std_path}, recomputing mean and std")
                mean_std = self.compute_stats()
                mean_std['md5'] = md5
                np.save(mean_std_path, mean_std)

            LircstAnaDataset.phan0_mean = mean_std['phan0_mean']
            LircstAnaDataset.phan0_std = mean_std['phan0_std']
            LircstAnaDataset.phan0_min = mean_std['phan0_min']
            LircstAnaDataset.phan0_max = mean_std['phan0_max']
            LircstAnaDataset.phan1_mean = mean_std['phan1_mean']
            LircstAnaDataset.phan1_std = mean_std['phan1_std']
            LircstAnaDataset.phan1_min = mean_std['phan1_min']
            LircstAnaDataset.phan1_max = mean_std['phan1_max']
            LircstAnaDataset.sino_mean = mean_std['sino_mean']
            LircstAnaDataset.sino_std = mean_std['sino_std']
            LircstAnaDataset.sino_min = mean_std['sino_min']
            LircstAnaDataset.sino_max = mean_std['sino_max']
            print(f"Loaded mean and std of dataset from {mean_std_path}, {mean_std}")
        else:
            # If the file does not exist, compute the mean and std
            print(f"Mean and std file {mean_std_path} does not exist, computing mean and std")
            mean_std = LircstAnaDataset.compute_stats()
            mean_std['md5'] = md5
            np.save(mean_std_path, mean_std)

            LircstAnaDataset.phan0_mean = mean_std['phan0_mean']
            LircstAnaDataset.phan0_std = mean_std['phan0_std']
            LircstAnaDataset.phan0_min = mean_std['phan0_min']
            LircstAnaDataset.phan0_max = mean_std['phan0_max']
            LircstAnaDataset.phan1_mean = mean_std['phan1_mean']
            LircstAnaDataset.phan1_std = mean_std['phan1_std']
            LircstAnaDataset.phan1_min = mean_std['phan1_min']
            LircstAnaDataset.phan1_max = mean_std['phan1_max']
            LircstAnaDataset.sino_mean = mean_std['sino_mean']
            LircstAnaDataset.sino_std = mean_std['sino_std']
            LircstAnaDataset.sino_min = mean_std['sino_min']
            LircstAnaDataset.sino_max = mean_std['sino_max']
            print(f"Computed and saved mean and std of dataset to {mean_std_path}, {mean_std}")


    def compute_stats(self):
        # Compute the summary statistics of the dataset using Welford's algorithm
        phan0_M = 0.0
        phan0_S = 0.0
        phan1_M = 0.0
        phan1_S = 0.0
        sino_M = 0.0
        sino_S = 0.0

        phan0_std = 0.0
        phan1_std = 0.0
        sino_std = 0.0

        phan0_min = float('inf')
        phan0_max = float('-inf')
        phan1_min = float('inf')
        phan1_max = float('-inf')
        sino_min = float('inf')
        sino_max = float('-inf')

        for k in range(1, len(self)+1):
            phan, sino, _ = self.__getitem__(k-1)

            phan0_oldM = phan0_M
            phan0_k_mean = phan[0].mean()
            phan0_M = phan0_M + (phan0_k_mean - phan0_M) / k
            phan0_S = phan0_S + (phan0_k_mean - phan0_M) * (phan0_k_mean - phan0_oldM)

            phan1_oldM = phan1_M
            phan1_k_mean = phan[1].mean()
            phan1_M = phan1_M + (phan1_k_mean - phan1_M) / k
            phan1_S = phan1_S + (phan1_k_mean - phan1_M) * (phan1_k_mean - phan1_oldM)

            sino_oldM = sino_M
            sino_k_mean = sino.mean()
            sino_M = sino_M + (sino_k_mean - sino_M) / k
            sino_S = sino_S + (sino_k_mean - sino_M) * (sino_k_mean - sino_oldM)

            # Update min and max values
            phan0_min = min(phan0_min, phan[0].min().item())
            phan0_max = max(phan0_max, phan[0].max().item())
            phan1_min = min(phan1_min, phan[1].min().item())
            phan1_max = max(phan1_max, phan[1].max().item())
            sino_min = min(sino_min, sino.min().item())
            sino_max = max(sino_max, sino.max().item())

        phan0_std = (phan0_S / (len(self) - 1)).sqrt()
        phan1_std = (phan1_S / (len(self) - 1)).sqrt()
        sino_std = (sino_S / (len(self) - 1)).sqrt()

        return {
            'phan0_mean': phan0_M,
            'phan0_std': phan0_std,
            'phan0_min': phan0_min,
            'phan0_max': phan0_max,
            'phan1_mean': phan1_M,
            'phan1_std': phan1_std,
            'phan1_min': phan1_min,
            'phan1_max': phan1_max,
            'sino_mean': sino_M,
            'sino_std': sino_std,
            'sino_min': sino_min,
            'sino_max': sino_max,
        }


