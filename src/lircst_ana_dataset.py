# Custom Torch Dataset for our phantom / sinogram data

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import astra

class LircstAnaDataset(Dataset):
    # Our data for our analytical simulation is laid out as such:
    # /data
    #   /<phantom-id>
    #       /meta.npy (contains metadata about the phantom, applies to all slices
    #       /phan-<slice-idx>.npy (contains the slice Ground Truth) (2x128x128)
    #       /sino-<slice-idx>.npy (contains the sinogram of the slice) (128x200x100)

    def __init__(self, data_dir: str, transform_phan: transforms=None, transform_sino: transforms=None, scale: bool=True):
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
        
    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        # Return a tuple of the phantom slice, the sinogram, and the phantom_id (in case we need to look up the metadata)
        phantom_id, slice_idx = self.idxs[idx]
        phantom_dir = os.path.join(self.data_dir, phantom_id)
        phan = np.load(os.path.join(phantom_dir, f'phan-{slice_idx}.npy'))
        sino = np.load(os.path.join(phantom_dir, f'sino-{slice_idx}.npy'))

        print(f'Phantom {phantom_id} slice {slice_idx} - phan[0] range: {np.min(phan[0])} - {np.max(phan[0])}')
        print(f'Phantom {phantom_id} slice {slice_idx} - phan[1] range: {np.min(phan[1])} - {np.max(phan[1])}')
        print(f'Phantom {phantom_id} slice {slice_idx} - sino range: {np.min(sino)} - {np.max(sino)}')


        if self.scale:
            # Normalize the phantom and sinogram data to [-1, 1]
            # This is done by first converting to float32,
            # Dividing by half of the max value of the tensor, and then subtracting 1
            phan = phan.astype(np.float32)
            phan[0] = phan[0] / (np.max(phan[0]) / 2) - 1
            phan[1] = phan[1] / (np.max(phan[1]) / 2) - 1
            sino = sino.astype(np.float32)
            sino = sino / (np.max(sino) / 2) - 1

        # print ranges of the data
        print(f'Phantom {phantom_id} slice {slice_idx} - phan[0] range: {np.min(phan[0])} - {np.max(phan[0])}')
        print(f'Phantom {phantom_id} slice {slice_idx} - phan[1] range: {np.min(phan[1])} - {np.max(phan[1])}')
        print(f'Phantom {phantom_id} slice {slice_idx} - sino range: {np.min(sino)} - {np.max(sino)}')

        if self.transform_phan:
            phan = self.transform_phan(phan)
        if self.transform_sino:
            sino = self.transform_sino(sino)

        return phan, sino, phantom_id

