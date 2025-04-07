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

    def __init__(self, data_dir: str, transform_phan: transforms=None, transform_sino: transforms=None, scale: bool=True, fbp: bool=True):
        self.data_dir: str = data_dir
        self.transform_phan: transforms = transform_phan
        self.transform_sino: transforms = transform_sino
        self.scale: bool = scale
        self.fbp: bool = fbp
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
        
        if self.fbp:
            # Do filtered back projection on the sinogram
            # Assume sinogram is of shape (num_projections, num_detectors)

            # TODO: we have 100 bins. Separate them somehow or just sum along that axis

            angles = np.linspace(0, 2*np.pi, sino.shape[1], endpoint=False)
            proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[0], angles) # Technically the second half of our analytical solution is parallel
            vol_geom = astra.create_vol_geom(sino.shape[0], sino.shape[0])
            
            sinogram_id = astra.data2d.create('-sino', proj_geom, sino)
            rec_id = astra.data2d.create('-vol', vol_geom, 0)
            cfg = astra.astra_dict('FBP_CUDA')
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            reconstruction = astra.data2d.get(rec_id) # TODO return this 2D array as a 3D array
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sinogram_id)

        if self.scale:
            # Normalize the phantom and sinogram data to [-1, 1]
            # This is done by first converting to float32,
            # Dividing by half of the max value of the tensor, and then subtracting 1
            phan = phan.astype(np.float32)
            phan = phan / (np.max(phan) / 2) - 1
            sino = sino.astype(np.float32)
            sino = sino / (np.max(sino) / 2) - 1

        if self.transform_phan:
            phan = self.transform_phan(phan)
        if self.transform_sino:
            sino = self.transform_sino(sino)

        return phan, sino, phantom_id

