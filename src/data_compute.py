import torch
import torch.nn as nn
import os
import numpy as np
import hashlib

from lircst_ana_dataset import LircstAnaDataset


class DataCompute():
    """
    Class to compute data-related operations.
    """
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

    sino_ut_mean: float | None = None
    sino_ut_std: float | None = None
    sino_ut_min: float | None = None
    sino_ut_max: float | None = None
    
    sino_ut_a_t_mean: float | None = None
    sino_ut_a_t_std: float | None = None
    sino_ut_a_t_min: float | None = None
    sino_ut_a_t_max: float | None = None

    
    def __init__(self, data_dir: str, operator_dir: str):
        self.data_dir = data_dir
        self.stats_dir = '../data/'
        self.operator_dir = operator_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset = LircstAnaDataset(data_dir=self.data_dir)

        self.A_ut: torch.Tensor | None = None
        self.A_ub: torch.Tensor | None = None
        self.A_tb: torch.Tensor | None = None

        self.A_ut_T: torch.Tensor | None = None
        self.A_ub_T: torch.Tensor | None = None
        self.A_tb_T: torch.Tensor | None = None

        # Load the forward operator matrices from the specified directory
        if self.operator_dir is not None:
            A_ut_path = os.path.join(self.operator_dir, 'A_ut.npy')
            A_ub_path = os.path.join(self.operator_dir, 'A_ub.npy')
            A_tb_path = os.path.join(self.operator_dir, 'A_tb.npy')
            if os.path.exists(A_ut_path):
                self.A_ut = torch.from_numpy(np.load(A_ut_path)).float().to(self.device)
                self.A_ut_T = self.A_ut.T  # Precompute the transpose for efficiency
            if os.path.exists(A_ub_path):
                self.A_ub = torch.from_numpy(np.load(A_ub_path)).float().to(self.device)
                self.A_ub_T = self.A_ub.T  # Precompute the transpose for efficiency
            if os.path.exists(A_tb_path):
                self.A_tb = torch.from_numpy(np.load(A_tb_path)).float().to(self.device)
                self.A_tb_T = self.A_tb.T  # Precompute the transpose for efficiency

        self.load_save_stats()  # Load or compute the statistics of the dataset


    def A(self, x: torch.Tensor, operator: str) -> torch.Tensor:
        """
        Apply the specified forward operator to the input tensor x.
        """
        y: torch.Tensor | None = None

        A_matrix: torch.Tensor | None = None
        if operator == 'ut':
            A_matrix = self.A_ut
        elif operator == 'ub':
            A_matrix = self.A_ub
        elif operator == 'tb':
            A_matrix = self.A_tb

        for i in range(x.shape[0]):
            y_i = (A_matrix @ x[i].sum(dim=-3).view(-1)).view(1, 1, x.shape[-2], -1)
            if y is None:
                y = y_i
            else:
                y = torch.cat((y, y_i), dim=0)
        return y
    
    def A_T(self, y: torch.Tensor, operator: str) -> torch.Tensor:
        """
        Apply the transpose of the specified forward operator to the input tensor y.
        """
        x: torch.Tensor | None = None

        A_T_matrix: torch.Tensor | None = None
        if operator == 'ut':
            A_T_matrix = self.A_ut_T
        elif operator == 'ub':
            A_T_matrix = self.A_ub_T
        elif operator == 'tb':
            A_T_matrix = self.A_tb_T

        for i in range(y.shape[0]):
            x_i = (A_T_matrix @ y[i].sum(dim=-3).view(-1)).view(1, 1, y.shape[-2], -1)
            if x is None:
                x = x_i
            else:
                x = torch.cat((x, x_i), dim=0)
        return x

    def load_save_stats(self):
        # Try to load the stats from a file
        stats_pth = os.path.join(self.stats_dir, 'dataset_meta.npy')
        # Compute an md5 hash of our directories to check if the mean and std are still valid
        md5 = hashlib.md5(bytes(''.join(self.dataset.phantom_ids), 'utf-8')).hexdigest()
        if os.path.exists(stats_pth):
            # If the file exists, load it
            stats = np.load(stats_pth, allow_pickle=True).item()
            
            if stats['md5'] != md5:
                # If the md5 hash does not match, recompute the mean and std
                print(f"MD5 hash mismatch for {stats_pth}, recomputing...")
                stats = self.compute_stats()
                stats['md5'] = md5
                np.save(stats_pth, stats)

            DataCompute.phan0_mean = stats['phan0_mean']
            DataCompute.phan0_std = stats['phan0_std']
            DataCompute.phan0_min = stats['phan0_min']
            DataCompute.phan0_max = stats['phan0_max']

            DataCompute.phan1_mean = stats['phan1_mean']
            DataCompute.phan1_std = stats['phan1_std']
            DataCompute.phan1_min = stats['phan1_min']
            DataCompute.phan1_max = stats['phan1_max']

            DataCompute.sino_mean = stats['sino_mean']
            DataCompute.sino_std = stats['sino_std']
            DataCompute.sino_min = stats['sino_min']
            DataCompute.sino_max = stats['sino_max']

            DataCompute.sino_ut_mean = stats['sino_ut_mean']
            DataCompute.sino_ut_std = stats['sino_ut_std']
            DataCompute.sino_ut_min = stats['sino_ut_min']
            DataCompute.sino_ut_max = stats['sino_ut_max']
            
            DataCompute.sino_ut_a_t_mean = stats['sino_ut_a_t_mean']
            DataCompute.sino_ut_a_t_std = stats['sino_ut_a_t_std']
            DataCompute.sino_ut_a_t_min = stats['sino_ut_a_t_min']
            DataCompute.sino_ut_a_t_max = stats['sino_ut_a_t_max']

            print(f"Loaded stats of dataset from {stats_pth}, {stats}")
        else:
            # If the file does not exist, compute the mean and std
            print(f"Stats file {stats_pth} does not exist, computing...")
            stats = self.compute_stats()
            stats['md5'] = md5
            np.save(stats_pth, stats)

            DataCompute.phan0_mean = stats['phan0_mean']
            DataCompute.phan0_std = stats['phan0_std']
            DataCompute.phan0_min = stats['phan0_min']
            DataCompute.phan0_max = stats['phan0_max']

            DataCompute.phan1_mean = stats['phan1_mean']
            DataCompute.phan1_std = stats['phan1_std']
            DataCompute.phan1_min = stats['phan1_min']
            DataCompute.phan1_max = stats['phan1_max']

            DataCompute.sino_mean = stats['sino_mean']
            DataCompute.sino_std = stats['sino_std']
            DataCompute.sino_min = stats['sino_min']
            DataCompute.sino_max = stats['sino_max']

            DataCompute.sino_ut_mean = stats['sino_ut_mean']
            DataCompute.sino_ut_std = stats['sino_ut_std']
            DataCompute.sino_ut_min = stats['sino_ut_min']
            DataCompute.sino_ut_max = stats['sino_ut_max']
            
            DataCompute.sino_ut_a_t_mean = stats['sino_ut_a_t_mean']
            DataCompute.sino_ut_a_t_std = stats['sino_ut_a_t_std']
            DataCompute.sino_ut_a_t_min = stats['sino_ut_a_t_min']
            DataCompute.sino_ut_a_t_max = stats['sino_ut_a_t_max']
            
            print(f"Computed and saved stats of dataset to {stats_pth}, {stats}")

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

        # Sinogram-paired-axes-specific statistics
        sino_ut_M = 0.0
        sino_ut_S = 0.0
        sino_ut_std = 0.0
        sino_ut_min = float('inf')
        sino_ut_max = float('-inf')

        sino_ut_a_t_M = 0.0
        sino_ut_a_t_S = 0.0
        sino_ut_a_t_std = 0.0
        sino_ut_a_t_min = float('inf')
        sino_ut_a_t_max = float('-inf')


        for k in range(1, len(self.dataset)+1):
            phan, sino, _ = self.dataset[k-1]
            phan = phan.to(self.device)
            sino = sino.to(self.device)

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

            # Compute sinogram-paired-axes-specific statistics
            # Remember, default sinogram is of the format [u, t, b]
            sino_ut = sino.clone().sum(dim=-1, keepdim=True)  # Sum over the b-axis
            sino_ut_oldM = sino_ut_M
            sino_ut_k_mean = sino_ut.mean()
            sino_ut_M = sino_ut_M + (sino_ut_k_mean - sino_ut_M) / k
            sino_ut_S = sino_ut_S + (sino_ut_k_mean - sino_ut_M) * (sino_ut_k_mean - sino_ut_oldM)

            sino_ut_min = min(sino_ut_min, sino_ut.min().item())
            sino_ut_max = max(sino_ut_max, sino_ut.max().item())

            # Compute sinogram-paired-axes-specific statistics for A_ut
            sino_ut_a_t = self.A_T(sino_ut.clone().permute(2, 0, 1).unsqueeze(0), 'ut')
            sino_ut_a_t_oldM = sino_ut_a_t_M
            sino_ut_a_t_k_mean = sino_ut_a_t.mean()
            sino_ut_a_t_M = sino_ut_a_t_M + (sino_ut_a_t_k_mean - sino_ut_a_t_M) / k
            sino_ut_a_t_S = sino_ut_a_t_S + (sino_ut_a_t_k_mean - sino_ut_a_t_M) * (sino_ut_a_t_k_mean - sino_ut_a_t_oldM)

            sino_ut_a_t_min = min(sino_ut_a_t_min, sino_ut_a_t.min().item())
            sino_ut_a_t_max = max(sino_ut_a_t_max, sino_ut_a_t.max().item())

            if k % 100 == 0:
                print(f"Processed {k} data samples")

        phan0_std = (phan0_S / (len(self.dataset) - 1)).sqrt()
        phan1_std = (phan1_S / (len(self.dataset) - 1)).sqrt()
        sino_std = (sino_S / (len(self.dataset) - 1)).sqrt()

        # Compute sinogram-paired-axes-specific statistics
        sino_ut_std = (sino_ut_S / (len(self.dataset) - 1)).sqrt()
        sino_ut_a_t_std = (sino_ut_a_t_S / (len(self.dataset) - 1)).sqrt()

        return {
            'phan0_mean': phan0_M.item(),
            'phan0_std': phan0_std.item(),
            'phan0_min': phan0_min,
            'phan0_max': phan0_max,

            'phan1_mean': phan1_M.item(),
            'phan1_std': phan1_std.item(),
            'phan1_min': phan1_min,
            'phan1_max': phan1_max,

            'sino_mean': sino_M.item(),
            'sino_std': sino_std.item(),
            'sino_min': sino_min,
            'sino_max': sino_max,

            'sino_ut_mean': sino_ut_M.item(),
            'sino_ut_std': sino_ut_std.item(),
            'sino_ut_min': sino_ut_min,
            'sino_ut_max': sino_ut_max,

            'sino_ut_a_t_mean': sino_ut_a_t_M.item(),
            'sino_ut_a_t_std': sino_ut_a_t_std.item(),
            'sino_ut_a_t_min': sino_ut_a_t_min,
            'sino_ut_a_t_max': sino_ut_a_t_max,
        }


