import os
import time

import torch
from torch.utils.data import *
from lircst_ana_dataset import LircstAnaDataset
from torch import Generator


beta_scheduler = 'cosine'

degradation: float = 0.0
global_normalisation: bool = False  # Use global normalization for the dataset

# TODO: transition to script not notebook
model_args = {
    "ECD-CAT": {
        "physics": False,  # Don't use physics-based loss
        "latent": False,  # Don't use latent diffusion
        "predict_mode": 'eps',  # Use eps prediction
        "condition_A_T": True,  # Use A_T for conditioning
    },
    "ECD-Phys-CAT": {
        "physics": True,  # Use physics-based loss
        "latent": False,  # Don't use latent diffusion
        "predict_mode": 'eps',  # Use eps prediction
        "condition_A_T": True,  # Use A_T for conditioning
    },
    "ECD-Phys": {
        "physics": True,  # Use physics-based loss
        "latent": False,  # Don't use latent diffusion
        "predict_mode": 'x0',
        "condition_A_T": False,
    },
    "ECD": {
        "physics": False,  # Don't use physics-based loss
        "latent": False,  # Don't use latent diffusion
        "predict_mode": 'x0',
        "condition_A_T": False,
    },
    "ECLD-CAT": {
        "physics": False,  # Don't use physics-based loss
        "latent": True,  # Use latent diffusion
        "predict_mode": 'eps',  # Use eps prediction
        "condition_A_T": True,  # Use A_T for conditioning
    },
    "ECLD-Phys-CAT": {
        "physics": True,  # Use physics-based loss
        "latent": True,  # Use latent diffusion
        "predict_mode": 'eps',  # Use eps prediction
        "condition_A_T": True,  # Use A_T for conditioning
    },
    "ECLD-Phys": {
        "physics": True,  # Use physics-based loss
        "latent": True,  # Use latent diffusion
        "predict_mode": 'x0',
        "condition_A_T": False,
    },
    "ECLD": {
        "physics": False,  # Don't use physics-based loss
        "latent": True,  # Use latent diffusion
        "predict_mode": 'x0',
        "condition_A_T": False,
    },
}


def get_dataset():
    dataset = LircstAnaDataset('/home/samnub/dev/lircst-ana/data/')

    rand_generator = Generator().manual_seed(42) # The meaning of life, the universe and everything

    dataset_train, dataset_valid, dataset_test = random_split(dataset, [0.8, 0.1, 0.1], generator=rand_generator)

    print(f"Train set size: {len(dataset_train)}")
    print(f"Validation set size: {len(dataset_valid)}")
    print(f"Test set size: {len(dataset_test)}")

    return dataset_train, dataset_valid, dataset_test


def generate_directory_name(model_name, preexisting: str|None=None):
    timestamp = int(time.time())
    return f"../models/{model_name}/{timestamp if preexisting is None else preexisting}/", timestamp


def get_latest_ckpt(model_name, latest_dir: str|None=None):
    try:
        model_dir = f"../models/{model_name}/"
        if not os.path.exists(model_dir):
            return None, None
        directories = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if not directories:
            return None, None

        latest_dir = latest_dir if latest_dir is not None else sorted(directories)[-1]

        lightning_logs_dir = f'{model_dir}{latest_dir}/lightning_logs/'

        ckpt_dir = f'{model_dir}{latest_dir}/lightning_logs/{sorted(os.listdir(lightning_logs_dir))[-1]}/checkpoints/'

        ckpt_filename = os.listdir(ckpt_dir)[0] if os.path.exists(ckpt_dir) else None
        if ckpt_filename is None:
            print(f"No checkpoint found for {model_name} in {ckpt_dir}")
            return None, None
    
        return f'{ckpt_dir}{ckpt_filename}', latest_dir
    except Exception as e:
        print(f"Error getting latest checkpoint for {model_name}: {e}")
        return None, None
    

def extract(a, t, x_shape):
    # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/util.py#L103
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/1d9d8dffb72e02172da8a77bee039b1c72b7c6d5/denoising_diffusion_pytorch/repaint.py#L431
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def poisson_noise(x: torch.Tensor, noise_factor=0.1):
    """
    Add Poisson noise to the input tensor x.
    """
    # TODO: Make sure x is non-negative, as Poisson noise is defined for non-negative values

    noise = torch.poisson(x * noise_factor) / noise_factor
    return x + noise - x.mean(dim=(1, 2, 3), keepdim=True)  # Center the noise around zero


def sino_undersample(y: torch.Tensor, mask_proportion: float=0.2) -> torch.Tensor:
    """
    Undersample the input tensor x using the provided mask.
    The mask should be a binary tensor of the same shape as x.
    """
    # We are making a sinogram-like mask here, so vertically undersample the input tensor.

    # assert we have BCHW format
    if y.dim() != 4:
        raise ValueError(f"Input tensor y must be in BCHW format, got {y.dim()} dimensions instead.")

    mask: torch.Tensor = (torch.rand([y.shape[0], y.shape[1], 1, y.shape[3]]) > mask_proportion).expand(y.shape)  # Create a random boolean mask
    mask = mask.float().cuda()  # Convert mask to float for multiplication

    return y * mask  # Element-wise multiplication to apply the mask


def gaussian_log_likelihood(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, return_full: bool=False) -> torch.Tensor:
    # https://github.com/jhbastek/PhysicsInformedDiffusionModels/blob/main/src/denoising_toy_utils.py#L372
    centered_x = x - mean
    squared_diffs = (centered_x ** 2) / var
    if return_full:
        log_likelihood = -0.5 * (squared_diffs + torch.log(var) + torch.log(2 * torch.pi)) # Full log likelihood with constant terms
    else:
        log_likelihood = -0.5 * squared_diffs

    # avoid log(0)
    log_likelihood = torch.clamp(log_likelihood, min=-27.6310211159)

    return log_likelihood