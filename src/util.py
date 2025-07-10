import os
import time

import numpy as np
import torch
from torch.utils.data import *
from lircst_ana_dataset import LircstAnaDataset
from torch import Generator


beta_scheduler = 'cosine'

global_normalisation: bool = True  # Use global normalization for the dataset

models_dir = '/home/samnub/dev/lircst-diffusion/models/'

model_args = {
    "ECD-CAT": {
        "physics": False,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": True,
        "degradation": 0.0,
    },

    "ECD": {  # The original ECD model without CAT
        "physics": False,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": False,
        "degradation": 0.0,
    },
    "ECD-Phys": {  # The original ECD model without CAT
        "physics": True,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": False,
        "degradation": 0.0,
    },

}

model_args_unused = {
    "ECD-CAT": {
        "physics": False,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": True,
        "degradation": 0.0,
    },
    "ECD-Phys-CAT": {
        "physics": True,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": True,
        "degradation": 0.0,
    },

    "ECD-Phys-CAT-D20": {
        "physics": True,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": True,
        "degradation": 0.2,
    },
    "ECD-CAT-D20": {
        "physics": False,
        "latent": False,
        "predict_mode": 'v',
        "condition_A_T": True,
        "degradation": 0.2,
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
    return f"{models_dir}{model_name}/{timestamp if preexisting is None else preexisting}/", timestamp


def get_latest_ckpt(model_name, latest_dir: str|None=None):
    try:
        model_dir = f"{models_dir}{model_name}/"
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


def poisson_noise(x: torch.Tensor, noise_factor=0.1, clamp=True, scale: float=1e5):
    """
    Add Poisson noise to the input tensor x.
    As we are adding it to raw sinogram (before further transformations down the road), it does not need to be log-poisson.
    """
    # Make sure x is non-negative, as Poisson noise is defined for non-negative values

    x_clean_min = x.min()
    x_clean_max = x.max()

    noise_factor = noise_factor * scale # For our sinograms which usually range in 1e-5, we need a big factor

    lam = x * noise_factor  # Lambda parameter for Poisson distribution
    noisy = torch.poisson(lam) / noise_factor  # Generate Poisson noise

    if clamp:
        # Clamp the noisy tensor to the range of the original tensor
        noisy = torch.clamp(noisy, min=x_clean_min, max=x_clean_max)  

    return noisy


def gaussian_noise(x: torch.Tensor, noise_factor=0.1, clamp=True, scale: float=1e-4):
    """
    Add Gaussian noise to the input tensor x.
    """
    # Make sure x is non-negative, as Gaussian noise can be negative
    x_clean_min = x.min()
    x_clean_max = x.max()

    noise = torch.randn_like(x) * noise_factor * scale # Generate Gaussian noise

    noisy = x + noise  # Add noise to the original tensor

    if clamp:
        # Clamp the noisy tensor to the range of the original tensor
        noisy = torch.clamp(noisy, min=x_clean_min, max=x_clean_max)

    return noisy


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


def snr_db(signal: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    Calculate the SNR in dB.
    https://en.wikipedia.org/wiki/Signal-to-noise_ratio
    We use mu/sigma SNR calculation, which is common in image processing (as our mean is not zero centred).
    """
    noise = noisy - signal  # Calculate noise by subtracting the original signal from the noisy signal
    noise = noise.std()
    if noise == 0:
        return float('inf')  # If noise is zero, SNR is infinite
    snr = signal.mean() / noise  # Calculate SNR as the ratio of the mean signal to the standard deviation of the noise
    snr_db_value = to_decibels(snr)  # Convert SNR to decibels
    return snr_db_value.item()

def to_decibels(value: float) -> float:
    """
    Convert a linear value to decibels (amplitude).
    """
    if value <= 0:
        raise ValueError("Value must be positive to convert to decibels.")
    return 20 * torch.log10(value)


from data_compute import DataCompute as DC
def mlem(y: torch.Tensor,
         op_name: str, # A or A_T operator name (e.g. 'ut')
         dc: DC,
         iterations: int = 20,
         y_shape=(128,200),
         x_shape=(128,128)) -> torch.Tensor:
    # Perform Maximum Likelihood Expectation Maximization (MLEM) algorithm
    # On our measured sinogram y.
    assert len(y.shape) == 2, 'Sinogram input must only be 2 dimensional'

    x_rec: torch.Tensor = torch.ones(x_shape)
    y_ones: torch.Tensor = torch.ones(y_shape)
    # Sensitivity image
    sens_image = dc.A_T(y_ones.unsqueeze(0).unsqueeze(0), op_name).squeeze(0).squeeze(0)

    for iter in range(iterations):
        fp = dc.A(x_rec.unsqueeze(0).unsqueeze(0), op_name).squeeze(0).squeeze(0) # Forward projection
        ratio = y / (fp + 0.000001) # Ratio of measured to estimated (epsilon to prevent 0 div)
        correction = dc.A_T(ratio.unsqueeze(0).unsqueeze(0), op_name).squeeze(0).squeeze(0)
        x_rec = x_rec * correction

    return x_rec