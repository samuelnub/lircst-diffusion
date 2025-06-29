import os
import time

from torch.utils.data import *
from lircst_ana_dataset import LircstAnaDataset
from torch import Generator


beta_scheduler = 'cosine'


model_args = {
    "ECD-Phys": {
        "physics": True,  # Use physics-based loss
        "latent": False,  # Don't use latent diffusion
    },
    "ECD": {
        "physics": False,  # Don't use physics-based loss
        "latent": False,  # Don't use latent diffusion
    },
    "ECLD-Phys": {
        "physics": True,  # Use physics-based loss
        "latent": True,  # Use latent diffusion
    },
    "ECLD": {
        "physics": False,  # Don't use physics-based loss
        "latent": True,  # Use latent diffusion
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