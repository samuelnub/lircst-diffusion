import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Diffusion.DenoisingDiffusionProcess.forward import GaussianForwardProcess

class PhysicsIncorporated(nn.Module):
    def __init__(self, 
                 gaussian_forward_process: GaussianForwardProcess,
                 forward_operator_dir: str):
        self.gaussian_forward_process = gaussian_forward_process

        # TODO:
        # Load the forward operator from the specified directory

    def forward(self, x_t: torch.Tensor, noise_hat: torch.Tensor, t):
        # Apply forward operator to our predicted x_0 based on x_t and noise_hat

        # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py#L236
        # Is the extract method even needed here?

        pass

def extract(a, t, x_shape):
    # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/util.py#L103
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/1d9d8dffb72e02172da8a77bee039b1c72b7c6d5/denoising_diffusion_pytorch/repaint.py#L431
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))