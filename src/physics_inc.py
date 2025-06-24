import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Diffusion.DenoisingDiffusionProcess.forward import GaussianForwardProcess

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

class PhysicsIncorporated(nn.Module):
    def __init__(self, 
                 gaussian_forward_process: GaussianForwardProcess,
                 A_ut_dir: str | None = None,
                 A_ub_dir: str | None = None,
                 A_tb_dir: str | None = None,):
        super(PhysicsIncorporated, self).__init__()
        self.gfp: GaussianForwardProcess = gaussian_forward_process

        self.A_ut: torch.Tensor | None = None
        self.A_ub: torch.Tensor | None = None
        self.A_tb: torch.Tensor | None = None

        # Load the forward operator matrices from the specified directory
        if A_ut_dir is not None:
            self.A_ut = torch.from_numpy(np.load(A_ut_dir)).float().cuda()
        if A_ub_dir is not None:
            self.A_ub = torch.from_numpy(np.load(A_ub_dir)).float().cuda()
        if A_tb_dir is not None:
            self.A_tb = torch.from_numpy(np.load(A_tb_dir)).float().cuda()


    def forward(self, x_t: torch.Tensor, noise_hat: torch.Tensor, t, y: torch.Tensor) -> torch.Tensor:
        # Apply forward operator to our predicted x_0 based on x_t and noise_hat, and calculate loss between the predicted and actual y.

        # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddim.py#L236
        # Is the extract method even needed here?

        # Stochastic sampling to only use one sample from the batch
        i = torch.randint(low=0, high=x_t.shape[0], size=(1,)).item()

        x_0_pred: torch.Tensor = (x_t - extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * noise_hat) / extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape)

        loss_total = 0.0

        # Apply the forward operator to x_0_pred
        if self.A_ut is not None:
            sino_pred_ut = (self.A_ut @ x_0_pred[i].sum(dim=-3).view(-1)).view(1, 1, x_t.shape[-2], -1)
            # Scale predicted sinogram to the same range as y
            sino_pred_ut_min = sino_pred_ut.min()
            sino_pred_ut_max = sino_pred_ut.max()
            sino_pred_ut = (sino_pred_ut - sino_pred_ut_min) / (sino_pred_ut_max - sino_pred_ut_min) * 2 - 1

            # Interpolate/resize the predicted sinogram to match the shape of y (we assume y has passed through the conditional encoder)
            sino_pred_ut = F.interpolate(sino_pred_ut, size=y.shape[-2:], mode='bilinear', align_corners=False)

            loss_ut = F.mse_loss(sino_pred_ut, y[i])
            loss_total += loss_ut

            # debugging: plot predicted x_0 and sino_pred_ut and y
            if False:
                clear_output(wait=True)
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 4, 1)
                plt.title(f'Predicted x_0 (scat) (i:{i})')
                plt.imshow(x_0_pred[i][0].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.title(f'Predicted x_0 (atten) (t:{t[i].item()})')
                plt.imshow(x_0_pred[i][1].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.title('Predicted Sinogram')
                plt.imshow(sino_pred_ut[0][0].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.title(f'Actual Sinogram (y), loss:{loss_ut.item():.4f}')
                plt.imshow(y[i][0].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')

                display(plt.gcf())

        # TODO: Implement A_ub and A_tb if needed

        return loss_total

def extract(a, t, x_shape):
    # https://github.com/Stability-AI/stablediffusion/blob/main/ldm/modules/diffusionmodules/util.py#L103
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/1d9d8dffb72e02172da8a77bee039b1c72b7c6d5/denoising_diffusion_pytorch/repaint.py#L431
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))