import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ssim import SSIM
from Diffusion.DenoisingDiffusionProcess.forward import GaussianForwardProcess
import math
from util import extract

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import wandb

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

        self.stochastic_proportion: float = 1/4 # Only use 1/nth of the batch to compute loss

        self.loss_metric = SSIM(data_range=1.0, size_average=True, channel=1).cuda()  # Assuming single channel for sinogram

        self.image_width = 128

        # Load the forward operator matrices from the specified directory
        if A_ut_dir is not None:
            self.A_ut = torch.from_numpy(np.load(A_ut_dir)).float().cuda()
        if A_ub_dir is not None:
            self.A_ub = torch.from_numpy(np.load(A_ub_dir)).float().cuda()
        if A_tb_dir is not None:
            self.A_tb = torch.from_numpy(np.load(A_tb_dir)).float().cuda()

    def forward(self, x_t: torch.Tensor, target_pred: torch.Tensor, t, y: torch.Tensor, epoch_and_step: tuple|None=None) -> torch.Tensor:
        # Apply forward operator to our predicted x_0 based on x_t and noise_hat, and calculate loss between the predicted and actual y.

        # Stochastic sampling
        indices = torch.randperm(x_t.shape[0])[:math.floor(x_t.shape[0] * self.stochastic_proportion)]
        loss_total = 0.0

        if x_t.shape[-1] != self.image_width:
            x_t = F.interpolate(x_t, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)
        if target_pred.shape[-1] != self.image_width:
            target_pred = F.interpolate(target_pred, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)
        if y.shape[-1] != self.image_width:
            y = F.interpolate(y, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)
            y = y.mean(dim=1, keepdim=True)  # Assuming y is a single channel sinogram

        x_0_pred: torch.Tensor = target_pred
        # ^^^ Previously, this was computed as: (when doing eps-prediction)
        # torch.Tensor = (x_t - extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * noise_hat) / extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape)

        for i in indices:
            # Apply the forward operator to x_0_pred
            if self.A_ut is not None:
                sino_pred_ut = (self.A_ut @ x_0_pred[i].sum(dim=-3).view(-1)).view(1, 1, x_t.shape[-2], -1)
                # Scale predicted sinogram to the same range as y
                sino_pred_ut_min = sino_pred_ut.min()
                sino_pred_ut_max = sino_pred_ut.max()
                sino_pred_ut = (sino_pred_ut - sino_pred_ut_min) / (sino_pred_ut_max - sino_pred_ut_min) * 2 - 1

                # Interpolate/resize the predicted sinogram to match the shape of y (we assume y has passed through the conditional encoder)
                sino_pred_ut = F.interpolate(sino_pred_ut, size=y.shape[-2:], mode='bilinear', align_corners=False)

                loss_ut = 1 - self.loss_metric((sino_pred_ut+1)/2, (y[i].unsqueeze(0)+1)/2) # (1-ssim) For SSIM, we need to scale the images to [0, 1] range
                loss_total += loss_ut * (1/x_t.shape[0]) # Scale by batch size

                # debugging: plot predicted x_0 and sino_pred_ut and y
                if epoch_and_step is not None and epoch_and_step[0] % 5 == 0 and i == indices[0]:  # Only plot for the first sample in the batch

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

                    plt.tight_layout()

                    fig = plt.gcf()
                    wandb.log({"phys/pred_fig": fig})

                    plt.close()

        # TODO: Implement A_ub and A_tb if needed

        return loss_total

