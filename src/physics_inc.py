import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Diffusion.DenoisingDiffusionProcess.forward import GaussianForwardProcess
import math
from data_compute import DataCompute as DC
from util import extract, gaussian_log_likelihood, extract, global_normalisation

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import wandb

class PhysicsIncorporated(nn.Module):
    def __init__(self, 
                 gaussian_forward_process: GaussianForwardProcess,
                 data_compute: DC,
                 predict_mode: str = 'eps'):
        super(PhysicsIncorporated, self).__init__()
        self.gfp: GaussianForwardProcess = gaussian_forward_process

        self.data_compute = data_compute

        self.predict_mode: str = predict_mode  # 'eps' or 'x0' or 'v'

        self.stochastic_proportion: float = 1/4 # Only use 1/nth of the batch to compute loss

        self.loss_metric = F.mse_loss

        self.image_width = 128


    def forward(self, x_t: torch.Tensor, target_pred: torch.Tensor, t, gt: torch.Tensor, epoch_and_step: tuple|None=None) -> torch.Tensor:
        # Apply forward operator to our predicted x_0 based on x_t and noise_hat, and calculate loss between the predicted and actual y.
        # P.S. GT phantom must be raw values, not normalised to any range.

        # Stochastic sampling
        indices = torch.randperm(x_t.shape[0])[:math.floor(x_t.shape[0] * self.stochastic_proportion)]
        loss_total = 0.0

        if x_t.shape[-1] != self.image_width:
            x_t = F.interpolate(x_t, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)
        if target_pred.shape[-1] != self.image_width:
            target_pred = F.interpolate(target_pred, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)
        if gt.shape[-1] != self.image_width:
            gt = F.interpolate(gt, size=(self.image_width, self.image_width), mode='bilinear', align_corners=False)

        if self.predict_mode == 'eps':
            x_0_pred: torch.Tensor = (x_t - extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * target_pred) / extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape)

        if self.predict_mode == 'x0':
            x_0_pred: torch.Tensor = target_pred

        if self.predict_mode == 'v':
            x_0_pred: torch.Tensor = (extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape) * x_t ) \
                                - (extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * target_pred)

        for i in indices:
            # Apply the forward operator to x_0_pred
            if self.data_compute.A_ut is not None:
                # De-normalise x_0_pred to the original range of the phantom data
                phan0_min = DC.phan0_min if global_normalisation else x_0_pred[i, 0, :, :].min()
                phan0_max = DC.phan0_max if global_normalisation else x_0_pred[i, 0, :, :].max()
                phan1_min = DC.phan1_min if global_normalisation else x_0_pred[i, 1, :, :].min()
                phan1_max = DC.phan1_max if global_normalisation else x_0_pred[i, 1, :, :].max()

                x_0_pred[i, 0, :, :] = ((x_0_pred[i, 0, :, :] + 1) / 2) * (phan0_max - phan0_min) + phan0_min
                x_0_pred[i, 1, :, :] = ((x_0_pred[i, 1, :, :] + 1) / 2) * (phan1_max - phan1_min) + phan1_min

                sino_pred_ut = self.data_compute.A(x_0_pred[i].unsqueeze(0), 'ut')  # Apply the forward operator (all channels)
                # Scale predicted sinogram to the same range as GT later
                sino_pred_ut_min = DC.sino_ut_min if global_normalisation else sino_pred_ut.min()
                sino_pred_ut_max = DC.sino_ut_max if global_normalisation else sino_pred_ut.max()
                sino_pred_ut = (sino_pred_ut - sino_pred_ut_min) / (sino_pred_ut_max - sino_pred_ut_min) * 2 - 1

                # Feed the GT phantom through the forward operator to get the expected sinogram
                sino_gt_ut = self.data_compute.A(gt[i].unsqueeze(0), 'ut')  # Apply the forward operator to GT (all channels)
                # Scale GT sinogram to the same range as predicted
                sino_gt_ut_min = DC.sino_ut_min if global_normalisation else sino_gt_ut.min()
                sino_gt_ut_max = DC.sino_ut_max if global_normalisation else sino_gt_ut.max()
                sino_gt_ut = (sino_gt_ut - sino_gt_ut_min) / (sino_gt_ut_max - sino_gt_ut_min) * 2 - 1

                if sino_gt_ut.shape[-2:] != sino_pred_ut.shape[-2:]:
                    sino_pred_ut = F.interpolate(sino_pred_ut, size=sino_gt_ut.shape[-2:], mode='bilinear', align_corners=False)

                loss_ut = self.loss_metric(sino_pred_ut, sino_gt_ut)
                # Compute the gaussian log likelihood of loss_ut
                # https://github.com/jhbastek/PhysicsInformedDiffusionModels/blob/main/src/denoising_toy_utils.py#L494
                variance = extract(self.gfp.posterior_variance_clipped, t[i].unsqueeze(0), loss_ut.shape)
                loss_ut_log_likelihood = gaussian_log_likelihood(torch.zeros_like(loss_ut), mean=loss_ut, var=variance)  # Assuming mean=0
                residual_constant = 0.005 # Original PDIM paper used 0.001
                residual_ut = residual_constant * -1 * loss_ut_log_likelihood.mean() # Maximse the log likelihood, so we take the negative of it

                loss_total += residual_ut * (1/x_t.shape[0]) # Scale by batch size

                # debugging: plot predicted x_0 and sino_pred_ut and y
                if epoch_and_step is not None and epoch_and_step[0] % 5 == 0 and i == indices[0]:  # Only plot for the first sample in the batch

                    plt.figure(figsize=(12, 6))
                    
                    plt.subplot(1, 4, 1)
                    plt.title(f'Predicted x_0 (scat) (i:{i})')
                    plt.imshow(x_0_pred[i][0].detach().cpu().numpy(), cmap='gray')
                    plt.colorbar(orientation='horizontal')
                    plt.axis('off')

                    plt.subplot(1, 4, 2)
                    plt.title(f'Predicted x_0 (atten) (t:{t[i].item()})')
                    plt.imshow(x_0_pred[i][1].detach().cpu().numpy(), cmap='gray')
                    plt.colorbar(orientation='horizontal')
                    plt.axis('off')

                    plt.subplot(1, 4, 3)
                    plt.title('Predicted Sinogram')
                    plt.imshow(sino_pred_ut[0][0].detach().cpu().numpy(), cmap='gray')
                    plt.colorbar(orientation='horizontal')
                    plt.axis('off')

                    plt.subplot(1, 4, 4)
                    plt.title(f'Sinogram from GT, loss:{loss_ut_log_likelihood.item():.4f}')
                    plt.imshow(sino_gt_ut[0][0].detach().cpu().numpy(), cmap='gray')
                    plt.colorbar(orientation='horizontal')
                    plt.axis('off')

                    plt.tight_layout()

                    fig = plt.gcf()

                    if wandb.run is not None:
                        wandb.log({"phys/pred_fig": fig})

                    plt.close()

        # TODO: Implement A_ub and A_tb if needed

        return loss_total

