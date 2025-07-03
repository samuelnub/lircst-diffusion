import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Diffusion.DenoisingDiffusionProcess.forward import GaussianForwardProcess
import math
from util import extract, gaussian_log_likelihood, extract

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import wandb

class PhysicsIncorporated(nn.Module):
    def __init__(self, 
                 gaussian_forward_process: GaussianForwardProcess,
                 A_ut_dir: str | None = None,
                 A_ub_dir: str | None = None,
                 A_tb_dir: str | None = None,
                 predict_mode: str = 'eps'):
        super(PhysicsIncorporated, self).__init__()
        self.gfp: GaussianForwardProcess = gaussian_forward_process

        self.A_ut: torch.Tensor | None = None
        self.A_ub: torch.Tensor | None = None
        self.A_tb: torch.Tensor | None = None

        self.predict_mode: str = predict_mode  # 'eps' or 'x0' or 'v'

        self.stochastic_proportion: float = 1/4 # Only use 1/nth of the batch to compute loss

        self.loss_metric = F.mse_loss

        self.image_width = 128

        # Load the forward operator matrices from the specified directory
        if A_ut_dir is not None:
            self.A_ut = torch.from_numpy(np.load(A_ut_dir)).float().cuda()
            self.A_ut_T = self.A_ut.T  # Precompute the transpose for efficiency
        if A_ub_dir is not None:
            self.A_ub = torch.from_numpy(np.load(A_ub_dir)).float().cuda()
            self.A_ub_T = self.A_ub.T  # Precompute the transpose for efficiency
        if A_tb_dir is not None:
            self.A_tb = torch.from_numpy(np.load(A_tb_dir)).float().cuda()
            self.A_tb_T = self.A_tb.T  # Precompute the transpose for efficiency

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

    def forward(self, x_t: torch.Tensor, target_pred: torch.Tensor, t, y: torch.Tensor, epoch_and_step: tuple|None=None) -> torch.Tensor:
        # Apply forward operator to our predicted x_0 based on x_t and noise_hat, and calculate loss between the predicted and actual y.
        # P.S. We assume y is noiseless and serves as a ground truth for the sinogram.
        # TODO: If sinogram y is noisy, we should instead use our GT phantom and feed it through the forward operator to get the expected sinogram.

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

        if self.predict_mode == 'eps':
            x_0_pred: torch.Tensor = (x_t - extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * target_pred) / extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape)

        if self.predict_mode == 'x0':
            x_0_pred: torch.Tensor = target_pred

        if self.predict_mode == 'v':
            x_0_pred: torch.Tensor = (extract(self.gfp.alphas_cumprod_sqrt, t, x_t.shape) * x_t ) \
                                - (extract(self.gfp.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * target_pred)

        for i in indices:
            # Apply the forward operator to x_0_pred
            if self.A_ut is not None:
                sino_pred_ut = self.A(x_0_pred[i].unsqueeze(0), 'ut')  # Apply the forward operator
                # Scale predicted sinogram to the same range as y
                sino_pred_ut_min = sino_pred_ut.min()
                sino_pred_ut_max = sino_pred_ut.max()
                sino_pred_ut = (sino_pred_ut - sino_pred_ut_min) / (sino_pred_ut_max - sino_pred_ut_min) * 2 - 1

                # Interpolate/resize the predicted sinogram to match the shape of y (we assume y has passed through the conditional encoder)
                sino_pred_ut = F.interpolate(sino_pred_ut, size=y.shape[-2:], mode='bilinear', align_corners=False)

                loss_ut = self.loss_metric(sino_pred_ut, y[i].unsqueeze(0))
                # Compute the gaussian log likelihood of loss_ut
                # https://github.com/jhbastek/PhysicsInformedDiffusionModels/blob/main/src/denoising_toy_utils.py#L494
                variance = extract(self.gfp.posterior_variance_clipped, t[i].unsqueeze(0), loss_ut.shape)
                loss_ut_log_likelihood = gaussian_log_likelihood(torch.zeros_like(loss_ut), mean=loss_ut, var=variance)  # Assuming mean=0
                residual_constant = 0.001
                residual_ut = residual_constant * -1 * loss_ut_log_likelihood.mean() # Maximse the log likelihood, so we take the negative of it

                loss_total += residual_ut * (1/x_t.shape[0]) # Scale by batch size

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
                    plt.title(f'Actual Sinogram (y), loss:{loss_ut_log_likelihood.item():.4f}')
                    plt.imshow(y[i][0].detach().cpu().numpy(), cmap='gray')
                    plt.axis('off')

                    plt.tight_layout()

                    fig = plt.gcf()

                    if wandb.run is not None:
                        wandb.log({"phys/pred_fig": fig})

                    plt.close()

        # TODO: Implement A_ub and A_tb if needed

        return loss_total

