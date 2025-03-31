from Diffusion.DenoisingDiffusionProcess import DenoisingDiffusionConditionalProcess
from conditional_encoder import ConditionalEncoder
from Diffusion.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncodedConditionalDiffusion(nn.Module):
    def __init__(self,
                 input_output_shape: tuple = (2, 128, 128),
                 condition_in_shape: tuple = (128, 200, 100),):
        super(EncodedConditionalDiffusion, self).__init__()
        self.input_output_shape = input_output_shape
        self.condition_in_shape = condition_in_shape
        self.condition_out_shape = (8, *self.input_output_shape[1:])
        self.condition_permute_shape = (2, 0, 1)

        self.train_timesteps = 1000
        self.sample_timesteps = 200

        self.diffusion_process = DenoisingDiffusionConditionalProcess(
            generated_channels=self.input_output_shape[0],
            condition_channels=self.condition_shape[self.condition_permute_shape[0]],
            num_timesteps=self.train_timesteps,
            loss_fn=F.mse_loss,
            sampler=DDIM_Sampler(self.sample_timesteps, self.train_timesteps),
        )
        self.conditional_encoder = ConditionalEncoder(
            in_shape=self.condition_in_shape,
            out_shape=self.condition_out_shape,
            permute_shape=self.condition_permute_shape,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        encoded_condition = self.conditional_encoder(condition)
        # Concatenate the encoded condition with the input
        x = torch.cat((x, encoded_condition), dim=1)
        # Pass through the diffusion process
        return self.diffusion_process(x)