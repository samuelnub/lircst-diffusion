from Diffusion.DenoisingDiffusionProcess import DenoisingDiffusionConditionalProcess
from conditional_encoder import ConditionalEncoder
from Diffusion.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class ECDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 batch_size=1,
                 lr=1e-4):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.batch_size = batch_size

        self.model = EncodedConditionalDiffusion(
            input_output_shape=(2, 128, 128),
            condition_in_shape=(128, 200, 100),
            num_timesteps=num_timesteps,
        )

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> torch.Tensor:
        condition = args[0]

        condition = self.model.conditional_encoder(condition)
        x_t = self.model.diffusion_process(condition, *args[1:], **kwargs)  

        return x_t
    
    def training_step(self, batch, batch_idx):
        image, condition, phantom_id = batch

        condition = self.model.conditional_encoder(condition)

        loss = self.model.diffusion_process.p_loss(image, condition)

        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)


class EncodedConditionalDiffusion(nn.Module):
    def __init__(self,
                 input_output_shape: tuple = (2, 128, 128),
                 condition_in_shape: tuple = (128, 200, 100),
                 num_timesteps: int = 1000,):
        super(EncodedConditionalDiffusion, self).__init__()
        self.input_output_shape = input_output_shape
        self.condition_in_shape = condition_in_shape
        self.condition_out_shape = (8, *self.input_output_shape[1:])
        self.condition_permute_shape = (2, 0, 1)

        self.train_timesteps = num_timesteps
        self.sample_timesteps = num_timesteps // 5

        self.diffusion_process = DenoisingDiffusionConditionalProcess(
            generated_channels=self.input_output_shape[0],
            condition_channels=self.condition_out_shape[0],
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
        pass