from Diffusion.DenoisingDiffusionProcess import DenoisingDiffusionConditionalProcess
from Diffusion.LatentDiffusion import LatentDiffusionConditional
from Diffusion.DenoisingDiffusionProcess.samplers.DDIM import DDIM_Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np

from sampler_wrapper import SamplerWrapper
from physics_inc import PhysicsIncorporated

from skimage.metrics import peak_signal_noise_ratio as PeakSignalNoiseRatio
from skimage.metrics import structural_similarity as StructuralSimilarity


class ECDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 num_timesteps=1000,
                 batch_size=16,
                 lr=1e-4,
                 physics=False,
                 latent=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.latent = latent

        self.image_shape = (2, 128, 128) if not self.latent else (3, 256, 256)  # 2 channels: scatter and attenuation

        self.model = EncodedConditionalDiffusion(
            input_output_shape=self.image_shape,
            num_timesteps=num_timesteps,
            batch_size=self.batch_size,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            latent=self.latent,
        )

        self.sampler_wrapper = SamplerWrapper(sample_timesteps=self.model.sample_timesteps,
                                             train_timesteps=self.model.train_timesteps)

        self.physics_model: PhysicsIncorporated | None = PhysicsIncorporated(
            gaussian_forward_process=self.model.diffusion_process.forward_process if not self.latent else self.model.diffusion_process.model.forward_process,
            A_ut_dir='/home/samnub/dev/lircst-iterecon/data_discretised/A_ut.npy',
        ) if physics else None

    def preprocess(self, image: torch.Tensor | None=None, condition: torch.Tensor | None=None):
        # Pre-process our phantom images and conditions (no need for a separate conditional encoder here)

        image_out: torch.Tensor | None = None if image is None else torch.zeros((
            image.shape[0], # batch size
            self.image_shape[-3], # 2 channels: scatter and attenuation (3 if latent)
            self.image_shape[-2], # height
            self.image_shape[-1], # width
        )).cuda()
        condition_out: torch.Tensor | None = None if condition is None else torch.zeros((
            condition.shape[0], # batch size
            1 if not self.latent else self.image_shape[-3], # 1 channel: sinogram (3 if latent)
            self.image_shape[-2], # height
            self.image_shape[-1], # width
        )).cuda()

        b: int = image.shape[0] if image is not None else condition.shape[0]

        for i in range(b):
            phan = image[i] if image is not None else None
            sino = condition[i] if condition is not None else None

            if phan is not None:
                min_phan1 = torch.min(phan[1])
                max_phan1 = torch.max(phan[1])
                min_phan0 = torch.min(phan[0])
                max_phan0 = torch.max(phan[0])
                phan[0] = ((phan[0] - min_phan0) / (max_phan0 - min_phan0)) * 2 - 1
                phan[1] = ((phan[1] - min_phan1) / (max_phan1 - min_phan1)) * 2 - 1

                if self.latent:
                    sandwich = torch.mean(phan, dim=0, keepdim=True)  # Create a sandwich channel
                    phan = torch.cat((phan[0].unsqueeze(0), sandwich, phan[1].unsqueeze(0)), dim=0)  # Concatenate the sandwich channel
                    phan = F.interpolate(phan.unsqueeze(0), size=self.image_shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

            if sino is not None:
                min_sino = torch.min(sino)
                max_sino = torch.max(sino)

                sino = sino.sum(dim=-1, keepdim=True)
                sino = sino.permute(2, 0, 1)  # Change to (C, H, W) format
                sino = F.interpolate(sino.unsqueeze(0), size=self.image_shape[-2:], mode='bilinear', align_corners=False).squeeze(0)
                sino = ((sino - min_sino) / (max_sino - min_sino)) * 2 - 1

                if self.latent:
                    sino = sino.repeat(self.image_shape[-3], 1, 1) # Needs 3 channels

            if image_out is not None:
                image_out[i] = phan
            if condition_out is not None:
                condition_out[i] = sino

        return image_out, condition_out

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        condition = args[0]

        _, condition = self.preprocess(condition=condition)
        x_t = self.model.diffusion_process(condition, self.sampler_wrapper.get_sampler(), *args[2:], **kwargs)  

        return x_t, condition
    
    def training_step(self, batch, batch_idx):
        image, condition, phantom_id = batch

        image, condition = self.preprocess(image=image, condition=condition)

        loss: torch.Tensor | None = None
        x_t: torch.Tensor | None = None
        noise_hat: torch.Tensor | None = None
        t: torch.Tensor | None = None

        if not self.latent:
            loss, x_t, noise_hat, t = self.model.diffusion_process.p_loss(image, condition)
        else:
            loss, x_t, noise_hat, t = self.model.diffusion_process.training_step((condition, image), batch_idx=batch_idx)

        if self.physics_model is not None:
            # Apply physics model to the loss
            physics_loss = self.physics_model(x_t, noise_hat, t, condition)
            loss += physics_loss * 0.5 # Adjust the weight as needed

        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    '''
    def validation_step(self, batch, batch_idx):
        return self.loss_evaluation(batch, batch_idx)
    '''
    def test_step(self, batch, batch_idx):
        return self.loss_evaluation(batch, batch_idx)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    '''
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)
    '''
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)

    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)

    def loss_evaluation(self, batch, batch_idx, to_print=False):
        image, condition, phantom_id = batch

        image, condition = self.preprocess(image=image, condition=condition)

        pred, _ = self.forward(condition)

        # Calculate PSNR and SSIM
        # If dimensions are not the same, resize the prediction to match the image
        if pred.shape != image.shape:
            pred = F.interpolate(pred, size=image.shape[-1], mode='bilinear', align_corners=False)

        data_range: float = 2.0 # [-1, 1]

        psnr_scat: float = 0
        ssim_scat: float = 0

        psnr_atten: float = 0
        ssim_atten: float = 0

        # As we have to do this with skimage, we need to convert the tensors to numpy arrays and iterate over the batch
        # This is not the most efficient way, but it works
        for i in range(image.shape[0]):
            pred_np = pred[i].cpu().numpy().astype(np.float32)
            image_np = image[i].cpu().numpy().astype(np.float32)

            psnr_scat += PeakSignalNoiseRatio(pred_np[0], image_np[0], data_range=data_range) / image.shape[0]
            ssim_scat += StructuralSimilarity(pred_np[0], image_np[0], data_range=data_range) / image.shape[0]

            psnr_atten += PeakSignalNoiseRatio(pred_np[-1], image_np[-1], data_range=data_range) / image.shape[0]
            ssim_atten += StructuralSimilarity(pred_np[-1], image_np[-1], data_range=data_range) / image.shape[0]

        self.log('psnr_scat', psnr_scat, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ssim_scat', ssim_scat, prog_bar=True, on_step=False, on_epoch=True)

        self.log('psnr_atten', psnr_atten, prog_bar=True, on_step=False, on_epoch=True)
        self.log('ssim_atten', ssim_atten, prog_bar=True, on_step=False, on_epoch=True)
        
        if to_print:
            print(f'Batch {batch_idx}: PSNR_scat: {psnr_scat:.4f}, SSIM_scat: {ssim_scat:.4f} | PSNR_atten: {psnr_atten:.4f}, SSIM_atten: {ssim_atten:.4f}')

        return psnr_scat, ssim_scat, psnr_atten, ssim_atten
    

class EncodedConditionalDiffusion(nn.Module):
    def __init__(self,
                 input_output_shape: tuple = (2, 128, 128),
                 num_timesteps: int = 1000,
                 batch_size: int = 16,
                 train_dataset=None,
                 valid_dataset=None,
                 latent: bool = False):
        super(EncodedConditionalDiffusion, self).__init__()
        self.input_output_shape = input_output_shape
        self.condition_out_shape = (1, *self.input_output_shape[1:])

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.sample_timesteps = num_timesteps // 5
        self.train_timesteps = num_timesteps

        self.latent = latent

        self.diffusion_process = DenoisingDiffusionConditionalProcess(
            generated_channels=self.input_output_shape[0],
            condition_channels=self.condition_out_shape[0],
            num_timesteps=self.train_timesteps,
            loss_fn=F.mse_loss,
        ) if not self.latent else LatentDiffusionConditional(
            num_timesteps=self.train_timesteps,
            batch_size=self.batch_size,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pass