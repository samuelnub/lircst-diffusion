from Diffusion.DenoisingDiffusionProcess import DenoisingDiffusionConditionalProcess
from conditional_encoder import ConditionalEncoder
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
                 batch_size=1,
                 lr=1e-4,
                 physics=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.batch_size = batch_size

        self.model = EncodedConditionalDiffusion(
            input_output_shape=(2, 128, 128),
            condition_in_shape=(128, 200, 100),
            num_timesteps=num_timesteps,
        )

        self.sampler_wrapper = SamplerWrapper(sample_timesteps=self.model.sample_timesteps,
                                             train_timesteps=self.model.train_timesteps)

        self.physics_model: PhysicsIncorporated | None = PhysicsIncorporated(
            gaussian_forward_process=self.model.diffusion_process.forward_process,
            A_ut_dir='/home/samnub/dev/lircst-iterecon/data_discretised/A_ut.npy',
        ) if physics else None

    def preprocess(self, image: torch.Tensor, condition: torch.Tensor):
        # Pre-process our phantom images and conditions (no need for a separate conditional encoder here)

        image_out: torch.Tensor = torch.zeros((
            image.shape[0], # batch size
            image.shape[1], # 2 channels: scatter and attenuation,
            image.shape[2], # height
            image.shape[3], # width
        )).cuda()
        condition_out: torch.Tensor = torch.zeros((
            condition.shape[0], # batch size
            1, # 1 channel: sinogram
            image.shape[2], # height
            image.shape[3], # width
        )).cuda()

        for i in range(image.shape[0]):
            phan = image[i]
            sino = condition[i]

            sino = sino.sum(dim=-1, keepdim=True)
            sino = sino.permute(2, 0, 1)  # Change to (C, H, W) format
            sino = F.interpolate(sino.unsqueeze(0), size=phan.shape[1:], mode='bilinear', align_corners=False).squeeze(0)

            nonzero_phan0 = torch.nonzero(phan[0])

            nonzero_min_phan0 = torch.min(phan[0][nonzero_phan0])
            nonzero_max_phan0 = torch.max(phan[0][nonzero_phan0])

            min_phan1 = torch.min(phan[1])
            max_phan1 = torch.max(phan[1])

            min_sino = torch.min(sino)
            max_sino = torch.max(sino)

            if False: # TODO not nonzero_max_phan0 == nonzero_min_phan0:
                phan[0][nonzero_phan0] = (phan[0][nonzero_phan0] - nonzero_min_phan0) / (nonzero_max_phan0 - nonzero_min_phan0)
                phan[0][torch.where(phan[0] == 0.0)] = -1.0
            else:
                min_phan0 = torch.min(phan[0])
                max_phan0 = torch.max(phan[0])
                phan[0] = ((phan[0] - min_phan0) / (max_phan0 - min_phan0)) * 2 - 1

            phan[1] = ((phan[1] - min_phan1) / (max_phan1 - min_phan1)) * 2 - 1
            sino = ((sino - min_sino) / (max_sino - min_sino)) * 2 - 1

            image_out[i] = phan
            condition_out[i] = sino

        return image_out, condition_out

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        condition = args[0]

        condition = self.model.conditional_encoder(condition)
        x_t = self.model.diffusion_process(condition, self.sampler_wrapper.get_sampler(), *args[2:], **kwargs)  

        return x_t, condition
    
    def training_step(self, batch, batch_idx):
        image, condition, phantom_id = batch

        image, condition = self.preprocess(image, condition)

        print(f'Batch {batch_idx}: Image shape: {image.shape}, Condition shape: {condition.shape}')

        #condition = self.model.conditional_encoder(condition)

        loss, x_t, noise_hat, t = self.model.diffusion_process.p_loss(image, condition)

        if self.physics_model is not None:
            # Apply physics model to the loss
            physics_loss = self.physics_model(x_t, noise_hat, t, condition)
            loss += physics_loss

        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        return self.loss_evaluation(batch, batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.loss_evaluation(batch, batch_idx)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)

    def configure_optimizers(self):
        return torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)

    def loss_evaluation(self, batch, batch_idx, to_print=False):
        image, condition, phantom_id = batch

        condition = self.model.conditional_encoder(condition)

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

            psnr_atten += PeakSignalNoiseRatio(pred_np[1], image_np[1], data_range=data_range) / image.shape[0]
            ssim_atten += StructuralSimilarity(pred_np[1], image_np[1], data_range=data_range) / image.shape[0]

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
                 condition_in_shape: tuple = (128, 200, 100),
                 num_timesteps: int = 1000,):
        super(EncodedConditionalDiffusion, self).__init__()
        self.input_output_shape = input_output_shape
        self.condition_in_shape = condition_in_shape
        self.condition_out_shape = (1, *self.input_output_shape[1:])
        self.condition_permute_shape = (2, 0, 1)

        self.sample_timesteps = num_timesteps // 5
        self.train_timesteps = num_timesteps

        self.diffusion_process = DenoisingDiffusionConditionalProcess(
            generated_channels=self.input_output_shape[0],
            condition_channels=self.condition_out_shape[0],
            num_timesteps=self.train_timesteps,
            loss_fn=F.mse_loss,
        )
        self.conditional_encoder = ConditionalEncoder(
            in_shape=self.condition_in_shape,
            out_shape=self.condition_out_shape,
            permute_shape=self.condition_permute_shape,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pass