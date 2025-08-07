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
from data_compute import DataCompute as DC
from util import sino_undersample, poisson_noise, gaussian_noise, snr_db, to_decibels, global_normalisation, mlem, degrade_sino

from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MeanSquaredError as MSE

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, clear_output

import wandb


class ECDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 test_dataset=None,
                 num_timesteps=1000,
                 batch_size=16,
                 lr=1e-4,
                 physics=False,
                 latent=False,
                 predict_mode='v',
                 condition_A_T=True,
                 degradation=0.0,
                 classical_evaluation=False):
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.latent = latent
        self.predict_mode = predict_mode  # 'eps' or 'x0' or 'v'
        self.condition_A_T = condition_A_T  # Whether to pass our condition (sinogram) through the physics model
        self.degradation = degradation  # Self sabotaging Degradation factor for the sinogram readings
        self.classical_evaluation = classical_evaluation  # Whether to evaluate the model using classical methods (MLEM, backprojection)

        self.image_shape = (2, 128, 128) if not self.latent else (3, 256, 256)  # 2 channels: scatter and attenuation

        self.model = EncodedConditionalDiffusion(
            input_output_shape=self.image_shape,
            num_timesteps=num_timesteps,
            batch_size=self.batch_size,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            latent=self.latent,
            predict_mode=self.predict_mode
        )
        self.sampler_wrapper = SamplerWrapper(sample_timesteps=self.model.sample_timesteps,
                                             train_timesteps=self.model.train_timesteps)
        if not self.latent:
            self.model.diffusion_process.sampler = self.sampler_wrapper.get_sampler()
        else:
            self.model.diffusion_process.model.sampler = self.sampler_wrapper.get_sampler()

        self.data_compute = DC(
            stats_dir='/home/samnub/dev/lircst-diffusion/data/',
            data_dir='/home/samnub/dev/lircst-ana/data/',
            operator_dir='/home/samnub/dev/lircst-iterecon/data_discretised/',
        )

        self.physics = physics  # Whether to incorporate physics loss in the model
        self.physics_model: PhysicsIncorporated | None = PhysicsIncorporated(
            gaussian_forward_process=self.model.diffusion_process.forward_process if not self.latent else self.model.diffusion_process.model.forward_process,
            data_compute=self.data_compute,
            predict_mode=self.predict_mode,
        )

        self.metrics = {
            'psnr': PSNR(data_range=1.0).cuda(),
            'ssim': SSIM(data_range=1.0).cuda(),  
            'mse': MSE().cuda(),
            'mae': nn.L1Loss(reduction='none').cuda(), # For visualisation
        }
        
    def on_load_checkpoint(self, checkpoint):
        print(f'Loading checkpoint: epoch {checkpoint["epoch"]} | step {checkpoint["global_step"]}')
        return super().on_load_checkpoint(checkpoint)

    def preprocess(self, 
                   image: torch.Tensor | None=None, 
                   condition: torch.Tensor | None=None, 
                   global_norm: bool=False, 
                   condition_perm: bool=True, 
                   condition_a_t: bool=None,
                   return_precat: bool=False, # Whether to return the sinogram before we condition it with A_T P.S. returned before normalisation, so it will be in the original range
                   ):
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

        precat: torch.Tensor | None = None if not return_precat else torch.zeros((
            condition.shape[0], # batch size
            1 if not self.latent else self.image_shape[-3], # 1 channel: sinogram (3 if latent)
            condition.shape[-3], # u
            condition.shape[-2], # theta
        )).cuda()

        b: int = image.shape[0] if image is not None else condition.shape[0]

        for i in range(b):
            phan = image[i] if image is not None else None
            sino = condition[i] if condition is not None else None

            if phan is not None:
                min_phan0 = DC.phan0_min if global_norm else torch.min(phan[0])
                max_phan0 = DC.phan0_max if global_norm else torch.max(phan[0])
                min_phan1 = DC.phan1_min if global_norm else torch.min(phan[1])
                max_phan1 = DC.phan1_max if global_norm else torch.max(phan[1])
                phan[0] = ((phan[0] - min_phan0) / (max_phan0 - min_phan0)) * 2 - 1
                phan[1] = ((phan[1] - min_phan1) / (max_phan1 - min_phan1)) * 2 - 1

                if self.latent:
                    sandwich = torch.mean(phan, dim=0, keepdim=True)  # Create a sandwich channel
                    phan = torch.cat((phan[0].unsqueeze(0), sandwich, phan[1].unsqueeze(0)), dim=0)  # Concatenate the sandwich channel
                    phan = F.interpolate(phan.unsqueeze(0), size=self.image_shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

            if sino is not None:
                sino = sino.sum(dim=-1, keepdim=True)

                if condition_perm:
                    sino = sino.permute(2, 0, 1)  # Change to (C, H, W) format

                # Degradation is applied to the sinogram readings
                if self.degradation > 0:
                    sino, snr = degrade_sino(sino.unsqueeze(0), degradation=self.degradation, calc_snr=True)
                    sino = sino.squeeze(0)  # Remove batch dimension
                    if wandb.run is not None and wandb.run.step % 10 == 0:
                        wandb.log({'data/degradation_snr': snr,})
                    elif wandb.run is None:
                        print(f"Degradation - SNR: {snr:.2f} dB")

                if return_precat:
                    # Return the pre-cat sinogram for later use
                    precat[i] = sino.clone()

                min_sino = DC.sino_ut_min if global_norm else torch.min(sino)
                max_sino = DC.sino_ut_max if global_norm else torch.max(sino)

                if (condition_a_t is None and self.condition_A_T) or condition_a_t is True:
                    # Apply the pseudoinverse physics model to the sinogram
                    sino = self.data_compute.A_T(sino.unsqueeze(0), 'ut').squeeze(0)
                    # P.S. Our "sino" is now a misnomer, it's actually in phantom space now
                    # Recompute min and max for the phantom space
                    min_sino = DC.sino_ut_a_t_min if global_norm else torch.min(sino)
                    max_sino = DC.sino_ut_a_t_max if global_norm else torch.max(sino)

                sino = ((sino - min_sino) / (max_sino - min_sino)) * 2 - 1

                sino = F.interpolate(sino.unsqueeze(0), size=self.image_shape[-2:], mode='bilinear', align_corners=False).squeeze(0)

                if self.latent:
                    sino = sino.repeat(self.image_shape[-3], 1, 1) # Needs 3 channels

            if image_out is not None:
                image_out[i] = phan
            if condition_out is not None:
                condition_out[i] = sino

        if return_precat:
            return image_out, condition_out, precat

        return image_out, condition_out

    @torch.no_grad()
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        condition = args[0]

        _, condition, precat = self.preprocess(condition=condition, 
                                       global_norm=global_normalisation,
                                       return_precat=True)
        pred = self.model.diffusion_process(condition, *args[2:], **kwargs)  

        return pred, condition, precat
    
    def training_step(self, batch, batch_idx):
        image, condition, phantom_id = batch

        image_raw: torch.Tensor | None = image.clone() if self.physics else None

        sino_condition: torch.Tensor | None = None # Used if condition_A_T is false OR we have physics true
        phan_condition: torch.Tensor | None = None # If condition_A_T is true

        image, _ = self.preprocess(image=image, global_norm=global_normalisation)  # Preprocess the image (GT phantom)

        if not self.condition_A_T:
            _, sino_condition = self.preprocess(condition=condition, condition_a_t=False, global_norm=global_normalisation)
        if self.condition_A_T:
            _, phan_condition = self.preprocess(condition=condition, global_norm=global_normalisation)
            
        loss: torch.Tensor | None = None
        x_t: torch.Tensor | None = None
        target_pred: torch.Tensor | None = None
        t: torch.Tensor | None = None

        if not self.latent:
            loss, x_t, target_pred, t = self.model.diffusion_process.p_loss(image, 
                                                                            phan_condition if self.condition_A_T else sino_condition)
        else:
            loss, x_t, target_pred, t = self.model.diffusion_process.training_step((phan_condition if self.condition_A_T else sino_condition, 
                                                                                    image), batch_idx=batch_idx)

        if self.physics and self.physics_model is not None:
            # Apply physics model to the loss
            physics_loss: torch.Tensor | None = None
            epoch_and_batch_idx: tuple = (self.current_epoch, batch_idx)
            if not self.latent:
                physics_loss = self.physics_model(x_t, target_pred, t, image_raw, epoch_and_batch_idx=epoch_and_batch_idx)
            else:
                decoded_target_pred: torch.Tensor | None = None
                with torch.no_grad():
                    decoded_target_pred = self.model.diffusion_process.ae.decode(target_pred) / self.model.diffusion_process.latent_scale_factor
                physics_loss = self.physics_model(x_t, decoded_target_pred, t, image_raw, epoch_and_batch_idx=epoch_and_batch_idx)
            loss += physics_loss

        self.log('train_loss', loss, prog_bar=True)

        if wandb.run is not None:
            if batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                })
                if batch_idx == 0:
                    wandb.log({
                        'train/epoch': self.current_epoch,
                        'train/global_step': self.global_step,
                    })

        return loss

    def validation_step(self, batch, batch_idx):
        # This will run every epoch, but we only want to evaluate the visual loss every n epochs
        eval_every_n_epochs = 5
        fig_every_n_batches = 1
        if self.current_epoch % eval_every_n_epochs == 0:
            # Only evaluate visual loss every n epochs (computationally expensive)
            return self.loss_evaluation(batch, batch_idx, to_print=True if batch_idx % fig_every_n_batches == 0 else False)
        return None

    def test_step(self, batch, batch_idx):
        fig_every_n_batches = 10
        return self.loss_evaluation(batch, batch_idx, to_print=True if batch_idx % fig_every_n_batches == 0 else False, is_test=True)
    
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

    @torch.no_grad()
    def loss_evaluation(self, batch, batch_idx, to_print=False, return_pred=False, is_test=False):
        image, condition, phantom_id = batch

        image, _ = self.preprocess(image=image, global_norm=global_normalisation) # Don't double-preprocess the condition

        is_classical: bool = self.classical_evaluation

        pred: torch.Tensor | None = None
        encoded_condition: torch.Tensor | None = None
        precat: torch.Tensor | None = None

        if not is_classical:
            pred, encoded_condition, precat = self.forward(condition) # Learned prediction is assumed to be in [-1, 1] range

        if is_classical:
            is_mlem: bool = True

            encoded_condition: torch.Tensor = torch.zeros((condition.shape[0], 1, *self.image_shape[-2:])).cuda() if not self.latent else torch.zeros((condition.shape[0], 3, *self.image_shape[-2:])).cuda()
            precat: torch.Tensor = torch.zeros((condition.shape[0], 1, condition.shape[-3], condition.shape[-2])).cuda() if not self.latent else torch.zeros((condition.shape[0], 3, condition.shape[-3], condition.shape[-2])).cuda()
            if is_mlem:
                pred: torch.Tensor = torch.zeros((condition.shape[0], *self.image_shape)).cuda()  # Initialize prediction tensor
                for i in range(condition.shape[0]):
                    to_feed = condition[i].permute(2,0,1).sum(dim=-3,keepdim=False).cuda()
                    if self.degradation > 0:
                        to_feed, snr = degrade_sino(to_feed.unsqueeze(0).unsqueeze(0), degradation=self.degradation, calc_snr=True)
                        to_feed = to_feed.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
                        if wandb.run is not None:
                            wandb.log({'data/degradation_snr': snr,})
                        elif wandb.run is None:
                            print(f"Degradation - SNR: {snr:.2f} dB")
                    precat[i] = to_feed.unsqueeze(0).clone()
                    pred[i] = mlem(to_feed, 'ut', self.data_compute,)
                pred = ((pred - DC.phan0_min) / (DC.phan0_max - DC.phan0_min)) * 2 - 1
            else:
                # Classical prediction using backprojection
                pred = self.data_compute.A_T(condition.permute(0,3,1,2).sum(dim=-3,keepdim=True), 'ut')  # Apply the pseudoinverse physics model to the sinogram
                # Normalise to [-1, 1] range
                pred = ((pred - DC.sino_ut_a_t_min) / (DC.sino_ut_a_t_max - DC.sino_ut_a_t_min)) * 2 - 1

        if pred.shape != image.shape:
            pred = F.interpolate(pred, size=image.shape[-1], mode='bilinear', align_corners=False)

        # Rescale the images to [0, 1] range for PSNR and SSIM calculations
        pred = ((pred + 1) / 2)
        # Clamp our prediction to [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        
        image = ((image + 1) / 2)

        metrics_per_sample: bool = True # Whether to compute metrics per sample or average over the batch

        psnr_scat: list[float] = []
        ssim_scat: list[float] = []
        rmse_scat: list[float] = []

        psnr_atten: list[float] = []
        ssim_atten: list[float] = []
        rmse_atten: list[float] = []

        def compute_metrics(pred: torch.Tensor, image: torch.Tensor) -> tuple[float|list[float], float|list[float], float|list[float]]:
            # expects a BCHW tensor
            psnr = self.metrics['psnr'](pred, image)
            ssim = self.metrics['ssim'](pred, image)
            rmse = torch.sqrt(self.metrics['mse'](pred.reshape(-1), image.reshape(-1)))
            return psnr.item(), ssim.item(), rmse.item()
        
        if not metrics_per_sample:
            psnr_s, ssim_s, rmse_s = compute_metrics(pred[:, 0, :, :].unsqueeze(0), image[:, 0, :, :].unsqueeze(0))
            psnr_scat.append(psnr_s)
            ssim_scat.append(ssim_s)
            rmse_scat.append(rmse_s)            

            psnr_a, ssim_a, rmse_a = compute_metrics(pred[:, -1, :, :].unsqueeze(0), image[:, -1, :, :].unsqueeze(0))
            psnr_atten.append(psnr_a)
            ssim_atten.append(ssim_a)
            rmse_atten.append(rmse_a)
        else:
            for i in range(pred.shape[0]):
                psnr_s, ssim_s, rmse_s = compute_metrics(pred[i, 0, :, :].unsqueeze(0).unsqueeze(0), image[i, 0, :, :].unsqueeze(0).unsqueeze(0))
                psnr_scat.append(psnr_s)
                ssim_scat.append(ssim_s)
                rmse_scat.append(rmse_s)

                psnr_a, ssim_a, rmse_a = compute_metrics(pred[i, -1, :, :].unsqueeze(0).unsqueeze(0), image[i, -1, :, :].unsqueeze(0).unsqueeze(0))
                psnr_atten.append(psnr_a)
                ssim_atten.append(ssim_a)
                rmse_atten.append(rmse_a)

        if wandb.run is not None:
            for i in range(len(psnr_scat)):
                commit = i >= len(psnr_scat) - 1  # Only commit the last item in the batch to the cloud
                if not is_test:
                    wandb.log({
                        'val/psnr_scat': psnr_scat[i],
                        'val/ssim_scat': ssim_scat[i],
                        'val/rmse_scat': rmse_scat[i],
                        #'val/psnr_atten': psnr_atten[i],
                        #'val/ssim_atten': ssim_atten[i],
                        #'val/rmse_atten': rmse_atten[i],
                    }, commit=commit)
                if is_test:
                    wandb.log({
                        'test/psnr_scat': psnr_scat[i],
                        'test/ssim_scat': ssim_scat[i],
                        'test/rmse_scat': rmse_scat[i],
                        #'test/psnr_atten': psnr_atten[i],
                        #'test/ssim_atten': ssim_atten[i],
                        #'test/rmse_atten': rmse_atten[i],
                    }, commit=commit)

        if to_print:
            if global_normalisation:
                # Rescale the [0, 1] images back to their real-world values
                image[0, 0, :, :] = image[0, 0, :, :] * (DC.phan0_max - DC.phan0_min) + DC.phan0_min
                image[0, -1, :, :] = image[0, -1, :, :] * (DC.phan1_max - DC.phan1_min) + DC.phan1_min
                pred[0, 0, :, :] = pred[0, 0, :, :] * (DC.phan0_max - DC.phan0_min) + DC.phan0_min
                pred[0, -1, :, :] = pred[0, -1, :, :] * (DC.phan1_max - DC.phan1_min) + DC.phan1_min

            # Display the results *aesthetically*
            # 2 x 2 plot, we will plot:
            #  the GT phantom scatter channel,
            #  the original sinogram,
            #  the Predicted scatter channel,
            #  the error map between the predicted and GT phantom scatter channel

            plt.figure(figsize=(10, 10), dpi=200)
            plt.subplot(2, 2, 1)
            plt.title(f'Ground Truth')
            plt.imshow(image[0, 0].detach().cpu().numpy(), cmap='bone')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04).ax.set_title('ρₑ (e/cm³)' if global_normalisation else '')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.subplot(2, 2, 2)
            plt.title(f'Detector Reading')
            plt.imshow(precat[0, 0].detach().cpu().numpy(), cmap='afmhot')
            cb = plt.colorbar(orientation='horizontal').ax.set_title('Intensity', y=-2.8)
            cb.axes.locator_params(nbins=5)
            plt.gca().set_xticks(np.linspace(0, 200, 5))
            plt.gca().set_xticklabels([0, 45, 90, 135, 180])
            plt.xlabel('θ (degrees)')
            plt.ylabel('Receiver')
            plt.subplot(2, 2, 3)
            plt.title(f'Predicted')
            plt.imshow(pred[0, 0].detach().cpu().numpy(), cmap='bone')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04).ax.set_title('ρₑ (e/cm³)' if global_normalisation else '')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.subplot(2, 2, 4)
            plt.title(f'x̂-x')
            plt.imshow((pred[0, 0] - image[0, 0]).detach().cpu().numpy(), cmap='seismic')
            plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04).ax.set_title('Error (e/cm³)' if global_normalisation else '')
            plt.xlabel('mm')
            plt.ylabel('mm')
            plt.tight_layout()

            '''
            # Display the prediction and the ground truth for first item in batch
            plt.figure(figsize=(20, 6), dpi=200)
            plt.subplot(1, 6, 1)
            plt.title(f'{phantom_id[0]}')
            plt.imshow(precat[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal').ax.set_title('Intensity')
            plt.axis('off')
            plt.subplot(1, 6, 2)
            plt.title(f'Condition')
            plt.imshow(encoded_condition[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal')
            plt.axis('off')
            plt.subplot(1, 6, 3)
            plt.title(f'(PSNR:{psnr_scat.item():.3f}, SSIM:{ssim_scat.item():.3f}, RMSE:{rmse_scat.item():.3f})')
            plt.imshow(pred[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal').ax.set_title('ρₑ (e/cm³)' if global_normalisation else '')
            plt.axis('off')
            plt.subplot(1, 6, 4)
            plt.title(f'Scat')
            plt.imshow(image[0, 0].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal').ax.set_title('ρₑ (e/cm³)' if global_normalisation else '')
            plt.axis('off')
            plt.subplot(1, 6, 5)
            plt.title(f'(PSNR:{psnr_atten.item():.3f}, SSIM:{ssim_atten.item():.3f}, RMSE:{rmse_atten.item():.3f})')
            plt.imshow(pred[0, -1].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal').ax.set_title('μ/ρ (cm²/g)' if global_normalisation else '')
            plt.axis('off')
            plt.subplot(1, 6, 6)
            plt.title(f'Atten')
            plt.imshow(image[0, -1].detach().cpu().numpy(), cmap='gray')
            plt.colorbar(orientation='horizontal').ax.set_title('μ/ρ (cm²/g)' if global_normalisation else '')
            plt.axis('off')
            plt.tight_layout()
            '''

            fig = plt.gcf()

            if wandb.run is not None:
                wandb.log({
                    'eval/pred_fig': fig,
                })
            else:
                display(fig)
                print(f'(PSNR:{psnr_scat}, SSIM:{ssim_scat}, RMSE:{rmse_scat})')

            plt.close()

        return psnr_scat, ssim_scat, rmse_scat, psnr_atten, ssim_atten, rmse_atten, pred if return_pred else None
    

class EncodedConditionalDiffusion(nn.Module):
    def __init__(self,
                 input_output_shape: tuple = (2, 128, 128),
                 num_timesteps: int = 1000,
                 batch_size: int = 16,
                 train_dataset=None,
                 valid_dataset=None,
                 latent: bool = False,
                 predict_mode: str = 'eps'):
        super(EncodedConditionalDiffusion, self).__init__()
        self.input_output_shape = input_output_shape
        self.condition_out_shape = (1, *self.input_output_shape[1:])

        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.sample_timesteps = num_timesteps // 10  # 100 steps with eta=1 seems to be the sweet spot
        self.train_timesteps = num_timesteps

        self.latent = latent
        self.predict_mode = predict_mode  # 'eps' or 'x0' or 'v'

        self.diffusion_process = DenoisingDiffusionConditionalProcess(
            generated_channels=self.input_output_shape[0],
            condition_channels=self.condition_out_shape[0],
            num_timesteps=self.train_timesteps,
            loss_fn=F.mse_loss,
            predict_mode=self.predict_mode,
        ) if not self.latent else LatentDiffusionConditional(
            num_timesteps=self.train_timesteps,
            batch_size=self.batch_size,
            train_dataset=self.train_dataset,
            valid_dataset=self.valid_dataset,
            predict_mode=self.predict_mode,
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        pass