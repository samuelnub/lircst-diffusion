import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from .forward import *
from .samplers import *
from .backbones.unet_convnext import *

from util import extract, beta_scheduler # ADDED: utility function to extract values from tensors based on indices


class DenoisingDiffusionProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,              
                 loss_fn=F.mse_loss,
                 schedule=beta_scheduler,
                 num_timesteps=1000,
                 sampler=None
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels=generated_channels
        self.num_timesteps=num_timesteps
        self.loss_fn=loss_fn
        
        # Forward Process Used for Training
        self.forward_process=GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                    schedule=schedule)
        self.model=UnetConvNextBlock(dim=64,
                                     dim_mults=(1,2,4,8),
                                     channels=self.generated_channels,
                                     out_dim=self.generated_channels,
                                     with_time_emb=True)
               
        
        # defaults to a DDPM sampler if None is provided
        self.sampler=DDPM_Sampler(num_timesteps=self.num_timesteps) if sampler is None else sampler
        
    @torch.no_grad()
    def forward(self,
                shape=(256,256),
                batch_size=1,
                sampler=None,
                verbose=False
               ):
        """
            forward() function triggers a complete inference cycle
            
            A custom sampler can be provided as an argument!
        """           
        
        # read dimensions
        b,c,h,w=batch_size,self.generated_channels,*shape
        device=next(self.model.parameters()).device
        
        # select sampler
        if sampler is None:
            sampler=self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps=sampler.num_timesteps 
        it=reversed(range(0, num_timesteps))    
        
        x_t = torch.randn([b, self.generated_channels, h, w],device=device)
                
        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t=self.model(x_t,t) # prediction of noise
            x_t = sampler(x_t,t,z_t) # prediction of next state
            
        return x_t
        
    def p_loss(self,output):
        """
            Assumes output is in [-1,+1] range
        """        
        
        b,c,h,w=output.shape
        device=output.device
        
        # loss for training
        
        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise=self.forward_process(output,t,return_noise=True)        

        # reverse pass
        noise_hat = self.model(output_noisy, t) 

        # apply loss
        return self.loss_fn(noise, noise_hat)
    
    
class DenoisingDiffusionConditionalProcess(nn.Module):
    
    def __init__(self,
                 generated_channels=3,
                 condition_channels=3,
                 loss_fn=F.mse_loss,
                 schedule=beta_scheduler,
                 num_timesteps=1000,
                 sampler=None,
                 predict_mode='v'  # 'eps' or 'x0' or 'v'
                ):
        super().__init__()
        
        # Basic Params
        self.generated_channels=generated_channels
        self.condition_channels=condition_channels
        self.num_timesteps=num_timesteps
        self.loss_fn=loss_fn
        self.predict_mode=predict_mode  # 'eps' or 'x0' or 'v'
        
        # Forward Process
        self.forward_process=GaussianForwardProcess(num_timesteps=self.num_timesteps,
                                                    schedule=schedule)
        
        # Neural Network Backbone
        self.model=UnetConvNextBlock(dim=64,
                                     dim_mults = (1,2,4,8),
                                     channels=self.generated_channels + condition_channels,
                                     out_dim=self.generated_channels,
                                     with_time_emb=True)
        
        # defaults to a DDPM sampler if None is provided
        self.sampler=DDPM_Sampler(num_timesteps=self.num_timesteps) if sampler is None else sampler

    @torch.no_grad()
    def forward(self,
                condition,
                sampler=None,
                verbose=False
               ):
        """
            forward() function triggers a complete inference cycle
            
            A custom sampler can be provided as an argument!
        """
        
        # read dimensions
        b,c,h,w=condition.shape
        device=next(self.model.parameters()).device
        condition=condition.to(device)
        
        # select sampler
        if sampler is None:
            sampler=self.sampler
        else:
            sampler.to(device)

        # time steps list
        num_timesteps=sampler.num_timesteps 
        it=reversed(range(0, num_timesteps))        
        
        x_t = torch.randn([b, self.generated_channels, h, w],device=device)
        
        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_input=torch.cat([x_t,condition],1).to(device)

            if self.predict_mode == 'eps':
                z_t=self.model(model_input,t) # prediction of noise            
                x_t=self.sampler(x_t,t,z_t) # prediction of next state

            if self.predict_mode == 'x0':
                x_0_pred = self.model(model_input, t)  # prediction of x_0
                noise_hat = (x_t - extract(self.forward_process.alphas_cumprod_sqrt, t, x_t.shape) * x_0_pred) \
                            / extract(self.forward_process.alphas_one_minus_cumprod_sqrt, t, x_t.shape)
                x_t = self.sampler(x_t, t, noise_hat)  # prediction of next state

            if self.predict_mode == 'v':
                # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
                v_pred = self.model(model_input, t)  # prediction of velocity
                x_0_pred = (extract(self.forward_process.alphas_cumprod_sqrt, t, x_t.shape) * x_t ) \
                            - (extract(self.forward_process.alphas_one_minus_cumprod_sqrt, t, x_t.shape) * v_pred)
                noise_hat = (extract(self.forward_process.alphas_cumprod_recip_sqrt, t, x_t.shape) * x_t - x_0_pred) \
                            / extract(self.forward_process.alphas_cumprod_minus_one_recip_sqrt, t, x_t.shape)
                x_t = self.sampler(x_t, t, noise_hat)  # prediction of next state

        return x_t
        
    def p_loss(self,output,condition):
        """
            Assumes output and input are in [-1,+1] range
        """        
        
        b,c,h,w=output.shape
        device=output.device
        
        # loss for training
        
        # input is the optional condition
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()
        output_noisy, noise=self.forward_process(output,t,return_noise=True)        

        # reverse pass
        model_input=torch.cat([output_noisy,condition],1).to(device)
        target_pred = self.model(model_input, t) 

        if self.predict_mode == 'eps':
            target = noise

        if self.predict_mode == 'x0':
            target = output

        if self.predict_mode == 'v':
            # Compute velocity
            target = (extract(self.forward_process.alphas_cumprod_sqrt, t, noise.shape) * noise) \
                    - (extract(self.forward_process.alphas_one_minus_cumprod_sqrt, t, output.shape) * output)
            
        # apply loss
        loss: torch.Tensor = self.loss_fn(target, target_pred, reduction='none')
        # Given loss of shape [b, ...]
        mean_over = tuple(range(1, len(loss.shape)))  # mean over all dimensions except batch
        loss = loss.mean(dim=mean_over)  # shape [b]
        loss = loss * extract(self.forward_process.loss_weight, t, loss.shape)  # apply loss weight

        return loss.mean(), output_noisy, target_pred, t # ADDED: return noisy output and noise prediction and timestep for our physics model