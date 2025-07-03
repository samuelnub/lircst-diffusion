from Diffusion.DenoisingDiffusionProcess.samplers import DDIM_Sampler
from Diffusion.DenoisingDiffusionProcess.samplers import DDPM_Sampler # Slower but might work better
import torch
import torch.nn as nn

class SamplerWrapper():
    def __init__(self, sample_timesteps=200, train_timesteps=1000):
        self.sample_timesteps = sample_timesteps
        self.train_timesteps = train_timesteps # TODO VVVVVVVV self.sample_timesteps is used for sampling
        self.sampler = DDIM_Sampler(num_timesteps=self.sample_timesteps, train_timesteps=self.train_timesteps)
        # self.sampler = DDPM_Sampler(num_timesteps=self.train_timesteps)
        # 


    def get_sampler(self) -> DDIM_Sampler | DDPM_Sampler:
        return self.sampler