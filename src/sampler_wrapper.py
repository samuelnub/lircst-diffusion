from Diffusion.DenoisingDiffusionProcess.samplers import DDIM_Sampler
import torch
import torch.nn as nn

class SamplerWrapper():
    def __init__(self, sample_timesteps=200, train_timesteps=1000):
        self.sample_timesteps = sample_timesteps
        self.train_timesteps = train_timesteps
        self.sampler = DDIM_Sampler(num_timesteps=self.sample_timesteps, train_timesteps=self.train_timesteps)

    def get_sampler(self) -> DDIM_Sampler:
        return self.sampler