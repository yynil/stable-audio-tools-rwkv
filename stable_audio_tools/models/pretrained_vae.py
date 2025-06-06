import torch
from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
from torch import nn
from typing import Optional
class PretrainedVAEWrapper(nn.Module):
    def __init__(self, pretrained_path: str):
        super().__init__()
        self.pretrained_vae = AutoencoderOobleck.from_pretrained(pretrained_path)
        self.downsampling_ratio = 1
        for r in self.pretrained_vae.downsampling_ratios:
            self.downsampling_ratio *= r

    def forward(self, sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
        return self.pretrained_vae.forward(sample, sample_posterior, return_dict, generator)

    def encode(self, x):
        if x.device != self.pretrained_vae.device:
            x = x.to(self.pretrained_vae.device)
        latent_dist = self.pretrained_vae.encode(x).latent_dist
        return latent_dist.sample()

    def decode(self, x):
        return self.pretrained_vae.decode(x).sample
    
    