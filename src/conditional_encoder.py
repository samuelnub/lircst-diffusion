import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalEncoder(nn.Module):
    # Conditional encoder which translates the conditional input into a latent space
    # that can be used in conjunction with our input image space

    def __init__(self, in_shape: tuple, out_shape: tuple, permute_shape: tuple = (2, 0, 1)):
        super(ConditionalEncoder, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.permute_shape = permute_shape

        # TODO: Unuused as this does not work. We will just use non-DL methods to encode for now.
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_shape[permute_shape[0]], 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, self.out_shape[0], kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute((0, self.permute_shape[0]+1, self.permute_shape[1]+1, self.permute_shape[2]+1))

        # No deep learning for now
        x = torch.mean(x, dim=-3, keepdim=True)

        x = F.interpolate(x, size=self.out_shape[1:], mode='bilinear', align_corners=False)
        
        return x

