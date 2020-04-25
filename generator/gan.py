# 発展的DL参考
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=20, image_size=60):
        super(Generator, self).__init__()
        
        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, image_size * 8, kernel_size=4, stride=1),
            nn.BatchNorm2d(image_size * 8),
            nn.ReLU(inplace=True)
            ) # z_dim × z_dim ->