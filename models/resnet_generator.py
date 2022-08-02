import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, c=[64, 128, 256], blocks=9):
        super(ResnetGenerator, self).__init__()

        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, c[0], 7),
            nn.InstanceNorm2d(64), nn.GELU(),
        ]
        
        # Downsampling
        for i in range(len(c) - 1):
            model += [
                nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1),
                nn.InstanceNorm2d(c[i + 1]),
                nn.GELU(),
            ]

        # Residual blocks
        for _ in range(blocks):
            model += [ResidualBlock(c[-1])]

        # Upsampling
        c = c[::-1]
        for i in range(len(c) - 1):
            model += [
                nn.Upsample(scale_factor=2), nn.Conv2d(c[i], c[i + 1], 3, 1, 1),
                nn.InstanceNorm2d(c[i + 1]), nn.GELU(),
            ]

        # Output layer
        model += nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(c[-1], 18, 7),
            nn.Upsample(scale_factor=2), nn.Conv2d(18, 3, 4, 2, 1), nn.GELU(),
            nn.Upsample(scale_factor=2), nn.Conv2d(3, 3, 4, 2, 1),
            nn.Tanh()
        )

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
