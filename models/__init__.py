import imp
from resnet_generator import *
from unet_generator import *


class Discriminator(nn.Module):
    def __init__(self, c=[64, 128, 256]):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(6, c[0], 4, 2, 1),
            nn.BatchNorm2d(c[0]), nn.LeakyReLU(),
        ]
        
        for i in range(len(c) - 1):
            model += [
                nn.Conv2d(c[i], c[i + 1], 4, 2, 1),
                nn.BatchNorm2d(c[i + 1]), nn.LeakyReLU(),
            ]
        
        model += [
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), 
            nn.Linear(c[-1], 1), nn.Sigmoid()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, A, x):
        return self.model(torch.cat((A, x), dim=1))