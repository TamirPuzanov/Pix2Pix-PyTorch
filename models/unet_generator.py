import torch.nn as nn
import torch


class UnetGenerator(nn.Module):
    def __init__(self, c=[64, 256, 512, 1024, 2048]):
        super(UnetGenerator, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 1, 1),
            nn.BatchNorm2d(c[0]), nn.LeakyReLU() 
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c[0] + 3, c[0] // 2, 3, 1, 1),
            nn.BatchNorm2d(c[0] // 2), nn.LeakyReLU(),
            
            nn.ConvTranspose2d(c[0] // 2, 3, 4, 2, 1),
            nn.BatchNorm2d(3), nn.LeakyReLU(),
            
            nn.Conv2d(3, 3, 4, 2, 1), nn.Tanh()
        )
        
        self.down_sample = nn.Sequential(*[
            self.down_block(c[i], c[i + 1]) for i in range(len(c) - 1)
        ])
        
        c = c[::-1]
        self.up_sample = nn.Sequential(*[
            self.up_block(c[i], c[i + 1]) for i in range(len(c) - 1)
        ])
    
    def down_block(self, inp, out):
        return nn.Sequential(*[
            nn.Conv2d(inp, out, 4, 2, 1),
            nn.BatchNorm2d(out), nn.LeakyReLU(),
        ])
    
    def up_block(self, inp, out):
        return nn.Sequential(*[
            nn.ConvTranspose2d(inp * 2, out, 4, 2, 1),
            nn.BatchNorm2d(out), nn.LeakyReLU(),
        ])
    
    def forward(self, x0):
        x = self.conv1(x0)
        r = []
        
        for block in self.down_sample:
            x = block(x); r.append(x)
        
        for block, b in zip(self.up_sample, r[::-1]):
            x = block(torch.cat((x, b), dim=1))
        
        x = self.conv2(torch.cat((x, x0), dim=1))
        return x
