import torch
import torchvision.transforms.functional as TF

import random


class Buffer:
    def __init__(self, size=100, device=torch.device("cuda")):
        self.size = size 
        self.buffer = []
        self.device = device

    def push_and_pop(self, data):
        data = data.cpu()
        r = []
        
        for el in data.data:
            el = torch.unsqueeze(el, 0)
            
            if len(self.buffer) < self.size:
                self.buffer.append(el)
                r.append(el)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.size - 1)
                    r.append(self.buffer[i])
                    self.buffer[i] = el
                else:
                    r.append(el)
        
        return torch.cat(r).to(self.device)


class Denormalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = TF.normalize(tensor, self.demean, self.destd, self.inplace)
        return torch.clamp(tensor, 0.0, 1.0)


def set_requires_grad(model, p=True):
    for param in model.parameters():
        param.requires_grad = p


def weights_normal(model):
    class_name = model.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
        if hasattr(model, "bias") and model.bias is not None:
            torch.nn.init.constant_(model.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
