import imp
from torch.autograd import Variable
from tqdm import tqdm

import torch.nn as nn
import torch

from torchvision import models
import torchvision.transforms as tt
import torchvision.transforms.functional as TF

from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
import os, random

from dataset import get_dataloader
import matplotlib.pyplot as plt

from optparse import OptionParser
import models
import utils


parser = OptionParser()

parser.add_option("-m", "--model", dest="model", type="string", default="unet", help="unet, resnet")
parser.add_option("-e", "--epoch", dest="epoch", type="int", default=250)
parser.add_option("-d", "--data_path", dest="data_path", type="string")
parser.add_option("-s", "--image_size", dest="image_size", type="int", default=128)
parser.add_option("-b", "--batch_size", dest="batch_size", type="int", default=64)
parser.add_option("-c", "--cuda", dest="cuda", action="store_false", default=True)

parser.add_option("-r", "--lr", dest="lr", type="float", default=0.0002)
parser.add_option("-l", "--lambda", dest="lambda_", type="float", default=2.5)
parser.add_option("-q", "--quiet", dest="quiet", action="store_true", default=False)

(options, args) = parser.parse_args()

if not options.data_path:
    parser.error('data_path not given')

if options.model not in ["unet", "resnet"]:
    parser.error('model ["unet", "resnet"]')

device = torch.device("cuda" if parser.cuda else "cpu")

train_dl = get_dataloader(
    path=options.data_path, image_size=options.image_size, 
    batch_size=options.batch_size
)

if options.model == "unet":
    model_g = models.UnetGenerator()
else:
    model_g = models.ResnetGenerator()

model_d = models.Discriminator()

optim_g = torch.optim.Adamax(model_g.parameters(), lr=0.0002)
optim_d = torch.optim.Adamax(model_d.parameters(), lr=0.0002)

scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=4500)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=4500)

criterion_d = nn.BCELoss()
criterion_g = nn.L1Loss()

lambda_ = options.lambda_

real_label = torch.ones(options.batch_size, device=device)
fake_label = torch.zeros(options.batch_size, device=device)


def train_batch(batch, buffer):
    A = batch[0].to(device)
    B = batch[1].to(device)
    
    utils.set_requires_grad(model_d, p=True)
    optim_d.zero_grad()
    
    fake = model_g(A)
    
    out = model_d(A, B).view(-1)
    lossD_real = criterion_d(out, real_label)
    real_score = out.mean().item()
    
    out = model_d(A, buffer.push_and_pop(fake.detach())).view(-1)
    lossD_fake = criterion_d(out, fake_label)
    fake_score = out.mean().item()
    
    lossD = (lossD_real + lossD_fake) / 2
    lossD.backward()
    
    optim_d.step()
    
    utils.set_requires_grad(model_d, p=False)
    optim_g.zero_grad()
    
    out = model_d(A, fake).view(-1)
    
    lossG_gan = criterion_d(out, real_label)
    lossG_pix = criterion_g(fake, B) * lambda_
    
    lossG = lossG_pix + lossG_gan
    lossG.backward()
    
    optim_g.step()
    
    return {
        "lossG": lossG.item(), "lossD": lossD.item(),
        "real": real_score, "fake": fake_score
    }


def train_epoch(epoch):
    model_g.train()
    model_d.train()
    
    buffer = utils.Buffer(250)
    
    tq = tqdm(train_dl, total=len(train_dl), desc=f"Epoch #{epoch}")
    scores = {"lossG": 0, "lossD": 0, "real": 0, "fake": 0, "n": 0}
    
    for batch in tq:
        m = train_batch(batch, buffer)
        scores["n"] += 1
        
        for c in m.keys():
            scores[c] += m[c]
    
        tq.set_postfix({
            k: v / scores["n"] for k, v in scores.items()
        })


for epoch in range(options.epoch):
    torch.cuda.empty_cache()
    train_epoch(epoch)
    
    scheduler_g.step()
    scheduler_d.step()


# Save generator
traced_m = torch.jit.trace(model_g.cpu(), (torch.rand(1, 3, 128, 128)))
torch.jit.save(traced_m, "model_g.pt")
print("Model saved!")