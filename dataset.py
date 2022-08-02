from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as tt
from PIL import Image

import random
import os


class Data(Dataset):
    def __init__(self, path, transform=None):
        super(Dataset, self).__init__()
        
        self.files_A = os.listdir(os.path.join(path, "A/"))
        
        self.path = path
        self.transform = transform
        
    def __len__(self):
        return len(self.files_A)
    
    def __getitem__(self, idx):
        path_A = os.path.join(os.path.join(self.path, "A/"), self.files_A[idx])
        path_B = os.path.join(os.path.join(self.path, "B/"), self.files_A[idx].replace("B", "A"))
        
        A = Image.open(path_A).convert("RGB")
        B = Image.open(path_B).convert("RGB")
        
        if random.random() > 0.5:
            A = TF.hflip(A)
            B = TF.hflip(B)
        
        if self.transform is not None:
            A = self.transform(A)
            B = self.transform(B)
        
        return A, B


def get_dataloader(path, image_size=128, batch_size=32):

    train_transform = tt.Compose([
        tt.Resize((image_size, image_size), Image.BICUBIC),
        tt.ToTensor(),
    
        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loader = DataLoader(
        Data(path, train_transform), batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )
    
    return data_loader
