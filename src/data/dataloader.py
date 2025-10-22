# PyTorch Dataset/DataLoader for images

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_data   = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_data  = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    return {
        "train": DataLoader(train_data, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_data, batch_size=batch_size),
        "test": DataLoader(test_data, batch_size=batch_size)
    }
