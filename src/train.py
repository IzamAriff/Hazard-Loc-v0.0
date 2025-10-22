# Hazard detection model training & validation

import torch
import torch.optim as optim
from torch import nn
from src.data.dataloader import get_dataloaders
from src.models.hazard_cnn import HazardCNN
from src.config import DATA_DIR, MODEL_SAVE

def train():
    loaders = get_dataloaders(DATA_DIR)
    model = HazardCNN(num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        model.train()
        for images, targets in loaders['train']:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} finished.")
    torch.save(model.state_dict(), MODEL_SAVE)
if __name__ == '__main__':
    train()
