# Accuracy & performance metrics

import torch
from src.models.hazard_cnn import HazardCNN
from src.data.dataloader import get_dataloaders
from src.config import DATA_DIR, MODEL_SAVE

def evaluate():
    loaders = get_dataloaders(DATA_DIR)
    model = HazardCNN(num_classes=2)
    # Set weights_only=False because the checkpoint includes optimizer state and metrics.
    # This is safe as we are loading a model we trained ourselves.
    checkpoint = torch.load(MODEL_SAVE, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # Evaluate and print accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in loaders['test']:
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print('Test accuracy:', correct/total)
if __name__ == '__main__':
    evaluate()
