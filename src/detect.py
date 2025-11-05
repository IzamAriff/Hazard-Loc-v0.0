# Inference module for hazard localization

import torch
from torchvision import transforms
from PIL import Image
from src.models.hazard_cnn import HazardCNN
from src.config import MODEL_SAVE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = HazardCNN(num_classes=2)
checkpoint = torch.load(MODEL_SAVE, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device).eval()

def predict_img(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    label = torch.argmax(prob, dim=1)
    return label.item(), prob.max().item()
