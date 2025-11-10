# Inference module for hazard localization

import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
from src.models.hazard_cnn import HazardCNN
from src.config import MODEL_SAVE

# --- Global variables for lazy loading ---
_model = None
_device = None
_transform = None

def _initialize_model():
    """Initializes the model, device, and transforms only when needed."""
    global _model, _device, _transform

    # If already initialized, do nothing.
    if _model is not None:
        return

    print("Initializing detection model for the first time...")
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = Path(MODEL_SAVE)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_SAVE}'. "
            "Please ensure the model is trained and the path in 'src/config.py' is correct."
        )

    _model = HazardCNN(num_classes=2)  # Assuming 2 classes: hazard, no-hazard
    # Set weights_only=False because the checkpoint includes optimizer state and metrics
    # which may contain non-tensor data like numpy arrays. This is safe as we are
    # loading a model we trained ourselves.
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model = _model.to(_device).eval()

    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def predict_img(image_path):
    _initialize_model()  # Ensures model is loaded
    image = Image.open(image_path).convert('RGB')
    img_tensor = _transform(image).unsqueeze(0).to(_device)
    output = _model(img_tensor)
    prob = torch.softmax(output, dim=1)
    label = torch.argmax(prob, dim=1)
    return label.item(), prob.max().item()
