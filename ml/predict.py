import os
import sys

import torch
from PIL import Image
from torchvision import transforms

from model import GreekLetterCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def _default_model_path():
    # Resolve relative to this file so script works from any CWD.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "models", "best_model.pth")


def load_model_and_classes(model_path):
    checkpoint = torch.load(model_path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(
            "Invalid model format. Expected a checkpoint containing 'model_state', 'config', and 'classes'."
        )

    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint does not contain a class list ('classes').")

    config = checkpoint.get("config", {})
    channels = config.get("channels", [32, 64])

    model = GreekLetterCNN(num_classes=len(classes), channels=channels).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes


def predict_image(model, classes, image_path):
    with Image.open(image_path) as image:
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    label = classes[predicted_idx.item()]
    conf = confidence.item() * 100
    print(f"Prediction for {image_path}: {label} ({conf:.2f}%)")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python ml/predict.py <path_to_image> [path_to_model]")
        return

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) == 3 else _default_model_path()
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return

    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return

    model, classes = load_model_and_classes(model_path)
    predict_image(model, classes, image_path)


if __name__ == "__main__":
    main()
