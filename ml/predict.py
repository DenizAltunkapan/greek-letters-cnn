import os
import argparse

import torch
from PIL import Image
from torchvision import transforms

from config_utils import default_config_path, load_config, resolve_ml_path
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
    hidden_size = config.get("hidden_size", 128)
    dropout = config.get("dropout", 0.3)

    model = GreekLetterCNN(
        num_classes=len(classes),
        channels=channels,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)
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
    parser = argparse.ArgumentParser(description="Predict a Greek letter from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument(
        "model_path",
        nargs="?",
        default=None,
        help="Optional model path override. If omitted, value from config is used.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path(),
        help="Path to shared YAML config file.",
    )
    args = parser.parse_args()

    loaded_config, selected_config_path = load_config(args.config)
    paths_cfg = loaded_config.get("paths", {})

    image_path = args.image_path
    model_path = args.model_path or resolve_ml_path(paths_cfg.get("model_path", "models/best_model.pth"))
    model_path = os.path.abspath(model_path)

    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return

    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Config file: {selected_config_path}")
    print(f"Using model: {model_path}")
    model, classes = load_model_and_classes(model_path)
    predict_image(model, classes, image_path)


if __name__ == "__main__":
    main()
