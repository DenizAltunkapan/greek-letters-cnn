import argparse
import torch

from config_utils import default_config_path, load_config, resolve_ml_path
from dataset import get_dataloaders
from model import GreekLetterCNN


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved best model on test data.")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_path(),
        help="Path to shared YAML config file.",
    )
    args = parser.parse_args()

    loaded_config, selected_config_path = load_config(args.config)
    paths_cfg = loaded_config.get("paths", {})
    data_cfg = loaded_config.get("data", {})

    data_dir = resolve_ml_path(data_cfg.get("data_dir", "data"))
    batch_size = int(data_cfg.get("batch_size", 32))
    model_path = resolve_ml_path(paths_cfg.get("model_path", "models/best_model.pth"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint["config"]
    classes = checkpoint["classes"]
    num_classes = len(classes)
    channels = model_config.get("channels", [32, 64])
    hidden_size = model_config.get("hidden_size", 128)
    dropout = model_config.get("dropout", 0.3)

    model = GreekLetterCNN(
        num_classes=num_classes,
        channels=channels,
        hidden_size=hidden_size,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_loader, _ = get_dataloaders(data_dir, batch_size)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    print(f"Config file: {selected_config_path}")
    print(f"Model path: {model_path}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
