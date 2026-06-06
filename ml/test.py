import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config_utils import default_config_path, load_config, resolve_ml_path
from dataset import get_dataloaders
from model import GreekLetterCNN
from trainer import (
    compute_confusion,
    format_per_class_report,
    per_class_metrics,
    top_confusions,
)


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_per_class_csv(rows, csv_path):
    ensure_parent_dir(csv_path)
    ordered = sorted(rows, key=lambda r: (r["recall"], r["f1"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class", "support", "correct", "recall", "precision", "f1"]
        )
        writer.writeheader()
        for r in ordered:
            writer.writerow(r)


def save_confusion_plot(confusion, classes, plot_path):
    ensure_parent_dir(plot_path)
    matrix = confusion.cpu().numpy()
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.zeros_like(matrix, dtype=float)
    np.divide(matrix, row_sums, out=normalised, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(normalised, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="Fraction of true class")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (row-normalised)")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


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

    _, _, test_loader, test_classes = get_dataloaders(data_dir, batch_size)

    if list(test_classes) != list(classes):
        raise ValueError(
            "Class order mismatch between checkpoint and test set.\n"
            f"  checkpoint: {classes}\n"
            f"  test set:   {test_classes}"
        )

    confusion = compute_confusion(model, test_loader, num_classes, device)

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    test_acc = correct / total if total else 0.0

    rows = per_class_metrics(confusion, classes)

    print(f"Config file: {selected_config_path}")
    print(f"Model path: {model_path}")
    print(f"Test Accuracy: {test_acc:.4f}  ({correct}/{total})")
    print()
    print(format_per_class_report(rows, title="Per-class test performance (worst first)"))

    print("\nMost frequent confusions (true -> predicted):")
    for true_class, pred_class, count in top_confusions(confusion, classes, top_k=10):
        print(f"  {true_class:<10} -> {pred_class:<10} {count}")

    csv_path = resolve_ml_path("plots/test_per_class_report.csv")
    plot_path = resolve_ml_path("plots/test_confusion_matrix.png")
    save_per_class_csv(rows, csv_path)
    save_confusion_plot(confusion, classes, plot_path)

    print(f"\nPer-class report saved to {csv_path}")
    print(f"Confusion matrix saved to {plot_path}")

    weakest = sorted(rows, key=lambda r: r["recall"])[:5]
    print("\nFocus suggestion - letters with the lowest recall (add/clean data here):")
    for r in weakest:
        print(f"  {r['class']:<10} recall={r['recall']:.3f}  (correct {r['correct']}/{r['support']})")


if __name__ == "__main__":
    main()
