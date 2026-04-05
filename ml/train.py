import torch
import matplotlib.pyplot as plt
import os

from dataset import get_dataloaders
from model import GreekLetterCNN
from trainer import train_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader, classes = get_dataloaders("data", 32)
num_classes = len(classes)

configs = [
    {"channels": [32, 64]},
    {"channels": [32, 64, 128]},
    {"channels": [32, 64, 128, 256]},
]

results = []

best_model = None
best_score = 0.0
best_config = None

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

for cfg in configs:
    print("\nTesting config:", cfg)

    model = GreekLetterCNN(
        num_classes=num_classes,
        channels=cfg["channels"]
    ).to(device)

    train_loss, val_acc = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=20
    )

    results.append({
        "config": cfg,
        "train_loss": train_loss,
        "val_acc": val_acc
    })

    # stability-based score (mean of last 4 epochs)
    window = 4
    stability_score = sum(val_acc[-window:]) / window

    if stability_score > best_score:
        best_score = stability_score
        best_model = model.state_dict()
        best_config = cfg

# plot validation accuracy
plt.figure()

for r in results:
    plt.plot(r["val_acc"], label=str(r["config"]["channels"]))

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Architecture Comparison")
plt.legend()

plot_path = "plots/architecture_comparison.png"
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")

plt.show()


# save best model
model_path = "models/best_model.pth"
torch.save({
    "model_state": best_model,
    "config": best_config,
    "stability_score": best_score,
    "classes": classes
}, model_path)

print("\nBest architecture:", best_config)
print("Best stability score:", best_score)
print(f"Best model saved to {model_path}")