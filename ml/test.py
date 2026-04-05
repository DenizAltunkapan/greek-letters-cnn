import torch
from dataset import get_dataloaders
from model import GreekLetterCNN

data_dir = "data"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/best_model.pth"

# load checkpoint
checkpoint = torch.load(model_path, map_location=device)

config = checkpoint["config"]
classes = checkpoint["classes"]
num_classes = len(classes)

# create model with correct architecture
model = GreekLetterCNN(
    num_classes=num_classes,
    channels=config["channels"]
).to(device)

# load weights
model.load_state_dict(checkpoint["model_state"])
model.eval()

# get test loader
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
print(f"Test Accuracy: {test_acc:.4f}")