import torch
from dataset import get_dataloaders
from model import GreekLetterCNN

data_dir = "data"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/greek_letter_cnn_v1.pth"

_, _, test_loader, classes = get_dataloaders(data_dir, batch_size)
num_classes = len(classes)

model = GreekLetterCNN(num_classes, 128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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