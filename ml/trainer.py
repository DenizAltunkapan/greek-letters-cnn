import torch
import torch.nn as nn
import torch.optim as optim


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def train_model(model, train_loader, val_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        epoch_loss = running_loss / total
        train_losses.append(epoch_loss)

        val_acc = evaluate(model, val_loader, device)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")

    return train_losses, val_accs