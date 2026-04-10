import torch
import torch.nn as nn
import torch.optim as optim
import copy


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


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=20,
    lr=0.001,
    weight_decay=0.0,
    patience=6,
    min_delta=1e-4,
    scheduler_patience=3,
    scheduler_factor=0.5,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    train_losses = []
    val_accs = []
    lrs = []

    best_val_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

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
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        scheduler.step(val_acc)

        improved = val_acc > (best_val_acc + min_delta)
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch+1} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}. "
                f"Best val acc {best_val_acc:.4f} at epoch {best_epoch}."
            )
            break

    return {
        "train_loss_history": train_losses,
        "val_acc_history": val_accs,
        "lr_history": lrs,
        "best_state": best_state,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_trained": len(train_losses),
    }
