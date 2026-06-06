import torch
import torch.nn as nn
import torch.optim as optim
import copy


@torch.no_grad()
def compute_confusion(model, loader, num_classes, device):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        for true_label, pred_label in zip(labels.cpu().view(-1), predicted.cpu().view(-1)):
            confusion[true_label.long(), pred_label.long()] += 1

    return confusion


def per_class_metrics(confusion, classes):
    confusion = confusion.float()
    support = confusion.sum(dim=1)
    predicted_positives = confusion.sum(dim=0)
    true_positives = confusion.diag()

    zeros = torch.zeros_like(true_positives)
    recall = torch.where(support > 0, true_positives / support, zeros)
    precision = torch.where(
        predicted_positives > 0, true_positives / predicted_positives, zeros
    )
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, zeros)

    rows = []
    for idx, name in enumerate(classes):
        rows.append(
            {
                "class": name,
                "support": int(support[idx].item()),
                "correct": int(true_positives[idx].item()),
                "recall": float(recall[idx].item()),
                "precision": float(precision[idx].item()),
                "f1": float(f1[idx].item()),
            }
        )
    return rows


def top_confusions(confusion, classes, top_k=10):
    pairs = []
    num_classes = len(classes)
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx].item())
            if count > 0:
                pairs.append((classes[true_idx], classes[pred_idx], count))

    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:top_k]


def format_per_class_report(rows, title="Per-class performance (worst first)"):
    ordered = sorted(rows, key=lambda r: (r["recall"], r["f1"]))

    lines = [title, "-" * len(title)]
    lines.append(f"{'class':<10}{'support':>8}{'correct':>8}{'recall':>9}{'precision':>11}{'f1':>8}")
    for r in ordered:
        lines.append(
            f"{r['class']:<10}{r['support']:>8}{r['correct']:>8}"
            f"{r['recall']:>9.3f}{r['precision']:>11.3f}{r['f1']:>8.3f}"
        )

    macro_recall = sum(r["recall"] for r in rows) / len(rows) if rows else 0.0
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    lines.append("-" * len(title))
    lines.append(f"macro avg recall: {macro_recall:.3f} | macro avg f1: {macro_f1:.3f}")
    return "\n".join(lines)


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
