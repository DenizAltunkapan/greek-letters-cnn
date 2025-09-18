import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

# Normal images: grayscale, resize, tensor, normalize
normal_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Augmented images: same as normal + augmentation
augmented_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = normal_transform

def get_dataloaders(data_dir="ml/data", batch_size=32):
    """
    Creates DataLoaders for training, validation, and test datasets.
    Training dataset contains both normal and augmented images.
    Expects folder structure:
    data/
      train/Alpha, train/Beta, ...
      val/Alpha, val/Beta, ...
      test/Alpha, test/Beta, ...
    """

    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")
    test_dir  = os.path.join(data_dir, "test")

    # Load training datasets
    normal_dataset = datasets.ImageFolder(train_dir, transform=normal_transform)
    aug_dataset    = datasets.ImageFolder(train_dir, transform=augmented_transform)
    train_dataset  = ConcatDataset([normal_dataset, aug_dataset])

    # Load validation/test datasets
    val_dataset  = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = normal_dataset.classes

    return train_loader, val_loader, test_loader, classes