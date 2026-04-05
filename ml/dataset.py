import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

# Normal images: grayscale, resize, tensor, normalize
base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_transform = base_transform

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
    train_dataset = datasets.ImageFolder(train_dir, transform=base_transform)

    # Load validation/test datasets
    val_dataset   = datasets.ImageFolder(val_dir, transform=base_transform)
    test_dataset  = datasets.ImageFolder(test_dir, transform=base_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = train_dataset.classes

    return train_loader, val_loader, test_loader, classes