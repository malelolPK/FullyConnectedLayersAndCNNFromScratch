import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(batch_size, val_split=0.2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Pobierz pełny zestaw danych treningowych
    full_train_data = datasets.MNIST("data/datasets", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("data/datasets", train=False, download=True, transform=transform)

    # Oblicz rozmiary zbiorów
    train_size = int(len(full_train_data) * (1 - val_split))
    val_size = len(full_train_data) - train_size

    # Podziel dane treningowe na train i validation
    train_data, val_data = torch.utils.data.random_split(
        full_train_data,
        [train_size, val_size],
        #generator=torch.Generator().manual_seed(42)  # dla powtarzalności
    )

    # Utwórz data loadery
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    return train_loader, val_loader, test_loader