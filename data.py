"""Data loading: Fashion-MNIST with a fixed train/validation split.

The official 60,000 training images are split into train and validation
(default 54,000 / 6,000) using a seeded permutation. The official 10,000-image
test set is only ever used by evaluate.py, so there is no test-set leakage.
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = (0.2860,), (0.3530,)  # Fashion-MNIST training-set statistics


def train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25),
    ])


def eval_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def get_train_val_loaders(data_dir="data", batch_size=128, val_size=6000,
                          seed=42, num_workers=2):
    # Two copies of the same images: one augmented (train), one plain (validation).
    train_aug = datasets.FashionMNIST(data_dir, train=True, download=True,
                                      transform=train_transform())
    train_plain = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=eval_transform())

    n = len(train_aug)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx, val_idx = perm[: n - val_size], perm[n - val_size:]

    train_loader = DataLoader(
        Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        Subset(train_plain, val_idx), batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def get_test_loader(data_dir="data", batch_size=128, num_workers=2):
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True,
                                     transform=eval_transform())
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
