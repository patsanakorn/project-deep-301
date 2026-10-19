"""Train the Fashion-MNIST CNN.

Uses only the train and validation splits. The best epoch (by validation
accuracy) is saved to checkpoints/best_fashion_cnn_v2.pth. The test set is
NOT touched here; run evaluate.py for that.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_train_val_loaders
from model import FashionCNN
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-size", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--deterministic", action="store_true",
                   help="force deterministic cuDNN (slower)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def plot_history(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    axes[1].set(title="Accuracy (%)", xlabel="Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)  # must run before model/data creation
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_train_val_loaders(
        args.data_dir, args.batch_size, args.val_size, args.seed, args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = FashionCNN().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc, best_epoch, best_state = 0.0, 0, None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        for key, value in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[key].append(value)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch}/{args.epochs}] | Train Loss: {tr_loss:.4f} | "
              f"Train Acc: {tr_acc:.2%} | Val Loss: {va_loss:.4f} | Val Acc: {va_acc:.2%}")

    out_dir, results_dir = Path(args.out_dir), Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = out_dir / "best_fashion_cnn_v2.pth"
    torch.save(best_state, ckpt_path)

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(results_dir / "train_config.json", "w") as f:
        json.dump({**vars(args), "best_val_acc": best_val_acc, "best_epoch": best_epoch},
                  f, indent=2)
    plot_history(history, results_dir / "learning_curves.png")

    print(f"\nBest Val Acc: {best_val_acc:.2%} (epoch {best_epoch})")
    print(f"Saved model to {ckpt_path}")


if __name__ == "__main__":
    main()
