"""Evaluate a trained checkpoint on the official Fashion-MNIST test set.

Writes test accuracy, a per-class report and a confusion matrix to results/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from data import get_test_loader
from model import CLASS_NAMES, FashionCNN
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/best_fashion_cnn_v2.pth")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    model = FashionCNN().to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()  # disables dropout, uses BatchNorm running stats

    loader = get_test_loader(args.data_dir, args.batch_size)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    preds, labels = np.array(preds), np.array(labels)

    accuracy = float((preds == labels).mean())
    report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=3)
    print(f"Test Accuracy: {accuracy:.2%}\n")
    print(report)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump({"test_accuracy": accuracy, "checkpoint": args.checkpoint}, f, indent=2)
    (results_dir / "classification_report.txt").write_text(report)

    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(
        confusion_matrix(labels, preds), display_labels=CLASS_NAMES
    ).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    main()
