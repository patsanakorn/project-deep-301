# Fashion-MNIST Image Classification with a CNN (PyTorch)

A convolutional neural network, built and trained from scratch, that classifies
28x28 grayscale images of clothing into 10 categories.

## Task

- **Input:** a 28x28 grayscale image of a clothing item
- **Output:** one of 10 classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
  Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Dataset:** [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  (60,000 training + 10,000 test images). It is downloaded automatically by
  `torchvision`, so **no data is stored in this repository**.

## Results

| Model | Val accuracy | Test accuracy |
|---|---|---|
| Baseline CNN (2 conv layers, no BatchNorm, no augmentation, 8 epochs) | 90.67% | 90.40% |
| **Final CNN (`FashionCNN`: 3 conv blocks, BatchNorm, augmentation, 25 epochs)** | **93.77%** | **93.23%** |

Per-class recall of the final model on the test set:

| Class | Recall | Class | Recall |
|---|---|---|---|
| T-shirt/top | 87.2% | Sandal | 97.8% |
| Trouser | 98.8% | Shirt | 80.5% |
| Pullover | 90.1% | Sneaker | 98.5% |
| Dress | 94.1% | Bag | 99.1% |
| Coat | 90.1% | Ankle boot | 96.1% |

The final-model numbers come from a run of `train.py` (best epoch: 22 of 25,
selected on validation accuracy) followed by `evaluate.py`; the full per-class
report is in `results/classification_report.txt`. An earlier run in the original
notebook gave 93.97% / 93.60% (validation / test); the small difference is
run-to-run variation (see *Reproducibility*). The baseline row comes from that
notebook.

**Error analysis.** About 64% of the baseline's errors were confusions among
T-shirt/top, Pullover, Coat and Shirt. Many "Shirt" images are visually almost
identical to T-shirts at 28x28 resolution, so Shirt remains the weakest class
(80.5% recall) even after the improvements.

## Repository structure

```
.
├── data.py            # dataset, augmentation, seeded train/val split, loaders
├── model.py           # FashionCNN architecture and class names
├── utils.py           # seeding and device helpers
├── train.py           # training script (train + validation only)
├── evaluate.py        # evaluation on the held-out test set
├── requirements.txt
├── checkpoints/       # trained weights (best_fashion_cnn_v2.pth, ~2 MB)
└── results/           # learning curves, confusion matrix, metrics
```

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A GPU is optional but speeds up training (about a few minutes on a Colab T4).

## Reproduce training

```bash
python train.py
```

This downloads Fashion-MNIST into `data/`, trains for 25 epochs, and writes:

- `checkpoints/best_fashion_cnn_v2.pth` (weights of the best epoch by validation accuracy)
- `results/history.json`, `results/learning_curves.png`, `results/train_config.json`

All options are listed with `python train.py --help`.

## Reproduce evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_fashion_cnn_v2.pth
```

This prints test accuracy and a per-class report, and writes
`results/test_metrics.json`, `results/classification_report.txt` and
`results/confusion_matrix.png`.

## Data split (no test-set leakage)

- The 60,000 official training images are split into **54,000 train / 6,000
  validation** with a seeded permutation (`seed=42`).
- Data augmentation is applied **only** to the training subset. The validation
  subset uses a separate, non-augmented copy of the same images.
- Model selection (best epoch) uses the **validation** set only.
- The official **10,000-image test set** is used only in `evaluate.py`.

## Model

`FashionCNN` has three blocks of (Conv-BatchNorm-ReLU) x 2 + MaxPool + Dropout(0.25)
with 32, 64 and 128 channels, followed by a fully connected layer (256 units,
Dropout 0.4) and a 10-way output layer. About 585k trainable parameters.

## Hyperparameters

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 (cosine annealing to 0 over training) |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Epochs | 25 |
| Loss | Cross-entropy |
| Augmentation (train only) | RandomHorizontalFlip, RandomCrop(28, padding=2), RandomErasing(p=0.25) |
| Normalization | mean 0.2860, std 0.3530 |
| Validation size | 6,000 |
| Random seed | 42 |

## Reproducibility

`train.py` seeds Python, NumPy and PyTorch (`--seed`, default 42), and the
train/validation split is fixed by the same seed. Some CUDA operations are
non-deterministic, so results on GPU can differ slightly between runs. For
closer reproducibility use `python train.py --deterministic` (slower).

## Limitations

- The model is trained on Fashion-MNIST only. Photos of real garments differ
  from the dataset (lighting, background, viewpoint, texture), so predictions on
  them are less reliable and often low-confidence. Cropping to the garment and
  using a bright-on-dark, 28x28 grayscale format helped in informal tests.
- Shirt vs. T-shirt/top vs. Pullover vs. Coat remain the hardest distinctions.

## Acknowledgements

Dataset: Xiao, Rasul and Vollgraf, *Fashion-MNIST: a Novel Image Dataset for
Benchmarking Machine Learning Algorithms* (2017).
