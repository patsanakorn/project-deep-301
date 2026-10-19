"""CNN for Fashion-MNIST (28x28 grayscale, 10 classes)."""
import torch.nn as nn

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _block(in_ch: int, out_ch: int, p_drop: float = 0.25) -> nn.Sequential:
    """Two 3x3 convs (+BatchNorm+ReLU), then 2x2 max-pool and dropout."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p_drop),
    )


class FashionCNN(nn.Module):
    """Three conv blocks followed by a small fully connected classifier.

    Shape flow: [1,28,28] -> [32,14,14] -> [64,7,7] -> [128,3,3] -> 10 logits.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
