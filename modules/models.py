"""
Contains PyTorch model code to instantiate a TinyVGG model from the CNN Explainer website.
"""
import torch
from torch import nn

class TinyVGG_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

    def forward(self, x):
        return self.block(x)

class TinyVGG(nn.Module):
    """Creates the TinyVGG architecture.

    Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
    See the original architecture here: https://poloclub.github.io/cnn-explainer/
    
    Args:
        input_shape: An integer indicating number of input channels.
        hidden_units: An integer indicating number of hidden units between layers.
        output_shape: An integer indicating number of output units.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.features = nn.Sequential(
            TinyVGG_ConvBlock(input_shape, hidden_units),
            TinyVGG_ConvBlock(hidden_units, hidden_units)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_units, output_shape)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
