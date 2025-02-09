import torch
from torch import nn
import torch.nn.functional as F

class CTANet(nn.Module):
    def __init__(self, num_classes=10):
        super(CTANet, self).__init__()

        # using conv2d to extract spatial features
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
        # classification layer
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch_size, 3, 32, 32)
        spatial_features = self.spatial_conv(x)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)

        logits = self.fc(spatial_features)  # (batch_size, num_classes)
        return logits
