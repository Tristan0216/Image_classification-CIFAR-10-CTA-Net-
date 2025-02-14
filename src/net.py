import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * 4 * 4, 512)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x
    
class TransformerBranch(nn.Module):
    def __init__(self, patch_size=4, embed_dim=128, num_heads=4, num_layers=2):
        super(TransformerBranch, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (32 // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(embed_dim * self.num_patches, 512)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        x = self.transformer_encoder(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc(x))
        return x

class CTANet(nn.Module):
    def __init__(self, num_classes=10):
        super(CTANet, self).__init__()
        self.cnn_branch = CNNBranch()
        self.transformer_branch = TransformerBranch()
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)  
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        cnn_features = self.cnn_branch(x)
        transformer_features = self.transformer_branch(x)

        # combined_features, _ = self.attention(
        #     cnn_features.unsqueeze(0),  # query
        #     transformer_features.unsqueeze(0),  # key
        #     transformer_features.unsqueeze(0)  # value
        # )
        # combined_features = combined_features.squeeze(0)

        combined_features = torch.cat((cnn_features, transformer_features), dim=1)
        
        out = self.fc(combined_features)
        return out
