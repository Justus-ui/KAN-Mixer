import torch
import torch.nn.functional as F
from torch import nn
import math
from KAN_NN_fast import KAN_layer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbedding(nn.Module):
    """
    Path embedding layer is nothing but a convolutional layer with kerneli size and stride equal to patch size.
    """

    def __init__(self, in_channels, embedding_dim, patch_size):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels, embedding_dim, patch_size, patch_size
        )

    def forward(self, x):
        return self.patch_embedding(x)

class KAN(nn.Module):
    def __init__(self, dim, intermediate_dim, dropout=0.0, grid_size=5, spline_order=3):
        super().__init__()
        self.kan = nn.Sequential(
            KAN_layer(dim, intermediate_dim, hidden = [4]),
            KAN_layer(intermediate_dim, dim, hidden = [4]),
        )
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        return self.kan(x)
    


class Transformation1(nn.Module):
    """
    The transformation that is used in Mixer Layer (the T) which just switches the 2nd and the 3rd dimensions and is applied before and after Token Mixing KANs
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 1))


class Transformation2(nn.Module):
    """
    The transformation that is applied right after the patch embedding layer and convert it's shape from (batch_size,  embedding_dim, sqrt(num_patches), sqrt(num_patches)) to (batch_size, num_patches, embedding_dim)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.permute(x, (0, 2, 3, 1)).reshape(x.shape[0], -1, x.shape[1])


class MixerLayer(nn.Module):
    """
    Mixer layer which consists of Token Mixer and Channel Mixer modules in addition to skip connections.
    intermediate_output = Token Mixer(input) + input
    final_output = Channel Mixer(intermediate_output) + intermediate_output
    """

    def __init__(
        self,
        embedding_dim,
        num_patch,
        token_intermediate_dim,
        channel_intermediate_dim,
        dropout=0.0,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            Transformation1(),
            KAN(num_patch, token_intermediate_dim, dropout, grid_size, spline_order),
            Transformation1(),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            KAN(
                embedding_dim,
                channel_intermediate_dim,
                dropout,
                grid_size,
                spline_order,
            ),
        )

    def forward(self, x):
        val_token_mixer = self.token_mixer(x).to(device)
        val_channel_mixer = self.channel_mixer(x).to(device)
        x = x.to(device)

        x = x + val_token_mixer  # Token mixer and skip connection
        x = x + val_channel_mixer  # Channel mixer and skip connection

        return x


class KANMixer(nn.Module):
    """
    KAN-Mixer Architecture:
    1-Applies 'Patch Embedding' at first.
    2-Applies 'Mixer Layer' N times in a row.
    3-Performs 'Global Average Pooling'
    4-The Learnt features are then passed to the classifier
    """

    def __init__(
        self,
        in_channels,
        embedding_dim,
        num_classes,
        patch_size,
        image_size,
        depth,
        token_intermediate_dim,
        channel_intermediate_dim,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        self.num_patch = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            PatchEmbedding(in_channels, embedding_dim, patch_size),
            Transformation2(),
        )

        self.mixers = nn.ModuleList(
            [
                MixerLayer(
                    embedding_dim,
                    self.num_patch,
                    token_intermediate_dim,
                    channel_intermediate_dim,
                    grid_size,
                    spline_order,
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(nn.Linear(embedding_dim, num_classes))

    def forward(self, x):
        x = self.patch_embedding(x)  # Patch Embedding layer
        for mixer in self.mixers:  # Applying Mixer Layer N times
            x = mixer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global Average Pooling

        return self.classifier(x)
