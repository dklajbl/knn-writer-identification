import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GeM(nn.Module):
    """
    Generalized Mean Pooling.

    f = (1/N sum x^p)^(1/p), where p is trainable parameter.

    If p = 1 -> mean pooling.
    If p = inf -> max pooling.
    """

    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super().__init__()

        self.eps = eps

        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor(p))

    def forward(self, x):
        # (B, C, H, W)

        # (1)
        p = self.p.clamp(min=0.1, max=10.0)

        x = x.clamp(min=self.eps)

        x = x.pow(p)

        # (B, C, H, W) -> (B, C)
        x = x.mean(dim=(-2, -1))

        x = x.pow(1.0 / p)

        return x

    def __repr__(self):
        return f"GeM(p={self.p.item():.4f})"


class WriterEncoderResNET18(nn.Module):
    """
    Writer encoder module implemented using ResNet18 backbone.
    """

    def __init__(self, embedding_dim=256, use_gem=True):
        super().__init__()

        # ResNet18 backbone, do not use pretrained
        backbone = models.resnet18(weights=None)

        # change first convolution of backbone, because input has 1 channel
        # make it also less agressive to capture more detail
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # disable first maxpool to keep details for longer
        backbone.maxpool = nn.Identity()

        # remove final pooling and last FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        if use_gem:
            self.pool = GeM(p=3.0, trainable=True)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # embedding linear transform
        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x, padding_mask: torch.Tensor | None = None, return_patch_weights: bool = False,):
        """
        ""
        Parameters:
            x (torch.Tensor, shape=(B, P=1, C=1, H, W)): Input images.
            padding_mask: Unused. Exists only for compatibility.
            return_patch_weights: Unused. Exists only for compatibility.
        Returns:
            res (torch.Tensor, shape=(B, embedding_dim)): One embedding for each image in batch.
        """

        # check dimensions
        if not (x.dim() == 5 and x.size(1) == 1 and x.size(2) == 1):
            raise ValueError(f"Invalid tensors dimensions (expected (B, 1, 1, H, W)), got {x.shape}")

        # remove patch dimension, which is always 1
        # (B, 1, 1, H, W) -> (B, 1, H, W)
        x = x.squeeze(1)

        # feature extraction using backbone
        # (B, 512, W', H')
        feat_map = self.backbone(x)

        # (B, 512, 1, 1)
        pooled_features = self.pool(feat_map)

        # (B, 512)
        feat = pooled_features.flatten(1)

        # linear layer to get wanted dimensions
        emb = self.embedding(feat)

        # normalize for ArcFace
        norm_emb = F.normalize(emb, p=2, dim=1)

        return norm_emb
