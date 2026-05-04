from __future__ import annotations

import torch
from torch import nn

from config import VisionTransformerConfig
from model import ConvPatchEmbedding, TransformerEncoderBlock


class ViTSegmenter(nn.Module):
    """ViT encoder with a transposed-convolution decoder for dense segmentation."""

    def __init__(self, vit_config: VisionTransformerConfig, num_seg_classes: int = 4) -> None:
        super().__init__()
        self.grid_h, self.grid_w = vit_config.grid_size
        D = vit_config.embed_dim

        self.patch_embedding = ConvPatchEmbedding(vit_config)
        self.class_token = nn.Parameter(torch.zeros(1, 1, D))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, vit_config.num_patches + 1, D)
        )
        self.pos_dropout = nn.Dropout(vit_config.dropout)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(vit_config) for _ in range(vit_config.depth)
        )
        self.norm = nn.LayerNorm(D, eps=vit_config.layer_norm_eps)

        # 4 × 2× upsample = 16× total (matches patch_size=16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(D, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_seg_classes, kernel_size=2, stride=2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embedding(x)                              # (B, N, D)
        cls = self.class_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)                          # (B, N+1, D)
        x = self.pos_dropout(x + self.position_embedding)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Remove CLS token and reshape patch tokens to 2-D grid
        patch_tokens = x[:, 1:]                                  # (B, N, D)
        D = patch_tokens.shape[-1]
        feature_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, D, self.grid_h, self.grid_w)
        )
        return self.decoder(feature_map)                         # (B, C, H, W)
