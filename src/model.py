from dataclasses import dataclass
from typing import Self
from config import VisionTransformerConfig
import torch
from torch import nn

class ConvPatchEmbedding(nn.Module):
    """Converts an image batch into a sequence of patch embeddings."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.image_shape = config.image_shape
        self.patch_shape = config.patch_shape
        self.projection = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=self.patch_shape,
            stride=self.patch_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, channels, height, width), got {tuple(x.shape)}.")

        _, _, height, width = x.shape
        if (height, width) != self.image_shape:
            raise ValueError(f"Expected image size {self.image_shape}, got {(height, width)}.")

        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.embed_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            bias=config.qkv_bias,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Mlp(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = self.norm1(x)
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )
        x = x + self.dropout(attention_output)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = ConvPatchEmbedding(config)

        self.class_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, config.num_patches + 1, config.embed_dim))
        self.position_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(TransformerEncoderBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @classmethod
    def from_config(cls, config: VisionTransformerConfig) -> Self:
        return cls(config)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)

        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.position_embedding
        x = self.position_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)
