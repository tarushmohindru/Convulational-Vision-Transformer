from __future__ import annotations
from dataclasses import dataclass, field

def _as_2tuple(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return value, value

    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")

    height, width = value
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value}.")
    return height, width

@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int | tuple[int, int] = 224
    patch_size: int | tuple[int, int] = 16
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        image_height, image_width = _as_2tuple(self.image_size, "image_size")
        patch_height, patch_width = _as_2tuple(self.patch_size, "patch_size")

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, got "
                f"image_size={self.image_size} and patch_size={self.patch_size}."
            )
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1).")

    @property
    def image_shape(self) -> tuple[int, int]:
        return _as_2tuple(self.image_size, "image_size")

    @property
    def patch_shape(self) -> tuple[int, int]:
        return _as_2tuple(self.patch_size, "patch_size")

    @property
    def grid_size(self) -> tuple[int, int]:
        image_height, image_width = self.image_shape
        patch_height, patch_width = self.patch_shape
        return image_height // patch_height, image_width // patch_width

    @property
    def num_patches(self) -> int:
        grid_height, grid_width = self.grid_size
        return grid_height * grid_width


@dataclass
class SteelConfig:
    # Paths
    data_dir: str = "data"
    train_csv: str = "data/train.csv"
    image_dir: str = "data/train_images"
    test_image_dir: str = "data/test_images"
    checkpoint_dir: str = "checkpoints"
    submission_path: str = "submission.csv"

    # Image
    image_height: int = 256
    image_width: int = 512

    # Training
    batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.1
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    amp_dtype: str = "float16"   # "bfloat16" or "float16"
    compile_model: bool = False

    # ViT
    embed_dim: int = 512
    depth: int = 8
    num_heads: int = 8
    patch_size: int = 16
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Inference
    threshold: float = 0.5
    min_mask_area: int = 200

    @classmethod
    def h200(cls) -> "SteelConfig":
        """H200-optimised preset: ViT-L, 256×1024 input (1024 patches), bf16, compile."""
        return cls(
            # 256×1024 → 16×64 = 1024 patches (4× more signal than default 512-wide)
            image_width=1024,
            # ViT-L scale (~307 M params)
            embed_dim=1024,
            depth=24,
            num_heads=16,
            dropout=0.1,
            attention_dropout=0.0,
            # Training recipe
            batch_size=16,
            num_epochs=50,
            learning_rate=2e-4,        # linear-scaled vs reference (batch 8 → 16, 1e-4 → 2e-4)
            adam_betas=(0.9, 0.95),    # recommended for large ViT
            warmup_epochs=5,
            weight_decay=0.05,         # stronger regularisation for large model
            # Hardware
            amp_dtype="bfloat16",
            compile_model=True,
            num_workers=8,
        )

    def vit_config(self) -> VisionTransformerConfig:
        return VisionTransformerConfig(
            image_size=(self.image_height, self.image_width),
            patch_size=self.patch_size,
            in_channels=3,
            num_classes=4,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )
