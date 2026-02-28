from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import DenseNet121
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False

# Optional transformers
try:
    from transformers import AutoTokenizer, AutoModel
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class ImageEncoder3D(nn.Module):
    def __init__(self, d: int, in_channels: int = 1, use_monai: bool = True):
        super().__init__()
        self.d = int(d)
        self.in_channels = int(in_channels)
        self.use_monai = bool(use_monai and _HAS_MONAI)

        if self.use_monai:
            self.backbone = DenseNet121(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.d,
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv3d(self.in_channels, 32, 3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 64, 3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 128, 3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(128, self.d),
            )

        self.ln = nn.LayerNorm(self.d)

    def forward(self, x_dce: torch.Tensor) -> torch.Tensor:
        
        if x_dce.dim() == 4:
            # [B, D, H, W] -> [B, 1, D, H, W]
            x_dce = x_dce.unsqueeze(1)
        elif x_dce.dim() == 5:
            pass
        else:
            raise ValueError(
                f"ImageEncoder3D expected 4D [B,D,H,W] or 5D [B,C,D,H,W], got {tuple(x_dce.shape)}"
            )

        if x_dce.size(1) != self.in_channels:
            raise ValueError(
                f"ImageEncoder3D in_channels={self.in_channels} but got C={x_dce.size(1)}. "
                f"If you changed your pipeline to provide channels, set in_channels accordingly."
            )

        h = self.backbone(x_dce)
        return self.ln(h)


class TabularVarEncoder(nn.Module):
    def __init__(self, d: int, kind: str, num_classes: int = 0):
        super().__init__()
        
        if kind == "binary":
            kind = "onehot"
            if num_classes == 0:
                num_classes = 2

        assert kind in ["scalar", "categorical", "onehot"], f"Unsupported kind={kind}"

        self.kind = kind
        self.d = int(d)
        self.num_classes = int(num_classes)

        if self.kind == "categorical":
            if self.num_classes <= 0:
                raise ValueError("categorical encoder requires num_classes > 0")
            self.emb = nn.Embedding(self.num_classes, self.d)
            self.proj = nn.LayerNorm(self.d)
        elif self.kind == "onehot":
            if self.num_classes <= 0:
                raise ValueError("onehot encoder requires num_classes > 0")
            self.lin = nn.Sequential(
                nn.Linear(self.num_classes, self.d),
                nn.GELU(),
                nn.LayerNorm(self.d),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(1, self.d),
                nn.GELU(),
                nn.Linear(self.d, self.d),
                nn.LayerNorm(self.d),
            )

    @staticmethod
    def _is_scalar_tensor(x: torch.Tensor) -> bool:
        return x.dim() == 0 or (x.dim() == 1 and x.numel() == 1)

    def _to_onehot(self, x: torch.Tensor) -> torch.Tensor:
        K = self.num_classes
        
        if x.dim() == 1 and x.numel() == K:
            return x.float().view(1, K)
        if x.dim() == 2 and x.size(-1) == K:
            return x.float()

        if self._is_scalar_tensor(x):
            idx = x.long().view(1)

            if idx.min().item() >= 0 and idx.max().item() <= (K - 1):
                pass
            elif idx.min().item() >= 1 and idx.max().item() <= K:
                idx = idx - 1

            idx = idx.clamp(0, K - 1)
            return F.one_hot(idx, num_classes=K).float() 

        if x.dim() == 1:
            idx = x.long()
            if idx.numel() > 0:
                if idx.min().item() >= 0 and idx.max().item() <= (K - 1):
                    pass
                elif idx.min().item() >= 1 and idx.max().item() <= K:
                    idx = idx - 1

            idx = idx.clamp(0, K - 1)
            return F.one_hot(idx, num_classes=K).float() 

        raise ValueError(f"Invalid onehot input shape={tuple(x.shape)} for num_classes={K}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "categorical":
            if self._is_scalar_tensor(x):
                out = self.proj(self.emb(x.long().view(1)))  
                return out.squeeze(0)
            out = self.proj(self.emb(x.long()))  
            return out

        if self.kind == "onehot":
            oh = self._to_onehot(x)  
            out = self.lin(oh) 
            if out.dim() == 2 and out.size(0) == 1 and self._is_scalar_tensor(x):
                return out.squeeze(0)
            return out

        # scalar
        if self._is_scalar_tensor(x):
            out = self.mlp(x.float().view(1, 1))  
            return out.squeeze(0)

        if x.dim() == 2 and x.size(-1) == 1:
            out = self.mlp(x.float()) 
            return out

        if x.dim() == 1:
            out = self.mlp(x.float().view(-1, 1)) 
            return out

        raise ValueError(f"Invalid scalar input shape={tuple(x.shape)}")