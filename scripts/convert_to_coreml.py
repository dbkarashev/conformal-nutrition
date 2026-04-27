"""Конвертация моделей в CoreML для деплоя на iOS.

ONNX-путь в coremltools убран начиная с версии 6 — конвертируем напрямую
из PyTorch через torch.jit.trace + ct.convert. CQR-голову собираем
заново той же архитектурой, что в ноутбуке 04, и подгружаем веса из
чекпоинта. Энкодеры тянем с Hugging Face.

Требования:
  Python 3.11 или 3.12 (НЕ 3.13/3.14 — нативные libs coremltools не подгружаются).
  pip install torch transformers sentence-transformers coremltools

Использование:
  python scripts/convert_to_coreml.py \\
      --cqr_head build/cqr_head.pt \\
      --output_dir build/coreml/

  # частичная сборка
  python scripts/convert_to_coreml.py --cqr_head build/cqr_head.pt \\
      --output_dir build/coreml/ --components head
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct

VIS_DIM = 384
TEXT_DIM = 384
HIDDEN = 256
DROPOUT = 0.2
N_TARGETS = 4
N_QUANTILES = 3
TEXT_MAX_LEN = 64

VIS_ENCODER = "facebook/dinov2-small"
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

IMAGE_SIZE = 224
PATCH_SIZE = 14  # DINOv2-small
TARGET_GRID = IMAGE_SIZE // PATCH_SIZE  # 16


def mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class GatedQuantile(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_proj = nn.Linear(VIS_DIM, HIDDEN)
        self.t_proj = nn.Linear(TEXT_DIM, HIDDEN)
        self.gate = nn.Sequential(nn.Linear(2 * HIDDEN, HIDDEN), nn.Sigmoid())
        self.head = mlp(HIDDEN, HIDDEN, N_TARGETS * N_QUANTILES, DROPOUT)

    def forward(self, v, t):
        v_h = self.v_proj(v)
        t_h = self.t_proj(t)
        g = self.gate(torch.cat([v_h, t_h], dim=-1))
        fused = g * v_h + (1.0 - g) * t_h
        out = self.head(fused).view(-1, N_TARGETS, N_QUANTILES)
        return torch.sort(out, dim=-1).values


class VisualWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(pixel_values=x).last_hidden_state[:, 0, :]


class TextWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def convert_cqr_head(ckpt_path: Path, out_path: Path) -> None:
    model = GatedQuantile()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    example = (torch.randn(1, VIS_DIM), torch.randn(1, TEXT_DIM))
    traced = torch.jit.trace(model, example)
    m = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="v", shape=(1, VIS_DIM)),
            ct.TensorType(name="t", shape=(1, TEXT_DIM)),
        ],
        outputs=[ct.TensorType(name="quantiles")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    m.save(str(out_path))


def freeze_pos_embeddings(encoder, target_grid: int) -> None:
    """Заранее интерполирует position_embeddings под фиксированный input
    и зашивает как константу, плюс перекрывает `interpolate_pos_encoding`,
    чтобы он возвращал готовые pos_embeddings без интерполяции.

    Без перекрытия в transformers стоит `if not torch.jit.is_tracing()` —
    при трейсе всегда идет через bicubic-ветку «на случай dynamic shapes»,
    которую CoreML не поддерживает. У нас input размер фиксирован (224×224
    под `target_grid=16`), поэтому interpolate в forward не нужен."""
    import torch.nn.functional as F
    pos = encoder.embeddings.position_embeddings.data
    cls_pos = pos[:, :1, :]
    patch_pos = pos[:, 1:, :]
    src_grid = int(round(patch_pos.shape[1] ** 0.5))
    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    new_patch = F.interpolate(
        patch_pos, size=(target_grid, target_grid),
        mode="bicubic", align_corners=False,
    )
    new_patch = new_patch.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
    new_pos = torch.cat([cls_pos, new_patch], dim=1)
    encoder.embeddings.position_embeddings = nn.Parameter(new_pos, requires_grad=False)

    def _identity_pos_encoding(self, embeddings, height, width):
        return self.position_embeddings

    encoder.embeddings.interpolate_pos_encoding = _identity_pos_encoding.__get__(
        encoder.embeddings, type(encoder.embeddings)
    )


def convert_visual(out_path: Path) -> None:
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(VIS_ENCODER).eval()
    freeze_pos_embeddings(encoder, target_grid=TARGET_GRID)
    wrapped = VisualWrapper(encoder).eval()
    example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(wrapped, example, strict=False)
    m = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, IMAGE_SIZE, IMAGE_SIZE))],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    m.save(str(out_path))


def convert_text(out_path: Path) -> None:
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(TEXT_ENCODER).eval()
    wrapped = TextWrapper(encoder).eval()
    example_ids = torch.zeros(1, TEXT_MAX_LEN, dtype=torch.long)
    example_mask = torch.ones(1, TEXT_MAX_LEN, dtype=torch.long)
    traced = torch.jit.trace(wrapped, (example_ids, example_mask))
    m = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, TEXT_MAX_LEN), dtype=int),
            ct.TensorType(name="attention_mask", shape=(1, TEXT_MAX_LEN), dtype=int),
        ],
        outputs=[ct.TensorType(name="embedding")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    m.save(str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cqr_head", type=Path, required=True,
                    help="Чекпоинт CQR-головы (cqr_head.pt) из ноутбука 04.")
    ap.add_argument("--output_dir", type=Path, required=True,
                    help="Куда складывать .mlpackage.")
    ap.add_argument("--components", choices=["all", "head", "visual", "text"], default="all")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    todo = ["head", "visual", "text"] if args.components == "all" else [args.components]

    if "head" in todo:
        out = args.output_dir / "cqr_head.mlpackage"
        print(f"[head]   -> {out}")
        convert_cqr_head(args.cqr_head, out)

    if "visual" in todo:
        out = args.output_dir / "visual_dinov2_small.mlpackage"
        print(f"[visual] -> {out}  (тянет веса с Hugging Face)")
        convert_visual(out)

    if "text" in todo:
        out = args.output_dir / "text_minilm_l6_v2.mlpackage"
        print(f"[text]   -> {out}  (тянет веса с Hugging Face)")
        convert_text(out)

    print("Готово.")


if __name__ == "__main__":
    main()
