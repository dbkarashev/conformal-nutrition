"""Локальный экспорт DINOv2-small и MiniLM в ONNX.

Дублирует то, что делается в ноутбуке 05 на Kaggle, но на Mac, чтобы
ONNX-файлы лежали в локальном build/onnx/ и попадали в Xcode-проект
без скачивания output из Kaggle.

Использование:
  python scripts/export_onnx.py --output_dir build/onnx/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

VIS_ENCODER = "facebook/dinov2-small"
TEXT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_MAX_LEN = 64


class VisualWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, pixel_values):
        return self.encoder(pixel_values=pixel_values).last_hidden_state[:, 0, :]


class TextWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def export_visual(out_path: Path) -> None:
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(VIS_ENCODER).eval()
    wrapped = VisualWrapper(encoder).eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        wrapped, (dummy,), str(out_path),
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14, do_constant_folding=True, dynamo=False,
    )


def export_text(out_path: Path) -> None:
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(TEXT_ENCODER).eval()
    wrapped = TextWrapper(encoder).eval()
    dummy_ids = torch.zeros(1, TEXT_MAX_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, TEXT_MAX_LEN, dtype=torch.long)
    torch.onnx.export(
        wrapped, (dummy_ids, dummy_mask), str(out_path),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=14, do_constant_folding=True, dynamo=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    visual_out = args.output_dir / "visual_dinov2_small.onnx"
    text_out = args.output_dir / "text_minilm_l6_v2.onnx"

    print(f"[visual] -> {visual_out}")
    export_visual(visual_out)
    print(f"  размер: {visual_out.stat().st_size / 1024**2:.2f} МБ")

    print(f"[text]   -> {text_out}")
    export_text(text_out)
    print(f"  размер: {text_out.stat().st_size / 1024**2:.2f} МБ")


if __name__ == "__main__":
    main()
