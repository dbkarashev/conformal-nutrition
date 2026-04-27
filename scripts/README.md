# scripts/

Локальные скрипты для подготовки артефактов под iOS-деплой. Не
используются в Kaggle-ноутбуках. Все скрипты ожидают что-то в `build/`
и кладут результаты тоже в `build/` (директория в `.gitignore`).

## Окружение

Нужен **Python 3.11 или 3.12**. На 3.13/3.14 нативные библиотеки
coremltools не собираются. ONNX-экспорт ломается на свежем
`transformers>=5` — фиксируем 4.x.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install "torch" "transformers<5" "sentence-transformers" \
            "coremltools" "numpy<2" "onnx"
```

## export_onnx.py — DINOv2 и MiniLM в ONNX

Дублирует то, что в Kaggle делает ноутбук 05, но локально, чтобы файлы
сразу попадали в Resources iOS-приложения. Источник правды для UI —
отдельный репо `dbkarashev/foon`, склонированный рядом:

```bash
python scripts/export_onnx.py --output_dir ../foon/Foon/Foon/Resources/
```

На выходе:
- `visual_dinov2_small.onnx` (~84 МБ)
- `text_minilm_l6_v2.onnx` (~86 МБ)

Альтернатива — просто скачать готовые из output ноутбука 05 на Kaggle.

## convert_to_coreml.py — CQR-голова в CoreML

ONNX → CoreML путь в coremltools 6+ убран. Делаем напрямую из PyTorch
через `torch.jit.trace` + `ct.convert`. Конвертируется только голова
(простая MLP). Энкодеры через CoreML не идут — DINOv2 в transformers
использует bicubic-интерполяцию позиционных эмбеддингов на
нестандартном входе 224×224, а coremltools `upsample_bicubic2d` не
поддерживает. Для энкодеров используем ONNX через onnxruntime с
CoreML execution provider — он сам отдаёт совместимые подграфы на ANE.

### Подготовка чекпоинта

Скачать `cqr_head.pt` из Kaggle Dataset `nutrition5k-conformal-intervals`:

1. Data → `models/cqr_head.pt` → Download.
2. Положить в `build/cqr_head.pt`.

### Запуск

```bash
python scripts/convert_to_coreml.py \
    --cqr_head build/cqr_head.pt \
    --output_dir ../foon/Foon/Foon/Resources/ \
    --components head
```

На выходе: `../foon/Foon/Foon/Resources/cqr_head.mlpackage` (~1 МБ).

`--components all` соберёт ещё и энкодеры — но они упадут на
`upsample_bicubic2d` для DINOv2. Используй только если хочется
поэкспериментировать с входом 518×518 (нативный размер DINOv2 без
интерполяции).
