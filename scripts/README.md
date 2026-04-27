# scripts/

Локальные скрипты для подготовки артефактов под iOS-деплой. Не
используются в Kaggle-ноутбуках. Все скрипты ожидают что-то в `build/`
(директория в `.gitignore`) и пишут результаты в `Resources/` соседнего
iOS-репо `dbkarashev/foon`.

## Окружение

Нужен **Python 3.11 или 3.12**. На 3.13/3.14 нативные библиотеки
coremltools не собираются. Tracing моделей в transformers ломается на
свежем `transformers>=5` — фиксируем 4.x. coremltools 9 не дружит с
NumPy 2 — берем 1.x.

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install "torch" "transformers<5" "sentence-transformers" \
            "coremltools" "numpy<2" "onnx"
```

## convert_to_coreml.py

Конвертация всех трех компонентов pipeline в нативный CoreML
(`.mlpackage`, FP16) для запуска на ANE. Путь PyTorch → torch.jit.trace
→ ct.convert. Делает три модели: визуальный энкодер DINOv2-small,
текстовый MiniLM-L6-v2 и CQR-голову.

### Особенности конвертации DINOv2

DINOv2 в transformers вызывает `interpolate_pos_encoding` с bicubic-
интерполяцией позиционных эмбеддингов на нестандартном входе, а
coremltools `upsample_bicubic2d` не поддерживает. Скрипт это решает
заранее:

1. Считает интерполяцию pos_embeddings под фиксированный input 224×224
   (16×16 patches + CLS) и заменяет их в весах модели.
2. Перекрывает метод `interpolate_pos_encoding` на возврат готовых
   pos_embeddings без дополнительных операций. В transformers стоит
   `if not torch.jit.is_tracing()` — по умолчанию при трейсе всегда
   идет через bicubic «ради dynamic shapes». У нас input размер
   фиксирован, поэтому интерполяция в forward не нужна.

После этого граф чистый, конвертация проходит и модель работает на ANE.

### Подготовка чекпоинта

Скачать `cqr_head.pt` из Kaggle Dataset `nutrition5k-conformal-intervals`:

1. Data → `models/cqr_head.pt` → Download.
2. Положить в `build/cqr_head.pt`.

### Запуск

```bash
python scripts/convert_to_coreml.py \
    --cqr_head build/cqr_head.pt \
    --output_dir ../foon/Foon/Foon/Resources/
```

На выходе три файла в Resources iOS-приложения:

- `cqr_head.mlpackage` (~1 МБ)
- `visual_dinov2_small.mlpackage` (~41 МБ FP16)
- `text_minilm_l6_v2.mlpackage` (~43 МБ FP16)

Дополнительно в `Resources/` нужны (если их там еще нет):

- `normalization.json` — `mean`, `std` целей из `target_norm.json`
  ноутбука 02.
- `conformal_quantiles.json` — поле `cqr_q` из `conformal_quantiles.json`
  ноутбука 04.
- `tokenizer/` — папка с `tokenizer.json`, `vocab.txt`,
  `tokenizer_config.json` и т. п. от `sentence-transformers/all-MiniLM-L6-v2`.

Выкачать tokenizer:

```bash
mkdir -p ../foon/Foon/Foon/Resources/tokenizer
cd ../foon/Foon/Foon/Resources/tokenizer
for f in tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json config.json; do
  curl -sLo "$f" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/$f"
done
```

Эти файлы статичные, скачать достаточно один раз.
