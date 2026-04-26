# scripts/

Локальные скрипты для деплоя моделей. Не используются в Kaggle-ноутбуках.

## convert_to_coreml.py

Конвертация PyTorch-моделей в CoreML `.mlpackage` для запуска на iPhone.
Прямой путь PyTorch -> torch.jit.trace -> ct.convert. ONNX-путь в
coremltools 6+ убран и больше не работает.

### Окружение

Нужен **Python 3.11 или 3.12**. На 3.13/3.14 нативные библиотеки
coremltools (libcoremlpython, libmilstoragepython) не собираются и не
подгружаются.

```bash
python3.12 -m venv .venv-coreml
source .venv-coreml/bin/activate
pip install torch transformers sentence-transformers coremltools
```

### Подготовка чекпоинта

Скачать `cqr_head.pt` из Kaggle Dataset `nutrition5k-conformal-intervals`:

1. На странице датасета: Data -> models/cqr_head.pt -> Download.
2. Положить в `build/cqr_head.pt` (или любой другой путь).

### Запуск

```bash
python scripts/convert_to_coreml.py \
    --cqr_head build/cqr_head.pt \
    --output_dir build/coreml/
```

На выходе:
- `build/coreml/cqr_head.mlpackage` (~1 МБ)
- `build/coreml/visual_dinov2_small.mlpackage` (~42 МБ FP16)
- `build/coreml/text_minilm_l6_v2.mlpackage` (~43 МБ FP16)

Для частичной сборки: `--components head` или `visual` или `text`.

### Альтернатива без CoreML

Если CoreML по какой-то причине не нужен, можно использовать готовые
ONNX-файлы из output ноутбука 05 через `onnxruntime-mobile` для iOS.
Подключить ORT в Xcode-проекте и активировать CoreML execution provider:
он даст ANE при доступности без перекомпиляции моделей.

```swift
let options = OrtSessionOptions()
try options.appendExecutionProvider("CoreML", providerOptions: [:])
let session = try OrtSession(env: env, modelPath: "visual_dinov2_small.onnx",
                              sessionOptions: options)
```

Этот путь не дает такого же мелкого контроля (нет explicit FP16-квантизации
без extra шага), но работает с уже собранными артефактами.
