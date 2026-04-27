# iOS-pipeline (рабочий шаблон)

Гибридный inference: визуальный и текстовый энкодеры через
`onnxruntime-objc` с CoreML execution provider (внутри использует ANE),
CQR-голова через нативный CoreML. Возвращает 90 % предсказательный
интервал по четырем нутриентам на одно блюдо.

## Зависимости (Swift Package Manager)

В Xcode → File → Add Package Dependencies:

- `https://github.com/microsoft/onnxruntime` (продукт `onnxruntime-objc`)
- `https://github.com/huggingface/swift-transformers` (продукт `Tokenizers`)

## Ресурсы в Bundle

Положить в Resources Xcode-проекта:

| файл                          | источник                              |
|-------------------------------|---------------------------------------|
| `visual_dinov2_small.onnx`    | output ноутбука 05                    |
| `text_minilm_l6_v2.onnx`      | output ноутбука 05                    |
| `cqr_head.mlpackage`          | output `scripts/convert_to_coreml.py` |
| `normalization.json`          | `target_norm.json` из ноутбука 02     |
| `conformal_quantiles.json`    | поле `cqr_q` из ноутбука 04           |
| `tokenizer/`                  | папка с tokenizer от MiniLM-L6-v2     |

`tokenizer/` берется с Hugging Face: `huggingface-cli download
sentence-transformers/all-MiniLM-L6-v2 tokenizer.json tokenizer_config.json
vocab.txt special_tokens_map.json` или скачать вручную и сложить рядом.

`normalization.json` должен иметь схему `{"mean": [...], "std": [...]}`
длиной 4. `conformal_quantiles.json` — `{"total_calories": Q, ...}`,
тоже 4 ключа.

## Использование

```swift
let pipeline = try await NutritionInferencePipeline()
let intervals = try pipeline.predict(
    image: uiImage,
    ingredients: "soy sauce, garlic, white rice, chicken"
)
for it in intervals {
    print("\(it.target): [\(it.lower), \(it.upper)]")
}
```

## Что в шаблоне неполноценно

- **Image preprocessing через UIGraphicsBeginImageContext** — простой и
  достаточный для прототипа. Для production стоит заменить на
  `vImage_Buffer` + `Accelerate.framework` — будет быстрее в десятки раз.
- **Output-нейминг CQR-головы** — coremltools переименовал выход в
  `var_<id>` (видно по предупреждению при конвертации). В шаблоне
  делается fallback-поиск по имени; для production стоит явно зафиксировать
  имя в `convert_to_coreml.py` через `outputs=[ct.TensorType(name="quantiles")]`.
- **Tokenizer выбор**: для строк ингредиентов длина обычно сильно меньше
  64. Можно динамически уменьшать `textMaxLen` ради экономии.
- **Без обработки batch**: одна картинка на вызов. Для batched-инференса
  переписать shape входов на `(batch, ...)`.
- **Нет кэширования модели между вызовами** — pipeline создается раз и
  переиспользуется (за это отвечает вызывающий код).

## Альтернативы

Если хочется без onnxruntime-mobile (зависимость +5-10 МБ к binary):
- DINOv2 в transformers использует bicubic interpolation позиционных
  эмбеддингов на нестандартном размере, и это операция, которую
  coremltools не поддерживает. Решается заменой входа на 518×518
  (нативный DINOv2 размер) — тогда интерполяция не нужна. Latency
  вырастет в ~5 раз (для ANE станет ~100-150 мс), но pipeline будет
  чисто-CoreML без сторонних runtime.
