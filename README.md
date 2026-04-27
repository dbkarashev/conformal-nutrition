# conformal-nutrition: мультимодальная регрессия нутриентов с конформной калибровкой

## Цель проекта

Магистерская работа по предсказанию калорийности и макронутриентов
(жиры, углеводы, белки) блюда по его обзорной фотографии и текстовому
списку ингредиентов. Базовая модель — мультимодальный энкодер с gated
fusion поверх замороженных DINOv2-small и MiniLM-L6-v2. Поверх точечных
предсказаний строятся конформные интервалы (split CP и CQR), дающие
маржинальное покрытие около 90 % при выполнении exchangeability.
Источник данных — публичный датасет Nutrition5k от Google Research,
лицензия CC BY 4.0. Конечная цель — прикладная: компактный pipeline
с интервальными оценками, который запускается на iPhone в нативном
CoreML за десятки миллисекунд.

## Структура репозитория

```
conformal-nutrition/
  README.md                          этот файл
  pyproject.toml                     метаданные пакета и зависимости
  .gitignore                         правила игнорирования
  notebooks/                         ноутбуки для запуска на Kaggle
    01_data_preparation.ipynb        парсинг, скачивание, сплиты, EDA
    02_visual_baseline.ipynb         только-визуальная регрессия
    03_multimodal_fusion.ipynb       text_only / concat / gated fusion
    04_conformal_calibration.ipynb   split CP и CQR на gated-предсказаниях
    05_iphone_microbench.ipynb       замеры размера, latency и памяти
  scripts/                           локальные скрипты для iOS-деплоя
    convert_to_coreml.py             PyTorch -> CoreML (.mlpackage, FP16)
  src/                               импортируемый код (наполняется по мере)
    data/        models/        training/        calibration/        utils/
  docs/                              главы магистерской работы
  experiments/                       артефакты прогонов (игнорируется)
  tests/                             тесты (наполняются по мере)
```

## Как работать на Kaggle

Каждый ноутбук синхронизируется через GitHub-интеграцию Kaggle: один
раз импортируем через File → Import Notebook → Link и указываем
приватный репозиторий conformal-nutrition. Дальше при каждом открытии
Kaggle подтягивает свежую версию из GitHub. После запуска делаем
Save Version → Save & Run All (Commit), чтобы зафиксировать output как
Kaggle Dataset для следующего ноутбука.

Ноутбуки и их аппаратные требования:

| ноутбук | Internet | Accelerator |
|---------|:--------:|:-----------:|
| 01      |    on    |    None     |
| 02      |    on    |   GPU T4    |
| 03      |    on    |   GPU T4    |
| 04      |    off   |   GPU T4    |
| 05      |    on    |   GPU T4    |

`Internet` нужен для скачивания: данных Nutrition5k (01), весов с
Hugging Face (02, 03, 05). Ноутбук 04 работает только с уже
сохраненными артефактами и интернет ему не нужен.

## Артефакты данных

Каждый ноутбук публикует свой output как приватный Kaggle Dataset, а
следующий подключает его через `+ Add Data`:

| ноутбук | output dataset                       |
|---------|--------------------------------------|
| 01      | `nutrition5k-overhead-rgb-224`       |
| 02      | `nutrition5k-visual-baseline`        |
| 03      | `nutrition5k-multimodal-fusion`      |
| 04      | `nutrition5k-conformal-intervals`    |
| 05      | `nutrition5k-iphone-microbench`      |

## Локальная разработка

```bash
git clone git@github.com:dbkarashev/conformal-nutrition.git
cd conformal-nutrition
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Цикл работы: правим ноутбук или код в `src/`, делаем `git push`, на
Kaggle открываем ноутбук — он подтягивает свежую версию. Ноутбуки на
Kaggle не редактируем напрямую: единственный источник правды — git.

## iOS-деплой

iOS-приложение лежит в отдельном репо `dbkarashev/foon`. Связь
односторонняя: ML-репо генерирует артефакты, foon их потребляет.

Конвертация всех трех моделей в нативный CoreML делается локально
(coremltools требует macOS):

```bash
pip install -e ".[deploy]"   # coremltools + numpy<2
python scripts/convert_to_coreml.py \
    --cqr_head build/cqr_head.pt \
    --output_dir ../foon/Foon/Foon/Resources/
```

`build/cqr_head.pt` берется из Kaggle Dataset
`nutrition5k-conformal-intervals/models/cqr_head.pt`. Подробности про
обход bicubic в DINOv2 и tokenizer — в `scripts/README.md`.

## Стиль кода

- Минимализм: каждый файл — одна ответственность.
- Идемпотентность ноутбуков: каждый шаг проверяет наличие итогового
  артефакта на диске и пропускается, если все уже сделано.
- Чистая структура ноутбука: markdown с обоснованием → код → короткий
  вывод.
- Комментарии только когда смысл неочевиден из кода.
- Никаких эмодзи и эффектов в коде, кроме одного маркера предупреждения
  в markdown ноутбука 01 о промежуточном Save Version.
