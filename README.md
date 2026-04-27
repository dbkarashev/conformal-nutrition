# conformal-nutrition

Multimodal nutrition regression with conformal calibration of prediction intervals.

Master's thesis project. Estimates calories and macronutrients (fat, carbohydrates, protein) from a dish photograph and an ingredient list, with calibrated prediction intervals via conformalized quantile regression (CQR).

[Русская версия](README.ru.md)

## Approach

- Visual encoder: DINOv2-small (frozen, ImageNet-normalized 224×224 input).
- Text encoder: sentence-transformers MiniLM-L6 (frozen, mean-pooled).
- Fusion: gated combination of visual and text embeddings.
- Regression head: MLP with quantile outputs (0.05, 0.50, 0.95).
- Conformal calibration: CQR (Romano, Patterson, Candès, NeurIPS 2019) and split conformal prediction as baseline.

## Dataset

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (Thames et al., CVPR 2021). Overhead RGB subset, ~3 200 dishes with measured calories, mass, and macronutrient values.

## Repository structure

```
notebooks/   pipeline notebooks 01–05 (data prep, visual baseline, fusion, calibration, microbench)
src/         shared utilities (in progress; reference implementation lives in notebooks)
scripts/     CoreML conversion script for on-device deployment
docs/        thesis text and figures
```

## Reproducing experiments

All experiments run on Kaggle. Each notebook publishes its outputs as a Kaggle Dataset, which the next notebook consumes as input.

| Notebook | Output dataset                              |
|----------|---------------------------------------------|
| 01       | `dbkarashev/nutrition5k-overhead-rgb-224`   |
| 02       | `dbkarashev/nutrition5k-visual-baseline`    |
| 03       | `dbkarashev/nutrition5k-multimodal-fusion`  |
| 04       | `dbkarashev/nutrition5k-conformal-intervals`|
| 05       | `dbkarashev/nutrition5k-iphone-microbench`  |

## License

This project is licensed under the [MIT License](LICENSE).
