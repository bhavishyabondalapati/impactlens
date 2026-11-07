# ImpactLens

An open reimplementation of an aerial impact-intelligence pipeline using public data. It compares before/after Sentinel-2 imagery, detects change with a Siamese U-Net, rolls results up into per-plot acreage and severity, and produces RAG-grounded reports that are validated against the underlying numbers.

> **Status (work in progress, Phases 1-4 done):** flood U-Net test F1 0.92 on Sen1Floods11 (vs 0.87 for the best tuned water-index baseline). The Valencia 2024 DANA analysis ranks Paiporta, Sedaví and Alfafar highest for vegetation loss (37-42% of their vegetated land, vs 2-3% in comparable farm towns), with no location hints given to the model. See [docs/GEO.md](docs/GEO.md). Phase 5 (agent reports) in progress. The full README (tools, file structure, setup, key concepts) is written in Phase 7.

## Layout

```
src/impactlens/{data,models,geo,agents,aws}   package code
notebooks/                                    Kaggle/Colab training notebooks
tests/                                        pytest tests
infra/                                        AWS Terraform + Makefile
```
