# ImpactLens

An open reimplementation of an aerial impact-intelligence pipeline using public data. It compares before/after Sentinel-2 imagery, detects change with a Siamese U-Net, rolls results up into per-plot acreage and severity, and produces RAG-grounded reports that are validated against the underlying numbers.

> Status: Phase 1 (scaffold). No features yet. The full README (tools, file structure, setup, results, key concepts) is written in Phase 7.

## Layout

```
src/impactlens/{data,models,geo,agents,aws}   package code
notebooks/                                    Kaggle/Colab training notebooks
tests/                                        pytest tests
infra/                                        AWS CDK (Python)
```

> Work in progress, building in phases. Current: data pipeline (Phase 2).
