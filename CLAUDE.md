# ImpactLens

An open reimplementation of an aerial impact-intelligence pipeline, built on public data.

## Pipeline

1. **Imagery**: before/after Sentinel-2 imagery (OSCD for training, Earth Search STAC for live AOIs).
2. **Change detection**: Siamese U-Net (FC-Siam-diff) produces change masks with per-pixel confidence.
3. **Geo-AI rollups**: PostGIS computes acreage, hotspots, and severity per plot.
4. **Agents**: 6 LangGraph agents (Ingest, Detect, GeoAnalyst, Retriever, Reporter, Validator) write RAG-grounded reports.
5. **Validation loop**: every number in a report is checked against the geo metrics summary; mismatches go back to the Reporter.
6. **Deployment**: AWS (S3, SQS, Lambda, Step Functions), defined with Terraform in `infra/`.

## Stack and rules

- Python 3.11. Dependencies managed with `uv` (`uv add`, `uv sync`, `uv run`). **This repo uses uv, not conda: it overrides the global conda-per-project rule.** Do not create a conda env here.
- `src/` layout: package lives in `src/impactlens/{data,models,geo,agents,aws}`.
- Tests with `pytest` in `tests/`. Lint and format with `ruff`.
- Mac with Apple Silicon: use the PyTorch `mps` device (or MLX), never CUDA, for local work. Heavy training goes to a free Kaggle/Colab GPU.
- Keep code simple and commented; the owner is learning and will explain it in interviews.

## AWS

- Infrastructure is **Terraform + Makefile** (no CDK), same pattern as `~/projects/lexintake-aws`: `make deploy` / `make destroy`.
- AWS profile `sky`, region **us-east-2 only**. us-east-1 is blocked by an SCP.
- Terraform code lives in `infra/`; build outputs go to `build/` (gitignored).

## Hard rules

- **Never commit data, secrets, or `.env`.** Raw data goes in `data/` (gitignored). Real keys live only in `.env`; `.env.example` has placeholders only.
- **No `Co-Authored-By` trailers** in commit messages.
- Small commits with descriptive messages. Ask before pushing.
- Nothing about the owner's previous employer anywhere in code, docs, or commits.

## Workflow

Work **phase by phase and stop for review after each phase**. Do not start the next phase until told to.

1. Scaffold (this repo layout, CLAUDE.md, README stub)
2. Data: OSCD loader, Sen1Floods11 loader, STAC fetcher, `impactlens fetch` CLI (done). Event: Valencia floods, Oct 2024.
3. Models: two of them. (a) FC-Siam-diff (Siamese U-Net) trained on OSCD for general change; (b) a plain U-Net trained on Sen1Floods11 for flood water, because OSCD is mostly urban change and would not detect flooded farmland. Training scripts and Kaggle notebook, ONNX export, sliding-window inference, `geo/indices.py` crop stress layer
4. Geo-AI: docker-compose PostGIS, polygonize, spatial SQL, summary JSON and GeoJSON
5. Agents + RAG: FAISS over SOP PDFs, LangGraph graph with 6 agents, Anthropic Haiku, token logging
6. AWS: Terraform stack, Step Functions flow, Secrets Manager. Write the deploy steps; the owner runs deploy.
7. Eval + README: IoU/F1, validator catch rate, latency, cost per run, mermaid diagram, results

## Commands

```bash
uv sync                 # install deps
uv run pytest           # tests
uv run ruff check .     # lint
uv run ruff format .    # format
```
