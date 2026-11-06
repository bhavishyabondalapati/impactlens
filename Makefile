# Common commands. `make kaggle-run` asks before every push / upload / GPU run.

.PHONY: test lint kaggle-run

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

kaggle-run:  ## push code, upload data if changed, run the notebook on a Kaggle GPU, fetch results
	scripts/kaggle_run.sh
