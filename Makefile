.PHONY: install generate analyze reproduce test

install:
	uv sync --all-groups

# Apple Silicon only (MLX); hours; resumable. Outputs are committed, so you rarely need this.
generate:
	uv run judgegate generate

analyze:
	uv run judgegate analyze

# analyze + charts + site + README, from the committed model outputs (any OS)
reproduce:
	uv run judgegate reproduce

test:
	uv run ruff check . && uv run ruff format --check .
	uv run mypy src
	uv run pytest -q --cov=judgegate --cov-report=term-missing
