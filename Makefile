# ============================
# Project Makefile
# ============================

# Use uv + python
PYTHON = uv run python

# Default model 
model ?= ucb

# Default rounds
rounds ?= 1000

# ============================
# Installation
# ============================
install:
	uv sync

#install dev tools 
install-dev:
	uv add pytest ruff

# ============================
# Run Experiment
# ============================
run:
	$(PYTHON) -m bandit_ad_opt.main --model $(model) --rounds $(rounds)


# ============================
# Testing
# ============================
test:
	uv run pytest -v

test-one:
	uv run pytest $(file)


# ============================
# Linting (Ruff)
# ============================
lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

fix:
	uv run ruff check --fix src tests

# ============================
# Cleaning
# ============================
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf data/processed/*.npy
	rm -rf .ruff_cache
	rm -rf build dist

# ============================
# Quality Check (pre-push)
# ============================
check: lint test

