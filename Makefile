# Makefile for the EPFL project
# This file contains commands for setting up the environment, formatting code,
# building the book, and other maintenance tasks.

.DEFAULT_GOAL := help

# Create a virtual environment using uv with Python 3.12
venv:
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv --python='3.12'


# Mark install target as phony (not producing a file named 'install')
.PHONY: install
install: venv ## Install a virtual environment
	@uv pip install --upgrade pip
	@uv pip install -r requirements.txt


# Format and lint the code using pre-commit
.PHONY: fmt
fmt: venv ## Run autoformatting and linting
	#@uv pip install pre-commit
	@uvx pre-commit install
	@uvx pre-commit run --all-files


# Clean up generated files and remove stale branches
.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f
	@git branch -v | grep "\[gone\]" | cut -f 3 -d ' ' | xargs git branch -D


# Display help information about available make targets
.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort


# Install and run Marimo for interactive notebooks
.PHONY: marimo
marimo: install ## Run Marimo notebooks
	#@uv pip install marimo
	@uvx marimo edit book/marimo

# Build the Jupyter Book documentation
#.PHONY: book
#book: install  ## Compile the book
#	@uv pip install jupyterlab jupyter-book
#	@uv run jupyter-book clean book
#	@uv run jupyter-book build book
