#!/bin/bash
# Code quality checks script - runs all linting and type checking

echo "Running flake8..."
uv run flake8 backend/ --max-line-length=88 --extend-ignore=E203,W503 --exclude=chromadb_data,.venv,__pycache__

echo -e "\nRunning mypy..."
uv run mypy backend/

echo -e "\nRunning isort check..."
uv run isort backend/ --check-only

echo -e "\nRunning Black check..."
uv run black backend/ --check

echo -e "\n✓ All quality checks complete!"
