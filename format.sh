#!/bin/bash
# Code formatting script - formats all Python files with Black and isort

echo "Running isort..."
uv run isort backend/ --check-only --diff

echo -e "\nRunning Black..."
uv run black backend/ --check --diff

echo -e "\nTo apply formatting changes, run:"
echo "  uv run isort backend/"
echo "  uv run black backend/"
