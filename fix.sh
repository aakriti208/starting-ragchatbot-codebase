#!/bin/bash
# Auto-fix script - automatically fixes formatting issues

echo "Running isort..."
uv run isort backend/

echo -e "\nRunning Black..."
uv run black backend/

echo -e "\n✓ Code formatting applied!"
echo "Run ./lint.sh to verify all quality checks pass."
