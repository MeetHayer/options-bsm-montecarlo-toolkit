#!/bin/bash
# Script to run tests after installing dependencies

echo "Installing dependencies..."
pip install -e . -q

echo ""
echo "Running tests..."
pytest tests/ -v

echo ""
echo "Done!"

