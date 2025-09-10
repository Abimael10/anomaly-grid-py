#!/bin/bash

# Virtual environment setup script for anomaly-grid-py

set -e

echo "Setting up virtual environment for anomaly-grid-py..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install build dependencies
pip install maturin[patchelf] pytest

echo "Virtual environment setup complete!"
echo "This package is standalone and will automatically download Rust dependencies."
echo "To activate: source venv/bin/activate"
echo "To build: maturin develop"
echo "To test: pytest tests/"