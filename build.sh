#!/bin/bash

# Build script for anomaly-grid-py

set -e

echo "Building anomaly-grid-py..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup.sh
fi

# Activate virtual environment
source venv/bin/activate

# Build the package
echo "Building with maturin..."
maturin develop

echo "Build complete!"
echo "You can now run: python example.py"
echo "Or run tests with: pytest tests/"