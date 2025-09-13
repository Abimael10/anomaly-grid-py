#!/bin/bash

# Complete setup and build script for anomaly-grid-py

set -e

echo "🔧 Setting up anomaly-grid-py development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install build dependencies
echo "📥 Installing dependencies..."
pip install maturin[patchelf] pytest

# Build the package
echo "🔨 Building with maturin..."
maturin develop

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "  • Activate environment: source venv/bin/activate"
echo "  • Run examples: python example.py"
echo "  • Run tests: pytest tests/"
echo "  • Rebuild after changes: maturin develop"