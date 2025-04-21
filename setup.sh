#!/bin/bash

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install --upgrade setuptools
pip install -r requirements.txt

# Create necessary directories
mkdir -p output/images
mkdir -p output/analysis

echo "Setup completed successfully!"
