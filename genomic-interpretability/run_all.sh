#!/bin/bash
set -e

echo "1. Downloading Data..."
bash scripts/download_data.sh

echo "2. Preprocessing..."
python src/data/preprocess.py

echo "3. Training Model..."
python src/models/train.py

echo "4. Generating Interpretability Results..."
python src/analysis/attention.py
python src/analysis/patching.py

echo "Pipeline complete."