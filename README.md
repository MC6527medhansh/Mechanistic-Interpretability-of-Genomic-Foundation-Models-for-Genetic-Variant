# Genomic Interpretability: Nucleotide Transformer v2 on ClinVar

**Author:** Medhansh Choubey
**Date:** January 2026

## Project Overview
This repository contains a reproducible pipeline for the mechanistic interpretability of genomic foundation models. We fine-tune the **Nucleotide Transformer v2-50M** to classify genetic variants (Pathogenic vs. Benign) and apply two interpretability techniques:
1.  **Attention Analysis:** Visualizing where the model focuses within the DNA sequence to determine if it prioritizes the variant site.
2.  **Activation Patching:** A causal analysis to identify which transformer layers drive the pathogenicity prediction by systematically swapping hidden states between pathogenic and benign sequences.

## Repository Structure
```text
├── data/               # Raw and processed genomic data (gitignored)
├── results/            # Generated figures and model checkpoints
├── scripts/            # Utility scripts (downloaders)
├── src/                # Source code
│   ├── analysis/       # Interpretability scripts (attention, patching)
│   ├── data/           # Preprocessing logic
│   └── models/         # Training loop and model definition
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies

```

## Setup & Installation

**1. Clone the repository**

```bash
git clone [https://github.com/medhanshchoubey/genomic-interpretability.git](https://github.com/medhanshchoubey/genomic-interpretability.git)
cd genomic-interpretability

```

**2. Install Dependencies**
We recommend using a clean Conda environment (Python 3.9+).

```bash
pip install torch transformers datasets biopython pandas numpy scikit-learn matplotlib seaborn tqdm pyfaidx accelerate

```

## Reproduction Steps

**Step 1: Download Data**
This script fetches the ClinVar VCF and the hg38 reference genome from public sources.

```bash
bash scripts/download_data.sh

```

**Step 2: Preprocess Data**
Parses the VCF, filters for SNPs, and extracts 512bp context windows around each variant.

```bash
python src/data/preprocess.py

```

**Step 3: Fine-Tune Model**
Trains a linear classification head on the Nucleotide Transformer backbone.

* **Runtime:** ~1 hour on MPS (Mac) or GPU.
* **Optimization:** Uses downsampling (50k examples) and 1 epoch for efficiency.

```bash
python src/models/train.py

```

**Step 4: Run Interpretability Analysis**
Generates the Attention Profile and Causal Patching figures in `results/figures/`.

* **Note:** We use `PYTHONPATH=.` to ensure module imports work correctly from the root directory.

```bash
# Generate Attention Plots (Figure 1)
PYTHONPATH=. python src/analysis/attention.py

# Generate Activation Patching Plots (Figure 2)
PYTHONPATH=. python src/analysis/patching.py

```

## Results

The analysis generates two key figures in `results/figures/`:

* `attention_profile.png`: Shows the average attention weight across the 512bp sequence. A spike at index 256 indicates the model is attending to the variant.
* `patching_results.png`: Displays the causal effect of swapping hidden states at the variant position for each layer.

## References

* **Data:** ClinVar (NCBI).
* **Model:** Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics," 2023.

```

```