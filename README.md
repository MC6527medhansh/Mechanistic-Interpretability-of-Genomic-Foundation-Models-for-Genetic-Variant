# Genomic Interpretability: Nucleotide Transformer v2 on ClinVar

**Author:** Medhansh Choubey  
**Date:** January 2026

## Project Overview

Mechanistic interpretability analysis of the **Nucleotide Transformer v2-50M** for genetic variant classification (Pathogenic vs. Benign). Implements two interpretability approaches:

1. **Attention Analysis:** Statistical comparison of attention patterns between pathogenic and benign variants.
2. **Activation Patching:** Causal analysis identifying which layers drive pathogenicity predictions.

## Requirements

- Python 3.9+
- 8GB+ RAM
- GPU recommended (but not required)
- ~35GB disk space for reference genome

## Quick Start (Run Everything)

```bash
bash run_all.sh

```

This single command will:

1. Download data (ClinVar + hg38 reference)
2. Preprocess variants
3. Train the model (~1 hour on GPU)
4. Run interpretability analyses
5. Generate all figures in `results/figures/`

## Manual Step-by-Step

If you prefer to run steps individually:

### 1. Setup

```bash
git clone [https://github.com/MC6527medhansh/Mechanistic-Interpretability-of-Genomic-Foundation-Models-for-Genetic-Variant.git](https://github.com/MC6527medhansh/Mechanistic-Interpretability-of-Genomic-Foundation-Models-for-Genetic-Variant.git)
cd Mechanistic-Interpretability-of-Genomic-Foundation-Models-for-Genetic-Variant

pip install -r requirements.txt

```

### 2. Download Data

```bash
bash scripts/download_data.sh

```

Downloads:

* ClinVar VCF (~500MB compressed)
* hg38 reference genome (~30GB compressed)

### 3. Preprocess

```bash
python src/data/preprocess.py

```


Output: `data/processed/{train,val,test}_set.csv`

### 4. Train Model

```bash
python src/models/train.py

```

Output: `results/checkpoints/final_model_state.pt`

### 5. Run Interpretability

```bash
python src/analysis/attention.py
python src/analysis/patching.py

```

Output: All figures in `results/figures/`

## Repository Structure

```
├── data/
│   ├── raw/              (ClinVar VCF, hg38 reference - gitignored)
│   └── processed/        (Train/val/test CSVs)
├── results/
│   ├── figures/          (Generated plots)
│   ├── tables/           (Numerical results as JSON/NPY)
│   └── checkpoints/      (Model weights - gitignored)
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   └── train.py
│   └── analysis/
│       ├── attention.py
│       └── patching.py
├── scripts/
│   └── download_data.sh
├── README.md
├── requirements.txt
└── run_all.sh

```

## Results

Key outputs in `results/figures/`:

* `attention_comparison.png` - Pathogenic vs Benign attention profiles
* `patching_results.png` - Layer-wise causal effects

## Computational Notes

**CPU-only mode:** All scripts work on CPU but will be slower. Training may take 4-6 hours.

**Memory:** Peak RAM usage ~8GB during preprocessing (loading hg38). Model inference ~4GB.

**Disk:** Reference genome is large (~30GB compressed, ~35GB uncompressed). Can delete after preprocessing to save space.

## References

* **Data:** [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) (NCBI)
* **Model:** Dalla-Torre et al., "The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics," bioRxiv 2023
* **Reference Genome:** GRCh38/hg38 from UCSC Genome Browser

## Citation

If you use this code, please cite:

```
Choubey, M. (2026). Mechanistic Interpretability of Genomic Foundation Models 
for Genetic Variants. GitHub repository.

```