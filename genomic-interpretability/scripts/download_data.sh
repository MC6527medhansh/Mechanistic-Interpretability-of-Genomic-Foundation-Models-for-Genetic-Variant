#!/bin/bash
set -e

mkdir -p data/raw
mkdir -p data/processed

echo "Downloading ClinVar VCF (Genetic Variants)..."
curl -o data/raw/clinvar.vcf.gz https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz
curl -o data/raw/clinvar.vcf.gz.tbi https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi

echo "Downloading Human Reference Genome (hg38)..."
curl -o data/raw/hg38.fa.gz https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz

echo "Unzipping Reference Genome..."
gunzip -k data/raw/hg38.fa.gz

echo "Done."