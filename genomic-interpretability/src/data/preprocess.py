import pandas as pd
import gzip
from pyfaidx import Fasta
import os

def load_clinvar_vcf(vcf_path):
    """
    Parses VCF manually to avoid heavy dependencies like pysam.
    Extracts Chromosome, Position, Ref, Alt, and Clinical Significance.
    """
    print(f"Parsing VCF from {vcf_path}...")
    variants = []
    
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            
            parts = line.strip().split("\t")
            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            info = parts[7]
            
            if len(ref) != 1 or len(alt) != 1:
                continue
            
            clnsig = None
            for item in info.split(";"):
                if item.startswith("CLNSIG="):
                    clnsig = item.split("=")[1]
                    break
            
            if clnsig:
                if "Pathogenic" in clnsig and "Benign" not in clnsig and "Likely" not in clnsig:
                    variants.append([chrom, pos, ref, alt, 1]) # 1 = Pathogenic
                elif "Benign" in clnsig and "Pathogenic" not in clnsig and "Likely" not in clnsig:
                    variants.append([chrom, pos, ref, alt, 0]) # 0 = Benign

    return pd.DataFrame(variants, columns=["chrom", "pos", "ref", "alt", "label"])

def get_sequence_context(df, fasta_path, window=512):
    """
    Fetches the DNA sequence around the variant.
    """
    print("Extracting sequence contexts...")
    genome = Fasta(fasta_path)
    
    valid_data = []
    half_window = window // 2
    
    for idx, row in df.iterrows():
        chrom = str(row['chrom'])
        if chrom not in genome:
            chrom = f"chr{chrom}"
        
        if chrom not in genome:
            continue
            
        pos = row['pos']
        ref = row['ref']
        alt = row['alt']
        
        start = pos - 1 - half_window
        end = pos - 1 + half_window
        
        try:
            seq = genome[chrom][start:end].seq.upper()
        except:
            continue
            
        if len(seq) != window:
            continue
            
        center_base = seq[half_window]
        
        if center_base != ref:
            continue
            
        seq_list = list(seq)
        seq_list[half_window] = alt
        seq_alt = "".join(seq_list)
        
        valid_data.append({
            "chrom": row['chrom'],
            "pos": row['pos'],
            "ref": ref,
            "alt": alt,
            "label": row['label'],
            "seq_ref": seq,
            "seq_alt": seq_alt
        })
        
    return pd.DataFrame(valid_data)

if __name__ == "__main__":
    vcf_file = "data/raw/clinvar.vcf.gz"
    fasta_file = "data/raw/hg38.fa"
    output_file = "data/processed/variants.csv"
    
    df_vars = load_clinvar_vcf(vcf_file)
    print(f"Found {len(df_vars)} raw SNPs.")
    
    df_final = get_sequence_context(df_vars, fasta_file)
    print(f"Successfully processed {len(df_final)} variants.")
    
    df_final.to_csv(output_file, index=False)