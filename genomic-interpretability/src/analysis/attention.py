import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from scipy.stats import mannwhitneyu
from src.models.train import GenomicModel 

def analyze_attention():
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GenomicModel(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model.load_state_dict(torch.load("results/checkpoints/final_model_state.pt", map_location=device))
    except:
        pass
    model.to(device)
    model.eval()
    
    df = pd.read_csv("data/processed/test_set.csv")
    
    pathogenic = df[df['label'] == 1].sample(n=min(100, len(df[df['label']==1])), random_state=42)
    benign = df[df['label'] == 0].sample(n=min(100, len(df[df['label']==0])), random_state=42)
    
    def get_avg_attn(samples):
        attentions = []
        for _, row in samples.iterrows():
            inputs = tokenizer(row['seq_alt'], return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
            with torch.no_grad():
                outputs = model.backbone(inputs['input_ids'], output_attentions=True)
            last_layer = outputs.attentions[-1].squeeze(0).cpu().numpy()
            avg_head = np.mean(last_layer, axis=0)
            attentions.append(avg_head[:, 256])
        return np.array(attentions)

    print("Extracting attention for Pathogenic variants...")
    path_attn = get_avg_attn(pathogenic)
    print("Extracting attention for Benign variants...")
    benign_attn = get_avg_attn(benign)

    plt.figure(figsize=(10, 5))
    plt.plot(np.mean(path_attn, axis=0), label="Pathogenic", color="red", alpha=0.8)
    plt.plot(np.mean(benign_attn, axis=0), label="Benign", color="blue", alpha=0.6, linestyle="--")
    plt.axvline(x=256, color='black', linestyle=':', label='Variant Site')
    plt.legend()
    plt.title("Attention Profile Comparison: Pathogenic vs Benign")
    plt.xlabel("Token Position")
    plt.ylabel("Avg Attention Weight")
    plt.savefig("results/figures/attention_profile.png")
    print("Saved comparison plot.")

    path_center_mass = np.mean(path_attn[:, 254:258], axis=1)
    benign_center_mass = np.mean(benign_attn[:, 254:258], axis=1)
    
    stat, p_val = mannwhitneyu(path_center_mass, benign_center_mass, alternative='greater')
    print(f"\nStatistical Test (Mann-Whitney U):")
    print(f"Hypothesis: Pathogenic attention > Benign attention at variant site")
    print(f"P-value: {p_val:.5e}")
    if p_val < 0.05:
        print("RESULT: Significant difference found.")
    else:
        print("RESULT: No significant difference.")

if __name__ == "__main__":
    analyze_attention()