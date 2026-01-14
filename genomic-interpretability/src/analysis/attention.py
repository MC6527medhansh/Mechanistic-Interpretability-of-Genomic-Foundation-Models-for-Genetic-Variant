import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src.models.train import GenomicModel 

def analyze_attention():
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = GenomicModel(model_name)
    model.load_state_dict(torch.load("results/checkpoints/final_model_state.pt"))
    model.eval()
    
    test_df = pd.read_csv("data/processed/test_set.csv")
    pathogenic_samples = test_df[test_df['label'] == 1].sample(50, random_state=42)
    
    attention_scores = []
    
    print("Extracting attention maps...")
    for idx, row in pathogenic_samples.iterrows():
        seq = row['seq_alt']
        inputs = tokenizer(seq, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        
        with torch.no_grad():
            outputs = model.backbone(inputs['input_ids'], output_attentions=True)
            
        last_layer_attn = outputs.attentions[-1].squeeze(0).cpu().numpy()
        
        avg_attn = np.mean(last_layer_attn, axis=0)
        
        center_idx = 256 
        
        attn_to_center = avg_attn[:, center_idx]
        attention_scores.append(attn_to_center)

    avg_attention_profile = np.mean(np.array(attention_scores), axis=0)
    
    plt.figure(figsize=(10, 5))
    plt.plot(avg_attention_profile)
    plt.title("Average Attention to Variant Site (Pathogenic Samples)")
    plt.xlabel("Token Position")
    plt.ylabel("Attention Weight")
    plt.axvline(x=256, color='r', linestyle='--', label='Variant Site')
    plt.legend()
    plt.savefig("results/figures/attention_profile.png")
    print("Saved attention plot to results/figures/attention_profile.png")

if __name__ == "__main__":
    analyze_attention()