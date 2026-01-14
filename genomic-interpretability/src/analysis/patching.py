import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from src.models.train import GenomicModel

def run_patching():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}...")
    
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GenomicModel(model_name).to(device)
    
    try:
        model.load_state_dict(torch.load("results/checkpoints/final_model_state.pt", map_location=device))
    except:
        print("Could not load trained weights. Running with base weights (demo mode).")

    model.eval()
    
    df = pd.read_csv("data/processed/test_set.csv")
    candidates = df[df['label'] == 1].head(10)
    
    results_matrix = []
    
    print("Starting patching loop...")
    
    for idx, row in candidates.iterrows():
        seq_pathogenic = row['seq_alt']
        seq_benign = row['seq_ref']
        
        inputs_path = tokenizer(seq_pathogenic, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)
        inputs_benign = tokenizer(seq_benign, return_tensors="pt", padding="max_length", max_length=512, truncation=True).to(device)
        
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output[0].detach()
            return hook
        
        hooks = []
        layers = model.backbone.esm.encoder.layer
        for i, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(get_activation(f"layer_{i}")))
            
        with torch.no_grad():
            model.backbone(inputs_path['input_ids'], attention_mask=inputs_path['attention_mask'])
            
        for h in hooks: h.remove()
        
        layer_diffs = []
        for target_layer_idx in range(len(layers)):
            
            def patch_hook(module, input, output):
                act = output[0].clone()
                cached = activations[f"layer_{target_layer_idx}"]
                act[:, 255:258, :] = cached[:, 255:258, :]
                return (act,) + output[1:]
            
            hook_handle = layers[target_layer_idx].register_forward_hook(patch_hook)
            
            with torch.no_grad():
                patched_output = model(inputs_benign['input_ids'], attention_mask=inputs_benign['attention_mask'])
            
            hook_handle.remove()
            
            logits = patched_output['logits'].cpu().numpy()[0]
            score = logits[1] - logits[0]
            layer_diffs.append(score)
            
        results_matrix.append(layer_diffs)

    avg_patching = np.mean(np.array(results_matrix), axis=0)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(range(len(avg_patching))), y=avg_patching)
    plt.xlabel("Transformer Layer")
    plt.ylabel("Logit Shift (toward Pathogenic)")
    plt.title("Causal Effect of Patching Variant Position by Layer")
    plt.savefig("results/figures/patching_results.png")
    print("Saved patching plot to results/figures/patching_results.png")

if __name__ == "__main__":
    run_patching()