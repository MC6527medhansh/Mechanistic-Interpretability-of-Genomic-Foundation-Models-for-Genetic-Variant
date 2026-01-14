import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class GenomicModel(torch.nn.Module):
    """
    Wraps the foundation model with a classification head.
    """
    def __init__(self, model_name, num_labels=2):
        super().__init__()

        self.backbone = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size
        
        self.classifier = torch.nn.Linear(self.hidden_size, num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        last_hidden = outputs.hidden_states[-1]
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return {"loss": loss, "logits": logits}

def run_training():
    df = pd.read_csv("data/processed/variants.csv", dtype={'chrom': str})
    
    train_chroms = [str(i) for i in range(1, 19)]
    val_chroms = ["19"]
    test_chroms = ["20", "21", "22", "X", "Y"]
    
    df['seq'] = df['seq_alt'] 
    
    train_df = df[df['chrom'].isin(train_chroms)]
    
    if len(train_df) > 50000:
        train_df = train_df.sample(n=50000, random_state=42)
        
    val_df = df[df['chrom'].isin(val_chroms)]
    test_df = df[df['chrom'].isin(test_chroms)]
    test_df.to_csv("data/processed/test_set.csv", index=False)
    
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    def tokenize_function(examples):
        return tokenizer(examples["seq"], padding="max_length", truncation=True, max_length=512)

    train_ds = Dataset.from_pandas(train_df[['seq', 'label']]).map(tokenize_function, batched=True)
    val_ds = Dataset.from_pandas(val_df[['seq', 'label']]).map(tokenize_function, batched=True)
    
    model = GenomicModel(model_name)
    
    training_args = TrainingArguments(
        output_dir="results/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        seed=42
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()
    
    torch.save(model.state_dict(), "results/checkpoints/final_model_state.pt")

if __name__ == "__main__":
    run_training()