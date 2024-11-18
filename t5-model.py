import os
import torch
import pandas as pd
from transformers import (
    T5Tokenizer, 
    T5ForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support


df = pd.read_csv('BABE.csv')

assert df['text'].notna().all(), "There are missing values in the text column"
assert df['label_bias'].notna().all(), "There are missing values in the label_bias column"

# Remove the "No agreement" label
df = df[df['label_bias'] != "No agreement"]

# Convert labels to integers: "Non-biased" = 0, "Biased" = 1
df['label_bias'] = df['label_bias'].map({"Non-biased": 0, "Biased": 1})

# Separate classes for balancing
df_majority = df[df['label_bias'] == 0]
df_minority = df[df['label_bias'] == 1]

df_minority_oversampled = resample(df_minority, 
                                   replace=True,   
                                   n_samples=len(df_majority), 
                                   random_state=42)

df_balanced = pd.concat([df_majority, df_minority_oversampled])

train_df, eval_df = train_test_split(df_balanced, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# Tokenization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
max_length = 512 

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)


train_dataset = train_dataset.rename_column("label_bias", "labels")
eval_dataset = eval_dataset.rename_column("label_bias", "labels")


train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model = T5ForSequenceClassification.from_pretrained("t5-small", num_labels=2)

# Device setup: Use MPS if available, otherwise fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps", 
    evaluation_strategy="epoch", 
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=1,  
    logging_steps=10, 
    fp16=False  
)

# Define Metrics
from sklearn.metrics import precision_recall_fscore_support


def compute_metrics(pred):
    labels = pred.label_ids
    
    # If pred.predictions is a tuple (e.g., (logits, attention_weights)), we take the first element (logits)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    
    # Apply argmax to get the predicted labels
    preds = logits.argmax(-1)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    return {"precision": precision, "recall": recall, "f1": f1}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start Training
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)
