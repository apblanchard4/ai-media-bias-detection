import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Use MPS if available, otherwise fall back to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load and preprocess dataset
df = pd.read_csv('BABE.csv').dropna(subset=['text', 'label_bias'])
df = df[df['label_bias'] != "No agreement"]

# Prepare tokenizer and model
MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)

# Encode labels and split data
df['label'] = LabelEncoder().fit_transform(df['label_bias'])
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Tokenize data
def tokenize_texts(texts):
    return tokenizer(list(texts), truncation=True, padding='max_length', max_length=128)

train_encodings, val_encodings = map(tokenize_texts, [train_texts, val_texts])

# Create PyTorch Dataset
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BiasDataset(train_encodings, train_labels)
val_dataset = BiasDataset(val_encodings, val_labels)

# Class weights for imbalance
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label']), dtype=torch.float).to(device)

# Custom Trainer with class weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(weight=class_weights)(outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Compute metrics (precision, recall, F1)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics
)

# Train and save model
trainer.train()
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

print("Model and tokenizer saved to './saved_model'")
