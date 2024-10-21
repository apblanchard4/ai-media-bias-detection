import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample

# Check if MPS (Apple Silicon GPU) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset
df = pd.read_csv('BABE.csv')

# Check for missing values
assert df['text'].notna().all(), "There are missing values in the text column"
assert df['label_bias'].notna().all(), "There are missing values in the label_bias column"

# Remove the "No agreement" label
df = df[df['label_bias'] != "No agreement"]

# Separate classes for balancing
df_majority = df[df['label_bias'] == "Non-biased"]
df_minority = df[df['label_bias'] == "Biased"]

# Oversample minority class
df_minority_oversampled = resample(df_minority, 
                                   replace=True,    # Sample with replacement
                                   n_samples=len(df_majority),  # Match number of majority class
                                   random_state=42)

# Combine majority class with oversampled minority class
df_balanced = pd.concat([df_majority, df_minority_oversampled])

# Prepare the tokenizer and model
MODEL = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)  # 2 labels: Biased, Non-biased

# Move model to device (MPS or CPU)
model.to(device)

# Encode the labels
label_encoder = LabelEncoder()
df_balanced['label'] = label_encoder.fit_transform(df_balanced['label_bias'])

# Split the data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=42
)

# Tokenize the data with padding and truncation
train_encodings = tokenizer(list(train_texts), truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding='max_length', max_length=128)

# Convert to torch dataset
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BiasDataset(train_encodings, list(train_labels))
val_dataset = BiasDataset(val_encodings, list(val_labels))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=5,            
    learning_rate=2e-5,           
    weight_decay=0.01,              
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Define custom metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Adding Early Stopping Callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train the model
trainer.train()

# Save the best model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

print("Model and tokenizer saved to './saved_model'")
