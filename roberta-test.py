import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score

# Check if MPS (Apple Silicon GPU) is available
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load the dataset (same dataset used during training)
df = pd.read_csv('BABE.csv')

# Remove the "No agreement" category to keep only two classes
df = df[df['label_bias'] != "No agreement"]

# Encode the labels again (same process used during training)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label_bias'])

# Check the classes and ensure only two unique labels (e.g., [0, 1])
unique_labels = df['label'].unique()
assert len(unique_labels) == 2, f"Expected 2 unique labels, found {len(unique_labels)}: {unique_labels}"

# Split the data into train and validation sets (only need the validation set now)
_, val_texts, _, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Reload the tokenizer and model from the saved directory
MODEL_DIR = './saved_model'
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Move model to device (MPS or CPU)
model.to(device)
model.eval()  # Ensure the model is in evaluation mode

# Tokenize the validation data
val_encodings = tokenizer(list(val_texts), truncation=True, padding='max_length', max_length=128)

# Convert the validation data to a torch dataset
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(device)
        return item

    def __len__(self):
        return len(self.labels)

val_dataset = BiasDataset(val_encodings, list(val_labels))

# Define the compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    recall = recall_score(labels, preds, average='weighted', zero_division=1)
    f1 = f1_score(labels, preds, average='weighted', zero_division=1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define training arguments (these will not be used but are required by Trainer)
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,  # Adjusted for evaluation
)

# Trainer with only the evaluation dataset, model, and metrics function
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Run evaluation
eval_results = trainer.evaluate()

# Debugging: Print sample predictions and labels
preds = trainer.predict(val_dataset).predictions.argmax(-1)
print(f"Sample Predictions: {preds[:10]}")
print(f"Sample Labels: {list(val_labels)[:10]}")

# Print evaluation results
print("Evaluation results:", eval_results)
