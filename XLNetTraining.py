import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import XLNetTokenizer, XLNetForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

#using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
df = pd.read_excel('final_labels_SG1.xlsx')
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace('[^\w\s]', '', regex=True)


X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_bias'], test_size=0.2, random_state=42)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


joblib.dump(label_encoder, 'label_encoder.joblib')

# tokenizer and model
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(label_encoder.classes_))
model.to(device)

# Tokenize
train_encodings = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')


class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encodings, y_train_encoded)
test_dataset = NewsDataset(test_encodings, y_test_encoded)


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.0001)  # Reduced learning rate for better fine-tuning

#learning rate scheduler
num_training_steps = len(train_loader) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)


max_grad_norm = 1.0


class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.005):  # Increased patience and reduced min_delta for more stability
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# training loop with early stopping
early_stopping = EarlyStopping(patience=7)

best_model_path = 'best_model.pth'
best_f1_score = 0.0

for epoch in range(10):
    print(f"Epoch {epoch + 1} starting...")
    model.train()
    total_loss = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        #Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}, Batch {i}, Loss: {loss.item()}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")


    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    print(f"Validation Loss: {avg_val_loss}")

    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    print(f"Validation F1 Score: {f1:.4f}")

    # Save the best model based on F1 score
    if f1 > best_f1_score:
        best_f1_score = f1
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with F1 Score: {best_f1_score:.4f}")

    if early_stopping(avg_val_loss):
        print("Early stopping triggered.")
        break


model.load_state_dict(torch.load(best_model_path))

# Save
torch.save(model.state_dict(), 'model.pth')
tokenizer.save_pretrained('tokenizer')

print(f"Model and tokenizer saved to model.pth and tokenizer")


model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['labels'].cpu().numpy())

# Evaluation
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
