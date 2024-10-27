import torch
from pandas import read_csv
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Load the CSV file with error handling

df = read_csv("final_labels_MBIC.csv", sep=';')

# Filter relevant columns: 'text' and 'label_bias'
df = df[['text', 'label_bias']]
# Map the string labels to integers: "Biased" -> 1, "Non-biased" -> 0
# Optionally, drop rows where 'label_bias' is "No agreement" or handle it differently.
df = df[df['label_bias'] != "No agreement"]  # Remove "No agreement" rows (or handle as a separate class if needed)

# Convert the labels into numerical values
df['label_bias'] = df['label_bias'].map({'Biased': 1, 'Non-biased': 0})

# Split the data into training and testing sets (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), df['label_bias'].tolist(), test_size=0.2
)
print(len(test_texts))
print(len(train_texts))

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Prepare data loaders
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)


# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
total_correct = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        total_correct += (predictions == labels).sum().item()

accuracy = total_correct / len(test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Function to predict bias in new sentences
def predict_bias(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Biased" if prediction == 1 else "Unbiased"

