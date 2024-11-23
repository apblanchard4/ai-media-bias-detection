import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load the best model
model_data = torch.load('best_model.pt')
weights = model_data['weights']
logits = model_data['logits']

# Print the loaded weights and logits for debugging
print("Loaded Weights:", weights)
print("Loaded Logits XLNet:", logits['xlnet'][:5])
print("Loaded Logits RoBERTa:", logits['roberta'][:5])
print("Loaded Logits BERT:", logits['bert'][:5])
print("Loaded Logits T5:", logits['t5'][:5])

# Load ground truth for evaluation
ground_truth = pd.read_csv('ground_truth.csv')
true_labels = ground_truth['true_label'].values

# Combine the logits from all models using the saved weights
ensemble_logits = (
    weights[0] * logits['xlnet'] +
    weights[1] * logits['roberta'] +
    weights[2] * logits['bert'] +
    weights[3] * logits['t5']
)

# Print the ensemble logits before final decision
print("Ensemble Logits:", ensemble_logits[:5])

# Get the final prediction
ensemble_labels = np.argmax(ensemble_logits, axis=-1)

# Print the ensemble predictions
print("Ensemble Predictions:", ensemble_labels[:5])

# Evaluate the ensemble performance
accuracy = accuracy_score(true_labels, ensemble_labels)
print(f"Accuracy: {accuracy:.4f}")

# Predict for a new sentence
sentence = input("Enter a sentence to predict bias: ")

# Assuming you have a tokenizer and model ready for inference
# Tokenize the input sentence and pass it through the model
# (This part might differ depending on your exact setup)

# Make sure to import your tokenizer and models if you haven't done so
from transformers import XLNetTokenizer, RobertaTokenizer, BertTokenizer, T5Tokenizer
from transformers import XLNetForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification, T5ForSequenceClassification

# Initialize tokenizers and models
tokenizer_xlnet = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer_roberta = RobertaTokenizer.from_pretrained("roberta-base")
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")

model_xlnet = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
model_roberta = RobertaForSequenceClassification.from_pretrained("roberta-base")
model_bert = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model_t5 = T5ForSequenceClassification.from_pretrained("t5-small")

# Tokenize the input sentence for each model
xlnet_input = tokenizer_xlnet(sentence, return_tensors='pt')
roberta_input = tokenizer_roberta(sentence, return_tensors='pt')
bert_input = tokenizer_bert(sentence, return_tensors='pt')
t5_input = tokenizer_t5(sentence, return_tensors='pt')

# Get logits for each model (ensure this is the same format as used during training)
xlnet_logits = model_xlnet(**xlnet_input).logits
roberta_logits = model_roberta(**roberta_input).logits
bert_logits = model_bert(**bert_input).logits
t5_logits = model_t5(**t5_input).logits

# Combine the logits from the models with saved weights
ensemble_logits = (
    weights[0] * xlnet_logits +
    weights[1] * roberta_logits +
    weights[2] * bert_logits +
    weights[3] * t5_logits
)

# Get the final prediction
ensemble_prediction = np.argmax(ensemble_logits.detach().numpy(), axis=-1)

# Print the result
if ensemble_prediction == 1:
    print("Prediction: Biased")
else:
    print("Prediction: Not Biased")
