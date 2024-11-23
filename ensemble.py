import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

xlnet_predictions = torch.load('xlnet_predictions.pt')
roberta_predictions = torch.load('roberta_predictions.pt')
bert_predictions = torch.load('bert_predictions.pt')
t5_predictions = torch.load('t5_predictions.pt')

# Modified to work with other computer types
xlnet_logits = xlnet_predictions["logits"].detach().cpu().numpy()
roberta_logits = roberta_predictions["logits"].detach().cpu().numpy()
bert_logits = bert_predictions["logits"].detach().cpu().numpy()
t5_logits = t5_predictions["logits"].detach().cpu().numpy()

sample_ids = xlnet_predictions["sample_ids"]
assert np.array_equal(sample_ids, roberta_predictions["sample_ids"]), "Sample IDs do not match!"
assert np.array_equal(sample_ids, bert_predictions["sample_ids"]), "Sample IDs do not match!"
assert np.array_equal(sample_ids, t5_predictions["sample_ids"]), "Sample IDs do not match!"

ground_truth = pd.read_csv('ground_truth.csv')
true_labels = ground_truth['true_label'].values

weight_range = np.arange(0.0, 1.1, 0.1)
manual_weights = None  # Set to None to use grid search

best_weights = None
best_f1 = 0.0
best_accuracy = 0.0
best_precision = 0.0
best_recall = 0.0

if manual_weights:

    assert np.isclose(sum(manual_weights), 1.0), "Manual weights must sum to 1.0"

    w_xlnet, w_roberta, w_bert, w_t5 = manual_weights


    ensemble_logits = (
        w_xlnet * xlnet_logits +
        w_roberta * roberta_logits +
        w_bert * bert_logits +
        w_t5 * t5_logits
    )

    ensemble_labels = np.argmax(ensemble_logits, axis=-1)


    accuracy = accuracy_score(true_labels, ensemble_labels)
    precision = precision_score(true_labels, ensemble_labels, average='weighted')
    recall = recall_score(true_labels, ensemble_labels, average='weighted')
    f1 = f1_score(true_labels, ensemble_labels, average='weighted')


    print(f"Manual Weights: XLNet={manual_weights[0]}, RoBERTa={manual_weights[1]}, BERT={manual_weights[2]}, T5={manual_weights[3]}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
else:
    for weights in product(weight_range, repeat=4):
        if not np.isclose(sum(weights), 1.0):
            continue

        w_xlnet, w_roberta, w_bert, w_t5 = weights

        ensemble_logits = (
            w_xlnet * xlnet_logits +
            w_roberta * roberta_logits +
            w_bert * bert_logits +
            w_t5 * t5_logits
        )

        ensemble_labels = np.argmax(ensemble_logits, axis=-1)

        accuracy = accuracy_score(true_labels, ensemble_labels)
        precision = precision_score(true_labels, ensemble_labels, average='weighted')
        recall = recall_score(true_labels, ensemble_labels, average='weighted')
        f1 = f1_score(true_labels, ensemble_labels, average='weighted')

        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

    print(f"Best Weights: XLNet={best_weights[0]}, RoBERTa={best_weights[1]}, BERT={best_weights[2]}, T5={best_weights[3]}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")

    # Save the best weights and predictions for later use
    torch.save({
        'weights': torch.tensor(best_weights),  # Save weights as tensor
        'logits': {
            'xlnet': torch.tensor(xlnet_logits),  # Ensure logits are saved as tensors
            'roberta': torch.tensor(roberta_logits),
            'bert': torch.tensor(bert_logits),
            't5': torch.tensor(t5_logits)
        }
    }, 'best_model.pt')
