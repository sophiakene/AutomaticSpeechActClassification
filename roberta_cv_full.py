# %%
import os
import sys
import re
import json
import torch
import pandas as pd
import numpy as np

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split

# %%
from datasets import Dataset

# %%
from transformers import RobertaTokenizer, AutoModelForSequenceClassification, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

# %%
import accelerate
import transformers

print("accelerate version:", accelerate.__version__)
print("transformers version:", transformers.__version__)

# %%
from torch.utils.data import DataLoader

# %%
# Use CUDA GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# 

# %%
# Verify CUDA setup
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

# %%
data_path = 'path/to/csv/file.csv'

data_df = pd.read_csv(data_path, sep=',', encoding='utf-8', usecols=['sentence', 'label', 'filename'])
data_df = data_df.rename(columns={'sentence': 'text'})
data_df

# %%
# Create label mappings from all data
label2id = {label: i for i, label in enumerate(sorted(set(data_df['label'])))}
id2label = {i: label for label, i in label2id.items()}

# Function to encode labels
def encode_labels(dataset):
    return dataset.map(lambda x: {"label": label2id[x["label"]]})

# %%
model_name = "roberta-base"

tokenizer = RobertaTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# %%
from transformers import Trainer
import torch.nn as nn

class WeightedTrainer(Trainer):
    def __init__(self, *args, weights=None, **kwargs):
        kwargs.pop("tokenizer", None)  # Safely remove deprecated arg
        super().__init__(*args, **kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=self.weights.to(self.args.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# %%
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# %%
def classification_report_to_dict(y_true, y_pred, labels):
    """Convert sklearn classification report to a dictionary"""
    report = classification_report(
        y_true, 
        y_pred,
        target_names=labels,
        digits=4,
        output_dict=True
    )
    return report

# %%
# Create document-level label distribution
doc_labels = data_df.groupby('filename')['label'].agg(lambda x: x.value_counts().index[0]).reset_index()

# Set up k-fold cross validation at document level
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Lists to store metrics for each fold
all_metrics = []
all_classification_reports = []

# Create directories for results if they don't exist
results_dir = os.path.expanduser('~/results')
os.makedirs(results_dir, exist_ok=True)

# Iterate through folds using document-level stratification
for fold, (train_val_doc_idx, test_doc_idx) in enumerate(skf.split(doc_labels, doc_labels['label'])):
    print(f'\nFold {fold + 1}/{n_folds}')
    
    # Get document IDs for each split
    train_val_docs = doc_labels.iloc[train_val_doc_idx]['filename'].values
    test_docs = doc_labels.iloc[test_doc_idx]['filename'].values
    
    # Split data based on document IDs
    train_val_data = data_df[data_df['filename'].isin(train_val_docs)]
    test_data = data_df[data_df['filename'].isin(test_docs)]
    
    # Further split train_val into train and validation at document level
    train_val_docs_df = doc_labels[doc_labels['filename'].isin(train_val_docs)]
    try:
        train_docs, eval_docs = train_test_split(
            train_val_docs_df['filename'].values,
            test_size=0.15,
            stratify=train_val_docs_df['label'],
            random_state=42
        )
    except ValueError as e:
        print("Warning: Could not perform stratified split due to insufficient samples.")
        print("Falling back to random split.")
        train_docs, eval_docs = train_test_split(
            train_val_docs_df['filename'].values,
            test_size=0.15,
            random_state=42
        )

    # Split data based on document IDs
    train_df = train_val_data[train_val_data['filename'].isin(train_docs)]
    eval_df = train_val_data[train_val_data['filename'].isin(eval_docs)]
    
    print(f'Documents - Train: {len(train_docs)}, Val: {len(eval_docs)}, Test: {len(test_docs)}')
    print(f'Sentences - Train: {len(train_df)}, Val: {len(eval_df)}, Test: {len(test_data)}')
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    test_dataset = Dataset.from_pandas(test_data)
    
    # Encode labels
    train_dataset = encode_labels(train_dataset)
    eval_dataset = encode_labels(eval_dataset)
    test_dataset = encode_labels(test_dataset)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_eval = eval_dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    # Remove unnecessary columns and set format
    for dataset in [tokenized_train, tokenized_eval, tokenized_test]:
        dataset = dataset.remove_columns(['text', 'filename'])
        dataset.set_format('torch')
    
    # Update datasets
    train_dataset = tokenized_train
    eval_dataset = tokenized_eval
    test_dataset = tokenized_test
    
    # Calculate class weights for this fold
    labels = np.array(train_dataset['label'])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    weights = torch.tensor(class_weights, dtype=torch.float)
    
    # Initialize model for this fold
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, f'fold_{fold}'),
        eval_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        learning_rate=1e-5,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        logging_dir=os.path.join(results_dir, 'logs', f'fold_{fold}'),
        save_total_limit=1  # Keep only the best model
    )
    
    # Initialize trainer
    trainer = WeightedTrainer(
        model=model,
        weights=weights,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions and classification report
    raw_pred, _, _ = trainer.predict(test_dataset)
    predictions = np.argmax(raw_pred, axis=1)
    true_labels = test_dataset['label']
    
    # Get classification report as dictionary
    class_report = classification_report_to_dict(
        true_labels,
        predictions,
        labels=[id2label[i] for i in range(len(id2label))]
    )
    
    # Print results
    print(f'\nFold {fold + 1} Test Results:')
    print('Classification Report:')
    print(classification_report(true_labels, predictions, 
                              target_names=[id2label[i] for i in range(len(id2label))], 
                              digits=4))
    
    # Store metrics and classification report for this fold
    all_metrics.append(test_results)
    all_classification_reports.append(class_report)
    
    # Save fold results to JSON
    fold_results = {
        'metrics': test_results,
        'classification_report': class_report
    }
    
    with open(os.path.join(results_dir, f'fold_{fold}_results.json'), 'w') as f:
        json.dump(fold_results, f, indent=2)
    
# Calculate and display average metrics across folds
print('\n' + '='*50)
print('Average Results Across All Folds:')
print('='*50)

avg_metrics = {}
std_metrics = {}

for metric in all_metrics[0].keys():
    values = [fold[metric] for fold in all_metrics]
    avg_metrics[metric] = np.mean(values)
    std_metrics[metric] = np.std(values)
    print(f'{metric}:')
    print(f'  Mean: {avg_metrics[metric]:.4f}')
    print(f'  Std:  {std_metrics[metric]:.4f}')

# Calculate average classification report
avg_class_report = {}
for label in all_classification_reports[0].keys():
    if isinstance(all_classification_reports[0][label], dict):
        avg_class_report[label] = {}
        for metric in all_classification_reports[0][label].keys():
            values = [report[label][metric] for report in all_classification_reports]
            avg_class_report[label][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
    elif label != 'accuracy':
        values = [report[label] for report in all_classification_reports]
        avg_class_report[label] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }

# Save average results to JSON
average_results = {
    'metrics': {
        'mean': avg_metrics,
        'std': std_metrics
    },
    'classification_report': avg_class_report
}

with open(os.path.join(results_dir, 'average_results.json'), 'w') as f:
    json.dump(average_results, f, indent=2)

# %%



