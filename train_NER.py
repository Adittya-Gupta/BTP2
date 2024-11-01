from datasets import Dataset, DatasetDict, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
import torch
import numpy as np
import evaluate

# Function to read NER data from CoNLL format files
def read_ner_data(file_path):
    tokens_list = []
    ner_tags_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = []
        ner_tags = []
        for line in f:
            line = line.strip()
            if line == '':
                if tokens:
                    tokens_list.append(tokens)
                    ner_tags_list.append(ner_tags)
                    tokens = []
                    ner_tags = []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    token = splits[0]
                    ner_tag = splits[-1]
                    tokens.append(token)
                    ner_tags.append(ner_tag)
        if tokens:
            tokens_list.append(tokens)
            ner_tags_list.append(ner_tags)
    return {'tokens': tokens_list, 'ner_tags': ner_tags_list}

# Load data from local files
train_data = read_ner_data('./MCN2_en_train.conll')
validation_data = read_ner_data('./MCN2_en_dev.conll')
test_data = read_ner_data('./MCN2_en_test.conll')

# Create datasets
train_dataset = Dataset.from_dict(train_data)
validation_dataset = Dataset.from_dict(validation_data)
test_dataset = Dataset.from_dict(test_data)

# Create a DatasetDict
dataset = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

print("\n------Dataset loaded from local files------\n")

# Build label mappings
labels = set()
for split in ['train', 'validation', 'test']:
    for ner_tags in dataset[split]['ner_tags']:
        labels.update(ner_tags)

label_names = sorted(list(labels))
label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for idx, label in enumerate(label_names)}

# Map labels to IDs
def encode_labels(example):
    example['ner_tags'] = [label2id[label] for label in example['ner_tags']]
    return example

dataset = dataset.map(encode_labels)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-large", add_prefix_space=True)
print("\n------Tokenizer loaded------\n")

# Tokenize dataset and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Padding or special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])  # First subword gets the label
            else:
                label_ids.append(-100)  # Other subwords get -100
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
print("\n------Tokenization and label alignment done------\n")

# Load model
model = AutoModelForTokenClassification.from_pretrained("roberta-large", num_labels=len(label_names))
model.config.id2label = id2label
model.config.label2id = label2id

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
print("\nDevice used:", device)
print("\n------Model loaded------\n")

# Create data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Define metrics
metric = evaluate.load("seqeval", trust_remote_code=True)
print("\n------Metrics defined------\n")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    # Debugging: Print some predictions and labels
    print("Sample true predictions:", true_predictions[:3])
    print("Sample true labels:", true_labels[:3])
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

# Training arguments
batch_size = 32
logging_steps = len(tokenized_dataset['train']) // batch_size
epochs = 2

training_args = TrainingArguments(
    output_dir="./results_MCN2test_en",
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    disable_tqdm=False,
    logging_steps=logging_steps
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("\n------Training started------\n")

# Fine-tune on GPU
trainer.train()
print("\n------Training ended------\n")

# Evaluate on GPU
trainer.evaluate()
print("\n------Evaluation ended------\n")

# Predict on test set
predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
predictions = np.argmax(predictions, axis=2)

true_predictions = [
    [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]
results = metric.compute(predictions=true_predictions, references=true_labels)

print("\nResults:", results)

# Save final results to a text file
with open("final_results_MCN2.txt", "w") as file:
    # Overall metrics
    file.write(f"Overall Precision: {results['overall_precision']}\n")
    file.write(f"Overall Recall: {results['overall_recall']}\n")
    file.write(f"Overall F1: {results['overall_f1']}\n")
    file.write(f"Overall Accuracy: {results['overall_accuracy']}\n\n")
    
    # Per-entity metrics
    for key in results.keys():
        if key not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']:
            file.write(f"Entity: {key}\n")
            file.write(f"\tPrecision: {results[key]['precision']}\n")
            file.write(f"\tRecall: {results[key]['recall']}\n")
            file.write(f"\tF1: {results[key]['f1']}\n")
            file.write(f"\tNumber: {results[key]['number']}\n\n")
    
    # Compute macro-averaged metrics
    per_type_metrics = [v for k, v in results.items() if k not in ['overall_precision', 'overall_recall', 'overall_f1', 'overall_accuracy']]
    
    macro_precision = np.mean([v['precision'] for v in per_type_metrics])
    macro_recall = np.mean([v['recall'] for v in per_type_metrics])
    macro_f1 = np.mean([v['f1'] for v in per_type_metrics])
    
    file.write(f"Macro Precision: {macro_precision}\n")
    file.write(f"Macro Recall: {macro_recall}\n")
    file.write(f"Macro F1: {macro_f1}\n")
