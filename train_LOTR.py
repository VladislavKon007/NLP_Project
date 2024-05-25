import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seqeval.metrics import f1_score, accuracy_score
from seqeval.metrics import classification_report as seqeval_classification_report    
from collections import defaultdict


MAX_LEN = 174
BATCH_SIZE = 64
EPOCHS = 7
MAX_GRAD_NORM = 5
MODEL_NAME = 'bert-base-uncased'
from torch import cuda


# Data Reading and Preprocessing Functions

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
# Check if CUDA is available
if torch.cuda.is_available():
    # Get the number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    
    # Get the name and other details of each GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
else:
    print("CUDA is not available. Running on CPU.")

class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")

        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [labels_to_ids[label] for label in word_labels]

        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                encoded_labels[idx] = labels[i]
                i += 1

        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len

def read_data(file_path):
    sentences, labels = [], []
    sentence, label = [], []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):
                continue
            elif line == "\n":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            else:
                parts = line.strip().split("\t")
                sentence.append(parts[1].lower())  # Convert the token to lowercase before appending
                label.append(clean_tag(parts[2]))
    if sentence:
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def clean_tag(tag):
    if tag.count('-') > 1:
        prefix, entity = tag.split('-', 1)
        tag = f"{prefix}-{entity.replace('-', '')}"
    return tag

def train_model(training_set, model, optimizer):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0

    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        labels = batch['labels'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        tr_logits = outputs.logits

        tr_loss += loss.item()
        nb_tr_steps += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    return epoch_loss

train_tokens, train_tags = read_data("./tagged_sentences_train.iob2")
test_tokens, test_tags = read_data("./tagged_sentences_test.iob2")
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

data = {'sentence': [" ".join(sentence) for sentence in train_tokens],
        'word_labels': [",".join(tags) for tags in train_tags]}

df = pd.DataFrame(data)

data_test = {'sentence': [" ".join(sentence) for sentence in test_tokens],
             'word_labels': [",".join(tags) for tags in test_tags]}

df_test = pd.DataFrame(data_test)

# Initialize a dictionary to hold the counts
tag_counts = defaultdict(int)

# Iterate through each list in test_tags and count the occurrences of each tag
for sentence in test_tags:
    for tag in sentence:
        tag_counts[tag] += 1

# Convert the defaultdict to a regular dictionary for easier printing
tag_counts = dict(tag_counts)

# Print the counts for each tag
for tag, count in tag_counts.items():
    print(f"{tag}: {count}")

# Create mappings
all_tags = [tag for tags in df['word_labels'] for tag in tags.split(",")]
unique_tags = set(all_tags)
labels_to_ids = {k: v for v, k in enumerate(unique_tags)}
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# Display the mappings
print("labels_to_ids:", labels_to_ids)
print("ids_to_labels:", ids_to_labels)

# Create training and testing datasets
training_set = dataset(df, tokenizer, MAX_LEN)
testing_set = dataset(df_test, tokenizer, MAX_LEN)

test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
testing_loader = DataLoader(testing_set, **test_params)

# Function to count tag occurrences
def count_tags(tags_list):
    tag_counts = defaultdict(int)
    for sentence in tags_list:
        for tag in sentence:
            tag_counts[tag] += 1
    return tag_counts

# Count initial tag occurrences in test_tags
initial_tag_counts = count_tags(test_tags)
print("Initial tag counts in test_tags:", dict(initial_tag_counts))

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            # Forward pass
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            eval_logits = outputs.logits

            eval_loss += loss.item()
            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # Compute evaluation accuracy
            active_logits = eval_logits.view(-1, model.config.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            for i in range(labels.size(0)):
                label = labels[i]
                pred = flattened_predictions.view(labels.size(0), labels.size(1))[i]

                active_accuracy = label != -100  # shape (seq_len,)
                label = torch.masked_select(label, active_accuracy)
                pred = torch.masked_select(pred, active_accuracy)

                eval_labels.append([ids_to_labels[id.item()] for id in label])
                eval_preds.append([ids_to_labels[id.item()] for id in pred])

                tmp_eval_accuracy = accuracy_score(label.cpu().numpy(), pred.cpu().numpy())
                eval_accuracy += tmp_eval_accuracy

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = accuracy_score(eval_labels, eval_preds)
    F1_score = f1_score(eval_labels, eval_preds)
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")
    print(f"F1 Score: {F1_score}")
    report = seqeval_classification_report(eval_labels, eval_preds, output_dict=True)
    print(report)
    
    return eval_loss, eval_accuracy, F1_score, report


# Train and evaluate the model on the entire dataset
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_to_ids))
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-5)

# Training the model
for epoch in range(EPOCHS):
    train_loss = train_model(training_set, model, optimizer)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss}")

# Evaluating the model
eval_loss, eval_accuracy, f1_score, eval_report = valid(model, testing_loader)
print(f"Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")
print(eval_report)


# Display the evaluation metrics in a DataFrame
metrics = {
    "eval_loss": eval_loss,
    "accuracy": eval_accuracy,
    "f1_score": f1_score,
    "report": eval_report
}
metrics_df = pd.DataFrame([metrics])
print(metrics_df)

# Flatten the classification report for easier viewing
flat_reports = []
for label, scores in eval_report.items():
    flat_reports.append({
        "label": label,
        "precision": scores["precision"],
        "recall": scores["recall"],
        "f1-score": scores["f1-score"],
        "support": scores["support"]
    })

reports_df = pd.DataFrame(flat_reports)
print(reports_df)


model.save_pretrained("curve_train")
tokenizer.save_pretrained('curve_tokenizer')