import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch.nn.functional as F  # Add this import

def clean_tag(tag):
    # Ensure tags are in the correct format
    if tag.count('-') > 1:
        prefix, entity = tag.split('-', 1)
        tag = f"{prefix}-{entity.replace('-', '')}"
    return tag
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
def read_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        names = [name.strip().lower() for name in file.readlines()]
    return names
class NERDataset(Dataset):
    def __init__(self, sentences, tags, tokenizer, max_len, tag2idx):
        self.sentences = sentences
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        word_labels = self.tags[idx]
        encoding = self.tokenizer(sentence, is_split_into_words=True, return_offsets_mapping=True, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        labels = [self.tag2idx['O']] * self.max_len  # Initialize labels with "O"
        offsets = encoding['offset_mapping'].squeeze().tolist()  # Get the offsets
        encoding.pop('offset_mapping')  # Remove offsets, not needed for model input

        idx = 0
        for i, (start, end) in enumerate(offsets):
            if start == end:  # Special tokens
                labels[i] = self.tag2idx['O']
            elif start == 0:  # Start of a new word
                if idx < len(word_labels):
                    labels[i] = self.tag2idx[word_labels[idx]]
                else:
                    labels[i] = self.tag2idx['O']
                idx += 1
            else:  # Subtoken of a word
                labels[i] = -100  # PyTorch's convention to ignore these tokens in loss computation

        item = {key: val.squeeze() for key, val in encoding.items()}  # Remove batch dimension
        item['labels'] = torch.tensor(labels)
        return item
    
    
def evaluate(model, dataloader, device, tag2idx, idx2tag):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)

            # Collect the predictions and true labels for calculating F1 score and accuracy
            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(batch['labels'].cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)

    # Flatten the lists to calculate metrics
    all_preds_flat = [p for preds in all_preds for p in preds]
    all_labels_flat = [l for labels in all_labels for l in labels]

    # Remove padding tokens, the label 0 (O), and -100 for accuracy and F1 calculation
    true_preds = [pred for pred, label in zip(all_preds_flat, all_labels_flat) if label != tag2idx['PAD'] and label != tag2idx['O'] and label != -100]
    true_labels = [label for label in all_labels_flat if label != tag2idx['PAD'] and label != tag2idx['O'] and label != -100]

    # Map indices back to tags
    true_preds_tags = [idx2tag[pred] for pred in true_preds]
    true_labels_tags = [idx2tag[label] for label in true_labels]

    # Get the list of unique tags in the dataset (excluding PAD and O)
    unique_tags = [tag for tag in tag2idx if tag != 'PAD' and tag != 'O']

    f1 = f1_score(true_labels_tags, true_preds_tags, average='weighted')
    accuracy = accuracy_score(true_labels_tags, true_preds_tags)

    print(f'Average Loss: {avg_loss}')
    print(f'F1 Score (excluding PAD and O): {f1}')
    print(f'Accuracy (excluding PAD and O): {accuracy}')
    print(classification_report(true_labels_tags, true_preds_tags, labels=unique_tags, target_names=unique_tags))

    return avg_loss, f1, accuracy, true_labels_tags, true_preds_tags


MAX_LEN = 174
BATCH_SIZE = 64
EPOCHS = 5
MODEL_NAME = 'bert-base-uncased'
MODEL_PATH = 'ner_model_from_final'  # Path to save the model

character_names = read_names('./scraping_res/character_names.txt')
location_names = read_names('./scraping_res/location_names.txt')
organization_names = read_names('./scraping_res/organization_names.txt')
all_names = character_names + location_names + organization_names

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
num_added_toks = tokenizer.add_tokens(all_names)
train_tokens, train_tags = read_data("./tagged_sentences_train.iob2")
tag_values = list(set(tag for doc in train_tags for tag in doc))
tag_values.append("PAD")
tag2idx = {tag: idx for idx, tag in enumerate(tag_values)}
idx2tag = dict([(value, key) for key, value in tag2idx.items()])
print(len(tag2idx))
set(tag for doc in train_tags for tag in doc)
print(set(tag for doc in train_tags for tag in doc))
print(len(tag_values))
test_tokens, test_tags = read_data("./tagged_sentences_test.iob2")

# Combine train and test tags to create a comprehensive tag set
all_tags = set(tag for doc in train_tags for tag in doc).union(set(tag for doc in test_tags for tag in doc))
all_tags.add("PAD")  # Add the PAD tag
print(f"All tags: {all_tags}")
train_data = NERDataset(train_tokens, train_tags, tokenizer, MAX_LEN, tag2idx)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag_values))

if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)

model.train()
print(f'Using device: {device}')
for epoch in range(EPOCHS):
    print(f'Using device: {device}')
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

            
# Step 1: Load the test data
test_tokens, test_tags = read_data("./tagged_sentences_test.iob2")

# Step 2: Create a DataLoader for the test data
test_data = NERDataset(test_tokens, test_tags, tokenizer, MAX_LEN, tag2idx)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

from sklearn.metrics import f1_score, accuracy_score, classification_report


avg_loss, f1, accuracy, true_labels_tags, true_preds_tags = evaluate(model, test_loader, device, tag2idx, idx2tag)


print(avg_loss)
print(f1)
print(accuracy)
print(true_labels_tags)
print(true_preds_tags)

# Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Load the model before evaluation
model.load_state_dict(torch.load(MODEL_PATH))
print(f"Model loaded from {MODEL_PATH}")

test_data = NERDataset(test_tokens, test_tags, tokenizer, MAX_LEN, tag2idx)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

avg_loss, f1, accuracy, true_labels_tags, true_preds_tags = evaluate(model, test_loader, device, tag2idx, idx2tag)

print(f"Final Test Loss: {avg_loss}")
print(f"Final Test F1 Score: {f1}")
print(f"Final Test Accuracy: {accuracy}")
