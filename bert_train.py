import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from sklearn.metrics import classification_report
import numpy as np
import time  # Added to measure time

MAX_LEN = 174
BATCH_SIZE = 64
EPOCHS = 4
MODEL_NAME = 'bert-base-uncased'

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
                label.append(parts[2])
    if sentence:
        sentences.append(sentence)
        labels.append(label)
    return sentences, labels

def read_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        names = [name.strip().lower() for name in file.readlines()]
    return names

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

        # Skip tokenization and directly use the original sentence
        input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab['[UNK]']) for token in sentence]
        input_ids = input_ids[:self.max_len] + [0] * (self.max_len - len(input_ids))  # Padding to max_len

        labels = [self.tag2idx.get(label, self.tag2idx['O']) for label in word_labels]
        labels = labels[:self.max_len] + [self.tag2idx['O']] * (self.max_len - len(labels))  # Padding to max_len

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)

        return {'input_ids': input_ids, 'attention_mask': (input_ids != 0).long(), 'labels': labels}

train_data = NERDataset(train_tokens, train_tags, tokenizer, MAX_LEN, tag2idx)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag_values))

if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Print the device being used (CPU or GPU)
model.to(device)  # Move the model to the selected device
optimizer = AdamW(model.parameters(), lr=3e-5)

model.train()

# Start the timer before training
start_time = time.time()  # Record the start time

for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to the selected device
        outputs = model(**batch)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

# Stop the timer after training
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time
print(f"Training time: {elapsed_time:.4f} seconds")  # Print the training time

def test_sentence(sentence, model, tokenizer, device, idx2tag, max_len=MAX_LEN):
    tokens = sentence.lower().split()
    input_ids = [tokenizer.vocab.get(token, tokenizer.vocab['[UNK]']) for token in tokens]
    input_ids = input_ids[:max_len] + [0] * (max_len - len(input_ids))  # Padding to max_len
    input_ids = torch.tensor([input_ids], device=device)

    attention_mask = (input_ids != 0).long()
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0]

    new_labels = [idx2tag[prediction.item()] for prediction in predictions if prediction.item() != tokenizer.vocab['[PAD]']]

    result = []
    for token, label in zip(tokens, new_labels):
        result.append((token, label))

    return result

sentence = "Frodo Baggins traveled to Rivendell with Aragorn."
result = test_sentence(sentence, model, tokenizer, device, idx2tag)

for token, label in result:
    print(f"{token}: {label}")
