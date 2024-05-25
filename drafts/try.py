from itertools import product
from torch.optim import SGD
import json
from datetime import datetime
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, AdamWeightDecay
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import torch.nn.functional as F  # Add this import


MODEL_NAME = 'bert-base-cased'
MAX_LEN = 174


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hyperparameters = {
    "learning_rate": [1e-5, 3e-5, 5e-5],
    "batch_size": [16, 32, 64],
    "epochs": [2, 3, 5],
    "optimizer": ["AdamW", "AdamWeightDecay", "AdamP", "SGD1", "SGD2"]
}


def initialize_model_and_dataloader(batch_size, learning_rate, optimizer_type):
  character_names = read_names('./scraping_res/character_names.txt')
  location_names = read_names('./scraping_res/location_names.txt')
  organization_names = read_names('./scraping_res/organization_names.txt')

  all_names = character_names + location_names + organization_names

  tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
  num_added_toks = tokenizer.add_tokens(all_names)

  train_tokens, train_tags = read_data("./tagged_sentences_train.iob2")
  train_tokens, train_tags = tokenize_and_preserve_labels(train_tokens, train_tags, tokenizer)

  tag_values = list(set(tag for doc in train_tags for tag in doc))
  tag_values.append("PAD")
  tag2idx = {tag: idx for idx, tag in enumerate(tag_values)}
  idx2tag = dict([(value, key) for key, value in tag2idx.items()])

  train_data = NERDataset(train_tokens, train_tags, tokenizer, MAX_LEN, tag2idx)
  train_loader = DataLoader(train_data, batch_size=batch_size)

  model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(tag_values))

  if optimizer_type == "AdamW":
    optimizer = AdamW(model.parameters(), lr=learning_rate)
  elif optimizer_type == "AdamWeightDecay":
    optimizer = AdamWeightDecay(model.parameters(), lr=learning_rate)
  elif optimizer_type == "AdamP":
    optimizer = AdamP(model.parameters(), lr=learning_rate)
  elif optimizer_type == "SGD1":
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  elif optimizer_type == "SGD2":
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.5)
  else:
    raise ValueError("Invalid optimizer type")

  if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))

  return model, optimizer, train_loader, tokenizer, idx2tag



results = []

for lr, bs, epochs, optimizer_type in product(hyperparameters["learning_rate"], hyperparameters["batch_size"], hyperparameters["epochs"], hyperparameters["optimizer"]):
  print(f"Testing with Learning Rate: {lr}, Batch Size: {bs}, Epochs: {epochs}, Optimizer: {optimizer_type}")

  model, optimizer, train_loader, tokenizer, idx2tag = initialize_model_and_dataloader(bs, lr, optimizer_type)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  model.train()
  for epoch in range(epochs):
      for step, batch in enumerate(train_loader):
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs[0]
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if step % 10 == 0:
              print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

  test_sentences, _ = read_data('tagged_sentences_test.iob2')
  test_sentences_upd = [' '.join(word) for word in test_sentences]
  model.eval()

  make_predictions(test_sentences, tokenizer, idx2tag)

  process = subprocess.run(["python3", "span_f1.py", "tagged_sentences_test.iob2", "test_predictions.iob2"], capture_output=True, text=True)
  accuracy_score = process.stdout.strip()

  # Store results
  result = {
      "learning_rate": lr,
      "batch_size": bs,
      "epochs": epochs,
      "optimizer": optimizer_type,
      "accuracy": accuracy_score
  }

  results.append(result)

# Save results to a JSON file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"hyperparameter_results_{timestamp}.json"
with open(output_filename, "w") as f:
    json.dump(results, f, indent=4)

print("Hyperparameter testing complete. Results saved to", output_filename)