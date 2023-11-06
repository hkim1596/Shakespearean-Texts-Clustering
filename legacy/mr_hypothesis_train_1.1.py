import os
import pandas as pd
import torch
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
from tqdm import tqdm
import logging
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def log_time(start_time, task_name="Task"):
    elapsed_time = time.time() - start_time
    print(f"{task_name} completed in {elapsed_time:.2f} seconds.")

def parse_file(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return [], []

    print(f"Processing file: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content:
            print(f"File {file_path} is empty.")
            return []

        # Parse the XML with BeautifulSoup using the lxml parser
        soup = BeautifulSoup(content, 'lxml-xml')

        # Search for all divs with type 'play' within the parsed content
        play_divs = soup.find_all('div', {'type': 'play'})
        
        text_content = []
        titles = []
        if play_divs:
            for div in play_divs:
                title_tag = div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text(strip=True).replace('\n', ' '))
                else:
                    titles.append("untitled")
                text_content.append(div.get_text(strip=True).replace('\n', ' '))
        else:
            text_div = soup.find('div', {'type': 'text'})
            if text_div:
                title_tag = text_div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text(strip=True).replace('\n', ' '))
                else:
                    titles.append("untitled")
                text_content.append(text_div.get_text(strip=True).replace('\n', ' '))

        if not titles:
            logging.error("No titles found in the file.")
            return [], []

        print(f"Extracted titles: {titles}")
        return titles, text_content

def chunk_text(play_text, tokenizer, max_chunk_length, play_id):
    chunk_counter = 0
    chunks = []
    chunk_play_ids = []

    words = play_text.split()
    current_chunk_words = []
    current_length = 0
    
    pbar = tqdm(total=len(words), desc=f"Tokenizing and chunking - Play: {play_id}")
    
    for word in words:
        encoded_word = tokenizer.encode(word, add_special_tokens=False)
        word_length = len(encoded_word)
        
        if current_length + word_length <= max_chunk_length:
            current_length += word_length
            current_chunk_words.extend([word])
        else:
            chunks.append(' '.join(current_chunk_words))
            chunk_play_ids.append(play_id)
            current_length = word_length
            current_chunk_words = [word]
            chunk_counter += 1
            pbar.set_description(f"Tokenizing and chunking - Play: {play_id} - Chunk {chunk_counter}")
        
        pbar.update(1)
        
    # handle the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
        chunk_play_ids.append(play_id)
    
    pbar.close()
    
    return chunks, chunk_play_ids

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
metadata_df = pd.read_csv(metadata_path)

# Filter the data based on criteria for training
good_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Folio") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
bad_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Bad") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

# [Continue with the processing, training, and evaluation code]
all_play_texts = []
all_play_ids = []
all_play_titles = []

for idx, tcp in enumerate(tqdm(good_texts_df["TCP"])):
    titles, texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    for title, play_text in zip(titles, texts):
        all_play_texts.append(play_text)
        all_play_titles.append(title)  # Store title
        all_play_ids.append(f"good_{title}")

for idx, tcp in enumerate(tqdm(bad_texts_df["TCP"])):
    titles, texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    for title, play_text in zip(titles, texts):
        all_play_texts.append(play_text)
        all_play_titles.append(title)  # Store title
        all_play_ids.append(f"bad_{title}")

log_time(start_time, "File Parsing and Text Extraction")
start_time = time.time()

all_chunks = []
all_chunk_play_ids = []

for play_text, play_id in zip(all_play_texts, all_play_ids):
    chunks, chunk_play_ids = chunk_text(play_text, tokenizer, 500, play_id)
    all_chunks.extend(chunks)
    all_chunk_play_ids.extend(chunk_play_ids)

log_time(start_time, "Splitting Plays into Chunks")
start_time = time.time()

total_chunks = len(all_chunks)
print(f"Total chunks to process: {total_chunks}")

# Step 1: Split data into train and test sets
good_indices = [i for i, id in enumerate(all_chunk_play_ids) if 'good' in id]
bad_indices = [i for i, id in enumerate(all_chunk_play_ids) if 'bad' in id]

good_train_idx, good_test_idx = train_test_split(good_indices, test_size=0.2, random_state=42)
bad_train_idx, bad_test_idx = train_test_split(bad_indices, test_size=0.2, random_state=42)

train_idx = good_train_idx + bad_train_idx
test_idx = good_test_idx + bad_test_idx

train_texts = [all_chunks[i] for i in train_idx]
train_labels = [1 if 'good' in all_chunk_play_ids[i] else 0 for i in train_idx]
train_ids = [all_chunk_play_ids[i] for i in train_idx]

test_texts = [all_chunks[i] for i in test_idx]
test_labels = [1 if 'good' in all_chunk_play_ids[i] else 0 for i in test_idx]
test_ids = [all_chunk_play_ids[i] for i in test_idx]

log_time(start_time, "Splitting data into train and test sets")
start_time = time.time()

# Step 2: Train the model
class PlayDataset:
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PlayDataset(tokenizer, train_texts, train_labels)
test_dataset = PlayDataset(tokenizer, test_texts, test_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",    # <-- Evaluate periodically
    eval_steps=500,                # <-- Evaluate every 500 steps
    save_steps=500,
    save_total_limit=1,
    num_train_epochs=2,
    learning_rate=2e-5,
    output_dir='./results',
    load_best_model_at_end=True,   # <-- Load the best model at the end
    # early_stopping_patience=3      # <-- Stop if no improvement after 3 evaluations
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

log_time(start_time, "Trainning")
start_time = time.time()

# Step 3: Evaluate the model
predictions = trainer.predict(test_dataset).predictions.argmax(-1)
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(test_labels, predictions))

log_time(start_time, "Evaluating")
start_time = time.time()

# Save the trained model
model.save_pretrained("./trained_model")

# Save results to specified JSON files
results = [{"play_title": title, "play_id": id, "prediction": pred, "text": text} for title, id, pred, text in zip(all_play_titles, test_ids, predictions, test_texts)]
with open('prediction_results_on_test_sets.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, cls=NumpyEncoder, ensure_ascii=False)

train_data = [{"play_title": title, "play_id": id, "text": text} for title, id, text in zip(all_play_titles, train_ids, train_texts)]
with open('train_sets.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, cls=NumpyEncoder, ensure_ascii=False)
