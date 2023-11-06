import os
import pandas as pd
import torch
from bs4 import BeautifulSoup
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json
import logging
import time
from tqdm import tqdm
import numpy as np
import warnings

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

def chunk_text(play_text, tokenizer, max_chunk_length, play_title):
    chunk_counter = 0
    chunks = []
    chunk_play_ids = []

    words = play_text.split()
    current_chunk_words = []
    current_length = 0
    
    pbar = tqdm(total=len(words), desc=f"Tokenizing and chunking - Play: {play_title}")
    
    for word in words:
        encoded_word = tokenizer.encode(word, add_special_tokens=False)
        word_length = len(encoded_word)
        
        if current_length + word_length <= max_chunk_length:
            current_length += word_length
            current_chunk_words.extend([word])
        else:
            chunks.append(' '.join(current_chunk_words))
            chunk_play_ids.append(play_title)
            current_length = word_length
            current_chunk_words = [word]
            chunk_counter += 1
            pbar.set_description(f"Tokenizing and chunking - Play: {play_title} - Chunk {chunk_counter}")
        
        pbar.update(1)
        
    # handle the last chunk
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
        chunk_play_ids.append(play_title)
    
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

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata_All.csv")
# metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")

metadata_df = pd.read_csv(metadata_path)

# Filter the data based on criteria for predictions
non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

all_play_texts = []
all_play_ids = []
all_play_titles = []

for idx, tcp in enumerate(tqdm(non_texts_df["TCP"])):
    titles, texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    if not titles or not texts:
        logging.warning(f"No titles or texts found for TCP: {tcp}")
        continue
    for title, play_text in zip(titles, texts):
        all_play_texts.append(play_text)
        all_play_titles.append(title)  # Store title
        all_play_ids.append(f"non_{title}")  # Ensure the 'non' prefix for IDs

logging.info(f"Total plays processed: {len(all_play_texts)}")
logging.info(f"First few play titles: {all_play_titles[:5]}")

all_play_chunks = []
all_chunk_play_titles = []

for play_text, play_title in zip(all_play_texts, all_play_titles):
    chunks, chunk_play_titles = chunk_text(play_text, tokenizer, 500, play_title)
    all_play_chunks.extend(chunks)
    all_chunk_play_titles.extend([f"non_{title}" for title in chunk_play_titles])

logging.info(f"Total chunks created: {len(all_play_chunks)}")
logging.info(f"First few chunk titles: {all_chunk_play_titles[:5]}")

# Load the trained model
model = BertForSequenceClassification.from_pretrained("./trained_model")
# model = DistilBertForSequenceClassification.from_pretrained("./trained_model")

trainer = Trainer(model=model)

# Step 4: Predict on non-text chunks
non_indices = [i for i, title in enumerate(all_chunk_play_titles) if 'non' in title]
non_texts = [all_play_chunks[i] for i in non_indices]
if not non_texts:
    raise ValueError("No text chunks to tokenize. Ensure the input data is correct.")
non_dataset = PlayDataset(tokenizer, non_texts, [0]*len(non_texts))  # dummy labels

non_raw_predictions = trainer.predict(non_dataset).predictions
non_class_predictions = non_raw_predictions.argmax(-1)

# Aggregate chunk predictions to predict play
logging.info(f"Number of non_predictions: {len(non_class_predictions)}")

play_predictions = {}
play_raw_scores = {}
for i, play_title in enumerate(all_chunk_play_titles):
    if 'non' not in play_title:
        continue
    if play_title not in play_predictions:
        play_predictions[play_title] = []
        play_raw_scores[play_title] = []

    play_predictions[play_title].append(non_class_predictions[i])
    play_raw_scores[play_title].append(non_raw_predictions[i].tolist())

avg_play_raw_scores = {}
for play_title, raw_scores in play_raw_scores.items():
    avg_scores = np.mean(raw_scores, axis=0)
    avg_play_raw_scores[play_title] = avg_scores.tolist()

label_map = {0: "bad", 1: "good"}

final_play_predictions = {}
for play_title, preds in play_predictions.items():
    avg_prediction = round(sum(preds)/len(preds))
    final_play_predictions[play_title] = label_map[avg_prediction]

# Save results to specified JSON files
# Using final_play_predictions for JSON
non_results = [{"play_title": title.replace("non_", ""), 
                "prediction": final_play_predictions[title], 
                "avg_raw_scores": avg_play_raw_scores[title]} for title in final_play_predictions.keys()]

with open('prediction_results_on_non_shakespearean_texts.json', 'w', encoding='utf-8') as f:
    json.dump(non_results, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)