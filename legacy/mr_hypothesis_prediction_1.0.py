import os
import pandas as pd
import torch
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import json
import logging
import time
from tqdm import tqdm
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

        # Extracting all text from the parsed content
        text_content = soup.get_text()

        # Search for all divs with type 'play' within the parsed content
        play_divs = soup.find_all('div', {'type': 'play'})

        titles = []
        if play_divs:
            for div in play_divs:
                title_tag = div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text())
                else:
                    titles.append("untitled")
        else:
            text_div = soup.find('div', {'type': 'text'})
            if text_div:
                title_tag = text_div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text())
                else:
                    titles.append("untitled")

        if not titles:
            logging.error("No titles found in the file.")
            return [], []

        print(f"Extracted titles: {titles}")
        # print(f"Extracted text:\n{text_content}")
        return titles, [text_content] * len(titles)

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
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
metadata_df = pd.read_csv(metadata_path)

# Filter the data based on criteria for predictions
non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

all_play_texts = []
all_play_ids = []
all_play_titles = []

for idx, tcp in enumerate(tqdm(non_texts_df["TCP"])):
    titles, texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    for title, play_text in zip(titles, texts):
        all_play_texts.append(play_text)
        all_play_titles.append(title)  # Store title
        all_play_ids.append(f"non_{title}")

all_chunks = []
all_chunk_play_ids = []

for play_text, play_id in zip(all_play_texts, all_play_ids):
    chunks, chunk_play_ids = chunk_text(play_text, tokenizer, 500, play_id)
    all_chunks.extend(chunks)
    all_chunk_play_ids.extend(chunk_play_ids)

# Load the trained model
model = DistilBertForSequenceClassification.from_pretrained("./trained_model")
trainer = Trainer(model=model)


# Step 4: Predict on non-text chunks
non_indices = [i for i, id in enumerate(all_chunk_play_ids) if 'non' in id]
non_texts = [all_chunks[i] for i in non_indices]
non_dataset = PlayDataset(tokenizer, non_texts, [0]*len(non_texts))  # dummy labels

non_predictions = trainer.predict(non_dataset).predictions.argmax(-1)

# Aggregate chunk predictions to predict play
logging.info(f"Number of non_oredictions: {len(non_predictions)}")

play_predictions = {}
for i, play_id in enumerate(all_chunk_play_ids):
    if 'non' not in play_id:
        continue
    if play_id not in play_predictions:
        play_predictions[play_id] = []
    
    if i < len(non_predictions):
        play_predictions[play_id].append(non_predictions[i])
    else:
        print(f"Index {i} out of bounds for non_predictions!")
    play_predictions[play_id].append(non_predictions[i])

for play_id, preds in play_predictions.items():
    play_predictions[play_id] = round(sum(preds)/len(preds))



# Save results to specified JSON files
non_results = [{"play_title": title, "play_id": id, "prediction": pred, "text": text} for title, id, pred, text in zip(all_play_titles, all_chunk_play_ids, non_predictions, non_texts)]
with open('prediction_results_on_non_shakespearean_texts.json', 'w') as f:
    json.dump(non_results, f)
