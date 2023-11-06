import os
import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import logging
import time
import warnings
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def log_time(start, message):
    """Log time taken for a particular process."""
    end = time.time()
    elapsed_time = end - start
    logging.info(f"{message} took {elapsed_time:.2f} seconds")

def save_to_file(filename, content):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content + '\n')

if os.path.exists("results.txt"):
    os.remove("results.txt")

start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
metadata_df = pd.read_csv(metadata_path)

# Filter the data based on criteria
good_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Folio") & 
                            (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
bad_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Bad") & 
                           (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & 
                           (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

def parse_file(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return []
    
    # PRINT THE FILENAME HERE
    print(f"Processing file: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content:
            print(f"File {file_path} is empty.")
            return []

        # Parse the XML with BeautifulSoup using the lxml parser
        soup = BeautifulSoup(content, 'lxml-xml')
        plays = soup.find_all('div', {'type': 'play'})
        texts = soup.find_all('div', {'type': 'text'})
        return [extract_text(div).strip() for div in plays+texts]
    return []

def extract_text(play):
    """Extract all the text content from a play node."""
    return ' '.join(play.stripped_strings)

# def extract_text(play):
#     """Extract the text content from a play node."""
#     text_list = []
#     for child in play.children:
#         if isinstance(child, NavigableString):
#             text_list.append(child)
#         elif child.name in ["sp", "p", "stage", "head"]:
#             for sub_child in child.children:
#                 if isinstance(sub_child, NavigableString):
#                     text_list.append(sub_child)
#                 elif sub_child.name in ["l", "speaker", "hi"]:
#                     text_list.append(sub_child.get_text())
#     return ' '.join(text_list)

def chunk_text(text, tokenizer, max_length):
    """
    Split the text into smaller chunks of max_length, tokenized length.
    """
    words = text.split()
    current_chunk = []
    chunks = []
    for word in words:
        # Tokenize the current word
        current_word_tokens = tokenizer.tokenize(word)
        
        # If adding the current word exceeds the max length, store the current chunk and reset
        if len(tokenizer.encode(current_chunk + current_word_tokens, add_special_tokens=True)) <= max_length:
            current_chunk.extend(current_word_tokens)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = current_word_tokens

    # Append any remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")

# For good_texts (reiterating for clarity):
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

good_play_texts = [play_text for tcp in tqdm(good_texts_df["TCP"]) 
                   for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
bad_play_texts = [play_text for tcp in tqdm(bad_texts_df["TCP"]) 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
non_play_texts = [play_text for tcp in tqdm(non_texts_df["TCP"]) 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]

log_time(start_time, "File Parsing and Text Extraction")


# Splitting function for encodings
def train_test_split_encodings(encodings, test_size=0.2, random_state=None):
    input_ids_train, input_ids_test = train_test_split(encodings["input_ids"], test_size=test_size, random_state=random_state)
    attention_mask_train, attention_mask_test = train_test_split(encodings["attention_mask"], test_size=test_size, random_state=random_state)
    
    return {"input_ids": input_ids_train, "attention_mask": attention_mask_train}, {"input_ids": input_ids_test, "attention_mask": attention_mask_test}

good_train_texts, good_test_texts = train_test_split(good_play_texts, test_size=0.2, random_state=42)
bad_train_texts, bad_test_texts = train_test_split(bad_play_texts, test_size=0.2, random_state=42)
non_train_texts, non_test_texts = train_test_split(non_play_texts, test_size=0.2, random_state=42)

good_train_chunks = [chunk for play_text in good_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
good_test_chunks = [chunk for play_text in good_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

bad_train_chunks = [chunk for play_text in bad_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
bad_test_chunks = [chunk for play_text in bad_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

non_train_chunks = [chunk for play_text in non_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
non_test_chunks = [chunk for play_text in non_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

start_time = time.time()

# Tokenization
BATCH_SIZE = 1000

def batch_tokenize(texts, tokenizer, max_length=512):
    total_size = len(texts)
    tokenized_batches = []
    for i in tqdm(range(0, total_size, BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        tokenized = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
        tokenized_batches.append(tokenized)
    # Combining the batches
    return {
        "input_ids": torch.cat([batch["input_ids"] for batch in tokenized_batches], dim=0),
        "attention_mask": torch.cat([batch["attention_mask"] for batch in tokenized_batches], dim=0)
    }

start_time = time.time()

max_length = 512  # or any other size that fits your needs

good_train_texts_encoded = batch_tokenize(good_train_chunks, tokenizer, max_length)
good_test_texts_encoded = batch_tokenize(good_test_chunks, tokenizer, max_length)
bad_train_texts_encoded = batch_tokenize(bad_train_chunks, tokenizer, max_length)
bad_test_texts_encoded = batch_tokenize(bad_test_chunks, tokenizer, max_length)
non_train_texts_encoded = batch_tokenize(non_train_chunks, tokenizer, max_length)
non_test_texts_encoded = batch_tokenize(non_test_chunks, tokenizer, max_length)

log_time(start_time, "Tokenization")

# Dataset Creation
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

if torch.backends.mps.is_available():
    torch.backends.mps.is_built()

# Training Procedure
def train_and_evaluate(train_encodings, train_labels, test_encodings, test_labels, test_play_texts, model_save_path):
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    

    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10,
        evaluation_strategy="steps",
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        eval_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    start_time = time.time()  # Start timing here

    trainer.train()

    log_time(start_time, "Model Training")  # End timing here

    results = trainer.evaluate()
    print(f"Results for {trainer.args.output_dir}: {results}")
    save_to_file('results.txt', f"Results for {trainer.args.output_dir}: {results}")
    log_time(start_time, "Model Training")  # End timing here

    # Confusion Matrix
    start_time = time.time()  # Start timing here

    predictions = trainer.predict(test_dataset).predictions.argmax(axis=1)

    log_time(start_time, "Model Prediction")  # End timing here

    labels = [item['labels'] for item in test_dataset]
    print(confusion_matrix(labels, predictions))

     # Evaluate on entire plays
    cm = evaluate_model_on_plays(model, tokenizer, test_play_texts, test_labels, 500)
    print(cm)
    save_to_file('results.txt', str(cm))


    # Save and Visualize Model Predictions
    for i, (input_ids, prediction) in enumerate(zip(test_dataset.encodings['input_ids'], predictions)):
        original_text = tokenizer.decode(input_ids)
        predicted_label = "Good" if prediction == 0 else "Bad" if prediction == 1 else "Non"
        save_to_file('results.txt', f"Sample {i}:\nText: {original_text}\nPredicted: {predicted_label}\n")
        
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

def combine_encodings(encodings_list):
    input_ids_lengths = [e["input_ids"].size(1) for e in encodings_list]
    attention_mask_lengths = [e["attention_mask"].size(1) for e in encodings_list]
    
    unique_input_ids_lengths = set(input_ids_lengths)
    unique_attention_mask_lengths = set(attention_mask_lengths)
    
    if len(unique_input_ids_lengths) > 1 or len(unique_attention_mask_lengths) > 1:
        raise ValueError(f"Mismatch in tensor sizes. Input IDs lengths: {unique_input_ids_lengths}. Attention mask lengths: {unique_attention_mask_lengths}.")
    
    return {
        "input_ids": torch.cat([e["input_ids"] for e in encodings_list], dim=0),
        "attention_mask": torch.cat([e["attention_mask"] for e in encodings_list], dim=0),
    }

def predict_play_chunks(model, tokenizer, play_text, max_chunk_size):
    chunks = chunk_text(play_text, tokenizer, max_chunk_size)
    encodings = tokenizer(chunks, padding="max_length", truncation=True, return_tensors="pt", max_length=max_chunk_size)
    inputs = {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.mean(dim=0).argmax().item()

def evaluate_model_on_plays(model, tokenizer, play_texts, true_labels, max_chunk_size):
    predictions = [predict_play_chunks(model, tokenizer, play, max_chunk_size) for play in play_texts]
    return confusion_matrix(true_labels, predictions)


def get_embeddings(text_encodings):
    with torch.no_grad():
        inputs = text_encodings["input_ids"].to("cuda" if torch.cuda.is_available() else "cpu")
        masks = text_encodings["attention_mask"].to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.distilbert(inputs, attention_mask=masks)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()

def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings(embeddings, labels, title="Texts Clustering Visualization", save_path="clustering_visualization.png"):
    plt.figure(figsize=(10, 6))
    for label, color in [(0, 'blue'), (1, 'red'), (2, 'green')]:  # 0-Good, 1-Bad, 2-Non_Texts
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], c=color, label=label)
    plt.legend()
    plt.title(title)
    
    # Save the figure to the specified path
    plt.savefig(save_path)
    plt.close()



# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")

good_play_texts = [play_text for tcp in tqdm(good_texts_df["TCP"]) 
                   for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
bad_play_texts = [play_text for tcp in tqdm(bad_texts_df["TCP"]) 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
non_play_texts = [play_text for tcp in tqdm(non_texts_df["TCP"]) 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]

good_train_chunks = [chunk for play_text in good_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
good_test_chunks = [chunk for play_text in good_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

bad_train_chunks = [chunk for play_text in bad_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
bad_test_chunks = [chunk for play_text in bad_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

non_train_chunks = [chunk for play_text in non_train_texts 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
non_test_chunks = [chunk for play_text in non_test_texts 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

# Generate separate lists for good, bad, and non-text chunks
list_of_good_texts = good_train_chunks + good_test_chunks
list_of_bad_texts = bad_train_chunks + bad_test_chunks
non_texts = non_train_chunks + non_test_chunks

good_encodings = tokenizer(list_of_good_texts, truncation=True, padding=True, return_tensors="pt")
bad_encodings = tokenizer(list_of_bad_texts, truncation=True, padding=True, return_tensors="pt")
non_texts_encodings = tokenizer(non_texts, truncation=True, padding=True, return_tensors="pt")

good_embeddings = get_embeddings(good_encodings)
bad_embeddings = get_embeddings(bad_encodings)
non_texts_embeddings = get_embeddings(non_texts_encodings)

all_embeddings = np.vstack([good_embeddings, bad_embeddings, non_texts_embeddings])
reduced_embeddings = reduce_dimensions(all_embeddings)

labels = [0] * len(good_embeddings) + [1] * len(bad_embeddings) + [2] * len(non_texts_embeddings)

# Call the modified function to plot and save the image
plot_embeddings(reduced_embeddings, labels)
