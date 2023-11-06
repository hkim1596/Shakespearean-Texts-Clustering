import os
import pandas as pd
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import logging
import time
import warnings
from sklearn.manifold import TSNE
import numpy as np
import plotly.graph_objects as go

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
        logging.error(f"File {file_path} does not exist.")
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


# Tokenization
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

if torch.cuda.is_available():
    model = model.to("cuda")

BATCH_SIZE = 64  # Define a suitable batch size depending on your GPU memory

def get_embeddings(text_encodings):
    embeddings_list = []

    input_ids = text_encodings["input_ids"]
    masks = text_encodings["attention_mask"]
    
    total_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(total_batches), desc="Getting embeddings"):
        batch_input_ids = input_ids[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        batch_masks = masks[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        
        with torch.no_grad():
            batch_input_ids = batch_input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
            batch_masks = batch_masks.to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.distilbert(batch_input_ids, attention_mask=batch_masks)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings_list.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings_list)
    return embeddings

def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=0)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_embeddings_interactive(embeddings, labels, title="Texts Clustering Visualization_Chunks", save_path="clustering_visualization_Chunks.html"):
    # Colors corresponding to labels: 0-Good, 1-Bad, 2-Non_Texts
    colors = ['blue' if label == "Shakespeare's Folio" else 'red' if label == "Shakespeare's Bad Quartos" else 'green' for label in labels]
    
    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=colors, 
            opacity=0.6,
        ),
        text=labels  # This will display the label when you hover over a point
    ))
    
    # Set the layout of the plot
    fig.update_layout(title=title,
                      xaxis=dict(title='Dimension 1'),
                      yaxis=dict(title='Dimension 2'),
                      legend_title_text='Text Type',
                      hovermode='closest')
    
    # Show the plot
    fig.show()

    # Save the figure to the specified path
    fig.write_html(save_path)

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")

good_play_texts = [play_text for tcp in tqdm(good_texts_df["TCP"], desc="Processing Good Texts") 
                   for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
bad_play_texts = [play_text for tcp in tqdm(bad_texts_df["TCP"], desc="Processing Bad Texts") 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
non_play_texts = [play_text for tcp in tqdm(non_texts_df["TCP"], desc="Processing Non-Texts") 
                  for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]

log_time(start_time, "File Parsing and Text Extraction")
start_time = time.time()

good_play_chunks = [chunk for play_text in tqdm(good_play_texts, desc="Chunking Good Texts") 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
bad_play_chunks = [chunk for play_text in tqdm(bad_play_texts, desc="Chunking Bad Texts") 
                     for chunk in chunk_text(play_text, tokenizer, 500)]
non_play_chunks = [chunk for play_text in tqdm(non_play_texts, desc="Chunking Non-Texts") 
                    for chunk in chunk_text(play_text, tokenizer, 500)]

log_time(start_time, "Chunking")
start_time = time.time()

good_encodings = tokenizer([text for text in tqdm(good_play_chunks, desc="Tokenizing Good Texts")], truncation=True, padding=True, return_tensors="pt")
bad_encodings = tokenizer([text for text in tqdm(bad_play_chunks, desc="Tokenizing Bad Texts")], truncation=True, padding=True, return_tensors="pt")
non_texts_encodings = tokenizer([text for text in tqdm(non_play_chunks, desc="Tokenizing Non-Texts")], truncation=True, padding=True, return_tensors="pt")

log_time(start_time, "Tokenizing")
start_time = time.time()

good_embeddings = get_embeddings(good_encodings)
bad_embeddings = get_embeddings(bad_encodings)
non_texts_embeddings = get_embeddings(non_texts_encodings)

log_time(start_time, "Getting Embeddings")
start_time = time.time()

all_embeddings = np.vstack([good_embeddings, bad_embeddings, non_texts_embeddings])
reduced_embeddings = reduce_dimensions(all_embeddings)

# labels = [0] * len(good_embeddings) + [1] * len(bad_embeddings) + [2] * len(non_texts_embeddings)  # Numeric labels
labels = ["Shakespeare's Folio"] * len(good_embeddings) + ["Shakespeare's Bad Quartos"] * len(bad_embeddings) + ["Non-Shakespearean Plays"] * len(non_texts_embeddings)

# Call the modified function to plot and save the image
plot_embeddings_interactive(reduced_embeddings, labels)
log_time(start_time, "Plotting")
