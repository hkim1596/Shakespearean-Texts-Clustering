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

logging.basicConfig(level=logging.INFO)  # Initializing logging

def log_time(start, message):
    """Log time taken for a particular process."""
    end = time.time()
    elapsed_time = end - start
    logging.info(f"{message} took {elapsed_time:.2f} seconds")

def save_to_file(filename, content):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content + '\n')

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

def batch_tokenize_chunks(chunks, tokenizer, batch_size=64):
    """
    Tokenize chunks in batches to enable tqdm progress bars.
    """
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    encodings_list = []

    for i in tqdm(range(total_batches), desc="Tokenizing chunks"):
        batch_chunks = chunks[i * batch_size: (i + 1) * batch_size]
        encodings = tokenizer(batch_chunks, truncation=True, padding=True, return_tensors="pt")
        encodings_list.append(encodings)
    
    # Concatenate all batched encodings to a single encoding
    concatenated_encodings = {
        "input_ids": torch.cat([enc["input_ids"] for enc in encodings_list], dim=0),
        "attention_mask": torch.cat([enc["attention_mask"] for enc in encodings_list], dim=0),
    }

    return concatenated_encodings

def get_embeddings(text_encodings):
    embeddings_list = []

    input_ids = text_encodings["input_ids"]
    masks = text_encodings["attention_mask"]

    total_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE

    # Wrap with tqdm for progress tracking
    for i in tqdm(range(total_batches), desc="Fetching embeddings", total=total_batches):
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

def reduce_dimensions(aggregated_embeddings):
    # Check if the input embeddings are in dictionary format
    if isinstance(aggregated_embeddings, dict):
        # Convert dictionary values (embeddings) to a matrix
        embeddings_matrix = np.array(list(aggregated_embeddings.values()))
    else:
        embeddings_matrix = aggregated_embeddings

    # Dynamically set perplexity based on number of samples
    perplexity_value = min(40, embeddings_matrix.shape[0] // 2)

    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity_value, n_iter=300)
    reduced_embeddings = tsne.fit_transform(embeddings_matrix)

    return reduced_embeddings

def plot_embeddings_interactive(embeddings, labels, title="Texts Clustering Visualization", save_path="clustering_visualization.html"):
    """
    Visualizes the embeddings using plotly.
    The function accepts embeddings (in 2D after using TSNE) and their corresponding labels.
    Labels are assumed to have one of three values: 
    "Shakespeare's Folio", "Shakespeare's Bad Quartos", or "Non-Shakespearean Plays".
    These three categories are colored blue, red, and green respectively.
    """

    # Generate a list of colors corresponding to the labels
    label_to_color = {
        "Shakespeare's Folio": 'blue',
        "Shakespeare's Bad Quartos": 'red',
        "Non-Shakespearean Plays": 'green'
    }
    colors = [label_to_color[label.split(":")[0]] for label in labels]  # Use the prefix before ":" to get the play type

    # Create a scatter plot
    fig = go.Figure(data=go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=colors,  # Set marker color based on the play type
            opacity=0.6
        ),
        text=labels  # Hover text shows the label
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

# Use your DistilBert models here
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

BATCH_SIZE = 64 

start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
metadata_df = pd.read_csv(metadata_path)
logging.info(f"Loaded metadata from {metadata_path} with shape: {metadata_df.shape}")

# Filter the data based on criteria
good_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Folio") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
bad_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Bad") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")
logging.info(f"Data directory set to: {data_dir}")

all_play_texts = []
all_play_ids = []
all_play_titles = []

# Aggregate embeddings by title


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

for idx, tcp in enumerate(tqdm(non_texts_df["TCP"])):
    titles, texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    for title, play_text in zip(titles, texts):
        all_play_texts.append(play_text)
        all_play_titles.append(title)  # Store title
        all_play_ids.append(f"non_{title}")

log_time(start_time, "File Parsing and Text Extraction")
logging.info(f"Finished File Parsing and Text Extraction. Number of plays extracted: {len(all_play_texts)}")
start_time = time.time()

all_chunks = []
all_chunk_play_ids = []

for play_text, play_id in zip(all_play_texts, all_play_ids):
    chunks, chunk_play_ids = chunk_text(play_text, tokenizer, 500, play_id)
    all_chunks.extend(chunks)
    all_chunk_play_ids.extend(chunk_play_ids)

log_time(start_time, "Splitting Plays into Chunks")
logging.info(f"Finished Splitting Plays into Chunks. Total number of chunks: {len(all_chunks)}")
start_time = time.time()

chunk_encodings = batch_tokenize_chunks(all_chunks, tokenizer, batch_size=64)

log_time(start_time, "Tokenizing")
start_time = time.time()

chunk_embeddings = get_embeddings(chunk_encodings)

# Use a dictionary to aggregate embeddings by play ID
aggregated_embeddings_dict = {}
for play_id, embedding in zip(all_chunk_play_ids, chunk_embeddings):
    if play_id not in aggregated_embeddings_dict:
        aggregated_embeddings_dict[play_id] = []
    aggregated_embeddings_dict[play_id].append(embedding)

# Average the embeddings
aggregated_embeddings = {play_id: np.mean(embeds, axis=0) for play_id, embeds in aggregated_embeddings_dict.items()}

log_time(start_time, "Getting Embeddings")
logging.info(f"Number of aggregated embeddings: {len(aggregated_embeddings)}")
start_time = time.time()

reduced_embeddings = reduce_dimensions(aggregated_embeddings)

labels = []
for play_id in aggregated_embeddings.keys():
    play_type, title = play_id.split("_", 1)
    if play_type == "good":
        labels.append(f"Shakespeare's Folio: {title}")
    elif play_type == "bad":
        labels.append(f"Shakespeare's Bad Quartos: {title}")
    elif play_type == "non":
        labels.append(f"Non-Shakespearean Plays: {title}")
print("The number of labels")
print(len(set(labels)))

# Call the modified function to plot and save the image
plot_embeddings_interactive(reduced_embeddings, labels)

logging.info(f"Finished Plotting. Number of reduced embeddings: {reduced_embeddings.shape[0]}")
log_time(start_time, "Plotting")
