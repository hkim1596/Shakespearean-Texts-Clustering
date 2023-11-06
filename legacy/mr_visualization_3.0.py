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

from bs4 import BeautifulSoup
import os
import logging

from bs4 import BeautifulSoup
import os
import logging

def parse_file(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} does not exist.")
        return []

    print(f"Processing file: {os.path.basename(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        if not content:
            print(f"File {file_path} is empty.")
            return []

        # Parse the XML with BeautifulSoup using the lxml parser
        soup = BeautifulSoup(content, 'lxml-xml')
        
        # Find the body node
        body = soup.find('body')
        if not body:
            logging.error("No body node found in the file.")
            return []

        # Extracting all text under the body
        text_content = body.get_text()

        # Search for all divs with type 'play' within the body
        play_divs = body.find_all('div', {'type': 'play'})
        
        titles = []
        if play_divs:
            for div in play_divs:
                title_tag = div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text())
                else:
                    titles.append("untitled")
        else:
            text_div = body.find('div', {'type': 'text'})
            if text_div:
                title_tag = text_div.find('head')
                if title_tag:
                    titles.append(title_tag.get_text())
                else:
                    titles.append("untitled")

        if not titles:
            logging.error("No titles found in the file.")
            return []

        print(f"Extracted titles: {titles}")
        print(f"Extracted text:\n{text_content}")

        return titles, text_content

def extract_text(play):
    """Extract all the text content from a play node."""
    return ' '.join(play.stripped_strings)

def chunk_text(text, tokenizer, max_length, play_id):
    """
    Split the text into smaller chunks of max_length, tokenized length.
    Also, keep track of play_id for each chunk.
    """
    words = text.split()
    current_chunk = []
    chunks = []
    chunk_play_ids = []  # This will store play_id for each chunk
    for word in words:
        current_word_tokens = tokenizer.tokenize(word)
        if len(tokenizer.encode(current_chunk + current_word_tokens, add_special_tokens=True)) <= max_length:
            current_chunk.extend(current_word_tokens)
        else:
            chunks.append(' '.join(current_chunk))
            chunk_play_ids.append(play_id)
            current_chunk = current_word_tokens

    # Append any remaining chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        chunk_play_ids.append(play_id)
    return chunks, chunk_play_ids

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
    # Colors corresponding to labels: 0-Good, 1-Bad, 2-Non_Texts
    colors = []
    for label in labels:
        if label == "Shakespeare's Folio":
            colors.append('blue')
        elif label == "Shakespeare's Bad Quartos":
            colors.append('red')
        elif label == "Non-Shakespearean Plays":
            colors.append('green')
    
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


#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

BATCH_SIZE = 64  # Define a suitable batch size depending on your GPU memory


start_time = time.time()

# Load the metadata CSV
metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
metadata_df = pd.read_csv(metadata_path)

# Filter the data based on criteria
good_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Folio") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
bad_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Bad") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")

all_play_texts = []
all_play_ids = []

for idx, tcp in enumerate(tqdm(good_texts_df["TCP"])):
    play_texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    all_play_texts.extend(play_texts)
    all_play_ids.extend([f"good_{idx}"] * len(play_texts))

for idx, tcp in enumerate(tqdm(bad_texts_df["TCP"])):
    play_texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    all_play_texts.extend(play_texts)
    all_play_ids.extend([f"bad_{idx}"] * len(play_texts))

for idx, tcp in enumerate(tqdm(non_texts_df["TCP"])):
    play_texts = parse_file(os.path.join(data_dir, tcp + ".xml"))
    all_play_texts.extend(play_texts)
    all_play_ids.extend([f"non_{idx}"] * len(play_texts))

log_time(start_time, "File Parsing and Text Extraction")
start_time = time.time()

all_chunks = []
all_chunk_play_ids = []

for play_text, play_id in zip(all_play_texts, all_play_ids):
    chunks, chunk_play_ids = chunk_text(play_text, tokenizer, 500, play_id)
    all_chunks.extend(chunks)
    all_chunk_play_ids.extend(chunk_play_ids)

log_time(start_time, "Chunking")
start_time = time.time()

chunk_encodings = tokenizer(all_chunks, truncation=True, padding=True, return_tensors="pt")
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
start_time = time.time()

reduced_embeddings = reduce_dimensions(aggregated_embeddings)

labels = []
for play_id in aggregated_embeddings.keys():
    if "good" in play_id:
        labels.append("Shakespeare's Folio")
    elif "bad" in play_id:
        labels.append("Shakespeare's Bad Quartos")
    elif "non" in play_id:
        labels.append("Non-Shakespearean Plays")

# Call the modified function to plot and save the image
plot_embeddings_interactive(reduced_embeddings, labels)
log_time(start_time, "Plotting")

