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


good_texts = [chunk for tcp in tqdm(good_texts_df["TCP"])
              for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))
              for chunk in chunk_text(play_text, tokenizer, 500)]  # 510 to accommodate special tokens

# For bad_texts:
bad_texts = [chunk for tcp in tqdm(bad_texts_df["TCP"])
             for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))
             for chunk in chunk_text(play_text, tokenizer, 500)]  # 510 to accommodate special tokens

# For non_texts:
non_texts = [chunk for tcp in tqdm(non_texts_df["TCP"])
             for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))
             for chunk in chunk_text(play_text, tokenizer, 500)]  # 510 to accommodate special tokens

log_time(start_time, "File Parsing and Text Extraction")

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

good_texts_encoded = batch_tokenize(good_texts, tokenizer, max_length)
bad_texts_encoded = batch_tokenize(bad_texts, tokenizer, max_length)
non_texts_encoded = batch_tokenize(non_texts, tokenizer, max_length)

# After tokenizing, but before padding/truncating
original_token_counts = [len(tokenizer.tokenize(text)) for text in good_texts]

# Number of tokens truncated for each text
truncated_tokens = [count - max_length if count > max_length else 0 for count in original_token_counts]

# Total truncated tokens
total_truncated_tokens = sum(truncated_tokens)
print(f"Total number of truncated tokens for good_texts_encoded: {total_truncated_tokens}")

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

# Training Procedure
def train_and_evaluate(train_encodings, train_labels, test_encodings, test_labels):
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

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

    # Confusion Matrix
    start_time = time.time()  # Start timing here

    predictions = trainer.predict(test_dataset).predictions.argmax(axis=1)

    log_time(start_time, "Model Prediction")  # End timing here

    labels = [item['labels'] for item in test_dataset]
    print(confusion_matrix(labels, predictions))
    save_to_file('results.txt', str(confusion_matrix(labels, predictions)))

    # Save and Visualize Model Predictions
    for i, (input_ids, prediction) in enumerate(zip(test_dataset.encodings['input_ids'], predictions)):
        original_text = tokenizer.decode(input_ids)
        predicted_label = "Good" if prediction == 0 else "Bad" if prediction == 1 else "Non"
        save_to_file('results.txt', f"Sample {i}:\nText: {original_text}\nPredicted: {predicted_label}\n")
        
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

def predict_play(model, tokenizer, play_text):
    chunks = chunk_text(play_text, tokenizer, 500)
    tokenized_chunks = [tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=512) for chunk in chunks]
    
    predictions = []
    with torch.no_grad():
        for tokenized_chunk in tokenized_chunks:
            outputs = model(**tokenized_chunk)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class)
    
    from statistics import mode
    return mode(predictions)

# Splitting function for encodings
def train_test_split_encodings(encodings, test_size=0.2, random_state=None):
    input_ids_train, input_ids_test = train_test_split(encodings["input_ids"], test_size=test_size, random_state=random_state)
    attention_mask_train, attention_mask_test = train_test_split(encodings["attention_mask"], test_size=test_size, random_state=random_state)
    
    return {"input_ids": input_ids_train, "attention_mask": attention_mask_train}, {"input_ids": input_ids_test, "attention_mask": attention_mask_test}

good_train, good_test = train_test_split_encodings(good_texts_encoded, test_size=0.2, random_state=42)
bad_train, bad_test = train_test_split_encodings(bad_texts_encoded, test_size=0.2, random_state=42)
non_train, non_test = train_test_split_encodings(non_texts_encoded, test_size=0.2, random_state=42)

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

# Experiment Runs
# Base Classifier (Good vs Bad)
train_and_evaluate(
    combine_encodings([good_train, bad_train]),
    [0] * len(good_train["input_ids"]) + [1] * len(bad_train["input_ids"]),
    combine_encodings([good_test, bad_test]),
    [0] * len(good_test["input_ids"]) + [1] * len(bad_test["input_ids"])
)

# Piracy Hypothesis (Good vs Non)
train_and_evaluate(
    combine_encodings([good_train, non_train]),
    [0] * len(good_train["input_ids"]) + [1] * len(non_train["input_ids"]),
    combine_encodings([good_test, non_test]),
    [0] * len(good_test["input_ids"]) + [1] * len(non_test["input_ids"])
)

# Contemporary Commonality Hypothesis (Bad vs Non)
train_and_evaluate(
    combine_encodings([bad_train, non_train]),
    [0] * len(bad_train["input_ids"]) + [1] * len(non_train["input_ids"]),
    combine_encodings([bad_test, non_test]),
    [0] * len(bad_test["input_ids"]) + [1] * len(non_test["input_ids"])
)

# Load a trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = DistilBertTokenizer.from_pretrained("./saved_model")

# Predict on the test set of plays
good_play_texts = [play_text for tcp in good_texts_df["TCP"] for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
bad_play_texts = [play_text for tcp in bad_texts_df["TCP"] for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]
non_play_texts = [play_text for tcp in non_texts_df["TCP"] for play_text in parse_file(os.path.join(data_dir, tcp + ".xml"))]

good_play_predictions = [predict_play(model, tokenizer, play_text) for play_text in good_play_texts]
bad_play_predictions = [predict_play(model, tokenizer, play_text) for play_text in bad_play_texts]
non_play_predictions = [predict_play(model, tokenizer, play_text) for play_text in non_play_texts]

# Now you have the play-level predictions
# You can evaluate these predictions as you see fit.

# For instance, to calculate the accuracy for good_play_texts:
good_play_correct_predictions = sum([1 for pred in good_play_predictions if pred == 0])  # assuming 0 is the label for "Good"
good_play_accuracy = good_play_correct_predictions / len(good_play_texts)

print(f"Accuracy for Good Plays: {good_play_accuracy * 100:.2f}%")