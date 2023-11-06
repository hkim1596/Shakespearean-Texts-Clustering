import pandas as pd
import os
import torch
from transformers import BertTokenizer

# Paths
GOOD_TEXTS_PATH = './processed_texts/good_texts.csv'
BAD_TEXTS_PATH = './processed_texts/bad_texts.csv'
NON_TEXTS_PATH = './processed_texts/non_texts.csv'
TOKENIZED_DATA_PATH = './tokenized_data/'

# Load processed data
def load_processed_texts():
    good_texts_df = pd.read_csv(GOOD_TEXTS_PATH)
    bad_texts_df = pd.read_csv(BAD_TEXTS_PATH)
    non_texts_df = pd.read_csv(NON_TEXTS_PATH)
    
    return good_texts_df['Text'].tolist(), bad_texts_df['Text'].tolist(), non_texts_df['Text'].tolist()

# Tokenize
def tokenize_texts(good_texts, bad_texts, non_texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 512  # adjust if necessary
    
    good_texts_encoded = tokenizer(good_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    bad_texts_encoded = tokenizer(bad_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    non_texts_encoded = tokenizer(non_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
    
    return good_texts_encoded, bad_texts_encoded, non_texts_encoded

# Save tokenized data
def save_tokenized_data(good_texts_encoded, bad_texts_encoded, non_texts_encoded):
    os.makedirs(TOKENIZED_DATA_PATH, exist_ok=True)
    torch.save(good_texts_encoded, os.path.join(TOKENIZED_DATA_PATH, 'good_texts_encoded.pt'))
    torch.save(bad_texts_encoded, os.path.join(TOKENIZED_DATA_PATH, 'bad_texts_encoded.pt'))
    torch.save(non_texts_encoded, os.path.join(TOKENIZED_DATA_PATH, 'non_texts_encoded.pt'))

if __name__ == "__main__":
    good_texts, bad_texts, non_texts = load_processed_texts()
    good_texts_encoded, bad_texts_encoded, non_texts_encoded = tokenize_texts(good_texts, bad_texts, non_texts)
    save_tokenized_data(good_texts_encoded, bad_texts_encoded, non_texts_encoded)
