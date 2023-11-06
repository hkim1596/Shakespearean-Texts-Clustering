# data_preprocessing.py

import os
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Load the metadata CSV
def load_metadata():
    metadata_path = os.path.join(os.getcwd(), "Metadata", "PlayMetadata.csv")
    return pd.read_csv(metadata_path)

# Filter the data based on criteria
def filter_metadata(metadata_df):
    good_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Folio") & 
                                (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
    bad_texts_df = metadata_df[(metadata_df["Author"] == "Shakespeare") & (metadata_df["Type"] == "Bad") & 
                               (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]
    non_texts_df = metadata_df[(metadata_df["Author"] != "Shakespeare") & 
                               (metadata_df["Exception"] != "Yes") & (metadata_df["Beta"] == "Yes")]

    return good_texts_df, bad_texts_df, non_texts_df

# File Parsing Function
def parse_file(file_path):
    if file_path.endswith('.xml'):
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'lxml')
            texts = soup.find_all('text')
            return [text.get_text() for text in texts]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [file.read()]

def extract_and_check(file_path):
    extracted_texts = parse_file(file_path)
    if not extracted_texts:
        print(f"No text extracted from: {file_path}")
        return None
    return extracted_texts[0]

def chunk_text(text, chunk_size=500):
    """Divides the text into non-overlapping segments of chunk_size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Extracting text data based on metadata
def extract_texts_from_metadata(good_texts_df, bad_texts_df, non_texts_df):
    data_dir = os.path.join(os.getcwd(), "Data")
    
    good_texts = [chunk for tcp in tqdm(good_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")))]
    good_texts = [text for text in good_texts if text]  # Remove any None values
    print(f"Processed {len(good_texts)} files for Good Texts.")
    
    bad_texts = [chunk for tcp in tqdm(bad_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")))]
    bad_texts = [text for text in bad_texts if text]  # Remove any None values
    print(f"Processed {len(bad_texts)} files for Bad Texts.")
    
    non_texts = [chunk for tcp in tqdm(non_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")))]
    non_texts = [text for text in non_texts if text]  # Remove any None values
    print(f"Processed {len(non_texts)} files for Non Texts.")

    return good_texts, bad_texts, non_texts

if __name__ == "__main__":
    metadata_df = load_metadata()
    good_texts_df, bad_texts_df, non_texts_df = filter_metadata(metadata_df)
    good_texts, bad_texts, non_texts = extract_texts_from_metadata(good_texts_df, bad_texts_df, non_texts_df)
    
    # Optionally, save the processed texts to disk for future use.
    # This can be done using pandas DataFrame or any other method.

# Save processed texts to disk
def save_texts_to_csv(good_texts, bad_texts, non_texts, save_path='./processed_texts'):
    os.makedirs(save_path, exist_ok=True)
    
    good_texts_df = pd.DataFrame(good_texts, columns=['Text'])
    good_texts_df['Label'] = 'Good'
    good_texts_df.to_csv(os.path.join(save_path, 'good_texts.csv'), index=False)

    bad_texts_df = pd.DataFrame(bad_texts, columns=['Text'])
    bad_texts_df['Label'] = 'Bad'
    bad_texts_df.to_csv(os.path.join(save_path, 'bad_texts.csv'), index=False)
    
    non_texts_df = pd.DataFrame(non_texts, columns=['Text'])
    non_texts_df['Label'] = 'Non'
    non_texts_df.to_csv(os.path.join(save_path, 'non_texts.csv'), index=False)

if __name__ == "__main__":
    metadata_df = load_metadata()
    good_texts_df, bad_texts_df, non_texts_df = filter_metadata(metadata_df)
    good_texts, bad_texts, non_texts = extract_texts_from_metadata(good_texts_df, bad_texts_df, non_texts_df)
    
    save_texts_to_csv(good_texts, bad_texts, non_texts)
