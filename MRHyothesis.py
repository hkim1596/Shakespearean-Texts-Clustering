import os
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

def chunk_text(text, chunk_size):
    """Divides the text into non-overlapping segments of chunk_size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Extracting text data based on metadata
data_dir = os.path.join(os.getcwd(), "Data")

chunk_size = 500  # Adjust this value according to your needs

good_texts = [chunk for tcp in tqdm(good_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")), chunk_size)]
good_texts = [text for text in good_texts if text]  # Remove any None values
print(f"Processed {len(good_texts)} files for Good Texts.")

bad_texts = [chunk for tcp in tqdm(bad_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")), chunk_size)]
bad_texts = [text for text in bad_texts if text]  # Remove any None values
print(f"Processed {len(bad_texts)} files for Bad Texts.")

non_texts = [chunk for tcp in tqdm(non_texts_df["TCP"]) for chunk in chunk_text(extract_and_check(os.path.join(data_dir, tcp + ".xml")), chunk_size)]
non_texts = [text for text in non_texts if text]  # Remove any None values
print(f"Processed {len(non_texts)} files for Non Texts.")

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_length = 512  # or any other size that fits your needs
good_texts_encoded = tokenizer(good_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
bad_texts_encoded = tokenizer(bad_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)
non_texts_encoded = tokenizer(non_texts, padding="max_length", truncation=True, return_tensors="pt", max_length=max_length)

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
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
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
    trainer.train()
    results = trainer.evaluate()
    print(f"Results for {trainer.args.output_dir}: {results}")

    # Confusion Matrix
    predictions = trainer.predict(test_dataset).predictions.argmax(axis=1)
    labels = [item['labels'] for item in test_dataset]
    print(confusion_matrix(labels, predictions))

    # Save and Visualize Model Predictions
    for i, (input_ids, prediction) in enumerate(zip(test_dataset.encodings['input_ids'], predictions)):
        original_text = tokenizer.decode(input_ids)
        predicted_label = "Good" if prediction == 0 else "Bad" if prediction == 1 else "Non"
        print(f"Sample {i}:\nText: {original_text}\nPredicted: {predicted_label}\n")

    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

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
