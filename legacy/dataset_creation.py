import os
import torch
from sklearn.model_selection import train_test_split

# Paths
TOKENIZED_DATA_PATH = './tokenized_data/'
DATASETS_PATH = './datasets/'

# Load tokenized data
def load_tokenized_data():
    good_texts_encoded = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'good_texts_encoded.pt'))
    bad_texts_encoded = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'bad_texts_encoded.pt'))
    non_texts_encoded = torch.load(os.path.join(TOKENIZED_DATA_PATH, 'non_texts_encoded.pt'))
    
    return good_texts_encoded, bad_texts_encoded, non_texts_encoded

# Custom Dataset Class
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

# Splitting function for encodings
def train_test_split_encodings(encodings, labels, test_size=0.2, random_state=None):
    input_ids_train, input_ids_test, labels_train, labels_test = train_test_split(
        encodings["input_ids"], labels, test_size=test_size, random_state=random_state
    )
    attention_mask_train, attention_mask_test = train_test_split(
        encodings["attention_mask"], test_size=test_size, random_state=random_state
    )
    
    train_encodings = {"input_ids": input_ids_train, "attention_mask": attention_mask_train}
    test_encodings = {"input_ids": input_ids_test, "attention_mask": attention_mask_test}

    return train_encodings, test_encodings, labels_train, labels_test

# Save datasets
def save_datasets(train_dataset, test_dataset, dataset_name):
    os.makedirs(DATASETS_PATH, exist_ok=True)
    torch.save(train_dataset, os.path.join(DATASETS_PATH, f'{dataset_name}_train.pt'))
    torch.save(test_dataset, os.path.join(DATASETS_PATH, f'{dataset_name}_test.pt'))

if __name__ == "__main__":
    good_texts_encoded, bad_texts_encoded, non_texts_encoded = load_tokenized_data()
    
    # Good vs Bad
    train_encodings, test_encodings, train_labels, test_labels = train_test_split_encodings(
        {**good_texts_encoded, **bad_texts_encoded}, 
        [0] * len(good_texts_encoded["input_ids"]) + [1] * len(bad_texts_encoded["input_ids"])
    )
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    save_datasets(train_dataset, test_dataset, 'good_vs_bad')
    
    # Good vs Non
    train_encodings, test_encodings, train_labels, test_labels = train_test_split_encodings(
        {**good_texts_encoded, **non_texts_encoded}, 
        [0] * len(good_texts_encoded["input_ids"]) + [1] * len(non_texts_encoded["input_ids"])
    )
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    save_datasets(train_dataset, test_dataset, 'good_vs_non')

    # Bad vs Non
    train_encodings, test_encodings, train_labels, test_labels = train_test_split_encodings(
        {**bad_texts_encoded, **non_texts_encoded}, 
        [0] * len(bad_texts_encoded["input_ids"]) + [1] * len(non_texts_encoded["input_ids"])
    )
    train_dataset = CustomDataset(train_encodings, train_labels)
    test_dataset = CustomDataset(test_encodings, test_labels)
    save_datasets(train_dataset, test_dataset, 'bad_vs_non')
