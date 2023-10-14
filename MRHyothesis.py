from bs4 import BeautifulSoup
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix

# 1. Imports and Initial Setup
base_dir = os.getcwd()
directories = {
    "Good_Train": os.path.join(base_dir, 'Good_Train'),
    "Good_Test": os.path.join(base_dir, 'Good_Test'),
    "Bad_Train": os.path.join(base_dir, 'Bad_Train'),
    "Bad_Test": os.path.join(base_dir, 'Bad_Test'),
    "Non_Train": os.path.join(base_dir, 'Non_Train'),
    "Non_Test": os.path.join(base_dir, 'Non_Test')
}

# 2. File Parsing
def parse_file(file_path):
    if file_path.endswith('.xml'):
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'lxml')
            texts = soup.find_all('text')
            return [text.get_text() for text in texts]
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            return [file.read()]

parsed_texts = {}
for dir_name, dir_path in directories.items():
    parsed_texts[dir_name] = []
    for file_name in tqdm(os.listdir(dir_path), desc=f"Parsing {dir_name}"):
        if file_name.endswith(('.xml', '.txt')):
            parsed_content = parse_file(os.path.join(dir_path, file_name))
            parsed_texts[dir_name].extend(parsed_content)

# Print Sample Parsed Texts
for dir_name, texts in parsed_texts.items():
    print(f"Sample from {dir_name}: {texts[0][:500]}...\n")

# 3. Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_texts = {}
for key, texts in tqdm(parsed_texts.items(), desc="Tokenizing"):
    encoded_texts[key] = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Print Tokenization Sample
for key, value in encoded_texts.items():
    print(f"Sample tokenized input from {key}: {value['input_ids'][0]}")

# 4. Combine Encodings Function
def combine_encodings(encodings_list):
    return {
        "input_ids": torch.cat([e["input_ids"] for e in encodings_list], dim=0),
        "attention_mask": torch.cat([e["attention_mask"] for e in encodings_list], dim=0),
    }

# 5. Dataset Creation
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

# 6. Training Procedure
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

# 7. Experiment Runs
# Base Classifier
train_and_evaluate(
    combine_encodings([encoded_texts["Good_Train"], encoded_texts["Bad_Train"]]),
    [0] * len(parsed_texts["Good_Train"]) + [1] * len(parsed_texts["Bad_Train"]),
    combine_encodings([encoded_texts["Good_Test"], encoded_texts["Bad_Test"]]),
    [0] * len(parsed_texts["Good_Test"]) + [1] * len(parsed_texts["Bad_Test"])
)

# Piracy Hypothesis
train_and_evaluate(
    combine_encodings([encoded_texts["Good_Train"], encoded_texts["Non_Train"]]),
    [0] * len(parsed_texts["Good_Train"]) + [1] * len(parsed_texts["Non_Train"]),
    encoded_texts["Bad_Test"],
    [1] * len(parsed_texts["Bad_Test"])
)

# Contemporary Commonality Hypothesis
train_and_evaluate(
    combine_encodings([encoded_texts["Bad_Train"], encoded_texts["Non_Train"]]),
    [0] * len(parsed_texts["Bad_Train"]) + [1] * len(parsed_texts["Non_Train"]),
    encoded_texts["Good_Test"],
    [0] * len(parsed_texts["Good_Test"])
)
