import torch
import os
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

# Paths
DATASETS_PATH = './datasets/'
MODEL_SAVE_PATH = './trained_model/'
TOKENIZER_PATH = './bert-base-uncased/'

# Load datasets
def load_datasets(dataset_name):
    train_dataset = torch.load(os.path.join(DATASETS_PATH, f'{dataset_name}_train.pt'))
    test_dataset = torch.load(os.path.join(DATASETS_PATH, f'{dataset_name}_test.pt'))
    return train_dataset, test_dataset

# Model Definition
model = BertForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=2)

# Training arguments and logic
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_dir='./logs',
    logging_steps=100,
    output_dir='./results',
    overwrite_output_dir=True,
    save_total_limit=3,
)

# Training function
def train_and_evaluate(dataset_name):
    train_dataset, test_dataset = load_datasets(dataset_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=BertTokenizer.from_pretrained(TOKENIZER_PATH),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()

    # Print results
    print(f"Evaluation results for {dataset_name}:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Save model
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model.save_pretrained(os.path.join(MODEL_SAVE_PATH, dataset_name))
    trainer.tokenizer.save_pretrained(os.path.join(MODEL_SAVE_PATH, dataset_name))

if __name__ == "__main__":
    for dataset_name in ["good_vs_bad", "good_vs_non", "bad_vs_non"]:
        train_and_evaluate(dataset_name)
