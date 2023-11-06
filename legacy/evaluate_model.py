import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix, classification_report

# Paths
DATASETS_PATH = './datasets/'
MODEL_PATH = './trained_model/'
TOKENIZER_PATH = './bert-base-uncased/'

# Load datasets
def load_datasets(dataset_name):
    test_dataset = torch.load(os.path.join(DATASETS_PATH, f'{dataset_name}_test.pt'))
    return test_dataset

# Load Model & Tokenizer
def load_model_and_tokenizer(dataset_name):
    model = BertForSequenceClassification.from_pretrained(os.path.join(MODEL_PATH, dataset_name))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_PATH, dataset_name))
    return model, tokenizer

# Evaluation function
def evaluate_model(dataset_name):
    test_dataset = load_datasets(dataset_name)
    model, tokenizer = load_model_and_tokenizer(dataset_name)
    
    training_args = TrainingArguments(
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        output_dir='./results',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )
    
    # Evaluate the model
    predictions = trainer.predict(test_dataset)
    predictions_labels = predictions.predictions.argmax(axis=1)
    true_labels = [item['labels'] for item in test_dataset]
    
    # Compute and print confusion matrix and classification report
    print(f"Evaluation results for {dataset_name}:")
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions_labels))
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions_labels, target_names=["Class 0", "Class 1"]))

if __name__ == "__main__":
    for dataset_name in ["good_vs_bad", "good_vs_non", "bad_vs_non"]:
        evaluate_model(dataset_name)
