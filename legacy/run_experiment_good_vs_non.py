# run_experiment_good_vs_non.py

import subprocess

def run_experiment():
    # Call data preprocessing
    subprocess.call(["python", "data_preprocessing.py"])

    # Call tokenization
    subprocess.call(["python", "tokenization.py"])

    # Call dataset creation for training and testing data
    subprocess.call(["python", "dataset_creation.py", "good_train", "non_train"])
    subprocess.call(["python", "dataset_creation.py", "good_test", "non_test"])

    # Call model training
    subprocess.call(["python", "train_model.py", "good_vs_non"])

    # Call evaluation
    subprocess.call(["python", "evaluate_model.py", "good_vs_non"])

if __name__ == "__main__":
    run_experiment()
