# run_experiment_bad_vs_non.py

import subprocess

def run_experiment():
    # Call data preprocessing
    subprocess.call(["python", "data_preprocessing.py"])

    # Call tokenization
    subprocess.call(["python", "tokenization.py"])

    # Call dataset creation for training and testing data
    subprocess.call(["python", "dataset_creation.py", "bad_train", "non_train"])
    subprocess.call(["python", "dataset_creation.py", "bad_test", "non_test"])

    # Call model training
    subprocess.call(["python", "train_model.py", "bad_vs_non"])

    # Call evaluation
    subprocess.call(["python", "evaluate_model.py", "bad_vs_non"])

if __name__ == "__main__":
    run_experiment()
