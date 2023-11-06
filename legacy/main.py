# main.py

import data_preprocessing
import tokenization
import dataset_creation
import train_model
import evaluate_model
import run_experiment_bad_vs_non
import run_experiment_good_vs_bad
import run_experiment_good_vs_non


def main():
    # Step 1: Data Preprocessing
    print("Data Preprocessing...")
    data_preprocessing.main()

    # Step 2: Tokenization
    print("Tokenization...")
    tokenization.main()

    # Step 3: Dataset Creation
    print("Dataset Creation...")
    dataset_creation.main()

    # Step 4: Model Training
    print("Model Training...")
    train_model.main()

    # Step 5: Evaluation
    print("Evaluation...")
    evaluate_model.main()

    # Step 6: Experiment Runs (assuming 3 experiments)
    print("Running Experiment 1...")
    run_experiment_bad_vs_non.main(exp_num=1)
    
    print("Running Experiment 2...")
    run_experiment_good_vs_bad.main(exp_num=2)
    
    print("Running Experiment 3...")
    run_experiment_good_vs_non.main(exp_num=3)
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()
