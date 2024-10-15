#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process datasets for cross-validation and rule mining.')
    parser.add_argument('data_folder_path', type=str, help='Path to the data folder containing .dat files')
    parser.add_argument('num_folds', type=int, help='Number of folds for cross-validation')
    parser.add_argument('output_folder_path', type=str, help='Output path for processed datasets and rules')
    return parser.parse_args()

def read_dat_file(file_path):
    try:
        data = pd.read_csv(file_path, header=None)
        return data
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_dataset(file_path, args):
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    data = read_dat_file(file_path)
    if data is None:
        return

    # Prepare output directories
    dataset_output_path = os.path.join(args.output_folder_path, dataset_name)

    # Create train and test subdirectories if they do not exist
    train_output_path = os.path.join(dataset_output_path, 'train')
    test_output_path = os.path.join(dataset_output_path, 'test')
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(test_output_path, exist_ok=True)

    # K-Fold cross-validation
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    fold_idx = 1

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        # Split into train and test sets
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        # Save train data
        train_file = os.path.join(train_output_path, f"train_{fold_idx}.dat")
        train_data.to_csv(train_file, index=False, header=False)

        # Save test data
        test_file = os.path.join(test_output_path, f"test_{fold_idx}.dat")
        test_data.to_csv(test_file, index=False, header=False)

        fold_idx += 1

def main():
    args = parse_arguments()

    # Process each .dat file in the data folder
    for file_name in os.listdir(args.data_folder_path):
        if file_name.endswith('.dat'):
            file_path = os.path.join(args.data_folder_path, file_name)
            process_dataset(file_path, args)

if __name__ == '__main__':
    main()
