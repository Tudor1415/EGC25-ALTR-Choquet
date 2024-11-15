# model_evaluation.py
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

class_items_dict = {
    'adult': ['145', '146'],
    'bank': ['89', '90'],
    'connect': ['127', '128'],
    'credit': ['111', '112'],
    'dota': ['346', '347'],
    'toms': ['911', '912'],
    'mushroom': ['116', '117']
}

def load_dataset(dataset_file, class_items):
    data = []
    with open(dataset_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            data.append([int(item) for item in items])

    # Collect all items
    all_items = set()
    for transaction in data:
        all_items.update(transaction)

    # Map items to columns
    item_to_col = {item: idx for idx, item in enumerate(sorted(all_items))}

    n_samples = len(data)
    n_features = len(all_items)
    X = np.zeros((n_samples, n_features), dtype=int)
    y = np.full(n_samples, -1)

    for i, transaction in enumerate(data):
        items_in_transaction = set(transaction)
        for item in items_in_transaction:
            col_idx = item_to_col[item]
            X[i, col_idx] = 1

        # Assign labels
        for label, class_item in enumerate(class_items):
            if int(class_item) in items_in_transaction:
                y[i] = label
                break

    # Remove samples without labels
    valid_indices = y != -1
    X = X[valid_indices]
    y = y[valid_indices]

    return X, y

def evaluate_model(X, y, model, n_splits=5):
    accuracies = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Clone the model to ensure a fresh model for each fold
        clf = model.__class__(**model.get_params())
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return accuracies


def evaluate_dat_files(dat_files_folder, output_base, models, nb_folds=5):
    results = {}
    
    for dataset_file in os.listdir(dat_files_folder):
        if dataset_file.endswith('.dat'):
            dataset_name = os.path.splitext(dataset_file)[0]
            if dataset_name not in class_items_dict:
                print(f"Class items not defined for dataset {dataset_name}, skipping.")
                continue

            print(f"Processing dataset {dataset_name}...")

            # Read class items
            class_items = class_items_dict[dataset_name]

            # Load original dataset
            dataset_path = os.path.join(dat_files_folder, dataset_file)
            X_orig, y_orig = load_dataset(dataset_path, class_items)

            # Load augmented dataset
            aug_dataset_file = os.path.join(output_base, dataset_name, "aug", dataset_name + "_processed.dat")
            if not os.path.exists(aug_dataset_file):
                print(f"Augmented dataset not found for {dataset_name}, skipping augmented dataset.")
                X_aug, y_aug = None, None
            else:
                X_aug, y_aug = load_dataset(aug_dataset_file, class_items)

            # Evaluate on original dataset
            if len(np.unique(y_orig)) < 2:
                print(f"Not enough classes in original dataset {dataset_name}, skipping.")
                continue

            acc_orig = {}
            acc_aug = {}

            for model_name, model in models.items():
                print(f"Evaluating Original dataset with {model_name}...")
                acc = evaluate_model(X_orig, y_orig, model, n_splits=nb_folds)
                acc_orig[model_name] = acc

                if X_aug is not None and y_aug is not None:
                    if len(np.unique(y_aug)) < 2:
                        print(f"Not enough classes in augmented dataset {dataset_name}, skipping augmented dataset.")
                        acc_aug[model_name] = None
                    else:
                        print(f"Evaluating Augmented dataset with {model_name}...")
                        acc = evaluate_model(X_aug, y_aug, model, n_splits=nb_folds)
                        acc_aug[model_name] = acc
                else:
                    acc_aug[model_name] = None

            # Store results
            results[dataset_name] = {'Original': acc_orig, 'Augmented': acc_aug}
    
    return results