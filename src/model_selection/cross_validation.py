import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

def generate_paired_folds(categories, exemplars, pseudocategory=False):
    
    n_splits = len(exemplars.unique()) // len(categories.unique())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    skgf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []

    trial_idxs = np.arange(len(categories))
    confounded_splits = skf.split(categories, exemplars)
    unconfounded_splits = skgf.split(categories, exemplars, groups= exemplars//12 if pseudocategory else exemplars)

    for idx, ((_, confounded_test), (_, unconfounded_test)) in enumerate(zip(confounded_splits, unconfounded_splits)):
        confounded_test_idxs = confounded_test[~np.isin(confounded_test, unconfounded_test)]
        unconfounded_test_idxs = unconfounded_test[~np.isin(unconfounded_test, confounded_test)]
        train_val_idxs = trial_idxs[~np.isin(trial_idxs, confounded_test) & ~np.isin(trial_idxs, unconfounded_test)]
        print("Confounded", categories[confounded_test].unique(return_counts=True), exemplars[confounded_test].unique(return_counts=True))
        print("Unconfounded", categories[unconfounded_test].unique(return_counts=True), exemplars[unconfounded_test].unique(return_counts=True))
        partitions = {'confounded_test': torch.tensor(confounded_test_idxs, dtype=torch.long),
                      'unconfounded_test': torch.tensor(unconfounded_test_idxs, dtype=torch.long),
                      'train_val': torch.tensor(train_val_idxs, dtype=torch.long),
                      'train': [],
                      'val': []}
                                                         
        nskf = StratifiedKFold(n_splits=n_splits-1, shuffle=True, random_state=42)

        for kdx, (train, val) in enumerate(nskf.split(train_val_idxs, exemplars[train_val_idxs])):
            partitions['train'].append(torch.tensor(train_val_idxs[train], dtype=torch.long))
            partitions['val'].append(torch.tensor(train_val_idxs[val], dtype=torch.long))
        folds.append(partitions)

    return folds
