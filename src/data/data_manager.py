from torch.utils.data import TensorDataset
import torch
import os

class DataManager(TensorDataset):

    def __init__(self, features, folds, metadata=None):
        super().__init__()

        self.features = features
        self.folds = folds

        self.n_folds = len(folds)
        self.n_nested_folds = len(folds[0]['train'])    

        self.fold_idx = None
        self.nested_fold_idx = None

        self.train_idx = None
        self.val_idx = None
        self.train_val_idx = None
        self.confounded_test_idx = None
        self.unconfounded_test_idx = None

        self.metadata = metadata or {}

    def set_fold_idx(self, fold_idx=None, nested_fold_idx=0):
        
        if fold_idx is not None:
            self.fold_idx = fold_idx
        self.nested_fold_idx = nested_fold_idx

        self.train_idx = self.folds[self.fold_idx]['train'][self.nested_fold_idx]
        self.val_idx = self.folds[self.fold_idx]['val'][self.nested_fold_idx]
        
        self.train_val_idx = self.folds[self.fold_idx]['train_val']

        self.confounded_test_idx = self.folds[self.fold_idx]['confounded_test']
        self.unconfounded_test_idx = self.folds[self.fold_idx]['unconfounded_test']

    def save_partition(self, partition, dir):
        os.makedirs(dir, exist_ok=True)
        partition_path = os.path.join(dir, f"{partition}_idx.pth")
        if partition == 'train':
            torch.save(self.train_idx, partition_path)
        elif partition == 'val':
            torch.save(self.val_idx, partition_path)
        elif partition == 'train_val':
            torch.save(self.train_val_idx, partition_path)
        elif partition == 'confounded_test':
            torch.save(self.confounded_test_idx, partition_path)
        elif partition == 'unconfounded_test':
            torch.save(self.unconfounded_test_idx, partition_path)
        else:
            raise ValueError("Partition must be one of 'train', 'val', 'train_val', 'confounded_test', or 'unconfounded_test'.")

    def fit_transforms(self, idx):
        for i, f in enumerate(self.features):
            if self.transforms and self.transforms[i]:
                self.transforms[i].fit(f[idx])

    def apply_transforms(self, idx):
        return (self.transforms[i].transform(f[idx]) if self.transforms and self.transforms[i] else f[idx] for i, f in enumerate(self.features))

    def __getitem__(self, idx):
        return {f_id: f[idx] for f_id, f in self.features.items()}
        # return (*self.apply_transforms(idx), self.labels[idx])
    
    def get_train_data(self):
        # self.fit_transforms(self.train_idx)
        return self[self.train_idx]
    
    def get_val_data(self):
        return self[self.val_idx]
    
    def get_train_val_data(self):
        # self.fit_transforms(self.train_val_idx)
        return self[self.train_val_idx]

    def get_confounded_test_data(self):
        return self[self.confounded_test_idx]
    
    def get_unconfounded_test_data(self):
        return self[self.unconfounded_test_idx]

    def get_fold_idx(self):
        return self.fold_idx
    
    def get_metadata(self):
        return self.metadata