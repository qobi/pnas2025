import torch
from sklearn.model_selection import StratifiedGroupKFold

def generate_pseudocategories(category, exemplar):
    pseudocategory = torch.zeros_like(category, dtype=torch.long)

    skgf = StratifiedGroupKFold(n_splits=12, shuffle=True, random_state=42)
    for idx, (_, test_idxs) in enumerate(skgf.split(category, exemplar, groups=exemplar)):
        pseudocategory[test_idxs] = idx
    return pseudocategory
