import os
import torch
from src.model_selection.cross_validation import generate_paired_folds
from src.model_selection.test_cross_validation import validate_folds

def save_cv_folds(subject, task, folds):
    folds_path = get_cv_folds_path(subject, task)
    os.makedirs(os.path.dirname(folds_path), exist_ok=True)
    torch.save(folds, folds_path)

def load_cv_folds(subject, task, device = 'cpu'):
    folds_path = get_cv_folds_path(subject, task)
    folds = torch.load(folds_path, map_location=device, weights_only = False)
    return folds

def get_cv_folds_path(subject, task):
    path = ['data',
            'SUDB',
            'cross_validation',
            subject,
            'task', 
            f'{task}_decoding.pth']
    
    return os.path.join(*path)

def process_cv_folds(subject, task, exemplar, category, pseudocategory=False):
    try:
        folds = load_cv_folds(subject, task)
        # validate_folds(folds, category, exemplar)
        return folds
    except:
        folds = generate_paired_folds(category, exemplar, pseudocategory=pseudocategory)
        validate_folds(folds, category, exemplar)
        save_cv_folds(subject, task, folds)
        return folds