import os
from scipy.io import loadmat
import torch
from src.data.feature_engineering.utils import load_feature, save_feature

def extract_features(subject):

    raw_data_path = os.path.join("data", 'SUDB', "raw", f"{subject}.mat")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Data file not found at {raw_data_path}")
    
    data = loadmat(raw_data_path)

    eeg = torch.tensor(data['X_3D'], dtype=torch.float32).permute(2, 0, 1)
    category = torch.tensor(data['categoryLabels'].reshape(-1), dtype=torch.long) - 1
    exemplar = torch.tensor(data['exemplarLabels'].reshape(-1), dtype=torch.long) - 1
    
    # Some exemplars have more than 72 trials, so we need to drop some trials to ensure balance across exemplars. We do this by selecting the first 72 trials for each exemplar, concatenating them and then sorting them so that the presentation order is preserved. This gives us the indices of the trials to keep.
    trials, _ = torch.sort(torch.cat([torch.argwhere(exemplar == i)[:72, 0] for i in range(72)]))
    
    eeg = eeg[trials]
    category = category[trials]
    exemplar = exemplar[trials]

    return eeg, category, exemplar

def save_features(subject, eeg, category, exemplar):
    save_feature(subject, 'EEG', eeg)
    save_feature(subject, 'category', category)
    save_feature(subject, 'exemplar', exemplar)
    
    return

def load_features(subject, device = 'cpu'):
    eeg = load_feature(subject, 'EEG', device)
    category = load_feature(subject, 'category', device)
    exemplar = load_feature(subject, 'exemplar', device)
    
    return eeg, category, exemplar

def reformat_dataset(subject):
    try:
        eeg, category, exemplar = load_features(subject)
        return eeg, category, exemplar
    except:
        pass
    eeg, category, exemplar = extract_features(subject)
    save_features(subject, eeg, category, exemplar)
    return eeg, category, exemplar

