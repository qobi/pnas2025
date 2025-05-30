import torch
import os
from src.data.feature_engineering.aep import generate_aep
from src.data.feature_engineering.wpli import generate_wpli
from src.data.feature_engineering.st import generate_st
from src.data.feature_engineering.pseudocategories import generate_pseudocategories
from src.data.feature_engineering.electrodes import get_electrode_positions, generate_electrode_distance_matrix

def get_data_dir():
    data_dir = os.path.join('data', 'SUDB', 'processed')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

def get_features_dir(subject):
    features_dir = os.path.join('data', 'SUDB', 'processed', subject)
    os.makedirs(features_dir, exist_ok=True)
    return features_dir
    
def load_feature(subject, feature_id, device = 'cpu'):
    
    path = os.path.join(get_features_dir(subject), f"{feature_id}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found at {path}.")
    feature = torch.load(path, map_location=device, weights_only = False)
    
    return feature

def save_feature(subject, feature_id, feature):
    path = os.path.join(get_features_dir(subject), f"{feature_id}.pt")
    torch.save(feature, path) 

def generate_feature(subject, feature_id, **kwargs):
    generating_functions = {
        'pseudocategory': generate_pseudocategories,
        'AEP': generate_aep,
        'WPLI': generate_wpli,
        'ST': generate_st,
    }
    if feature_id not in generating_functions:
        raise ValueError(f"Feature ID {feature_id} is not recognized.")
    try:
        return load_feature(subject, feature_id)
    except FileNotFoundError:
        feature = generating_functions[feature_id](**kwargs)
        save_feature(subject, feature_id, feature)
        return feature
    
def process_electrode_positions(device='cpu'):
    path = os.path.join(get_data_dir(), 'electrode_positions.pt')
    try:
        electrode_positions = torch.load(path, map_location=device, weights_only=False)
    except:
        electrode_positions = get_electrode_positions(device)
        torch.save(electrode_positions, path)
   
    return electrode_positions

def process_electrode_distance_matrix(normalized=True, device='cpu'):
    path = os.path.join(get_data_dir(), 'electrode_distance_matrix.pt')
    if not os.path.exists(path):
        electrode_dist = generate_electrode_distance_matrix(device=device)
        torch.save(electrode_dist, path)
    else:
        electrode_dist = torch.load(path, map_location=device, weights_only=False)

    if normalized:
        electrode_dist = (electrode_dist - electrode_dist.min())/(electrode_dist.max() - electrode_dist.min())
    
    return electrode_dist
    