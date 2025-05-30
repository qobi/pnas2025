import numpy as np
import torch
from mne.channels import make_standard_montage

def get_electrode_positions(device='cpu'):

    montage = make_standard_montage("GSN-HydroCel-128")
    electrode_positions = np.array([pos for _, pos in montage.get_positions()['ch_pos'].items()][:124])
    electrode_positions = torch.tensor(electrode_positions, dtype=torch.float32, device=device)

    return electrode_positions

def generate_electrode_distance_matrix(device='cpu'):
        
    pos = get_electrode_positions(device)
    distance_matrix = torch.cdist(pos, pos)

    return distance_matrix
