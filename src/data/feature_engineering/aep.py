import torch
from scipy.interpolate import CloughTocher2DInterpolator
import numpy as np
from src.data.feature_engineering.parallel_processing import run_jobs

def azimuthal_equidistant_projection(points, ref_point = None):

    normalized_points = points / torch.norm(points, dim=1, keepdim=True)
    x, y, z = normalized_points.T

    theta = torch.atan2(y, x)
    phi = torch.tensor(torch.pi) - torch.asin(z)

    if isinstance(ref_point, int):
        phi_0 = phi[ref_point]
        theta_0 = theta[ref_point]
    else:
        x_0, y_0, z_0 = ref_point
        theta_0 = torch.atan2(y_0, x_0)
        phi_0 = torch.tensor(np.pi) - torch.asin(z_0)

    alpha = torch.acos(torch.sin(phi_0) * torch.sin(phi) + torch.cos(phi_0) * torch.cos(phi) * torch.cos(theta - theta_0))
    alpha /= torch.sin(alpha)
    x = torch.cos(phi) * torch.sin(theta - theta_0)
    y = torch.cos(phi_0) * torch.sin(phi) - torch.sin(phi_0) * torch.cos(phi) * torch.cos(theta - theta_0)

    points_proj = torch.stack([x, y], dim=1) * alpha[:, None]

    if isinstance(ref_point, int):    
        points_proj[ref_point] = torch.zeros_like(points_proj[ref_point])

    return points_proj

def generate_aep_trial(signals, positions, grid, fill_value = 0):

    signals_proj = []
    for i in range(signals.shape[1]):
        signal = signals[:, i]  
        interpolator = CloughTocher2DInterpolator(positions, signal, fill_value=fill_value)
        signals_proj.append(interpolator(grid))

    signals_proj = np.stack(signals_proj, axis=2)
    return signals_proj

def generate_aep(eeg, mesh_size = (34, 34), crop_size = (2, 2), n_jobs = 1):
    from src.data.feature_engineering.utils import process_electrode_positions
    ch_pos = process_electrode_positions()
    ch_pos_proj = azimuthal_equidistant_projection(ch_pos, ref_point=54)
    ch_pos_proj = ch_pos_proj.numpy()

    grid = np.meshgrid(np.linspace(min(ch_pos_proj[:, 0]), max(ch_pos_proj[:, 0]), mesh_size[0]), 
                        np.linspace(min(ch_pos_proj[:, 1]), max(ch_pos_proj[:, 1]), mesh_size[1]))
    
    
    aep = run_jobs(generate_aep_trial, eeg, args=(ch_pos_proj, grid, 0), max_jobs=n_jobs)

    aep = aep[:, crop_size[0]//2:mesh_size[0]-crop_size[0]//2, crop_size[1]//2:mesh_size[1]-crop_size[1]//2, :]
    return aep
