import torch
from pywt import cwt
import numpy as np
import torch
from src.data.feature_engineering.parallel_processing import run_jobs
from tqdm import tqdm

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 50)
}

def freq_to_scale(freq, sampling_period, central_freq = 1):
    return central_freq / (sampling_period * freq)

def bands_to_scales(bands, sampling_period, resolution = 5, central_freq = 1.0, bandwidth = 1.0):
    scales = np.zeros((len(bands), resolution))
    for i, band in enumerate(bands.values()):
        scales[i] = np.linspace(band[0], band[1], resolution)
        scales[i] = freq_to_scale(scales[i], sampling_period, central_freq)
    return scales

def generate_st_trial(signals, wavelet, bands, sampling_freq, resolution, padding = 8):
    sampling_period = 1/sampling_freq
    scales = bands_to_scales(bands, sampling_period, resolution=resolution)
    spectral_bands = np.zeros((signals.shape[0], scales.shape[0] * scales.shape[1], signals.shape[1]))
    
    signal_padded = np.pad(signals, ((0, 0), (padding, padding)), mode='reflect')
    scales_combined = np.concatenate(scales)
    transform, _ = cwt(signal_padded, scales_combined, wavelet, sampling_period, 'conv')
    spectral_bands = abs(transform)[:, :, padding:32+padding]
    return spectral_bands

def generate_st(eeg, wavelet="cmor1.0-0.5", sampling_freq = 62.5, bands = BANDS, resolution = 5, n_jobs=1):
    if n_jobs == 1:
        st = torch.zeros((eeg.shape[0], len(bands) * resolution, eeg.shape[1], eeg.shape[2]))
        for trial_idx, trial in tqdm(enumerate(eeg.cpu().numpy()), desc="Generating Spectral Transform", total=eeg.shape[0], leave=False):
            st[trial_idx] = torch.tensor(generate_st_trial(trial, wavelet, bands, sampling_freq, resolution), dtype=torch.float32)
    else:
        st = run_jobs(generate_st_trial, eeg, args=(wavelet, bands, sampling_freq, resolution), max_jobs=n_jobs)
    return st
