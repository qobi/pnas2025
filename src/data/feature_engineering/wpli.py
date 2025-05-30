import numpy as np
from scipy.signal import windows, csd
from .parallel_processing import run_jobs

def wpli(x, y):

    window = windows.hann(x.shape[0])
    _, Sxy = csd(x, y, fs=62.5, window=window)
    wpli = np.sum(np.abs(np.imag(Sxy))*np.sign(np.imag(Sxy)))
    wpli /= np.sum(np.abs(np.imag(Sxy)))
    
    return wpli

def generate_wpli_trial(signals):

    n_channels, _ = signals.shape

    wpli_trial = np.zeros((n_channels, n_channels))


    for i in range(n_channels):
        for j in range(i+1, n_channels):
            wpli_trial[i, j] = wpli(signals[i], signals[j])
            wpli_trial[j, i] = wpli_trial[i, j]

    return wpli_trial
            
def generate_wpli(eeg, n_jobs=1):

    wpli_matrix = run_jobs(generate_wpli_trial, eeg, max_jobs=n_jobs)
    return wpli_matrix
