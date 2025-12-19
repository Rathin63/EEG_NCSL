
"""
eeg_lognormal_multivariate.py
============================

Multivariate log-normal feature extraction for EEG energy.

This module:
1. Partitions EEG into fixed windows
2. Computes energy per channel per window
3. Normalizes each channel by its maximum energy
4. Applies log-transform
5. Estimates a single multivariate log-normal model per subject

Final outputs:
- Mean vector (n_channels,)
- Covariance matrix (n_channels, n_channels)

Input convention:
- EEG array shape = (n_channels, n_samples)
- Channels are rows, time is columns
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from typing import Tuple, Optional


def extract_multivariate_lognormal_energy(
    eeg: np.ndarray,
    fs_hz: float,
    window_ms: float = 125.0,
    step_ms: Optional[float]= None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    eeg : np.ndarray
        Shape (n_channels, n_samples)
    fs_hz : float
        Sampling frequency (Hz)
    window_ms : float
        Window length in milliseconds
    step_ms : float or None
        Step size in ms (None = non-overlapping)
    eps : float
        Small constant to avoid log(0)

    Returns
    -------
    mu_vec : np.ndarray
        Mean vector of log-energy (n_channels,)
    cov_mat : np.ndarray
        Covariance matrix of log-energy (n_channels, n_channels)
    log_energy_windows : np.ndarray
        Log-energy samples used for estimation (n_windows, n_channels)
    """
    if eeg.ndim != 2:
        raise ValueError("EEG must be 2D (n_channels, n_samples)")

    n_channels, n_samples = eeg.shape

    if step_ms is None:
        step_ms = window_ms

    win_len = int(round(window_ms * fs_hz / 1000.0))
    step_len = int(round(step_ms * fs_hz / 1000.0))

    if win_len < 2:
        raise ValueError("Window too small")

    n_windows = 1 + (n_samples - win_len) // step_len
    if n_windows <= 1:
        raise ValueError("Not enough data for multivariate estimation")

    # 1. Energy per channel per window
    energy = np.zeros((n_channels, n_windows))
    for w in range(n_windows):
        start = w * step_len
        end = start + win_len
        seg = eeg[:, start:end]
        energy[:, w] = np.sum(seg ** 2, axis=1)

    # 2. Channel-wise normalization
    max_energy = np.max(energy, axis=1, keepdims=True)
    max_energy[max_energy == 0] = eps
    energy_norm = energy / max_energy

    # 3. Log-transform
    log_energy = np.log(energy_norm + eps)
    log_energy_windows = log_energy.T  # (n_windows, n_channels)

    # 4. Multivariate MLE
    mu_vec = np.mean(log_energy_windows, axis=0)
    cov_mat = np.cov(log_energy_windows, rowvar=False, ddof=0)



    return mu_vec, cov_mat, log_energy_windows
