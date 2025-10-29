#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Processing Pipeline: Load, Preprocess, Compute A matrices, and Visualize Sink Indices
"""

# %% Section 1: Imports and Setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Import preprocessing modules
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'preprocessing'))
sys.path.insert(0, os.path.join(os.getcwd(), 'preprocessing', 'TDBRAIN'))

from preprocessing.TDBRAIN.autopreprocessing import dataset as ds

# Configuration
PATIENT_ID = "sub-87974621_ses-1_task-restEC_eeg"  # Change this to your patient ID
DATA_PATH = "E:\\JHU_Postdoc\\Research\\TDBrain\\TD_BRAIN_code\\BRAIN_code\\Sample\\diff_data"  # Update with your data path
OUTPUT_PATH = ".\\output"  # Where to save results

# Create output directory
Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

print(f"Processing patient: {PATIENT_ID}")
print(f"Data path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# %% Section 2: Load Raw Data
print("\n" + "=" * 60)
print("SECTION 2: LOADING RAW DATA")
print("=" * 60)

# Construct file path (adjust based on your file structure)
# Assuming .csv format, modify if using .edf
eeg_file = os.path.join(DATA_PATH, f"{PATIENT_ID}.csv")

# Check if file exists
if not os.path.exists(eeg_file):
    print(f"ERROR: File not found: {eeg_file}")
    print("Please update DATA_PATH and PATIENT_ID")
else:
    print(f"Loading data from: {eeg_file}")

    # Initialize dataset object
    eeg_data = ds(eeg_file)

    # Load the data
    eeg_data.loaddata()

    print(f"Data loaded successfully!")
    print(f"Data shape: {eeg_data.data.shape}")
    print(f"Number of channels: {len(eeg_data.labels)}")
    print(f"Sampling frequency: {eeg_data.Fs} Hz")
    print(f"Channel labels: {eeg_data.labels[:10]}...")  # Show first 10 channels

# %% Section 3: Preprocessing
print("\n" + "=" * 60)
print("SECTION 3: PREPROCESSING")
print("=" * 60)

# Convert to bipolar EOG
print("Converting to bipolar EOG...")
eeg_data.bipolarEOG()

# Apply filters (notch, highpass, lowpass)
print("Applying filters...")
eeg_data.apply_filters(hpfreq=0.5, lpfreq=100, notchfreq=50)

# Correct EOG artifacts
print("Correcting EOG artifacts...")
eeg_data.correct_EOG()

# Detect EMG artifacts
print("Detecting EMG artifacts...")
eeg_data.detect_emg()

# Detect jumps/baseline shifts
print("Detecting jumps and baseline shifts...")
eeg_data.detect_jumps()

# Detect kurtosis artifacts
print("Detecting kurtosis artifacts...")
eeg_data.detect_kurtosis()

# Detect extreme voltage swings
print("Detecting extreme voltage swings...")
eeg_data.detect_extremevoltswing()

# Detect residual eyeblinks
print("Detecting residual eyeblinks...")
eeg_data.residual_eyeblinks()

# Define and mark artifacts
print("Defining artifacts...")
eeg_data.define_artifacts()

print(f"Preprocessing complete!")
print(f"Data quality: {eeg_data.info.get('data quality', 'Unknown')}")
print(f"Repaired channels: {eeg_data.info.get('repaired channels', 'None')}")

# %% Section 4: Display Preprocessed Data
print("\n" + "=" * 60)
print("SECTION 4: VISUALIZING PREPROCESSED DATA")
print("=" * 60)

# Segment data for visualization (10 second segments)
# eeg_data.segment(trllength=10, remove_artifact='no')

# Plot a sample of the preprocessed data
fig, ax = plt.subplots(figsize=(15, 8))

# Plot first few channels for 10 seconds
n_channels_to_plot = max(5, len(eeg_data.labels))
time_vector = np.arange(eeg_data.data.shape[2]) / eeg_data.Fs

for i in range(n_channels_to_plot):
    # Plot first trial, offset each channel
    offset = i * 50  # Adjust offset for better visualization
    ax.plot(time_vector, eeg_data.data[0, i, :] + offset, label=eeg_data.labels[i])

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (μV)')
ax.set_title(f'Preprocessed EEG Data - {PATIENT_ID}')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_preprocessed_sample.png'), dpi=150)
plt.show()

print(f"Sample plot saved to: {OUTPUT_PATH}")

# %% Section 5: Compute A Matrices
print("\n" + "=" * 60)
print("SECTION 5: COMPUTING A MATRICES")
print("=" * 60)

# Re-segment data to continuous (all data in one segment)
# eeg_data.segment(trllength='all', remove_artifact='no')

# Extract continuous data
data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)
n_channels = data_continuous.shape[0]
n_samples = data_continuous.shape[1]

print(f"Data for A matrix computation: {data_continuous.shape}")
print(f"Channels: {n_channels}, Samples: {n_samples}")

# Parameters for A matrix computation
window_length = 0.15 #0.045  # 45ms window (from MATEO code)
alpha = 1e-12  # Regularization parameter
fs = eeg_data.Fs

window_length_samples = int(fs * window_length)
window_advance_samples = window_length_samples  # Non-overlapping windows

print(f"Window length: {window_length}s ({window_length_samples} samples)")
print(f"Computing A matrices...")


def compute_A_matrix(X, alpha, n_channels):
    """
    Compute A transition matrix from data window X
    X: (n_channels, n_samples)
    Returns: A matrix (n_channels, n_channels)
    """
    nchns, T = X.shape

    Z = X[:, 0:T - 1]
    Y = X[:, 1:T]

    Z2 = Z @ Z.T + alpha * np.eye(nchns)
    D = np.linalg.pinv(Z2)
    D2 = Z.T @ D
    A_hat = Y @ D2

    return A_hat


# Compute A matrices for all windows
window_start = 0
A_matrices = []

while window_start < (n_samples - window_length_samples):
    X = data_continuous[:, window_start:window_start + window_length_samples]

    # Standardize the data
    mean_vec = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_vec
    std_vec = np.std(X_centered, axis=1, keepdims=True)
    std_vec[std_vec == 0] = 1  # Avoid division by zero

    # Compute A matrix
    A_hat = compute_A_matrix(X, alpha, n_channels)
    A_matrices.append(A_hat)

    window_start += window_advance_samples

A_matrices = np.array(A_matrices)
A_matrices = np.transpose(A_matrices, (1, 2, 0))  # Shape: (n_channels, n_channels, n_windows)

print(f"A matrices computed!")
print(f"Shape: {A_matrices.shape}")
print(f"Number of windows: {A_matrices.shape[2]}")

# %% Section 5.5: Reconstruct Signal from A Matrices
print("\n" + "=" * 60)
print("SECTION 5.5: RECONSTRUCTING SIGNAL")
print("=" * 60)

# Initialize reconstructed signal
data_reconstructed = np.zeros_like(data_continuous)

n_windows = A_matrices.shape[2]  # ADD THIS LINE
channel_labels = eeg_data.labels

print(f"Reconstructing signal using A matrices...")
print(f"Window length: {window_length_samples} samples")
print(f"Number of windows: {n_windows}")

# Reconstruct signal window by window
for win_idx in range(n_windows):
    # Get the A matrix for this window
    A = A_matrices[:, :, win_idx]

    # Define window boundaries
    win_start = win_idx * window_length_samples
    win_end = min((win_idx + 1) * window_length_samples, n_samples)

    # Set initial condition: first sample of this window from true signal
    data_reconstructed[:, win_start] = data_continuous[:, win_start]

    # Recursively reconstruct within this window: X(t+1) = A * X(t)
    for t in range(win_start + 1, win_end):
        data_reconstructed[:, t] = A @ data_reconstructed[:, t - 1]

    if (win_idx + 1) % 10 == 0:
        print(f"  Processed {win_idx + 1}/{n_windows} windows...")

print(f"Signal reconstruction complete!")

# Calculate reconstruction quality metrics
mse = np.mean((data_continuous - data_reconstructed) ** 2)
correlation = np.corrcoef(data_continuous.flatten(), data_reconstructed.flatten())[0, 1]
print(f"\nReconstruction Quality:")
print(f"  Mean Squared Error: {mse:.2e}")
print(f"  Correlation: {correlation:.4f}")

# %% Section 5.6: Visualize Original vs Reconstructed Signal
print("\n" + "=" * 60)
print("SECTION 5.6: VISUALIZING RECONSTRUCTION")
print("=" * 60)

# Select channels to plot
n_channels_to_plot = min(5, n_channels)
channels_to_plot = np.linspace(0, n_channels - 1, n_channels_to_plot, dtype=int)

# Time window to display (first 10 seconds)
plot_duration = 10  # seconds
plot_samples = int(plot_duration * fs)
plot_samples = min(plot_samples, n_samples)
time_vector = np.arange(plot_samples) / fs

# Create figure with subplots for each channel
fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(16, 10), sharex=True)

if n_channels_to_plot == 1:
    axes = [axes]

print(f"Plotting first {plot_duration}s of {n_channels_to_plot} channels...")

for idx, ch in enumerate(channels_to_plot):
    ax = axes[idx]

    # Plot original signal in color (thicker line)
    ax.plot(time_vector, data_continuous[ch, :plot_samples] ,
            color='green', linewidth=2.0, alpha=0.7, label='Original')

    # Plot reconstructed signal in black (thinner line, on top)
    ax.plot(time_vector, data_reconstructed[ch, :plot_samples] ,
            color='black', linewidth=0.8, alpha=0.8, label='Reconstructed')

    # Calculate per-channel correlation
    ch_corr = np.corrcoef(
        data_continuous[ch, :plot_samples],
        data_reconstructed[ch, :plot_samples]
    )[0, 1]

    # Styling
    ax.set_ylabel(f'{channel_labels[ch]}\n(μV)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Channel {channel_labels[ch]} (r={ch_corr:.3f})',
                 fontsize=10, loc='right', pad=5)

    ax.set(ylim=[-50,50])

    # Add legend only to first subplot
    if idx == 0:
        ax.legend(loc='upper right', fontsize=10)

    # Mark window boundaries with vertical lines
    for win in range(int(plot_samples / window_length_samples) + 1):
        win_time = win * window_length
        if win_time <= plot_duration:
            ax.axvline(win_time, color='red', linestyle='--',
                       alpha=0.3, linewidth=0.8)

# Set common x-label
axes[-1].set_xlabel('Time (s)', fontsize=12)

# Overall title
fig.suptitle(
    f'Original vs Reconstructed Signal - {PATIENT_ID}\n'
    f'Window size: {window_length}s | Overall correlation: {correlation:.4f}',
    fontsize=14,
    fontweight='bold',
    y=0.995
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_reconstruction_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print(f"Reconstruction plot saved to: {OUTPUT_PATH}")
print(f"\nPer-channel correlations:")
for ch in channels_to_plot:
    ch_corr = np.corrcoef(data_continuous[ch, :], data_reconstructed[ch, :])[0, 1]
    # print(f"  {channel_labels[ch]}: {ch_corr:.4f}")
# %% Section 6: Compute Sink Indices
print("\n" + "="*60)
print("SECTION 6: COMPUTING SINK INDICES")
print("="*60)

# Import the identifySS function from utils
from preprocessing.utils import identifySS

# Compute sink and source indices for each window
n_windows = A_matrices.shape[2]
sink_indices = np.zeros((n_channels, n_windows))
source_indices = np.zeros((n_channels, n_windows))
row_ranks = np.zeros((n_channels, n_windows))
col_ranks = np.zeros((n_channels, n_windows))

print("Computing sink and source indices for each window...")

for i in range(n_windows):
    sink_idx, source_idx, row_rank, col_rank = identifySS(A_matrices[:, :, i])
    sink_indices[:, i] = sink_idx
    source_indices[:, i] = source_idx
    row_ranks[:, i] = row_rank
    col_ranks[:, i] = col_rank

print(f"Sink indices computed!")
print(f"Shape: {sink_indices.shape}")

# Print overall statistics
overall_sink = np.mean(sink_indices, axis=1)
overall_source = np.mean(source_indices, axis=1)
print(f"\nOverall Statistics:")
print(f"  Top sink channel: {eeg_data.labels[np.argmax(overall_sink)]} (index: {np.max(overall_sink):.4f})")
print(f"  Top source channel: {eeg_data.labels[np.argmax(overall_source)]} (index: {np.max(overall_source):.4f})")

# %% Section 7: Visualize Sink Index Heatmap
print("\n" + "=" * 60)
print("SECTION 7: VISUALIZING SINK INDEX HEATMAP")
print("=" * 60)

# Sort channels by mean sink index
mean_sink = np.mean(sink_indices, axis=1)
sort_idx = np.argsort(mean_sink)[::-1]
sink_sorted = sink_indices[sort_idx, :]

# Get channel labels (first 26 channels)
channel_labels = eeg_data.labels[:26]
labels_sorted = [channel_labels[i] for i in sort_idx]

# Create heatmap
fig, ax = plt.subplots(figsize=(20, 12))

sns.heatmap(
    sink_sorted,
    yticklabels=labels_sorted,
    cmap=sns.color_palette("rainbow", as_cmap=True),
    cbar_kws={"pad": 0.01, "label": "Sink Index"},
    ax=ax
)

ax.set_title(
    f"Sink Index Heatmap - {PATIENT_ID}\n"
    f"Window size: {window_length}s | Channels sorted by mean sink index",
    fontsize=14
)
ax.set_xlabel("Window Index")
ax.set_ylabel("Channel")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_sink_heatmap.png'), dpi=150)
plt.show()

print(f"Heatmap saved to: {OUTPUT_PATH}")

# Print summary statistics
print("\nTop 5 Sink Channels (by mean):")
for i in range(min(5, len(labels_sorted))):
    print(f"  {i + 1}. {labels_sorted[i]}: {mean_sink[sort_idx[i]]:.4f}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print("=" * 60)

# %% Section 8: Visualize Sink Index Topomap
print("\n" + "="*60)
print("SECTION 8: VISUALIZING SINK INDEX TOPOMAP")
print("="*60)

import mne

# Calculate mean sink index over entire recording
mean_sink_index = np.mean(sink_indices, axis=1)

print(f"Mean sink index range: [{mean_sink_index.min():.4f}, {mean_sink_index.max():.4f}]")

# Create MNE info structure for topographic plotting
# Use only the EEG channel labels (first 26 channels)
ch_names = channel_labels[:n_channels]
ch_names = ch_names.tolist()
sfreq = eeg_data.Fs
ch_types = ['eeg'] * n_channels
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Set standard 10-20 montage
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Create figure for topomap
fig, ax = plt.subplots(figsize=(10, 8))

# Plot topomap with mean sink indices
im, _ = mne.viz.plot_topomap(
    mean_sink_index,
    info,
    axes=ax,
    show=False,
    cmap='RdBu_r',
    vlim=(mean_sink_index.min(), mean_sink_index.max()),
    contours=6,
    names=ch_names,
    # show_names=True
)

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Mean Sink Index', rotation=270, labelpad=20, fontsize=12)

# Set title
ax.set_title(
    f'Mean Sink Index Topographic Map - {PATIENT_ID}\n'
    f'Averaged over {n_windows} windows ({n_windows * window_length:.1f}s total)',
    fontsize=14,
    fontweight='bold',
    pad=20
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_sink_topomap.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"Topomap saved to: {OUTPUT_PATH}")
print(f"\nTop 5 Sink Channels (by mean sink index):")
top_sink_idx = np.argsort(mean_sink_index)[::-1][:5]
for i, idx in enumerate(top_sink_idx):
    print(f"  {i+1}. {ch_names[idx]}: {mean_sink_index[idx]:.4f}")