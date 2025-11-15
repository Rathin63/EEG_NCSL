#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Processing Pipeline: Load, Preprocess, Compute A matrices, and Visualize Sink Indices
"""

# %% Section 0: Imports and Setup
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

# Input and Output Path
DATA_PATH = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2"
BATCH_OUTPUT_PATH = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Batch_Outputs"

Path(BATCH_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# ---- NEW SECTION: Batch File Loop ----
csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
if not csv_files:
    print(f"No CSV files found in {DATA_PATH}")
    sys.exit()

print(f"\nFound {len(csv_files)} CSV files for processing:\n")
for f in csv_files:
    print(f"  - {f}")

# Loop through all CSVs
for file_idx, file_name in enumerate(csv_files, start=1):
    PATIENT_ID = Path(file_name).stem
    print("\n" + "="*80)
    print(f"Processing patient: {PATIENT_ID}")
    print("="*80)

    # 🔹 Create per-subject output folder
    OUTPUT_PATH = os.path.join(BATCH_OUTPUT_PATH, PATIENT_ID)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    eeg_file = os.path.join(DATA_PATH, file_name)
    eeg_data = ds(eeg_file)
    eeg_data.loaddata()

    print(f"Data loaded successfully!")
    print(f"Shape: {eeg_data.data.shape}")
    print(f"Sampling frequency: {eeg_data.Fs} Hz")
    print(f"Channels: {len(eeg_data.labels)}")


# %% Section 1: Load Raw Data
    print("\n" + "=" * 60)
    print("SECTION 1: LOADING RAW DATA")
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
        print(f"Duration: {eeg_data.data.shape[1] / eeg_data.Fs} seconds")
        print(f"Channel labels: {eeg_data.labels[:]}...")  # Show first 10 channels

# %% Section 2: Preprocessing
    print("\n" + "=" * 60)
    print("SECTION 2: PREPROCESSING")
    print("=" * 60)

    # Convert to bipolar EOG
    print("Converting to bipolar EOG...")
    eeg_data.bipolarEOG()

    # Apply filters (notch, highpass, lowpass)
    print("Applying filters...")
    eeg_data.apply_filters(hpfreq=0.5, lpfreq=48, notchfreq=50)

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

# %% Section 3: Display Preprocessed Data
    print("\n" + "=" * 60)
    print("SECTION 3: VISUALIZING PREPROCESSED DATA")
    print("=" * 60)

    # Segment data for visualization (10 second segments)
    eeg_data.segment(trllength=20, remove_artifact='no')

    # Plot a sample of the preprocessed data
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot first few channels for 5 seconds
    n_channels_to_plot = min(5, len(eeg_data.labels))
    time_vector = np.arange(eeg_data.data.shape[2]) / eeg_data.Fs

    for i in range(n_channels_to_plot):
        # Plot first trial, offset each channel
        offset = i * 100  # Adjust offset for better visualization
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

# %% Section 4: Compute A Matrices
    print("\n" + "=" * 60)
    print("SECTION 4: COMPUTING A MATRICES")
    print("=" * 60)

    # Re-segment data to continuous (all data in one segment)
    eeg_data.segment(trllength='all', remove_artifact='no')

    # Extract continuous data
    data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)
    n_channels = data_continuous.shape[0]
    n_samples = data_continuous.shape[1]

    print(f"Data for A matrix computation: {data_continuous.shape}")
    print(f"Channels: {n_channels}, Samples: {n_samples}")

    # Parameters for A matrix computation
    window_length = 0.5 #0.045  # 45ms window (from MATEO code)
    alpha = 1e-6  # Regularization parameter
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

        Z = X[:, 0:T - 1] # Takes all columns except the last → represents data from time t to T−1 (past samples).
        Y = X[:, 1:T] #Takes all columns except the first → represents data from time t+1 to T (future samples).

        # Computes regularized autocovariance matrix of Z:  #Z @ Z.T gives covariance between channels;
        # alpha * np.eye(nchns) adds diagonal regularization (ridge term) to ensure matrix invertibility.
        Z2 = Z @ Z.T + alpha * np.eye(nchns)

        #Computes the Moore–Penrose pseudo-inverse of the regularized covariance matrix Z2. This step is needed for numerical stability if Z2 is near-singular.
        D = np.linalg.pinv(Z2)

        D2 = Z.T @ D #This forms part of the regression solution (preparing to estimate the transition mapping).

        A_hat = Y @ D2 #Computes the transition (connectivity) matrix:
        # This gives how each channel’s next-time activity depends on all channels’ current activity.

        #Returns the estimated A_hat, a square matrix of size (n_channels × n_channels)
        # describing how each channel at time t influences every channel at time t + 1.
        return A_hat


    # Compute A matrices for all windows
    window_start = 0 #Initializes starting index of the first time window.
    A_matrices = [] #Empty list to store A-matrices from each window.

    while window_start < (n_samples - window_length_samples): #Loop through the continuous data in chunks
        X = data_continuous[:, window_start:window_start + window_length_samples] #Extracts one time window Shape: (n_channels, window_length_samples)

        # Standardize the data
        mean_vec = np.mean(X, axis=1, keepdims=True)  #Computes mean of each channel (along time). Used for centering.
        X_centered = X - mean_vec #makes each channel zero-mean within this window.
        std_vec = np.std(X_centered, axis=1, keepdims=True) #Computes standard deviation of each channel.
        std_vec[std_vec == 0] = 1  # Avoid division by zero

        # Compute A matrix
        A_hat = compute_A_matrix(X, alpha, n_channels) #Computes the transition (A) matrix for this window using the earlier routine.
        A_matrices.append(A_hat) #Stores the A-matrix for this time window.

        window_start += window_advance_samples #Moves the sliding window ahead by a fixed number of samples (defines overlap)

    A_matrices = np.array(A_matrices) #Converts the list to a 3-D NumPy array. Shape: (n_windows, n_channels, n_channels).
    A_matrices = np.transpose(A_matrices, (1, 2, 0))  # Shape: (n_channels, n_channels, n_windows)

    print(f"A matrices computed!")
    print(f"Shape: {A_matrices.shape}")
    print(f"Number of windows: {A_matrices.shape[2]}")

# %% Section 5: Signal Reconstruction
    print("\n" + "=" * 60)
    print("SECTION 5: RECONSTRUCTING SIGNAL")
    print("=" * 60)

    # Initialize reconstructed signal
    data_reconstructed = np.zeros_like(data_continuous) #data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)

    n_windows = A_matrices.shape[2]  #Total window count
    channel_labels = eeg_data.labels # channel labels

    print(f"Reconstructing signal using A matrices...")
    print(f"Window length: {window_length_samples} samples")
    print(f"Number of windows: {n_windows}")

    # Reconstruct signal window by window
    for win_idx in range(n_windows): #Iterates over each window index, reconstructing one segment at a time.

        A = A_matrices[:, :, win_idx]

        # Define window boundaries
        win_start = win_idx * window_length_samples #Starting sample index for this window.
        win_end = min((win_idx + 1) * window_length_samples, n_samples) #Ending index for the window (ensures not to exceed total samples).

        # Set initial condition: first sample of this window from true signal
        data_reconstructed[:, win_start] = data_continuous[:, win_start]

        # Recursively reconstruct within this window: X(t+1) = A * X(t)
        for t in range(win_start + 1, win_end):
            data_reconstructed[:, t] = A @ data_reconstructed[:, t - 1] #Use Current A Mat to get the next Sample. Recursively approximate the entire signal.
            #X^(t+1) = A X^(t)

        if (win_idx + 1) % 10 == 0: #Every 10 windows, prints progress to monitor long runs.
            print(f"  Processed {win_idx + 1}/{n_windows} windows...")

    print(f"Signal reconstruction complete!")

    # Calculate reconstruction quality metrics
    mse = np.mean((data_continuous - data_reconstructed) ** 2)
    correlation = np.corrcoef(data_continuous.flatten(), data_reconstructed.flatten())[0, 1]
    print(f"\nReconstruction Quality:")
    print(f"  Mean Squared Error: {mse:.2e}")
    print(f"  Correlation: {correlation:.4f}")

    # Suppose:
    # data_continuous: (n_channels, n_samples)
    # data_reconstructed: (n_channels, n_samples)
    # window_length_samples, window_advance_samples already defined

    n_channels, n_samples = data_continuous.shape

    # Define window starts (same as in reconstruction)
    window_starts = np.arange(0, n_samples - window_length_samples + 1, window_advance_samples)
    n_windows = len(window_starts)

    # Initialize R² array
    R2_values = np.zeros((n_channels, n_windows))
    MSE_values = np.zeros((n_channels, n_windows))

    for w, start in enumerate(window_starts):
        end = start + window_length_samples

        # Extract segment for this window
        Y_true = data_continuous[:, start:end]
        Y_pred = data_reconstructed[:, start:end]

        # Mean Squared Error (per channel)
        mse = np.mean((Y_true - Y_pred) ** 2, axis=1)
        MSE_values[:, w] = mse

        # Compute per-channel R²
        ss_res = np.sum((Y_true - Y_pred) ** 2, axis=1)
        ss_tot = np.sum((Y_true - np.mean(Y_true, axis=1, keepdims=True)) ** 2, axis=1)
        R2_values[:, w] = 1 - (ss_res / ss_tot)

    # ---- Create side-by-side subplots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # ---- Left subplot: MSE ----
    for ch in range(n_channels):
        axes[0].scatter(
            np.full(n_windows, ch+1),
            MSE_values[ch, :],
            color='tab:orange',
            alpha=0.6,
            s=25
        )

    axes[0].set_xlabel("Channels")
    axes[0].set_ylabel("MSE across windows")
    axes[0].set_title("Channel-wise Reconstruction Error (MSE)")
    axes[0].set_xticks(range(1, n_channels+1))
    axes[0].set_xticklabels(eeg_data.labels[:26])
    #axes[0].set_xticklabels([f"Ch{i+1}" for i in range(n_channels)])
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.6)
    axes[0].set_yscale('log')  # optional, use if MSE varies widely

    # ---- Right subplot: R² ----
    for ch in range(n_channels):
        axes[1].scatter(
            np.full(n_windows, ch+1),
            R2_values[ch, :],
            color='tab:blue',
            alpha=0.6,
            s=25
        )

    axes[1].set_xlabel("Channels")
    axes[1].set_ylabel("R² across windows")
    axes[1].set_title("Channel-wise Reconstruction Quality (R²)")
    axes[1].set_xticks(range(1, n_channels+1))
    axes[1].set_xticklabels(eeg_data.labels[:26])
    #axes[1].set_xticklabels([f"Ch{i+1}" for i in range(n_channels)])
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_reconstruction_metric_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()


    # # ---- Plot after loop ----
    # plt.figure(figsize=(10,5))
    # plt.boxplot(
    #     MSE_values.T,
    #     labels=[f"Ch{i+1}" for i in range(n_channels)],
    #     showfliers=False
    # )
    # plt.xlabel("Channels")
    # plt.ylabel("MSE across all windows")
    # plt.title("Channel-wise Reconstruction Error (MSE)")
    # plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
    #
    # # Optional: also show R² boxplot
    # plt.figure(figsize=(10,5))
    # plt.boxplot(
    #     R2_values.T,
    #     labels=[f"Ch{i+1}" for i in range(n_channels)],
    #     showfliers=False
    # )
    # plt.xlabel("Channels")
    # plt.ylabel("R² across all windows")
    # plt.title("Channel-wise Reconstruction Quality (R²)")
    # plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()

# %% Section 6: Visualize Original vs Reconstructed Signal
    print("\n" + "=" * 60)
    print("SECTION 6: VISUALIZING RECONSTRUCTION")
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


# %% Section 7: Compute Sink Indices
    print("\n" + "="*60)
    print("SECTION 7: COMPUTING SINK INDICES")
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

# %% Section 8: Visualize Sink Index Heatmap
    print("\n" + "=" * 60)
    print("SECTION 8: VISUALIZING SINK INDEX HEATMAP")
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

# %% Section 9: Visualize Sink Index Topomap
    print("\n" + "="*60)
    print("SECTION 9: VISUALIZING SINK INDEX TOPOMAP")
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
        #vlim=(mean_sink_index.min(), mean_sink_index.max()),
        vlim=(0,1.4),
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
    for i, ch_idx in enumerate(top_sink_idx):
        print(f"  {i+1}. {ch_names[ch_idx]}: {mean_sink_index[ch_idx]:.4f}")
    print(f"\n✅ Completed {file_idx}/{len(csv_files)} "
          f"({(file_idx / len(csv_files)) * 100:.1f}%) files. "
          f"Results saved in: {OUTPUT_PATH}\n")
