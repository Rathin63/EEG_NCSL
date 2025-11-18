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

results = []

# Loop through all CSVs
for file_idx, file_name in enumerate(csv_files, start=1):
    PATIENT_ID = Path(file_name).stem
    print("\n" + "="*80)
    print(f"Preprocessing Subject: {PATIENT_ID}")
    print("="*80)

    # 🔹 Create per-subject output folder
    OUTPUT_PATH = os.path.join(BATCH_OUTPUT_PATH, PATIENT_ID)
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    # Full file path
    eeg_file = os.path.join(DATA_PATH, file_name)

# %% Section 1: Load Raw Data
    print("\n" + "=" * 60)
    print("SECTION 1: LOADING RAW DATA")
    print("=" * 60)

    # File existence check (single, correct place)
    if not os.path.exists(eeg_file):
        print(f"ERROR: File not found: {eeg_file}")
        print("Skipping...")
        continue

    print(f"Loading data from: {eeg_file}")

    # Load dataset
    eeg_data = ds(eeg_file)
    eeg_data.loaddata()

    # Basic info
    print(f"Data loaded successfully!")
    print(f"Shape: {eeg_data.data.shape}")
    print(f"Sampling frequency: {eeg_data.Fs} Hz")
    print(f"Channels: {len(eeg_data.labels)}")
    print(f"Channel labels: {eeg_data.labels[:]}...")  # Show first 10 channels

    # Duration (if needed)
    orig_duration = eeg_data.data.shape[1] / eeg_data.Fs
    print(f"Original duration: {orig_duration:.2f} sec")

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
    eeg_data.define_artifacts()

    # Option 1: Remove artifacts
    print("EEG Shape Before removal:", eeg_data.data.shape)
    print("Opted for Artifact Removal..")
    print("Defining artifacts...")
    eeg_data.remove_artifacts(remove_art=True)
    print("EEG Shape After removal:", eeg_data.data.shape)
    clean_duration = eeg_data.data.shape[1] / eeg_data.Fs # Result---2

    # Option 2: Keep artifacts #TODO Add a helper function here later
    # eeg_data.remove_artifacts(remove_art=False)
    # print("Not Opted for Artifact Removal..")
    # clean_duration = eeg_data.data.shape[1] / eeg_data.Fs # Result---2

    # After define_artifacts()
    if 'artifacts' in eeg_data.labels:
        artidx = np.where(eeg_data.labels == 'artifacts')[0]
        if len(artidx) > 0 and artidx[0] < eeg_data.data.shape[0]:
            # FIX: data is 2D (channels, samples), not 3D
            artifact_mask = eeg_data.data[artidx[0], :] == 1  # Removed [0] indexing
            artifact_percent = np.mean(artifact_mask) * 100

            n_samples = eeg_data.data.shape[1]
            fs = eeg_data.Fs

            print(f"\n📊 Artifact Statistics:")
            print(f"  Total duration: {n_samples / fs:.1f}s")
            print(f"  Artifact duration: {np.sum(artifact_mask) / fs:.1f}s ({artifact_percent:.1f}%)")
            print(f"  Clean duration: {np.sum(~artifact_mask) / fs:.1f}s ({100 - artifact_percent:.1f}%)")

            if artifact_percent > 30:
                print("  ⚠️ Warning: >30% artifacts - consider excluding artifact windows")
            else:
                print("  ✅ Artifact level acceptable for continuous analysis")

    print(f"Preprocessing complete!")
    print(f"Data quality: {eeg_data.info.get('data quality', 'Unknown')}")
    print(f"Repaired channels: {eeg_data.info.get('repaired channels', 'None')}")

    # %% Section 2B: Compute Band Power Features (Standardized)
    print("\n" + "=" * 60)
    print("SECTION 2B: BAND POWER ANALYSIS (STANDARDIZED)")
    print("=" * 60)

    from scipy.signal import welch

    # Extract EEG data (first 26 channels)
    data_clean = eeg_data.data[:26, :]  # channels x samples
    fs = eeg_data.Fs
    n_channels = data_clean.shape[0]

    # EEG bands
    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 48)  # FULL gamma instead of clipped at 48 Hz
    }

    # Containers
    total_power = np.zeros(n_channels)
    band_power = {band: np.zeros(n_channels) for band in bands}
    band_power_percent = {band: np.zeros(n_channels) for band in bands}
    TBR = np.zeros(n_channels)  # Theta/Beta ratio

    print("Computing PSD using Welch method...")

    for ch in range(n_channels):

        freqs, psd = welch(
            data_clean[ch],
            fs=fs,
            nperseg=fs * 2,
            noverlap=fs,
            scaling='density'
        )

        # Total spectral power
        total_power[ch] = np.trapz(psd, freqs)

        # Compute band powers
        for band_name, (fmin, fmax) in bands.items():
            idx = (freqs >= fmin) & (freqs <= fmax)
            bp = np.trapz(psd[idx], freqs[idx])  # integrate PSD
            band_power[band_name][ch] = bp

        # Standardize to percentage of total power
        for band_name in bands:
            if total_power[ch] > 0:
                band_power_percent[band_name][ch] = \
                    (band_power[band_name][ch] / total_power[ch]) * 100

        # Theta/Beta Ratio
        theta = band_power["theta"][ch]
        beta = band_power["beta"][ch]
        TBR[ch] = theta / beta if beta > 0 else np.nan

    print("Band power (absolute, %, and TBR) computation complete!\n")

    # ---------------------- Summaries ---------------------------
    rel_band = {}

    print("Average % Band Powers Across Channels:")
    for band_name in bands:
        value = np.mean(band_power_percent[band_name])
        rel_band[band_name] = value  # store the result
        print(f"  {band_name.capitalize():6s}: {np.mean(band_power_percent[band_name]):.2f}%")

    print("\nTheta/Beta Ratio (TBR):")
    print(f"  Mean TBR: {np.nanmean(TBR):.3f}")
    print(f"  TBR range: [{np.nanmin(TBR):.3f}, {np.nanmax(TBR):.3f}]")
    mean_TBR_val = np.nanmean(TBR) # Result---3

    # %% Section 2C: Combined PSD + Band Power Diagram
    print("\n" + "=" * 60)
    print("SECTION 2D: PSD + BAND POWER DIAGRAM")
    print("=" * 60)

    import matplotlib.pyplot as plt
    from scipy.signal import welch

    # Compute mean PSD across channels
    psd_list = []

    for ch in range(n_channels):
        freqs, psd = welch(
            data_clean[ch],
            fs=fs,
            nperseg=fs * 2,
            noverlap=fs,
            scaling='density'
        )
        psd_list.append(psd)

    # Mean PSD across channels
    mean_psd = np.mean(psd_list, axis=0)

    # Mean % band power for bar chart
    band_names = list(bands.keys())
    mean_band_percent = [np.mean(band_power_percent[b]) for b in band_names]
    # Synchronized color palette for EEG bands
    band_colors_solid = {
        "delta": "#4e79a7",  # deep blue
        "theta": "#59a14f",  # deep green
        "alpha": "#f28e2b",  # deep orange
        "beta": "#e15759",  # deep red
        "gamma": "#76b7b2"  # deep cyan/teal
    }

    band_colors_light = {
        "delta": "#d0e1f9",  # light blue
        "theta": "#b5e2b1",  # light green
        "alpha": "#ffe5a5",  # light orange
        "beta": "#ffb8b1",  # light red/pink
        "gamma": "#cce5e8"  # light teal
    }

    # ---------------------------------------------------------
    #                CREATE COMBINED FIGURE
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=False, constrained_layout=True
    )

    # ---------------------------------------------------------
    #                     PANEL 1 – PSD
    # ---------------------------------------------------------

    ax1.plot(freqs, mean_psd)
    ax1.set_yscale("log")
    ymin = mean_psd[0:100].min()*0.9
    ymax = mean_psd[0:100].max()*1.1

    ax1.set_ylim(ymin, ymax)

    # Shade the EEG bands
    colors = {
        "delta": "#d0e1f9",
        "theta": "#b5e2b1",
        "alpha": "#ffe5a5",
        "beta": "#ffb8b1",
        "gamma": "#b5cdee"
    }

    for band, (fmin, fmax) in bands.items():
        ax1.axvspan(fmin, fmax, color=band_colors_light[band], alpha=0.35)
    # -------------------------------
    # Add legend for EEG bands
    # -------------------------------
    import matplotlib.patches as mpatches

    legend_patches = []
    for band in band_names:
        patch = mpatches.Patch(
            facecolor=band_colors_solid[band],
            edgecolor='black',
            label=f"{band.capitalize()} band"
        )
        legend_patches.append(patch)

    ax1.legend(
        handles=legend_patches,
        loc='upper right',
        fontsize=10,
        frameon=True,
        title="EEG Bands"
    )

    ax1.set_ylabel("PSD (µV²/Hz)", fontsize=12)
    ax1.set_xlabel("Frequecy (Hz)")
    ax1.set_xlim([0, 48])

    ax1.set_title("Power Spectrum Density (Averaged Across Channels)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    #                     PANEL 2 – BAND POWER %
    # ---------------------------------------------------------
    bars = ax2.bar(
        band_names,
        mean_band_percent,
        width=0.5,
        color=[band_colors_solid[b] for b in band_names],
        edgecolor='black'
    )

    ax2.set_ylabel("Relative Power (%)", fontsize=12)
    ax2.set_title("Relative Band Power (%)", fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add % labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha='center', va='bottom', fontsize=10
        )

    # ---------------------------------------------------------
    #                 COMMON SUPER-TITLE
    # ---------------------------------------------------------

    fig.suptitle(
        f"{PATIENT_ID} – PSD + Relative Band Power + TBR\n"
        f"Theta/Beta Ratio (TBR): {np.nanmean(TBR):.3f}",
        fontsize=15, fontweight='bold'
    )

    plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_band_composition.png'), dpi=150)
    plt.show()

    # %% Section 3: Display Preprocessed Data
    print("\n" + "=" * 60)
    print("SECTION 3: VISUALIZING PREPROCESSED DATA")
    print("=" * 60)

    # Segment data for visualization (10 second segments)
    eeg_data.segment(trllength='all', remove_artifact='no')#ToDo Add a copy of eeg_data here to plot only first few seconds

    # # Plot a sample of the preprocessed data
    # fig, ax = plt.subplots(figsize=(15, 8))
    #
    # # Plot first few channels for 5 seconds
    # n_channels_to_plot = min(5, len(eeg_data.labels))
    # time_vector = np.arange(eeg_data.data.shape[2]) / eeg_data.Fs
    #
    # for i in range(n_channels_to_plot):
    #     # Plot first trial, offset each channel
    #     offset = i * 100  # Adjust offset for better visualization
    #     ax.plot(time_vector, eeg_data.data[0, i, :] + offset, label=eeg_data.labels[i])
    #
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Amplitude (μV)')
    # ax.set_title(f'Preprocessed EEG Data - {PATIENT_ID}')
    # ax.legend(loc='upper right')
    # ax.grid(True, alpha=0.3)
    #
    # plt.tight_layout()
    # #plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_preprocessed_sample.png'), dpi=150)
    # plt.show()
    #
    # print(f"Sample plot saved to: {OUTPUT_PATH}")

# %% Section 4: Compute A Matrices
    print("\n" + "=" * 60)
    print("SECTION 4: COMPUTING A MATRICES")
    print("=" * 60)

    # Re-segment data to continuous (all data in one segment)
    #eeg_data.segment(trllength='all', remove_artifact='no')

    # Extract continuous data
    data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)
    n_channels = data_continuous.shape[0]
    n_samples = data_continuous.shape[1]

    print(f"Data for A matrix computation: {data_continuous.shape}")
    print(f"Channels: {n_channels}, Samples: {n_samples}")

    # Parameters for A matrix computation
    window_length = 0.15 # (in seconds)
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

    n_windows = A_matrices.shape[2]  #Total window count
    channel_labels = eeg_data.labels # channel labels

    # Initialize reconstructed signal
    data_reconstructed = np.zeros_like(data_continuous) #data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)
    data_reconstructed_good = np.full_like(data_continuous, np.nan, dtype=float)
    data_continuous_good = np.full_like(data_continuous, np.nan, dtype=float)
    window_correlations = np.zeros(n_windows)  # Initializing an array for window wise correlation

    print(f"Reconstructing signal using A matrices...")
    print(f"Window length: {window_length_samples} samples")
    print(f"Number of windows: {n_windows}")

    # ---------------------------------------------------------
    # PART A: FULL RECONSTRUCTION (ALL WINDOWS)
    # -------------------------------------------------------
    print("\n--- FULL RECONSTRUCTION ---")

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

        # Compute window correlation by Extracting true + reconstructed segments for this window
        true_seg = data_continuous[:, win_start:win_end]
        recon_seg = data_reconstructed[:, win_start:win_end]

        # Compute channel-wise correlation
        corr_vals = []


        for ch in range(true_seg.shape[0]):
            # np.corrcoef returns a 2x2 matrix; take (0,1)
            c = np.corrcoef(true_seg[ch], recon_seg[ch])[0, 1]
            corr_vals.append(c)

        # Average correlation for this window
        window_correlations[win_idx] = np.mean(corr_vals)

        if (win_idx + 1) % 10 == 0: #Every 10 windows, prints progress to monitor long runs.
            print(f"  Processed {win_idx + 1}/{n_windows} windows...")

    print(f"Full Signal reconstruction complete!")

    # ---------------------------------------------
    # PART B: CORRELATION THRESHOLDING
    # ---------------------------------------------
    print("\n--- CORRELATION QUALITY CHECK ---")

    corr_check = window_correlations  # shape: (n_windows,)
    threshold = 0.70

    # --- Good/bad window masks ---
    good_mask = corr_check >= threshold  # boolean mask
    bad_mask = corr_check < threshold  # boolean mask

    good_windows = np.where(good_mask)[0]
    bad_windows = np.where(bad_mask)[0]

    # --- Summary ---
    n_good = good_mask.sum()
    n_bad = bad_mask.sum()
    n_total = len(corr_check)
    percent_good = (n_good / n_total) * 100
    percent_bad = (n_bad / n_total) * 100

    print(f"\n=== Window Quality Summary ===")
    print(f"Good windows (>= {threshold}): {n_good}/{n_total} = {percent_good:.2f}%")
    print(f"Bad windows (< {threshold}): {n_bad}/{n_total} = {percent_bad:.2f}%")
    effective_duration = n_good * (window_length_samples / fs)

    # --- Keep A matrices for good windows only (for final analysis) ---
    A_good = A_matrices[:, :, good_mask]  # filtered A matrices

    print(f"A_good shape (for analysis): {A_good.shape}")

    # ---------------------------------------------------------
    # PART C: RECONSTRUCT USING GOOD WINDOWS ONLY
    # ---------------------------------------------------------
    print("\n--- GOOD-ONLY RECONSTRUCTION ---")


    for win_idx in range(n_windows):

        if not good_mask[win_idx]:
            # leave NaN section for bad windows
            continue

        A = A_matrices[:, :, win_idx]

        win_start = win_idx * window_length_samples
        win_end = min((win_idx + 1) * window_length_samples, n_samples)

        # Initial sample from true data
        data_reconstructed_good[:, win_start] = data_continuous[:, win_start]

        # Recursive reconstruction using ONLY good A’s
        for t in range(win_start + 1, win_end):
            data_reconstructed_good[:, t] = A @ data_reconstructed_good[:, t - 1]

    print("Good-only reconstruction complete!\n")

    # ---------------------------------------------------------
    # PART D: BUILD TRUE CONTINUOUS SIGNAL FOR GOOD WINDOWS
    # ---------------------------------------------------------
    print("\n--- BUILDING GOOD-ONLY CONTINUOUS SIGNAL ---")
    for win_idx in range(n_windows):

        if not good_mask[win_idx]:
            continue  # skip bad windows

        # Window boundaries
        win_start = win_idx * window_length_samples
        win_end = min((win_idx + 1) * window_length_samples, n_samples)

        # Copy true continuous data ONLY for good windows
        data_continuous_good[:, win_start:win_end] = data_continuous[:, win_start:win_end]
    print("Good-only continuous data built!")

    # ---------------------------------------------------------
    # PART E: HISTOGRAM
    # ---------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.hist(corr_check, bins=20, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold}')
    plt.xlabel('Window Correlation')
    plt.ylabel('Count')
    plt.title('Histogram of Window Correlations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate reconstruction quality metrics #TODO: To be removed
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
    #plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_reconstruction_metric_comparison.png'),
                #dpi=150, bbox_inches='tight')
    plt.show()

# %% Section 6A: Visualize Original vs Reconstructed Signal
    print("\n" + "=" * 60)
    print("SECTION 6A: VISUALIZING RECONSTRUCTION")
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
        print(f"  {channel_labels[ch]}: {ch_corr:.4f}")

    # %% Section 6B: Visualize Original vs Good-Only Reconstructed Signal
    print("\n" + "=" * 60)
    print("SECTION 6B: VISUALIZING GOOD-ONLY RECONSTRUCTION")
    print("=" * 60)

    # Use same channels and time window as before
    n_channels_to_plot = min(5, n_channels)
    channels_to_plot = np.linspace(0, n_channels - 1, n_channels_to_plot, dtype=int)

    plot_duration = 10  # seconds
    plot_samples = int(plot_duration * fs)
    plot_samples = min(plot_samples, n_samples)
    time_vector = np.arange(plot_samples) / fs

    fig, axes = plt.subplots(n_channels_to_plot, 1, figsize=(16, 10), sharex=True)

    if n_channels_to_plot == 1:
        axes = [axes]

    print(f"Plotting GOOD-ONLY reconstruction for first {plot_duration}s...")

    for idx, ch in enumerate(channels_to_plot):
        ax = axes[idx]

        # Original signal
        ax.plot(time_vector,
                data_continuous[ch, :plot_samples],
                color='green', linewidth=2.0, alpha=0.7,
                label='Original')

        # GOOD-ONLY reconstructed signal
        ax.plot(time_vector,
                data_reconstructed_good[ch, :plot_samples],
                color='blue', linewidth=1.2, alpha=0.9,
                label='Reconstructed (Good Only)')

        # Correlation calculated ONLY on good samples (ignore NaNs)
        good_idx = ~np.isnan(data_reconstructed_good[ch, :plot_samples])

        if np.sum(good_idx) > 2:  # avoid empty segments
            ch_corr_good = np.corrcoef(
                data_continuous[ch, :plot_samples][good_idx],
                data_reconstructed_good[ch, :plot_samples][good_idx]
            )[0, 1]
        else:
            ch_corr_good = np.nan

        # Labeling
        ax.set_ylabel(f'{channel_labels[ch]}\n(μV)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Channel {channel_labels[ch]} (Good-window r={ch_corr_good:.3f})',
                     fontsize=10, loc='right', pad=5)
        ax.set(ylim=[-50, 50])

        # Only add legend to first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

        # Mark window boundaries
        for win in range(int(plot_samples / window_length_samples) + 1):
            win_time = win * window_length
            if win_time <= plot_duration:
                ax.axvline(win_time, color='red', linestyle='--',
                           alpha=0.3, linewidth=0.8)

    axes[-1].set_xlabel('Time (s)', fontsize=12)

    fig.suptitle(
        f'Original vs Good-Only Reconstructed Signal - {PATIENT_ID}\n'
        f'Window size: {window_length}s | Good-window correlation only',
        fontsize=14, fontweight='bold', y=0.995
    )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_good_reconstruction_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print("Good-only reconstruction plot saved.")

    print(f"\nPer-channel correlations for good windows:") #TODO ADD LATER

    # %% Section 7A: Compute Sink Indices
    print("\n" + "="*60)
    print("SECTION 7A: COMPUTING SINK INDICES")
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
        sink_idx, source_idx, row_rank, col_rank = identifySS(A_matrices[:, :, i]) #TODO: Normalize so 0 --> 1
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


    # %% Section 7B: Compute Sink Indices (GOOD Windows Only)
    print("\n" + "=" * 60)
    print("SECTION 7B: COMPUTING SINK INDICES (GOOD WINDOWS ONLY)")
    print("=" * 60)

    # Use only windows that passed the threshold
    n_windows_good = A_good.shape[2]

    sink_indices_good = np.zeros((n_channels, n_windows_good))
    source_indices_good = np.zeros((n_channels, n_windows_good))
    row_ranks_good = np.zeros((n_channels, n_windows_good))
    col_ranks_good = np.zeros((n_channels, n_windows_good))

    print(f"Computing sink/source indices for {n_windows_good} GOOD windows...")

    # Loop through only the GOOD windows
    for i in range(n_windows_good):
        sink_idx, source_idx, row_rank, col_rank = identifySS(A_good[:, :, i])
        sink_indices_good[:, i] = sink_idx
        source_indices_good[:, i] = source_idx
        row_ranks_good[:, i] = row_rank
        col_ranks_good[:, i] = col_rank

    print("Good-window Sink/Source indices computed!")
    print(f"Shape: {sink_indices_good.shape}")

    # ----- Summary statistics for GOOD windows -----
    overall_sink_good = np.mean(sink_indices_good, axis=1)
    overall_source_good = np.mean(source_indices_good, axis=1)

    print("\nGood-window Overall Statistics:")
    print(f"  Top sink channel (GOOD): {eeg_data.labels[np.argmax(overall_sink_good)]} "
          f"(index: {np.max(overall_sink_good):.4f})")

    print(f"  Top source channel (GOOD): {eeg_data.labels[np.argmax(overall_source_good)]} "
          f"(index: {np.max(overall_source_good):.4f})")

    # %% Section 8A: Visualize Sink Index Heatmap
    print("\n" + "=" * 60)
    print("SECTION 8A: VISUALIZING SINK INDEX HEATMAP")
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

    # %% Section 8B: Visualize Sink Index Heatmap (GOOD Windows Only)
    print("\n" + "=" * 60)
    print("SECTION 8B: VISUALIZING GOOD-WINDOW SINK INDEX HEATMAP")
    print("=" * 60)

    # Sink indices for good windows only (from Section 7B)
    # sink_indices_good: shape (n_channels, n_windows_good)

    # Sort channels using the SAME ordering as the full data
    # This ensures direct comparability between heatmaps
    sink_good_sorted = sink_indices_good[sort_idx, :]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 12))

    sns.heatmap(
        sink_good_sorted,
        yticklabels=labels_sorted,
        cmap=sns.color_palette("rainbow", as_cmap=True),
        cbar_kws={"pad": 0.01, "label": "Sink Index (Good Only)"},
        ax=ax
    )

    ax.set_title(
        f"Good-Only Sink Index Heatmap - {PATIENT_ID}\n"
        f"Window size: {window_length}s | Channels sorted by ALL-window mean sink index",
        fontsize=14
    )
    ax.set_xlabel("Good Window Index")
    ax.set_ylabel("Channel")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_sink_heatmap_good_only.png'), dpi=150)
    plt.show()

    print(f"Good-window heatmap saved to: {OUTPUT_PATH}")

    # Print top-5 sinks for good windows only
    mean_sink_good = np.mean(sink_indices_good, axis=1)

    print("\nTop 5 Sink Channels (Good Windows Only):")
    for i in range(min(5, len(labels_sorted))):
        print(f"  {i + 1}. {labels_sorted[i]}: {mean_sink_good[sort_idx[i]]:.4f}")

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

    plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.08)
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

    # %% Section 9B: Visualize Sink Index Topomap (GOOD Windows Only)
    print("\n" + "=" * 60)
    print("SECTION 9B: VISUALIZING GOOD-WINDOW SINK INDEX TOPO MAP")
    print("=" * 60)

    # Mean sink index for GOOD windows only
    mean_sink_index_good = np.mean(sink_indices_good, axis=1)

    print(f"Good-window mean sink index range: "
          f"[{mean_sink_index_good.min():.4f}, {mean_sink_index_good.max():.4f}]")

    # Reuse same MNE info (same channel labels, montage)
    # ch_names, info, montage already created in 9A

    fig, ax = plt.subplots(figsize=(10, 8))

    # Topomap using ONLY good windows
    im, _ = mne.viz.plot_topomap(
        mean_sink_index_good,
        info,
        axes=ax,
        show=False,
        cmap='RdBu_r',
        vlim=(0, 1.4),  # Use SAME limits so heatmaps are comparable
        contours=6,
        names=ch_names
    )

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Mean Sink Index (Good Only)',
                   rotation=270, labelpad=20, fontsize=12)

    # Title
    ax.set_title(
        f'Mean Sink Index Topomap (GOOD WINDOWS) - {PATIENT_ID}\n'
        f'Averaged over {sink_indices_good.shape[1]} good windows',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.subplots_adjust(left=0.05, right=0.88, top=0.92, bottom=0.08)

    plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_sink_topomap_good_only.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Good-only topomap saved to: {OUTPUT_PATH}")

    print(f"\nTop 5 Sink Channels (GOOD ONLY):")
    top_sink_good_idx = np.argsort(mean_sink_index_good)[::-1][:5]
    for i, ch_idx in enumerate(top_sink_good_idx):
        print(f"  {i + 1}. {ch_names[ch_idx]}: {mean_sink_index_good[ch_idx]:.4f}")


    # ============================================
    # SECTION X: SUMMARY METRICS FOR EXCEL EXPORT
    # ============================================

    # ---------- Safe helpers ----------
    def safe_topk(arr, labels, k=2):
        """Return top-k indices, labels, and values from a 1D array."""
        if arr.size == 0 or np.all(np.isnan(arr)):
            return [], [], []
        idx_sorted = np.argsort(arr)[::-1]
        idx_top = idx_sorted[:k]
        labels_top = [labels[i] for i in idx_top]
        values_top = [arr[i] for i in idx_top]
        return idx_top, labels_top, values_top


    # ==================================================
    # 1) Mean Sink Index (ALL windows)
    # ==================================================
    mean_sink_all = np.mean(sink_indices, axis=1)

    # top–2 sinks (all windows)
    sink_idx_all, sink_label_all, sink_val_all = safe_topk(mean_sink_all, ch_names, k=2)

    # top–2 sources (all windows)
    mean_source_all = np.mean(source_indices, axis=1)
    source_idx_all, source_label_all, source_val_all = safe_topk(mean_source_all, ch_names, k=2)

    # ==================================================
    # 2) Mean Sink Index (GOOD windows only)
    # ==================================================
    mean_sink_good = np.mean(sink_indices_good, axis=1)

    # top–2 sink (good only)
    sink_idx_good, sink_label_good, sink_val_good = safe_topk(mean_sink_good, ch_names, k=2)

    # top–2 source (good only)
    mean_source_good = np.mean(source_indices_good, axis=1)
    source_idx_good, source_label_good, source_val_good = safe_topk(mean_source_good, ch_names, k=2)

    # ==================================================
    # 3) Build result dictionary for this subject
    # ==================================================
    # ==================================================
    # CLEAN SUMMARY RESULT ENTRY (MATCHES RIGHT COLUMN)
    # ==================================================

    result_entry = {
        "ID": PATIENT_ID,
        "Original Duration": orig_duration,
        "Clean Duration": clean_duration,
        "Effective Duration": effective_duration,  # you already compute this earlier
        "TBR": mean_TBR_val,

        # ===== Band Power % =====
        "delta": rel_band.get("delta"),
        "theta": rel_band.get("theta"),
        "alpha": rel_band.get("alpha"),
        "beta": rel_band.get("beta"),
        "gamma": rel_band.get("gamma"),

        # ===== Good SS Analysis % =====
        "Mean Sink Index (Good)": float(np.nanmean(mean_sink_good)),
        "Mean Source Index (Good)": float(np.nanmean(mean_source_good)),
        "Top Sink Ch 1,2 (Good)": ", ".join(sink_label_good[:2]) if sink_label_good else None,
        "Top Source Ch 1,2 (Good)": ", ".join(source_label_good[:2]) if source_label_good else None,

        # ===== Good SS Analysis % =====
        "Mean Sink Index (All)": float(np.nanmean(mean_sink_all)),
        "Mean Source Index (All)": float(np.nanmean(mean_source_all)),
        "Top Sink Ch 1,2 (All)": ", ".join(sink_label_all[:2]) if sink_label_all else None,
        "Top Source Ch 1,2 (All)": ", ".join(source_label_all[:2]) if source_label_all else None,

    }

    # Add this subject to batch results
    results.append(result_entry)

    # results.append({
    #     "ID": PATIENT_ID,
    #     "Original Duration": orig_duration,
    #     "Clean Duration": clean_duration,
    #     "TBR": mean_TBR_val,
    #     **rel_band,  # expands alpha=value, beta=value, ...
    #     "Good Window Percent": percent_good,
    #     "Sink Index": overall_sink,
    #     "Source Index": overall_source,
    #     "Good Sink Index": overall_sink_good,
    #     "Good Source Index": overall_source_good,
    #     "Top five Sink": top_sink_idx,
    #     "Top Good five Sink": top_sink_good_idx,
    #     # add all 10–12 metrics here
    # })

import pandas as pd
df = pd.DataFrame(results)

# Ensure batch-level output file goes to BATCH_OUTPUT_PATH
batch_excel_path = os.path.join(BATCH_OUTPUT_PATH, "BatchSummary.xlsx")

df.to_excel(batch_excel_path, index=False)
# --- Auto-adjust Excel column widths ---
from openpyxl import load_workbook

wb = load_workbook(batch_excel_path)
ws = wb.active

for column_cells in ws.columns:
    max_length = 0
    col_letter = column_cells[0].column_letter

    for cell in column_cells:
        try:
            cell_value = str(cell.value)
        except:
            cell_value = ""
        if cell_value:
            max_length = max(max_length, len(cell_value))

    ws.column_dimensions[col_letter].width = max_length + 2   # padding

wb.save(batch_excel_path)


print(f"\nBatch summary saved to:\n{batch_excel_path}")

