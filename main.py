 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Processing Pipeline for ADHD Classification
"""

# %% Section 0: Imports and Setup
# ==== GLOBAL IMPORTS ====

import numpy as np
import pandas as pd
import os
from pathlib import Path
import seaborn as sns
import traceback
import time

# Matplotlib (batch-safe)
import matplotlib
matplotlib.use("Agg")   # Important for batch mode!
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# MNE and other libraries
import mne
import scipy
from scipy.signal import welch
from scipy.stats import lognorm, norm

from get_save_path import get_save_path   # Custom function to get analysis wise save paths

# Import preprocessing modules
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'preprocessing'))
sys.path.insert(0, os.path.join(os.getcwd(), 'preprocessing', 'TDBRAIN'))

from preprocessing.TDBRAIN.autopreprocessing import dataset as ds


DEBUG_MODE = True   # True = debug (single test folder)
                    # False = full run

if DEBUG_MODE:
    # Debug / trial mode (process parent folder / subset)
    input_dirs = [
        r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2"
    ]
else:
    # Normal full processing mode
    # List of input directories (one per run)
    input_dirs = [
        r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\HC",
        r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Low",
        r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Med",
        r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_High",
    ]


valid_dirs = []
total_csv_files = 0

for d in input_dirs:
    if os.path.isdir(d):
        csv_files = [f for f in os.listdir(d) if f.lower().endswith('.csv')]
        total_csv_files += len(csv_files)
        valid_dirs.append(d)
    else:
        print(f"Warning: directory `{d}` not found, skipping.")

n_categories = len(valid_dirs)
print(f"\nFound Total {n_categories} Directories and {total_csv_files} files")

# Use `dirs_to_process` in the main loop instead of `input_dirs`
dirs_to_process = valid_dirs

def process_directory(DATA_PATH):
    """
    Move your existing pipeline into this function (or call it from here).
    This snippet ensures the Results folder exists per input folder.
    """
    BATCH_OUTPUT_PATH = os.path.join(DATA_PATH, "Results")
    Path(BATCH_OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing directory: `{DATA_PATH}`")
    print(f"Results will be written to: `{BATCH_OUTPUT_PATH}`\n")

# ---- NEW SECTION: Batch File Loop ----
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {DATA_PATH}")
        return

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
        #eeg_data.detect_kurtosis()

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
        eeg_data.remove_artifacts(remove_art=True)
        print("EEG Shape After removal:", eeg_data.data.shape)
        print(f"Preprocessing complete!")
        print(f"Data quality: {eeg_data.info.get('data quality', 'Unknown')}")
        print(f"Repaired channels: {eeg_data.info.get('repaired channels', 'None')}")
        clean_duration = eeg_data.data.shape[1] / eeg_data.Fs # Result---2

        # Option 2: Keep artifacts #TODO Add a helper function here later
        # eeg_data.remove_artifacts(remove_art=False)
        # print("Not Opted for Artifact Removal..")
        # clean_duration = eeg_data.data.shape[1] / eeg_data.Fs # Result---2

        # After define_artifacts()
        # if 'artifacts' in eeg_data.labels:
        #     artidx = np.where(eeg_data.labels == 'artifacts')[0]
        #     if len(artidx) > 0 and artidx[0] < eeg_data.data.shape[0]:
        #         # FIX: data is 2D (channels, samples), not 3D
        #         artifact_mask = eeg_data.data[artidx[0], :] == 1  # Removed [0] indexing
        #         artifact_percent = np.mean(artifact_mask) * 100
        #
        #         n_samples = eeg_data.data.shape[1]
        #         fs = eeg_data.Fs
        #
        #         print(f"\n📊 Artifact Statistics:")
        #         print(f"  Total duration: {n_samples / fs:.1f}s")
        #         print(f"  Artifact duration: {np.sum(artifact_mask) / fs:.1f}s ({artifact_percent:.1f}%)")
        #         print(f"  Clean duration: {np.sum(~artifact_mask) / fs:.1f}s ({100 - artifact_percent:.1f}%)")
        #
        #         if artifact_percent > 30:
        #             print("  ⚠️ Warning: >30% artifacts - consider excluding artifact windows")
        #         else:
        #             print("  ✅ Artifact level acceptable for continuous analysis")
        #
        # print(f"Preprocessing complete!")
        # print(f"Data quality: {eeg_data.info.get('data quality', 'Unknown')}")
        # print(f"Repaired channels: {eeg_data.info.get('repaired channels', 'None')}")

        # %% Section 2B: Compute Band Power Features (Standardized)
        print("\n" + "=" * 60)
        print("SECTION 2B: BAND POWER ANALYSIS (STANDARDIZED)")
        print("=" * 60)

        # Extract EEG data (first 26 channels)
        print("Extract EEG data (first 26 channels)")
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

        #Per Channel PSD and Band Power Calculation
        for ch in range(n_channels):

            freqs, psd = welch(
                data_clean[ch],
                fs=fs,
                nperseg=fs * 2,
                noverlap=fs,
                scaling='density'
            )

            # Total spectral power per channel
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
        std_TBR_val = np.nanstd(TBR)

        # -------- LOBE MATRICES --------

        from spectral_analysis import compute_lobe_band_matrices

        abs_matrix, pct_matrix, lobes, band_order = compute_lobe_band_matrices(
            band_power=band_power,
            band_power_percent=band_power_percent,
            eeg_labels=eeg_data.labels[:26]
        )

        # ======================TEMP storage for lobe–band percentage features======================================

        band_pct_features = {}

        # -----------------Extract Band_Pct_[Lobe]_[Band]-------------------------------------------

        for i, lobe in enumerate(lobes):
            lobe_clean = lobe.replace(" ", "_").replace("-", "_")

            for j, band in enumerate(band_order):
                band_clean = band.replace(" ", "_").replace("-", "_")
                key = f"Band_Pct_{lobe_clean}_{band_clean}"
                band_pct_features[key] = float(pct_matrix[i, j])

        # -------- PLOT + SAVE --------

        from plot_lobe_band_heatmaps import plot_lobe_band_heatmaps

        fig, TBR_lobes, TBR_mean = plot_lobe_band_heatmaps(
            abs_matrix,
            pct_matrix,
            lobes,
            bands
        )

        save_path = get_save_path("LBTopo", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)
        plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_band_topo.png'), dpi=150)
        plt.close()

        def clean_name(name):
            return name.replace("–", "-").replace(" ", "").replace("_", "")

        TBR_metrics = {
            f"TBR_lb_{clean_name(lobe)}": value
            for lobe, value in zip(lobes, TBR_lobes)
        }

        TBR_metrics["Global_TBR"] = TBR_mean
        TBR_metrics["TBR_Ch_Mean"] = mean_TBR_val
        TBR_metrics["TBR_Ch_STD"] = std_TBR_val

        print("\nTheta/Beta Ratios (TBR):")
        for k, v in TBR_metrics.items():
            print(f"{k:25s}: {v:.3f}")

        # %% Section 2C: Combined PSD + Band Power Diagram
        print("\n" + "=" * 60)
        print("SECTION 2C: PSD + BAND POWER DIAGRAM")
        print("=" * 60)

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
        save_path = get_save_path("FreqBand", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)
        plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_band_composition.png'), dpi=150)
        #plt.show()
        plt.close()

        # %% SECTION 2D: COMBINED 3×2 TOPOGRAPHY PANEL (TOTAL + EEG BANDS)
        print("\n" + "=" * 60)
        print("SECTION 2D: COMBINED TOPOGRAPHY PANEL (TOTAL + EEG BANDS)")
        print("=" * 60)

        # ------------------------------------------------------------
        # 1. Compute Total Power Topomap (0.5–48 Hz)
        # ------------------------------------------------------------
        total_power_topo = np.zeros(n_channels)

        for ch in range(n_channels):
            freqs, psd = welch(
                data_clean[ch],
                fs=fs,
                nperseg=fs * 2,
                noverlap=fs,
                scaling='density'
            )
            idx = (freqs >= 0.5) & (freqs <= 48)
            total_power_topo[ch] = np.trapz(psd[idx], freqs[idx])

        print(f"Total power range: [{total_power_topo.min():.3f}, {total_power_topo.max():.3f}]")

        # ------------------------------------------------------------
        # 2. Compute band-specific power topomaps
        # ------------------------------------------------------------
        band_topos = {}

        for band_name, (fmin, fmax) in bands.items():
            bp = np.zeros(n_channels)

            for ch in range(n_channels):
                freqs, psd = welch(
                    data_clean[ch],
                    fs=fs,
                    nperseg=fs * 2,
                    noverlap=fs,
                    scaling='density'
                )
                idx = (freqs >= fmin) & (freqs <= fmax)
                bp[ch] = np.trapz(psd[idx], freqs[idx])

            band_topos[band_name] = bp

        # ------------------------------------------------------------
        # 3. Prepare MNE info
        # ------------------------------------------------------------
        ch_names_topo = eeg_data.labels[:n_channels].tolist()
        ch_types = ['eeg'] * n_channels
        info_topo = mne.create_info(ch_names_topo, sfreq=fs, ch_types=ch_types)

        montage = mne.channels.make_standard_montage('standard_1020')
        info_topo.set_montage(montage)

        pos_dict = info_topo.get_montage().get_positions()['ch_pos']

        # ------------------------------------------------------------
        # 4. Create 3×2 figure
        # ------------------------------------------------------------
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        axes = axes.flatten()

        titles = [
            "Total Power (0.5–48 Hz)",
            "Delta (0.5–4 Hz)",
            "Theta (4–8 Hz)",
            "Alpha (8–13 Hz)",
            "Beta (13–30 Hz)",
            "Gamma (30–48 Hz)"
        ]

        topo_values = [
            total_power_topo,
            band_topos["delta"],
            band_topos["theta"],
            band_topos["alpha"],
            band_topos["beta"],
            band_topos["gamma"],
        ]

        # ------------------------------------------------------------
        # 5. Plot all 6 topomaps (each with own scale)
        # ------------------------------------------------------------
        for i, ax in enumerate(axes):

            vals = topo_values[i]
            vlim = (np.min(vals), np.max(vals))  # each band has unique scale

            im, _ = mne.viz.plot_topomap(
                vals,
                info_topo,
                axes=ax,
                show=False,
                cmap="Greens",
                contours=6,
                vlim=vlim
            )

            # Add channel name annotations
            for name, xy in pos_dict.items():
                if name in ch_names_topo:
                    ax.text(
                        xy[0], xy[1],
                        name,
                        color='black',
                        fontsize=7,
                        ha='center',
                        va='center'
                    )

            ax.set_title(titles[i], fontsize=12, fontweight='bold')

        # ------------------------------------------------------------
        # 6. Separate colorbars for each subplot
        # ------------------------------------------------------------
        for i, ax in enumerate(axes):
            # Create separate colorbars (nicer visuals)
            cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        plt.suptitle(f"{PATIENT_ID}: Total + Band Power Topographies",
                     fontsize=16, fontweight="bold")

        plt.subplots_adjust(left=0.05, right=0.98, top=0.93,
                            bottom=0.05, hspace=0.35)

        combined_path = os.path.join(OUTPUT_PATH, f"{PATIENT_ID}_combined_topomaps.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        save_path = get_save_path("FreqTopo", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)

        plt.close()

        print(f"Combined topomap figure saved to: {combined_path}")

        # ------------------------------------------------------------
        # 7. Normalized 3x2 Topomaps (shared 0–1 scale, Blues)
        # ------------------------------------------------------------
        from matplotlib import gridspec

        # ---------- NORMALIZE ----------
        topo_values_norm = []
        for vals in topo_values:
            vmin = np.min(vals)
            vmax = np.max(vals)
            topo_values_norm.append((vals - vmin) / (vmax - vmin + 1e-12))

        # ---------- FIGURE ----------
        fig = plt.figure(figsize=(13, 7))

        # 3 columns for maps + 1 narrow column for colorbar
        gs = gridspec.GridSpec(
            2, 4,
            width_ratios=[1, 1, 1, 0.05],
            wspace=0.30,
            hspace=0.40
        )

        # correct subplot placement: row-major order
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]

        # ---------- PLOT ----------
        for i, ax in enumerate(axes):
            vals = topo_values_norm[i]

            im, _ = mne.viz.plot_topomap(
                vals,
                info_topo,
                axes=ax,
                show=False,
                cmap="Blues",
                contours=6,
                vlim=(0, 1)
            )

            for name, xy in pos_dict.items():
                if name in ch_names_topo:
                    ax.text(xy[0], xy[1], name,
                            color='black', fontsize=7,
                            ha='center', va='center')

            ax.set_title(titles[i], fontsize=12, fontweight='bold')

        # ---------- SUPERTITLE (first) ----------
        plt.suptitle(
            f"{PATIENT_ID}: Normalized Band Power Topographies (0–1)",
            fontsize=16,
            fontweight="bold",
            y=0.98
        )

        # ---------- NOW ADJUST LAYOUT (ONCE ONLY) ----------
        plt.subplots_adjust(
            left=0.06,
            right=0.88,  # reserve right space
            top=0.90,  # push grid DOWN from title
            bottom=0.08,
        )

        # ---------- COLORBAR IN DEDICATED AXIS ----------
        cax = fig.add_axes([0.90, 0.18, 0.02, 0.64])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=8)

        # ---------- SAVE ----------
        combined_path = os.path.join(
            OUTPUT_PATH,
            f"{PATIENT_ID}_combined_topomaps_normalized.png"
        )

        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        save_path = get_save_path("FreqTopoNorm", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Normalized topomap figure saved to: {combined_path}")

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


        # Extract continuous data
        data_continuous = eeg_data.data[0, :26, :]  # First 26 channels (EEG only, exclude EOG)
        n_channels = data_continuous.shape[0]
        n_samples = data_continuous.shape[1]

        print(f"Data for A matrix computation: {data_continuous.shape}")
        print(f"Channels: {n_channels}, Samples: {n_samples}")

        # Parameters for A matrix computation
        window_length = 0.125 # (in seconds)
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

        # %% Section 4B: Singular values, Shannon entropy, and condition ratio per window
        print("\n" + "=" * 60)
        print("SECTION 4B: SINGULAR VALUES, ENTROPY, AND CONDITION RATIO")
        print("=" * 60)

        # A_matrices has shape: (n_channels, n_channels, n_windows)
        n_channels = A_matrices.shape[0]
        n_windows = A_matrices.shape[2]

        print(f"Number of channels        : {n_channels}")
        print(f"Number of A-matrix windows: {n_windows}")

        # Arrays to store metrics for each window
        singular_values_windows = np.zeros((n_windows, n_channels))  # shape: (n_windows, 26)
        entropy_windows = np.zeros(n_windows)  # Shannon entropy per window
        condratio_windows = np.zeros(n_windows)  # sigma_max / sigma_min per window
        rank_windows = np.zeros(n_windows)  # Rank per window
        sing_val_diff = np.zeros(n_windows)  # sigma_max - sigma_min per window
        auc_per_window = np.zeros(n_windows)  # Sum of singular values per window

        eps = 1e-12  # small positive value for numerical safety

        for w in range(n_windows):
            # Extract A-matrix for this time window
            A_w = A_matrices[:, :, w]

            # --- 1) Singular values (no need for U, V) ---
            s = np.linalg.svd(A_w, compute_uv=False)  # shape: (26,)
            singular_values_windows[w, :] = s
            auc_per_window[w] = s.sum()          # Per-window AUC (shape: [n_windows])

            # --- 2) Shannon entropy of singular values ---
            # Make all entries strictly positive to avoid log(0)
            # s_pos = np.clip(s, a_min=eps, a_max=None)
            s_pos = s
            # Convert to probability distribution
            p = s_pos / np.sum(s_pos)  # sum(p) = 1

            # Shannon entropy in bits
            entropy_windows[w] = -np.sum(p * np.log2(p))

            # --- 3) Condition ratio: sigma_max / sigma_min ---
            s_min = s_pos.min()
            #log_sigma_max = np.log(s_pos.max())
            log_sigma_max = s_pos.max()
            #log_sigma_min = np.log(s_pos.min())
            log_sigma_min = s_pos.min()
            sing_val_diff[w] = s_pos.max() - s_pos.min()
            condratio_windows[w] = log_sigma_max / log_sigma_min

            # --- 4) Rank ---
            eps_machine = np.finfo(float).eps  # 2.22e-16 for float64
            tol = 26 * s_pos.max() * eps_machine  # recommended stable tolerance
            rank_windows[w] = np.sum(s_pos > tol)

        print("Per-window metrics computed:")
        print(f"  Entropy    shape: {entropy_windows.shape}")
        print(f"  Cond ratio shape: {condratio_windows.shape}")
        print(f"  SVD values shape: {singular_values_windows.shape}")

        # %% Section 4C: Summary metrics for this recording
        print("\n" + "=" * 60)
        print("SECTION 4C: SVD BASED METRICS FOR THIS RECORDING")
        print("=" * 60)

        # Basic summary statistics (ignore NaNs from nearly singular windows)
        H_SVD_mean = np.nanmean(entropy_windows)
        H_SVD_std = np.nanstd(entropy_windows)
        H_SVD_min = np.nanmin(entropy_windows)
        H_SVD_max = np.nanmax(entropy_windows)

        SVD_mean = np.nanmean(s_pos)
        SVD_std = np.nanstd(s_pos)
        SVD_min = np.nanmin(s_pos)
        SVD_max = np.nanmax(s_pos)

        SVD_ratio_mean = np.nanmean(condratio_windows)
        SVD_ratio_std = np.nanstd(condratio_windows)
        SVD_ratio_min = np.nanmin(condratio_windows)
        SVD_ratio_max = np.nanmax(condratio_windows)

        SVD_Sum_mean = np.mean(auc_per_window)    # Mean AUC across all windows (scalar)
        SVD_Sum_std = np.nanstd(auc_per_window)  # SVD AUC across all windows (scalar)
        SVD_Sum_min = np.nanmin(auc_per_window) # Min AUC across all windows (scalar)
        SVD_Sum_max = np.nanmax(auc_per_window) # Max AUC across all windows (scalar)

        sing_val_diff_mean=np.nanmean(sing_val_diff)
        sing_val_diff_std = np.nanstd(sing_val_diff)
        sing_val_diff_min = np.nanmin(sing_val_diff)
        sing_val_diff_max = np.nanmax(sing_val_diff)

        print(f"Entropy  -> mean: {H_SVD_mean:.3f}, std: {H_SVD_std:.3f}, min: {H_SVD_min:.3f}, max: {H_SVD_max:.3f}")
        print(f"CondRatio-> mean: {SVD_ratio_mean:.3f}, std: {SVD_ratio_std:.3f}, min: {SVD_ratio_min:.3f}, max: {SVD_ratio_max:.3f}")
        print(f"SVD AUC  -> mean: {SVD_Sum_mean:.3f}, std: {SVD_Sum_std:.3f}, min: {SVD_Sum_min:.3f}, max: {SVD_Sum_max:.3f}")
        print(f"SVD Diff -> mean: {sing_val_diff_mean:.3f}, std: {sing_val_diff_std:.3f}")

        # ---------------------------------------------------------
        # SECTION 4D: Per-subject line charts (Entropy & CondRatio)
        # ---------------------------------------------------------

        # Create a folder for per-subject plots inside your batch output folder
        plots_dir = os.path.join(BATCH_OUTPUT_PATH, "A_metrics_subject_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Build a safe filename for this subject

        plot_path = os.path.join(plots_dir, f"{PATIENT_ID}_Ametrics_timeseries.png")

        from matplotlib.ticker import FormatStrFormatter

        def set_4_yticks(ax, data):

            data = np.asarray(data, dtype=float)
            data = data[np.isfinite(data)]

            y_min, y_max = np.min(data), np.max(data)
            y_min_r = int(np.floor(y_min))
            y_max_r = int(np.ceil(y_max))

            ax.set_ylim(y_min_r, y_max_r)

            yticks = np.linspace(y_min_r, y_max_r, 4)
            ax.set_yticks(yticks)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # 1x2 subplot: Entropy and Condition Ratio across windows
        plt.figure(figsize=(12, 5))

        # --- Top: Entropy time series ---
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(entropy_windows)
        ax1.set_title(f"Entropy per Window\n{PATIENT_ID}")
        ax1.set_xlabel("Window index")
        ax1.set_ylabel("Shannon Entropy (bits)")
        ax1.grid(True)
        set_4_yticks(ax1, entropy_windows)

        # # --- Middle: Condition ratio time series ---
        # ax2 = plt.subplot(3, 1, 2)
        # ax2.yaxis.converter = None
        # ax2.plot(condratio_windows)
        # ax2.set_title(f"Condition Ratio per Window\n{PATIENT_ID}")
        # ax2.set_xlabel("Window index")
        # ax2.set_ylabel("σ_max / σ_min")
        # ax2.grid(True)
        # set_4_yticks(ax2, condratio_windows)

        # --- Bottom: SVD Sum time series ---
        ax3 = plt.subplot(2, 1, 2)
        ax3.plot(auc_per_window)
        ax3.set_title("Singular Value Sum per Window")
        ax3.set_xlabel("Window Index")
        ax3.set_ylabel("Σ singular values")
        ax3.grid(True)
        set_4_yticks(ax3, auc_per_window)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_A-Mat_SVD.png'),
                     dpi=300, bbox_inches='tight')
        save_path = get_save_path("SVD", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"Saved per-subject A-metrics SVD Feature time-series plot to:\n  {OUTPUT_PATH}")

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

            if (win_idx + 1) % 100 == 0: #Every 100 windows, prints progress to monitor long runs.
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

        plt.figure(figsize=(8, 5))
        plt.hist(corr_check, bins=20, edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold = {threshold}')
        plt.xlabel('Window Correlation')
        plt.ylabel('Count')
        plt.title('Histogram of Window Correlations')
        plt.legend()
        plt.grid(True, alpha=0.3)
        #plt.show()

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

        # ---- Sanitize MSE for log scale for Runtime Error Removal ----
        eps = 1e-12
        MSE_values = np.asarray(MSE_values, dtype=float)
        MSE_values[~np.isfinite(MSE_values)] = eps  # NaN/Inf -> eps
        MSE_values[MSE_values <= 0] = eps  # Zero/negative -> eps


        # ---- Left subplot: MSE ----
        for ch in range(n_channels):
            axes[0].scatter(
                np.full(n_windows, ch+1),
                MSE_values[ch, :],
                color='tab:orange',
                alpha=0.6,
                s=25
            )
        # --- Force-clean Y-axis limits before log scale ---
        y_min = np.nanmin(MSE_values)
        y_max = np.nanmax(MSE_values)

        if not np.isfinite(y_min) or y_min <= 0:
            y_min = 1e-12
        if not np.isfinite(y_max):
            y_max = 1e-6  # something reasonable

        axes[0].set_ylim(y_min, y_max)

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
        #plt.show()

        # %% Section 6: Log Normal Modelling and Feature Extraction
        print("\n" + "=" * 60)
        print("SECTION 6: LOG NORMAL MODELING AND FEATURE EXTRACTION")
        print("=" * 60)

        from eeg_lognormal_multivariate import extract_multivariate_lognormal_energy
        from lognormal_features import extract_lognormal_cov_features

        # ----------------------------------
        # 1. Estimate multivariate log-normal model
        # ----------------------------------
        mu_vec, cov_mat, log_energy_windows,energy = extract_multivariate_lognormal_energy(
            eeg=data_clean,  # shape (26, n_samples)
            fs_hz=fs,
            window_ms=window_length*1000,
            step_ms=window_length*1000,
        )

        # ----------------------------------
        # 2. Channel-wise parameters
        # ----------------------------------
        mu = np.mean(log_energy_windows, axis=0)  # (26,)
        sigma = np.std(log_energy_windows, axis=0)  # (26,)

        n_channels = len(mu)

        # ----------------------------------
        # 3. Scalar Feature Extraction
        # ----------------------------------
        ln_features = extract_lognormal_cov_features(
            mu=mu,
            sigma=sigma,
            cov_mat=cov_mat,
            log_energy_windows = log_energy_windows
        )

        # ---------------------------
        # Log Normal Features Plotting
        # ---------------------------
        EEG_labels = list(channel_labels[:26])
        EEG_cov = cov_mat[:26, :26]

        # =========================================
        # EEG channel groups (10–20 system)
        # =========================================
        GROUPS = {
            "Prefrontal": [
                "Fp1", "Fp2"
            ],
            "Frontal": [
                 "F7", "F3", "Fz", "F4", "F8"
            ],
            "Front-Central": [
                "FC3", "FCz", "FC4"
            ],
            "Central": [
                "T7", "C3", "Cz", "C4", "T8"
            ],
            "Centro-Parietal": [
                "CP3","CPz","CP4"
            ],
            "Parietal": [
                "P7", "P3", "Pz", "P4", "P8"
            ],
            "Occipital": [
                "O1", "Oz", "O2"
            ]
        }
        GROUP_COLORS = {
            "Prefrontal": "tab:purple",
            "Frontal": "tab:blue",
            "Front-Central": "tab:cyan",
            "Central": "tab:orange",
            "Centro-Parietal": "tab:pink",
            "Parietal": "tab:green",
            "Occipital": "tab:red"
        }

        # ---------------------------
        # Sort channels by mu
        # ---------------------------
        order = np.argsort(mu[:26])
        mu_sorted = mu[order]
        sigma_sorted = sigma[order]
        labels_sorted = [EEG_labels[i] for i in order]

        # --------------------------------------------------
        # Summary statistics for channel-wise μ and σ
        # (Spread & dissimilarity measures) Added on Jan 10, 2026
        # --------------------------------------------------

        # ----- μ (log-mean energy) -----
        mu_range = np.max(mu_sorted) - np.min(mu_sorted)
        mu_std = np.std(mu_sorted)
        mu_mean = np.mean(mu_sorted)
        mu_cv = mu_std / (np.abs(mu_mean) + 1e-12)

        # ----- σ (log-variance / instability) -----
        sigma_range = np.max(sigma_sorted) - np.min(sigma_sorted)
        sigma_std = np.std(sigma_sorted)
        sigma_mean = np.mean(sigma_sorted)
        sigma_cv = sigma_std / (np.abs(sigma_mean) + 1e-12)

        # --------------------------------------------------
        # Append channel-wise μ / σ spread metrics Added on Jan 10, 2026
        # --------------------------------------------------

        ln_features.update({
            # μ (log-energy mean) dissimilarity
            "mu_range": mu_range,
            "mu_std": mu_std,
            "mu_cv": mu_cv,

            # σ (log-energy variability) dissimilarity
            "sigma_range": sigma_range,
            "sigma_std": sigma_std,
            "sigma_cv": sigma_cv
        })

        # ---------------------------
        # Gaussian axis
        # ---------------------------
        z = np.linspace(mu_sorted.min() - 4 * sigma_sorted.max(),
                        mu_sorted.max() + 4 * sigma_sorted.max(),
                        600)

        # ======================================================
        # BUILD MASTER FIGURE
        # ======================================================
        fig = plt.figure(figsize=(18, 10),constrained_layout=True)
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[3, 2, 3],
            height_ratios=[1, 1],
            wspace=0.25,
            hspace=0.25
        )

        from matplotlib import cm

        # Fixed, reproducible colors for 26 channels
        cmap = cm.get_cmap("tab20", len(labels_sorted))
        channel_color_map = {
            ch: cmap(i) for i, ch in enumerate(labels_sorted)
        }

        # Fixed, reproducible colors for 26 channels
        cmap = cm.get_cmap("tab20", len(labels_sorted))
        channel_color_map = {
            ch: cmap(i) for i, ch in enumerate(labels_sorted)
        }

        # ======================================================
        # PANEL 1 — Gaussian PDFs (FULL HEIGHT LEFT COLUMN)
        # spans 2 columns
        # ======================================================
        # ======================================================
        # SUBGRID for COLUMN 0 (split vertically)
        # ======================================================
        gs_col0 = gs[:, 0].subgridspec(
            2, 1,
            height_ratios=[1, 1],
            hspace=0.15
        )

        ax_pdf_energy = fig.add_subplot(gs_col0[0, 0])  # TOP
        ax_pdf_log = fig.add_subplot(gs_col0[1, 0])  # BOTTOM

        channel_handles = []
        channel_legend = []

        # --------- Plot channels - Normal Top ----------

        # Collect all positive energies across channels (and/or subjects if you have them)
        all_pos = energy[energy > 0].ravel()

        xmin = np.percentile(all_pos, 1)
        xmax = np.percentile(all_pos, 99)
        xmin = max(xmin, eps)

        # ======================================================
        # Global normalization (single scalar, shape-preserving)
        # ======================================================
        #K = np.median(energy[energy > 0])
        #K = max(K, eps)
        K=1

        energy_norm = energy / K  # same shape, no coupling

        xmin_n = xmin / K
        xmax_n = xmax / K


        x_grid = np.geomspace(xmin, xmax, 600)

        # --------- Plot channels — Normalized Energy PDFs (TOP) ----------

        channel_handles = []
        channel_legend = []

        for gname, chans in GROUPS.items():
            for ch in chans:
                if ch not in labels_sorted:
                    continue

                i = labels_sorted.index(ch)

                samples = energy_norm[i, :]
                samples = samples[samples > 0]

                if len(samples) < 10:
                    continue

                log_s = np.log(samples + eps)
                mu_i = np.mean(log_s)
                sigma_i = np.std(log_s, ddof=1)
                sigma_i = max(sigma_i, 1e-6)

                pdf = lognorm.pdf(x_grid, s=sigma_i, scale=np.exp(mu_i))

                line, = ax_pdf_energy.plot(
                    x_grid,
                    pdf,
                    lw=1,
                    alpha=0.7,
                    color=channel_color_map[ch]
                )

                # collect legend handles once
                if ch not in channel_legend:
                    channel_handles.append(line)
                    channel_legend.append(ch)

        # --------- TOP axis formatting ----------
        ax_pdf_energy.set_title("Energy PDFs (Globally Normalized)")
        ax_pdf_energy.set_xlabel("Energy / global median")
        ax_pdf_energy.set_ylabel("Probability Density")
        #ax_pdf_energy.set_xscale("log")
        ax_pdf_energy.set_xlim(xmin_n, xmax_n)

        # --------- SHARED LEGEND (TOP ONLY) ----------
        ax_pdf_energy.legend(
            channel_handles,
            channel_legend,
            fontsize=10,
            ncol=4,
            loc="best",
            frameon=False,
            title="Channels"
        )

        # --------- Plot channels - Log Normal Bottom ----------
        for gname, chans in GROUPS.items():
            #color = GROUP_COLORS[gname]
            for ch in chans:
                if ch in labels_sorted:
                    i = labels_sorted.index(ch)
                    ax_pdf_log.plot(
                        z,
                        norm.pdf(z, mu_sorted[i], sigma_sorted[i]),
                        #color=color,
                        lw=1,
                        alpha=0.7
                    )

        # --------- Plot channels — Gaussian PDFs (log-energy) [BOTTOM] ----------

        # Build common log-energy axis (already defined earlier)
        # z = np.linspace(mu_sorted.min() - 4 * sigma_sorted.max(),
        #                 mu_sorted.max() + 4 * sigma_sorted.max(), 600)

        for gname, chans in GROUPS.items():
            for ch in chans:
                if ch not in labels_sorted:
                    continue

                i = labels_sorted.index(ch)

                mu_i = mu_sorted[i]
                sigma_i = sigma_sorted[i]
                sigma_i = max(sigma_i, 1e-6)

                ax_pdf_log.plot(
                    z,
                    norm.pdf(z, mu_i, sigma_i),
                    lw=1,
                    alpha=0.7,
                    color=GROUP_COLORS[gname]
                )

        # --------- BOTTOM axis formatting ----------
        ax_pdf_log.set_title("Channel-wise Gaussian PDFs (log-energy)")
        ax_pdf_log.set_xlabel("log(Energy)")
        ax_pdf_log.set_ylabel("Probability Density")
        ax_pdf_log.set_xlim(z.min(), z.max())

        # ======================================================
        # PANEL 2 — Channel μ–σ table (TOP MIDDLE)
        # ======================================================
        gs_mid = gs[:, 1].subgridspec(
            2, 1,
            height_ratios=[3, 1],  # 75% / 25%
            hspace=0.00
        )

        ax_tbl1 = fig.add_subplot(gs_mid[0, 0])   # top table
        ax_tbl1.set_title("Channel-wise Gaussian Parameters", pad=0, loc="center")
        ax_tbl1.axis("off")

        tbl1 = ax_tbl1.table(
            cellText=[
                [labels_sorted[i],
                 f"{mu_sorted[i]:.3f}",
                 f"{sigma_sorted[i]:.3f}"]
                for i in range(len(mu_sorted))
            ],
            colLabels=["Channel", "μ (log)", "σ (log)"],
            loc="best",
            cellLoc="center"
        )

        tbl1.auto_set_font_size(False)
        tbl1.set_fontsize(12)
        tbl1.scale(1.0, 1.0)

        # ======================================================
        # PANEL 3 — EEG covariance matrix (TOP RIGHT)
        # ======================================================
        gs_right = gs[:, 2].subgridspec(
            2, 1,
            height_ratios=[3, 2],  # 60% / 40%
            hspace=0.25
        )

        ax_cov1 = fig.add_subplot(gs_right[0, 0])

        im = ax_cov1.imshow(EEG_cov, cmap="jet", aspect="auto",vmin=0,vmax=1)
        plt.colorbar(im, ax=ax_cov1, fraction=0.046, pad=0.04)

        ax_cov1.set_title("EEG Covariance Matrix (26×26)")

        # keep labels, remove tick values
        ax_cov1.set_xticks(np.arange(len(EEG_labels)))
        ax_cov1.set_xticklabels(EEG_labels, rotation=90, fontsize=8)
        ax_cov1.set_yticks(np.arange(len(EEG_labels)))
        ax_cov1.set_yticklabels(EEG_labels, fontsize=8)


        ax_cov1.tick_params(axis="both", length=0, labelbottom=True)

        # =========================================
        # BUILD GROUP SUMMARY (μ mean & σ mean)
        # =========================================
        group_names = list(GROUPS.keys())

        group_summary = []

        for gname in group_names:
            # channel indices that belong to this group
            idx = [labels_sorted.index(ch)
                   for ch in GROUPS[gname]
                   if ch in labels_sorted]

            # compute group means
            mu_mean = np.mean(mu_sorted[idx])
            sigma_mean = np.mean(sigma_sorted[idx])

            group_summary.append([
                gname,
                f"{mu_mean:.3f}",
                f"{sigma_mean:.3f}"
            ])

        # ======================================================
        # PANEL 4 — Group μ–σ table (BOTTOM MIDDLE)
        # ======================================================
        ax_tbl2 = fig.add_subplot(gs_mid[1, 0])   # bottom table
        ax_tbl2.set_title("Group-wise Mean Gaussian Parameters", pad=0, loc="center")
        ax_tbl2.axis("off")

        tbl2 = ax_tbl2.table(
            cellText=group_summary,
            colLabels=["Group", "μ mean", "σ mean"],
            loc="best",
            cellLoc="center"
        )

        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(12)
        tbl2.scale(1.0, 1.0)

        # =========================================
        # BUILD GROUP-WISE COVARIANCE MATRIX (5×5)
        # =========================================
        G = len(group_names)
        group_cov = np.zeros((G, G))

        for i, g1 in enumerate(group_names):
            idx1 = [EEG_labels.index(ch)
                    for ch in GROUPS[g1]
                    if ch in EEG_labels]

            for j, g2 in enumerate(group_names):
                idx2 = [EEG_labels.index(ch)
                        for ch in GROUPS[g2]
                        if ch in EEG_labels]

                # mean absolute covariance between groups
                group_cov[i, j] = np.mean(np.abs(EEG_cov[np.ix_(idx1, idx2)]))

        # ======================================================
        # PANEL 5 — Group-wise covariance (5×5)
        # ======================================================
        ax_cov2 = fig.add_subplot(gs_right[1, 0])

        pos = ax_cov2.get_position()
        ax_cov2.set_position([
            pos.x0 + 0.10,  # move right
            pos.y0,  # keep same vertical
            pos.width * 0.8,  # shrink width
            pos.height * 0.8  # keep height
        ])

        im2 = ax_cov2.imshow(group_cov, cmap="magma", aspect="auto",vmin=0,vmax=0.8)
        plt.colorbar(im2, ax=ax_cov2, fraction=0.046, pad=0.04)

        ax_cov2.set_title("Group-wise Covariance (5×5)")

        ax_cov2.set_xticks(range(len(group_names)))
        ax_cov2.set_xticklabels(group_names, rotation=45)
        ax_cov2.set_yticks(range(len(group_names)))
        ax_cov2.set_yticklabels(group_names)

        ax_cov2.tick_params(length=0)

        # =======================================================
        # FINAL LAYOUT + SAVE
        # =======================================================
        fig.subplots_adjust(
            left=0.01, right=0.99,
            bottom=0.01, top=0.99
        )
        #plt.tight_layout(rect=[0.04, 0.04, 0.96, 0.96])


        plt.savefig(
            os.path.join(OUTPUT_PATH, f"{PATIENT_ID}_Gaussian_Table_Covariance.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1
        )

        save_path = get_save_path("LN", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.1)

        print("Gaussian modeling plots saved successfully!")
        # %% Section 7A: Visualize Original vs Reconstructed Signal
        print("\n" + "=" * 60)
        print("SECTION 7A: VISUALIZING RECONSTRUCTION")
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
        #plt.show()

        print(f"Reconstruction plot saved to: {OUTPUT_PATH}")
        print(f"\nPer-channel correlations:")
        for ch in channels_to_plot:
            ch_corr = np.corrcoef(data_continuous[ch, :], data_reconstructed[ch, :])[0, 1]
            print(f"  {channel_labels[ch]}: {ch_corr:.4f}")

        # %% Section 7B: Visualize Original vs Good-Only Reconstructed Signal
        print("\n" + "=" * 60)
        print("SECTION 7B: VISUALIZING GOOD-ONLY RECONSTRUCTION")
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
        #plt.show()

        print("Good-only reconstruction plot saved.")

        print(f"\nPer-channel correlations for good windows:") #TODO ADD LATER

        # %% Section 8A: Compute Sink Indices
        print("\n" + "="*60)
        print("SECTION 8A: COMPUTING SINK INDICES")
        print("="*60)

        # Import the identifySS function from utils
        from SS_analysis import compute_sink_source_scores

        # Compute sink and source indices for each window
        n_windows = A_matrices.shape[2]
        sink_indices = np.zeros((n_channels, n_windows))
        source_indices = np.zeros((n_channels, n_windows))
        row_ranks = np.zeros((n_channels, n_windows))
        col_ranks = np.zeros((n_channels, n_windows))

        print("Computing sink and source indices for each window...")

        for i in range(n_windows):
            sink_idx, source_idx, row_rank, col_rank = compute_sink_source_scores(A_matrices[:, :, i]) #TODO: Normalize so 0 --> 1
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


        # %% Section 8B: Compute Sink Indices (GOOD Windows Only)
        print("\n" + "=" * 60)
        print("SECTION 8B: COMPUTING SINK INDICES (GOOD WINDOWS ONLY)")
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
            sink_idx, source_idx, row_rank, col_rank = compute_sink_source_scores(A_good[:, :, i])
            sink_indices_good[:, i] = sink_idx
            source_indices_good[:, i] = source_idx
            row_ranks_good[:, i] = row_rank
            col_ranks_good[:, i] = col_rank

        #Plotting Source Sink Map

        from matplotlib.lines import Line2D

        # --------------------------------------------------
        # 1. Mean ranks across windows
        # --------------------------------------------------
        row_rank_mean = np.mean(row_ranks_good, axis=1)
        col_rank_mean = np.mean(col_ranks_good, axis=1)

        labels = eeg_data.labels[:26]
        nCh = len(labels)

        # --------------------------------------------------
        # 2. Lobe definitions (edit if needed)
        # --------------------------------------------------
        lobe_map = {
            "Fp1": "Prefrontal", "Fp2": "Prefrontal",
            "F3": "Frontal", "F4": "Frontal", "F7": "Frontal", "F8": "Frontal", "Fz": "Frontal",
            "FC3": "FrontoCentral", "FC4": "FrontoCentral", "FCz": "FrontoCentral",

            "C3": "Central", "C4": "Central", "Cz": "Central",

            "CP3": "CentroParietal", "CP4": "CentroParietal", "CPz": "CentroParietal",
            "P3": "Parietal", "P4": "Parietal", "P7": "Parietal", "P8": "Parietal", "Pz": "Parietal",

            "O1": "Occipital", "O2": "Occipital", "Oz": "Occipital",

            "T7": "Temporal", "T8": "Temporal"
        }

        lobe_colors = {
            "Prefrontal": "#ffbb78",  # light orange
            "Frontal": "#1f77b4",  # blue
            "FrontoCentral": "#17becf",  # cyan
            "Central": "#2ca02c",  # green
            "CentroParietal": "#8c564b",  # brown
            "Parietal": "#ff7f0e",  # orange
            "Temporal": "#d62728",  # red
            "Occipital": "#9467bd",  # purple
            "Other": "gray"
        }

        # --------------------------------------------------
        # 3. Figure
        # --------------------------------------------------
        plt.figure(figsize=(9, 8))

        # --------------------------------------------------
        # 4. Cluster center & radial layout
        # --------------------------------------------------
        cx = np.mean(row_rank_mean)
        cy = np.mean(col_rank_mean)

        radius = 0.32
        angles = np.linspace(0, 2 * np.pi, nCh, endpoint=False)

        # --------------------------------------------------
        # 5. Plot channels
        # --------------------------------------------------
        for i in range(nCh):
            ch = labels[i]
            lobe = lobe_map.get(ch, "Other")
            color = lobe_colors[lobe]

            # Data point
            plt.scatter(row_rank_mean[i], col_rank_mean[i],
                        edgecolors=color,
                        facecolors='none',
                        s=120,
                        linewidths=1.8,
                        zorder=3)

            # Radial label position
            lx = cx + radius * np.cos(angles[i])
            ly = cy + radius * np.sin(angles[i])

            # Leader line
            plt.plot([row_rank_mean[i], lx],
                     [col_rank_mean[i], ly],
                     color=color,
                     lw=0.6,
                     alpha=0.7,
                     zorder=2)

            # Label
            plt.text(lx, ly, ch,
                     fontsize=9,
                     color=color,
                     ha='center',
                     va='center',
                     zorder=4)

        # --------------------------------------------------
        # 6. Perfect source / sink
        # --------------------------------------------------
        root2 = np.sqrt(2)
        plt.scatter(root2, 0, c='black', s=200, marker='X', zorder=5, label='Perfect Sink (√2, 0)')
        plt.scatter(0, root2, c='black', s=200, marker='P', zorder=5, label='Perfect Source (0, √2)')

        # --------------------------------------------------
        # 7. Axes & grid
        # --------------------------------------------------
        plt.xlim(-0.1, 1.6)
        plt.ylim(-0.1, 1.6)

        plt.axvline(0.5, color='gray', ls='--', alpha=0.4)
        plt.axhline(0.5, color='gray', ls='--', alpha=0.4)

        plt.xlabel("Row Rank (Receiving ↑)", fontsize=12)
        plt.ylabel("Column Rank (Sending ↑)", fontsize=12)
        plt.title("Channel-wise Source–Sink Map", fontsize=14)

        plt.grid(alpha=0.3)

        # --------------------------------------------------
        # 8. Legends
        # --------------------------------------------------
        lobe_legend = [
            Line2D([0], [0], color=c, lw=2, label=l)
            for l, c in lobe_colors.items() if l != "Other"
        ]

        plt.legend(handles=lobe_legend,
                   title="Lobes",
                   loc="best",
                   frameon=True)

        # --------------------------------------------------
        # Ideal Source / Sink reference points
        # --------------------------------------------------
        root2 = np.sqrt(2)

        # Ideal Sink
        plt.scatter(root2, 0,
                    marker='X',
                    s=260,
                    c='black',
                    linewidths=2.5,
                    zorder=10)

        plt.text(root2 + 0.03, 0,
                 "Ideal Sink",
                 fontsize=10,
                 fontweight='bold',
                 va='center',
                 ha='left')

        # Ideal Source
        plt.scatter(0, root2,
                    marker='+',
                    s=320,
                    c='black',
                    linewidths=2.5,
                    zorder=10)

        plt.text(0, root2 + 0.03,
                 "Ideal Source",
                 fontsize=10,
                 fontweight='bold',
                 va='bottom',
                 ha='center')

        plt.tight_layout()

        # --------------------------------------------------
        # 9. Save to both paths
        # --------------------------------------------------
        plt.savefig(combined_path, dpi=200, bbox_inches='tight')

        save_path = get_save_path("SS_Map", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

        plt.close()

        from ss_map_features import (
            ss_map_ch_features,
            ss_map_lb_features,
            ss_map_nw_features
        )

        # already computed
        row_rank_mean = np.mean(row_ranks_good, axis=1)
        col_rank_mean = np.mean(col_ranks_good, axis=1)
        labels = eeg_data.labels[:26]

        # channel-level
        ch_feats = ss_map_ch_features(row_rank_mean, col_rank_mean)

        # lobe-level
        lb_feats = ss_map_lb_features(
            row_rank_mean,
            col_rank_mean,
            labels,
            lobe_map
        )

        # network-level
        nw_feats = ss_map_nw_features(row_rank_mean, col_rank_mean)

        # --------------------------------------------------
        # Final container (directly appendable)
        # --------------------------------------------------
        ss_map = {
            "channel": ch_feats,
            "lobe": lb_feats,
            "network": nw_feats
        }

        # example


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

        # %% Section 9A: Visualize Sink Index Heatmap
        print("\n" + "=" * 60)
        print("SECTION 9A: VISUALIZING SINK INDEX HEATMAP")
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
        #plt.savefig(os.path.join(OUTPUT_PATH, f'{PATIENT_ID}_sink_heatmap.png'), dpi=150)
        #plt.show()

        print(f"Heatmap saved to: {OUTPUT_PATH}")

        # Print summary statistics
        print("\nTop 5 Sink Channels (by mean):")
        for i in range(min(5, len(labels_sorted))):
            print(f"  {i + 1}. {labels_sorted[i]}: {mean_sink[sort_idx[i]]:.4f}")

        # %% Section 9B: Visualize Sink Index Heatmap (GOOD Windows Only)
        print("\n" + "=" * 60)
        print("SECTION 9B: VISUALIZING GOOD-WINDOW SINK INDEX HEATMAP")
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
        save_path = get_save_path("SI_HM", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)
        #plt.show()

        print(f"Good-window heatmap saved to: {OUTPUT_PATH}")

        # Print top-5 sinks for good windows only
        mean_sink_good = np.mean(sink_indices_good, axis=1)

        print("\nTop 5 Sink Channels (Good Windows Only):")
        for i in range(min(5, len(labels_sorted))):
            print(f"  {i + 1}. {labels_sorted[i]}: {mean_sink_good[sort_idx[i]]:.4f}")

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)


    # %% Section 10: Visualize Sink Index Topomap
        print("\n" + "="*60)
        print("SECTION 10: VISUALIZING SINK INDEX TOPOMAP")
        print("="*60)

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
            vlim=(0, np.sqrt(2)),  # Use SAME limits so heatmaps are comparable
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
        save_path = get_save_path("SI_TM", PATIENT_ID, BATCH_OUTPUT_PATH)
        plt.savefig(save_path, dpi=150)
        #plt.show()

        print(f"Good-only topomap saved to: {OUTPUT_PATH}")

        print(f"\nTop 5 Sink Channels (GOOD ONLY):")
        top_sink_good_idx = np.argsort(mean_sink_index_good)[::-1][:5]
        for i, ch_idx in enumerate(top_sink_good_idx):
            print(f"  {i + 1}. {ch_names[ch_idx]}: {mean_sink_index_good[ch_idx]:.4f}")

        print(f"\n✅ Completed {file_idx}/{len(csv_files)} "
              f"({(file_idx / len(csv_files)) * 100:.1f}%) files. "
              f"Results saved in: {OUTPUT_PATH}\n")
        # ================================
        # REGION DEFINITIONS (FT vs CPO)
        # ================================

        left_all = [ch for ch in ch_names if ch[-1] in "13579"]
        right_all = [ch for ch in ch_names if ch[-1] in "2468"]
        mid_all = [ch for ch in ch_names if ch[-1] == "z" or ch[-1] == "Z"]

        frontotemporal_labels = [
            "Fp1", "Fp2",
            "F7", "F3", "Fz", "F4", "F8",
            "FC3", "FCz","FC4",
             "T7", "T8"
        ]

        left_FT = [
            "Fp1",
            "F7", "F3",
            "FC3",
            "T7"
        ]

        right_FT = [
            "Fp2",
            "F8", "F4",
            "FC4",
            "T8"
        ]

        centro_parieto_occipital_labels = [
            "C3", "Cz", "C4",
            "CP3", "CPz", "CP4",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz","O2"
        ]

        left_CPO = [
            "C3",
            "CP3",
            "P7", "P3",
            "O1"
        ]

        right_CPO = [
            "C4",
            "CP4",
            "P4", "P8",
            "O2"
        ]


        def roi_mean(arr, labels, roi):
            idx = [labels.index(ch) for ch in roi if ch in labels]
            if len(idx) == 0:
                return np.nan
            return float(np.nanmean(arr[idx]))

        # Map channel names to indices
        label_to_idx = {ch: i for i, ch in enumerate(ch_names)}

        # Get all left and right channel indices
        L_all_idx = [label_to_idx[ch] for ch in left_all if ch in label_to_idx]
        R_all_idx = [label_to_idx[ch] for ch in right_all if ch in label_to_idx]

        # Get FT,L_FT,R_FT and CPO,L_CPO,R_CPO channel indices
        FT_idx = [label_to_idx[ch] for ch in frontotemporal_labels if ch in label_to_idx]
        CPO_idx = [label_to_idx[ch] for ch in centro_parieto_occipital_labels if ch in label_to_idx]

        L_FT_idx = [label_to_idx[ch] for ch in left_FT if ch in label_to_idx]
        R_FT_idx = [label_to_idx[ch] for ch in right_FT if ch in label_to_idx]

        L_CPO_idx = [label_to_idx[ch] for ch in left_CPO if ch in label_to_idx]
        R_CPO_idx = [label_to_idx[ch] for ch in right_CPO if ch in label_to_idx]

        # Extract region-wise sink values for all windows

        #Sink index for all left and all right channels for all windows
        L_all_all = float(np.nanmean(mean_sink_index[L_all_idx]))
        R_all_all = float(np.nanmean(mean_sink_index[R_all_idx]))

        # Sink index for FT and CPO channels for all windows
        sink_FT = mean_sink_index[FT_idx]
        sink_CPO = mean_sink_index[CPO_idx]

        # Sink index for left and right FT channels for all windows
        sink_L_FT = float(np.nanmean(mean_sink_index[L_FT_idx]))
        sink_R_FT = float(np.nanmean(mean_sink_index[R_FT_idx]))

        # Sink index for left and right CPO channels for all windows
        sink_L_CPO = float(np.nanmean(mean_sink_index[L_CPO_idx]))
        sink_R_CPO = float(np.nanmean(mean_sink_index[R_CPO_idx]))

        # Extract region-wise sink values for all windows

        # Sink index for all left and all right channels for GOOD windows
        L_all_good = float(np.nanmean(mean_sink_index_good[L_all_idx]))
        R_all_good = float(np.nanmean(mean_sink_index_good[R_all_idx]))

        # Sink index for FT and CPO channels for GOOD windows
        sink_FT_good = mean_sink_index_good[FT_idx]
        sink_CPO_good = mean_sink_index_good[CPO_idx]

        # Sink index for left and right FT channels for GOOD windows
        sink_L_FT_good = float(np.nanmean(mean_sink_index_good[L_FT_idx]))
        sink_R_FT_good = float(np.nanmean(mean_sink_index_good[R_FT_idx]))

        # Sink index for left and right FT channels for GOOD windows
        sink_L_CPO_good = float(np.nanmean(mean_sink_index_good[L_CPO_idx]))
        sink_R_CPO_good = float(np.nanmean(mean_sink_index_good[R_CPO_idx]))

        # ----------------------------------------------
        # ASYMMETRY INDEX (Right vs Left)
        # ----------------------------------------------

        #For asymmetry calculation
        eps = 1e-6

        #Asymmetry Calculation for all windows
        AI_all_all = (L_all_all - R_all_all) / (L_all_all + R_all_all + eps) # All left and All right
        AI_FT_all = (sink_L_FT - sink_R_FT) / (sink_L_FT + sink_R_FT + eps) # All FT left and All FT right
        AI_CPO_all = (sink_L_CPO - sink_R_CPO) / (sink_L_CPO + sink_R_CPO + eps) # All CPO left and All CPO right

        # Asymmetry Calculation for GOOD windows
        AI_all_good = (L_all_good - R_all_good) / (L_all_good + R_all_good + eps) # GOOD left and All right
        AI_FT_good = (sink_L_FT_good - sink_R_FT_good) / (sink_L_FT_good + sink_R_FT_good + eps) # GOOD FT left and All FT right
        AI_CPO_good = (sink_L_CPO_good - sink_R_CPO_good) / (sink_L_CPO_good + sink_R_CPO_good + eps) # All CPO left and All CPO right

        # ----------------------------------------------
        # ENTROPY METRICS
        # ----------------------------------------------
        from preprocessing.utils.entropy_utils import hist_entropy


        # GOOD-window entropies
        H_SI_total_good = hist_entropy(mean_sink_index_good)
        H_SI_FT_good = hist_entropy(sink_FT_good)
        H_SI_CPO_good = hist_entropy(sink_CPO_good)
        H_SI_ratio_good = H_SI_FT_good / H_SI_CPO_good if H_SI_CPO_good > 0 else np.nan
        H_SI_Gradient_Good = H_SI_FT_good - H_SI_CPO_good


        # ============================================
        # SECTION 11: SUMMARY METRICS FOR EXCEL EXPORT
        # ============================================
        # %% Section 11: Summary Metrics for Excel Export
        print("\n" + "=" * 60)
        print("SECTION 11: SUMMARY METRICS FOR EXCEL EXPORT")
        print("=" * 60)
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
        #mean_sink_all = np.mean(sink_indices, axis=1)

        # top–5 sinks (all windows)
        #sink_idx_all, sink_label_all, sink_val_all = safe_topk(mean_sink_all, ch_names, k=5)

        # top–5 sources (all windows)
        #mean_source_all = np.mean(source_indices, axis=1)
        #source_idx_all, source_label_all, source_val_all = safe_topk(mean_source_all, ch_names, k=5)

        # ==================================================
        # 2) Mean Sink Index (GOOD windows only)
        # ==================================================
        mean_sink_good = np.mean(sink_indices_good, axis=1)

        # top–2 sink (good only)
        sink_idx_good, sink_label_good, sink_val_good = safe_topk(mean_sink_good, ch_names, k=5)

        # top–2 source (good only)
        mean_source_good = np.mean(source_indices_good, axis=1)
        source_idx_good, source_label_good, source_val_good = safe_topk(mean_source_good, ch_names, k=5)

        # ==================================================
        # 3) Regional Metrics (FT & CPO) — ALL windows
        # ==================================================
        #FT_Sink_All = roi_mean(mean_sink_all, ch_names, frontotemporal_labels)
        #FT_Source_All = roi_mean(mean_source_all, ch_names, frontotemporal_labels)

        #CPO_Sink_All = roi_mean(mean_sink_all, ch_names, centro_parieto_occipital_labels)
        #CPO_Source_All = roi_mean(mean_source_all, ch_names, centro_parieto_occipital_labels)

        # ==================================================
        # 5) Regional Metrics (FT & CPO) — GOOD windows
        # ==================================================
        FT_Sink_Good = roi_mean(mean_sink_good, ch_names, frontotemporal_labels)
        FT_Source_Good = roi_mean(mean_source_good, ch_names, frontotemporal_labels)

        CPO_Sink_Good = roi_mean(mean_sink_good, ch_names, centro_parieto_occipital_labels)
        CPO_Source_Good = roi_mean(mean_source_good, ch_names, centro_parieto_occipital_labels)

        # ----------------------------------------------
        # FRONTO_PARIETAL GRADIENT (FT ↔ CPO) (Top vs Bottom)
        # ----------------------------------------------

        #Sink_Gradient_All = FT_Sink_All - CPO_Sink_All
        Sink_Gradient_Good = FT_Sink_Good - CPO_Sink_Good

        #Source_Gradient_All = FT_Source_All - CPO_Source_All
        Source_Gradient_Good = FT_Source_Good - CPO_Source_Good

        # ==================================================
        # 4) Build result dictionary for this subject
        # ==================================================
        # ==================================================
        # CLEAN SUMMARY RESULT ENTRY (MATCHES RIGHT COLUMN)
        # ==================================================
        def safe_div(a, b):
            return float(a / b) if (b not in [0, None, np.nan] and not np.isnan(b)) else np.nan


        # --- 1) Temporal Variability (STD & CV) across windows ---

        # Sink variability (ALL windows)
        #sink_std_all = float(np.nanmean(np.std(sink_indices, axis=1)))
        #sink_cv_all = float(np.nanmean(
        #    np.std(sink_indices, axis=1) / (np.nanmean(sink_indices, axis=1) + 1e-6)
        #))

        # Sink variability (GOOD windows)
        sink_std_good = float(np.nanmean(np.std(sink_indices_good, axis=1)))
        sink_cv_good = float(np.nanmean(
            np.std(sink_indices_good, axis=1) / (np.nanmean(sink_indices_good, axis=1) + 1e-6)
        ))

        # --- 2) Global Sink Entropy ---

        from scipy.stats import entropy

        #global_sink_entropy_all = float(entropy(mean_sink_all + 1e-6))
        global_sink_entropy_good = float(entropy(mean_sink_good + 1e-6))

        # --- 3) Standard 10-20 2D coordinates (normalized) ---
        # Only for electrodes in your 26-channel list

        ten_twenty_xy = {
            "Fp1": (-0.5, 1.0), "Fp2": (0.5, 1.0),
            "F7": (-1.0, 0.7), "F3": (-0.5, 0.7), "Fz": (0, 0.7), "F4": (0.5, 0.7), "F8": (1.0, 0.7),
            "FT7": (-1.2, 0.4), "T7": (-1.3, 0.0), "CP7": (-1.2, -0.4),
            "FT8": (1.2, 0.4), "T8": (1.3, 0.0), "CP8": (1.2, -0.4),
            "C3": (-0.5, 0.0), "Cz": (0, 0.0), "C4": (0.5, 0.0),
            "P7": (-1.0, -0.7), "P3": (-0.5, -0.7), "Pz": (0, -0.7),"P4": (0.5, -0.7), "P8": (1.0, -0.7),
            "O1": (-0.5, -1.0), "Oz": (0, -1.0), "O2": (0.5, -1.0)
        }

        # build array of XY coords only for ch_names order
        ch_pos = np.array([ten_twenty_xy.get(ch, (0, 0)) for ch in ch_names])
        cz = np.array([0, 0])

        # Sink COM (ALL)
        #sink_centroid_all = np.average(ch_pos, axis=0, weights=mean_sink_all + 1e-6)
        #sink_com_dist_all = float(np.linalg.norm(sink_centroid_all - cz))

        # Sink COM (GOOD)
        sink_centroid_good = np.average(ch_pos, axis=0, weights=mean_sink_good + 1e-6)
        sink_com_dist_good = float(np.linalg.norm(sink_centroid_good - cz))

        # --- 4) Switching Rate ---

        # ALL windows
        #top_sink_each_window_all = np.argmax(sink_indices, axis=0)
        #switch_rate_all = float(np.mean(np.diff(top_sink_each_window_all) != 0))

        # GOOD windows
        top_sink_each_window_good = np.argmax(sink_indices_good, axis=0)
        switch_rate_good = float(np.mean(np.diff(top_sink_each_window_good) != 0))

        # --- 5) FT–CPO temporal ratio jitter ---

        # ALL windows
        #ft_vals_all = np.mean(sink_indices[FT_idx, :], axis=0)
        #cpo_vals_all = np.mean(sink_indices[CPO_idx, :], axis=0)
        #ft_cpo_ratio_win_all = ft_vals_all / (cpo_vals_all + 1e-6)
        #ft_cpo_jitter_all = float(np.std(ft_cpo_ratio_win_all))

        # GOOD windows
        ft_vals_good = np.mean(sink_indices_good[FT_idx, :], axis=0)
        cpo_vals_good = np.mean(sink_indices_good[CPO_idx, :], axis=0)
        ft_cpo_ratio_win_good = ft_vals_good / (cpo_vals_good + 1e-6)
        ft_cpo_jitter_good = float(np.std(ft_cpo_ratio_win_good))

        result_entry = {
            "ID": PATIENT_ID,
            "Original Duration": orig_duration,
            "Clean Duration": clean_duration,
            "Effective Duration": effective_duration,  # you already compute this earlier

            # ===== Band Power % =====
            # "delta": rel_band.get("delta"),
            # "theta": rel_band.get("theta"),
            # "alpha": rel_band.get("alpha"),
            # "beta": rel_band.get("beta"),
            # "gamma": rel_band.get("gamma"),

            # ===== Regional Sink/Source Metrics =====
            # "FT Sink (All)": FT_Sink_All,
            # "CPO Sink (All)": CPO_Sink_All,

            "FT_Sink_Good": FT_Sink_Good,
            "CPO_Sink_Good": CPO_Sink_Good,

            #"FT/CPO Sink_All)": safe_div(FT_Sink_All, CPO_Sink_All),
            #"FT/CPO Source (All)": safe_div(FT_Source_All, CPO_Source_All),

            "FT_CPO_Sink_Good": safe_div(FT_Sink_Good, CPO_Sink_Good),
            #"FT/CPO Source (Good)": safe_div(FT_Source_Good, CPO_Source_Good),

            # ===== Good SS Analysis % =====
            "Mean_Sink_Index_Good": float(np.nanmean(mean_sink_good)),
            "Mean_Source_Index_Good": float(np.nanmean(mean_source_good)),
            # "Top Sink Ch 1_to_5 (Good)": ", ".join(sink_label_good[:5]) if sink_label_good else None,
            # "Top Source Ch 1_to_5 (Good)": ", ".join(source_label_good[:5]) if source_label_good else None,

            # ===== SS Analysis % =====
            #"Mean Sink Index (All)": float(np.nanmean(mean_sink_all)),
            #"Mean Source Index (All)": float(np.nanmean(mean_source_all)),
            # "Top Sink Ch 1_to_5 (All)": ", ".join(sink_label_all[:5]) if sink_label_all else None,
            # "Top Source Ch 1_to_5 (All)": ", ".join(source_label_all[:5]) if source_label_all else None,

            # ===== Spatial Entropy Metrics (Good) =====
            "H_SI_total_good": H_SI_total_good,
            "H_SI_FT_good": H_SI_FT_good,
            "H_SI_CPO_good": H_SI_CPO_good,
            "H_SI_ratio_good": H_SI_ratio_good,
            "H_SI_Gradient_Good": H_SI_Gradient_Good,

            # ===== Spatial Entropy Metrics (All) =====
            #"SI value Entropy (All)": H_total,
            #"SI value Entropy (FT All)": H_FT,
            #"SI value Entropy (CPO All)": H_CPO,
            #"SI value Entropy Ratio (FT/CPO All)": H_ratio,
            #"Entropy Gradient (All)": Entropy_Gradient_All,

            # ===== Asymmetry Index (All) =====
            #"AI_FT_All": AI_FT_all,
            #"AI_CPO_All": AI_CPO_all,
            #"AI_All_All": AI_all_all,

            # ===== Asymmetry Index (GOOD) =====
            "AI_FT_Good": AI_FT_good,
            "AI_CPO_Good": AI_CPO_good,
            "AI_All_Good": AI_all_good,

            # ===== Gradient (GOOD) =====
           # "Sink Gradient (All)":  Sink_Gradient_All,
            "Sink Gradient (Good)": Sink_Gradient_Good,
            #"Source Gradient (All)":  Source_Gradient_All,
            "Source Gradient (Good)": Source_Gradient_Good,

            # ==== Heterogeneity Features ====
            #"SINK STD (All)": sink_std_all,
            "SINK STD (Good)": sink_std_good,

            #"SINK CV (All)": sink_cv_all,
            "SINK CV (Good)": sink_cv_good,

           # "Global Sink Entropy (All)": global_sink_entropy_all,
            "Global Sink Entropy (Good)": global_sink_entropy_good,

           # "Sink COM Dist (All)": sink_com_dist_all,
            "Sink COM Dist (Good)": sink_com_dist_good,

           # "Sink Switch Rate (All)": switch_rate_all,
            "Sink Switch Rate (Good)": switch_rate_good,

           # "FT-CPO Jitter (All)": ft_cpo_jitter_all,
            "FT-CPO Jitter (Good)": ft_cpo_jitter_good,

            # ==== A-Mat Singular Value Based Features ====
            "H_SVD_mean": H_SVD_mean,
            "H_SVD_std": H_SVD_std,
            "H_SVD_min": H_SVD_min,
            "H_SVD_max": H_SVD_max,

            "SVD_ratio_mean":SVD_ratio_mean,
            "SVD_ratio_std": SVD_ratio_std,
            "SVD_ratio_min": SVD_ratio_min,
            "SVD_ratio_max": SVD_ratio_max,

            "SVD_mean": SVD_mean,
            "SVD_std": SVD_std,
            "SVD_min": SVD_min,
            "SVD_max": SVD_max,

            "SVDSum_mean": SVD_Sum_mean,
            "SVDSum_std": SVD_Sum_std,
            "SVDSum_min": SVD_Sum_min,
            "SVDSum_max": SVD_Sum_max,

            "sing_val_diff_mean": sing_val_diff_mean,
            "sing_val_diff_std": sing_val_diff_std,
            "sing_val_diff_min": sing_val_diff_min,
            "sing_val_diff_max": sing_val_diff_max
        }

        # ===== Append TBR metrics with prefix 'TBR_' =====
        for k, v in TBR_metrics.items():
            result_entry[f"TBR_{k}"] = v


        # Append log-normal features (AFTER)
        for k, v in ln_features.items():
            result_entry[f"logn_{k}"] = v

        # ==================================================
        # Append SS-map features (FLATTENED, SAFE)
        # ==================================================

        # ---------- Channel-wise ----------
        for feat_name, feat_vals in ss_map["channel"].items():
            feat_vals = np.asarray(feat_vals)

            result_entry[f"SS_ch_{feat_name}_mean"] = float(np.mean(feat_vals))
            result_entry[f"SS_ch_{feat_name}_std"] = float(np.std(feat_vals))
            result_entry[f"SS_ch_{feat_name}_min"] = float(np.min(feat_vals))
            result_entry[f"SS_ch_{feat_name}_max"] = float(np.max(feat_vals))

        # ---------- Lobe-wise ----------
        for lobe, lobe_feats in ss_map["lobe"].items():
            for feat_name, val in lobe_feats.items():
                result_entry[f"SS_lb_{lobe}_{feat_name}"] = float(val)

        # ---------- Network-wise ----------
        for feat_name, val in ss_map["network"].items():
            if isinstance(val, dict):
                for sub_name, sub_val in val.items():
                    result_entry[f"SS_nw_{feat_name}_{sub_name}"] = float(sub_val)
            else:
                result_entry[f"SS_nw_{feat_name}"] = float(val)

        # ============================================================
        # Append stored Band_Pct features
        # ============================================================
        result_entry.update(band_pct_features)

        # ==================================================
        # Add this subject to batch results
        results.append(result_entry)
        plt.close('all')

    df = pd.DataFrame(results)

    # ==================================================
    # Min–max normalization of log-likelihood (across all subjects)
    # ==================================================

    ll_col = "logn_log_likelihood_mean"

    if ll_col in df.columns:
        ll_vals = df[ll_col].values

        # Handle case where all values are identical or NaN
        ll_min = np.nanmin(ll_vals)
        ll_max = np.nanmax(ll_vals)

        if ll_max > ll_min:
            df["logn_log_likelihood_norm"] = (ll_vals - ll_min) / (ll_max - ll_min)
        else:
            # All values identical → set normalized value to 0
            df["logn_log_likelihood_norm"] = 0.0

    # Extract last folder name from DATA_PATH
    last_folder = os.path.basename(DATA_PATH.rstrip("\\/"))

    # Build filename: BatchSummary_<lastfolder>.xlsx
    excel_name = f"BatchSummary_{last_folder}.xlsx"

    # Save Excel
    batch_excel_path = os.path.join(BATCH_OUTPUT_PATH, excel_name)
    df.to_excel(batch_excel_path, index=False)

    # Auto-adjust Excel column widths (unchanged)
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

    # Return per-directory results (do not change per-directory Excel output)
    return results



# 3) Add a main loop at the bottom of `main.py` to call run_pipeline for each input folder:

if __name__ == "__main__":
    start_time = time.time()
    all_results = []

    for DATA_PATH in dirs_to_process:
        try:
            res=process_directory(DATA_PATH)
            if res:
                all_results.extend(res)
        except Exception as e:
            # short error message and continue to next directory
            print(f"\n❌ Error processing `{DATA_PATH}`")
            traceback.print_exc()  # <<< SHOWS EXACT LINE
            continue

    # ----- Final Summary -----
    total_subjects = len(all_results)
    total_time_sec = time.time() - start_time

    # Convert seconds → HH:MM:SS
    hrs = int(total_time_sec // 3600)
    mins = int((total_time_sec % 3600) // 60)
    secs = int(total_time_sec % 60)

    print("\n" + "=" * 60)
    print("TIMING SUMMARY OF BATCH PROCESSING")
    print("=" * 60)
    print(f"Total subjects analyzed: {total_subjects}")
    print(f"Total time taken: {hrs:02d}:{mins:02d}:{secs:02d} (HH:MM:SS)")
    print("=" * 60 + "\n")