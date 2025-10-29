"""
EEG Visualization Utilities Module

This module provides utility functions for visualizing EEG/iEEG data with
support for annotations, channel highlighting, and sliding window plotting.

Functions:
    viz_eeg: Create sliding window plots of multichannel EEG data

Dependencies:
    numpy
    matplotlib
    tqdm
    time_utils (from this package)
"""

import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from tqdm import tqdm

# Import time utilities from the package
try:
    from .time_utils import time_sum, sec2time
except ImportError:
    # Fallback for standalone usage
    from time_utils import time_sum, sec2time


def viz_eeg(data,
            labels,
            fs=2000,
            WIN_LEN_SEC=10,
            WIN_STEP_SEC=1,
            scaling_factor=1e-4,
            fig_out_dir=".",
            fig_name="Figure",
            plot_title="Title",
            start_time='0:0:0',
            detect_time='0:0:0',
            colorMaps=None,
            annotations=None,
            soz_channels=None,
            prop_channels=None,
            bad_channels=None,
            data_recon=None,
            most_involved_channels=None):
    """
    Visualize multichannel EEG/iEEG data with sliding windows.

    This function creates a series of plots showing EEG data in sliding windows,
    with support for channel highlighting, annotations, and reconstructed signals.
    Each window is saved as a separate PNG file.

    Parameters
    ----------
    data : numpy.ndarray
        2D array of EEG data with shape (n_channels, n_samples)
    labels : list or array
        Channel labels/names for each row in data
    fs : int, optional
        Sampling frequency in Hz. Default is 2000.
    WIN_LEN_SEC : int, optional
        Window length in seconds for each plot. Default is 10.
    WIN_STEP_SEC : int, optional
        Step size in seconds between consecutive windows. Default is 1.
    scaling_factor : float, optional
        Vertical scaling factor for channel separation. Default is 1e-4.
        Increase for more separation, decrease for more compact display.
    fig_out_dir : str, optional
        Output directory for saving figures. Default is current directory.
    fig_name : str, optional
        Base name for output figure files. Default is "Figure".
    plot_title : str, optional
        Title for the plots. Default is "Title".
    start_time : str, optional
        Start time of recording in HH:MM:SS format. Default is '0:0:0'.
    detect_time : str, optional
        Detection event time in HH:MM:SS format for marking. Default is '0:0:0'.
    colorMaps : dict, optional
        Dictionary mapping channel region names to colors. If None, uses default colors.
    annotations : dict, optional
        Dictionary of {annotation_text: time} for marking events.
        Time can be in HH:MM:SS or MM:SS.MS format.
    soz_channels : list, optional
        List of seizure onset zone channel names (displayed in red).
    prop_channels : list, optional
        List of propagation channel names (displayed in orange).
    bad_channels : list, optional
        List of bad channel names to exclude from plotting.
    data_recon : numpy.ndarray, optional
        Reconstructed signal array with same shape as data (plotted in black).
    most_involved_channels : list, optional
        List of most involved channels for special highlighting.

    Returns
    -------
    None
        Saves PNG files to fig_out_dir/fig_name/

    Output Files
    ------------
    Creates directory: {fig_out_dir}/{fig_name}/
    Saves files as: {fig_name}_000.png, {fig_name}_001.png, etc.

    Examples
    --------
    Basic usage:
        data = np.random.randn(64, 20000)  # 64 channels, 10 seconds at 2000 Hz
        labels = [f"CH{i:02d}" for i in range(64)]
        viz_eeg(data, labels, fs=2000, WIN_LEN_SEC=5, fig_name="test_eeg")

    With annotations and special channels:
        annotations = {"Spike": "0:5:30", "Seizure": "0:8:15"}
        soz = ["CH01", "CH02"]
        bad = ["CH63", "CH64"]
        viz_eeg(data, labels, annotations=annotations,
                soz_channels=soz, bad_channels=bad)

    With custom colors by region:
        color_map = {"FP": "blue", "T": "green", "O": "red"}
        viz_eeg(data, labels, colorMaps=color_map)

    Adjust channel spacing:
        viz_eeg(data, labels, scaling_factor=2e-4)  # More separation
        viz_eeg(data, labels, scaling_factor=5e-5)  # More compact

    Notes
    -----
    - Each plot window shows WIN_LEN_SEC seconds of data
    - Vertical gray lines mark 1-second intervals
    - Channel names are color-coded: red (SOZ), orange (propagation), black (normal)
    - Detection time is marked with a red vertical line
    - Annotations appear as green vertical lines with text boxes
    - Bad channels are automatically excluded from plotting
    - Reconstructed signals (if provided) appear in black overlaid on original

    Visual Features
    ---------------
    - Background: Ivory color
    - Grid: Silver dashed lines at 1-second intervals
    - SOZ channels: Red text and thicker lines if most_involved_channels provided
    - Propagation channels: Orange text
    - Annotations: Green vertical lines with text in green boxes
    - Detection marker: Red vertical line with "Detect ON" label
    """
    # Handle default mutable arguments
    if annotations is None:
        annotations = {}
    if soz_channels is None:
        soz_channels = []
    if prop_channels is None:
        prop_channels = []
    if bad_channels is None:
        bad_channels = []
    if most_involved_channels is None:
        most_involved_channels = []
    if data_recon is None:
        data_recon = []

    # Default color scheme if not provided
    if colorMaps is None:
        color_names = ["royalblue",
                       "mediumorchid",
                       "green",
                       "teal",
                       "dodgerblue",
                       "darkorange",
                       "olive",
                       "gray",
                       "salmon",
                       "chocolate",
                       "green",
                       "teal",
                       "dodgerblue"]

    nCH = data.shape[0]

    # Calculate channel offset scaling
    channel_offset = 1.2 * scaling_factor

    # Calculate number of windows
    n_wins = int(np.floor((data.shape[1]-(WIN_LEN_SEC*fs))/(WIN_STEP_SEC*fs)))+1
    print(f"Total windows to plot: {n_wins}")

    # Process each window with a single progress bar
    pbar = tqdm(range(n_wins+1), desc="Plotting iEEG", dynamic_ncols=True,
                leave=True, position=0, ncols=100)

    for win_i in pbar:

        # Calculate window boundaries
        win_i_0 = int( win_i * WIN_STEP_SEC * fs )
        win_i_1 = int( win_i_0 + WIN_LEN_SEC * fs )

        # Update progress bar description with current window info
        pbar.set_description(f"Plotting window {win_i}/{n_wins} (samples {win_i_0}-{win_i_1})")

        # Optional: use pbar.write() for important messages that need to persist
        # pbar.write(f"Processing window {win_i}: samples {win_i_0} to {win_i_1}")

        # Create figure
        fig, ax = plt.subplots(1, figsize=(28, 14))
        plt.ion()
        ax.set_facecolor("ivory")

        # Add vertical second markers
        for i in range(WIN_LEN_SEC):
            plt.axvline(fs*i, c='silver', linestyle='--', linewidth=3)

        # Track channel grouping for coloring
        labi_1 = ''
        labi_2 = ''
        c_idx = 0

        # Plot each channel
        for i in range(nCH):
            # Skip bad channels
            if labels[i].strip() in bad_channels:
                continue

            # Extract data for this window
            plot_data = data[i, win_i_0:win_i_1]
            if len(data_recon) != 0:
                plot_data_recon = data_recon[i, win_i_0:win_i_1]

            # Time axis for plotting
            tW = np.arange(0, plot_data.shape[0], 1)

            # Extract channel region (letters only)
            current_label = ''.join((x for x in labels[i] if (not x.isdigit()) and (not x == '-')))

            # Add channel label with appropriate color
            label_x_pos = -fs/3 + fs/20
            label_y_pos = i * channel_offset

            if labels[i] in soz_channels:
                ax.text(label_x_pos, label_y_pos, labels[i], c='red', fontsize=12, weight="bold")
            elif labels[i] in prop_channels:
                ax.text(label_x_pos, label_y_pos, labels[i], c='orange', fontsize=12, weight="bold")
            else:
                ax.text(label_x_pos, label_y_pos, labels[i], c='k', fontsize=12)

            # Determine color index based on channel grouping
            if labi_1 == '':
                labi_1 = labels[i]
            else:
                labi_2 = labels[i]
                ch_labi_1 = ''.join((x for x in labi_1 if (not x.isdigit()) and (not x == '-')))
                ch_labi_2 = ''.join((x for x in labi_2 if (not x.isdigit()) and (not x == '-')))

                if ch_labi_1 != ch_labi_2:
                    labi_1 = labels[i]
                    c_idx += 1

            # Plot the signal
            signal_y = plot_data + i * channel_offset

            if colorMaps is None:
                # Use default color scheme
                if len(data_recon) == 0:
                    ax.plot(tW, signal_y, color=color_names[c_idx % len(color_names)])
                else:
                    ax.plot(tW, signal_y, color=color_names[c_idx % len(color_names)], lw=2)
                    ax.plot(tW, plot_data_recon + i * channel_offset, color='k')
            else:
                # Use provided color map
                if len(most_involved_channels) == 0:
                    ax.plot(tW, signal_y, color=colorMaps.get(current_label, 'black'),
                           linewidth=1.5, alpha=1)
                else:
                    if labels[i] in soz_channels:
                        ax.plot(tW, signal_y, color=colorMaps.get(current_label, 'black'),
                               linewidth=3, alpha=0.6)
                    ax.plot(tW, signal_y, color=colorMaps.get(current_label, 'black'),
                           linewidth=1.5, alpha=1)

        # Setup time axis labels
        tW_ticks = np.arange(0, plot_data.shape[0] + 1, fs)
        tW_tick_labels = []

        # Track if detection marker has been drawn in this window
        detection_drawn = False

        for ti in range(len(tW_ticks)):
            ticki = (tW_ticks[ti] + win_i_0) / fs
            tick = time_sum(start_time, sec2time(ticki))
            tW_tick_labels.append(tick)

            # Check for detection time marker (only draw once per window)
            if not detection_drawn and detect_time != '0:0:0':
                h0, m0, s0 = tick.split(':')[0], tick.split(':')[1], tick.split(':')[2]
                h1, m1, s1 = detect_time.split(':')[0], detect_time.split(':')[1], detect_time.split(':')[2]

                # Check for exact second match
                if h0 == h1 and m0 == m1 and int(float(s0)) == int(float(s1)):
                    # Calculate precise position within the window
                    detect_total_seconds = int(h1)*3600 + int(m1)*60 + int(float(s1))
                    current_window_start = win_i_0 / fs
                    detect_offset = detect_total_seconds - current_window_start

                    if 0 <= detect_offset <= WIN_LEN_SEC:
                        x_position = detect_offset * fs
                        ax.axvline(x_position, linewidth=5, color='red')
                        ax.text(x_position, 0, "Detect ON", fontsize=20,
                               bbox=dict(facecolor='red', alpha=1), color='gold')
                        detection_drawn = True

        # Add annotations
        annotations_drawn = set()  # Track which annotations have been drawn

        for annot_txt, annot_time in annotations.items():
            # Skip if already drawn in this window
            if annot_txt in annotations_drawn:
                continue

            # Parse annotation time (handle different formats)
            if annot_time.count(':') == 2:  # HH:MM:SS
                h1, m1, s1 = annot_time.split(':')
                if '.' in s1:
                    s1_ms = s1.split('.')
                    s1 = s1_ms[0]
                    ms1 = float('0.' + s1_ms[1]) if len(s1_ms) > 1 else 0
                else:
                    ms1 = 0
                h1, m1, s1 = int(h1), int(m1), int(float(s1))
            elif annot_time.count(':') == 1:  # MM:SS.MS
                m1, s1 = annot_time.split(':')
                if '.' in s1:
                    s1_ms = s1.split('.')
                    s1 = s1_ms[0]
                    ms1 = float('0.' + s1_ms[1]) if len(s1_ms) > 1 else 0
                else:
                    ms1 = 0
                h1, m1, s1 = 0, int(m1), int(float(s1))
            else:
                continue  # Skip invalid format

            # Calculate total seconds for annotation
            annot_total_seconds = h1*3600 + m1*60 + s1 + ms1

            # Calculate start time in seconds
            h0, m0, s0 = start_time.split(':')
            start_total_seconds = int(h0)*3600 + int(m0)*60 + int(float(s0))

            # Calculate absolute time of annotation
            absolute_annot_time = start_total_seconds + annot_total_seconds

            # Check if annotation falls within current window
            window_start_time = start_total_seconds + (win_i_0 / fs)
            window_end_time = window_start_time + WIN_LEN_SEC

            if window_start_time <= absolute_annot_time < window_end_time:
                # Calculate position within window
                x_position = (absolute_annot_time - window_start_time) * fs
                ax.axvline(x_position, linewidth=5, color='limegreen')
                y_pos = np.random.uniform(5, 8) * channel_offset
                ax.text(x_position, y_pos, annot_txt, fontsize=20,
                       bbox=dict(facecolor='springgreen', alpha=1), color='saddlebrown')
                annotations_drawn.add(annot_txt)

        # Set axis properties
        ax.set(xlim=[-fs/3, fs*WIN_LEN_SEC])
        ax.set(ylim=[-0.255*1e-3, nCH * 1.25 * scaling_factor])
        ax.set_xticks(tW_ticks)
        ax.set_xticklabels(tW_tick_labels, fontsize=16, rotation=45)
        ax.axes.get_yaxis().set_visible(False)
        ax.grid(False)

        # Add title
        plt.title(f"{plot_title}\nGray vertical lines are 1 second intervals", fontsize=16)
        plt.tight_layout()

        # Save figure
        output_path = f'{fig_out_dir}/{fig_name}'
        makedirs(output_path, exist_ok=True)
        plt.ioff()
        plt.savefig(f'{output_path}/{fig_name}_{win_i:03}.png')
        plt.close()


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Generate and visualize a synthetic 20-channel EEG signal.
    
    This example demonstrates how to:
    1. Generate synthetic multichannel EEG-like data
    2. Add different frequency components to different channels
    3. Add spike events and artifacts
    4. Visualize with annotations and special channel markings
    """

    import numpy as np

    # Signal parameters
    n_channels = 20
    fs = 2000  # Sampling frequency (Hz)
    duration = 30  # Duration in seconds
    n_samples = fs * duration

    # Generate channel labels (simulating 10-20 EEG system)
    regions = ['FP', 'F', 'C', 'T', 'P', 'O']
    labels = []
    for region in regions[:4]:  # Use first 4 regions
        for i in range(5):  # 5 channels per region
            if i < len(regions):
                labels.append(f"{region}{i+1}")
    labels = labels[:n_channels]  # Ensure exactly 20 channels

    print(f"Generating {n_channels}-channel synthetic EEG signal...")
    print(f"Duration: {duration} seconds at {fs} Hz")
    print(f"Channels: {', '.join(labels)}")

    # Generate base signal (different frequency bands for different regions)
    np.random.seed(42)  # For reproducibility
    data = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        # Base noise (pink noise approximation)
        noise = np.random.randn(n_samples)
        freqs = np.fft.fftfreq(n_samples, 1/fs)
        fft = np.fft.fft(noise)
        fft[1:] = fft[1:] / np.sqrt(np.abs(freqs[1:]))  # 1/f scaling
        base_signal = np.real(np.fft.ifft(fft))

        # Add region-specific rhythms
        t = np.arange(n_samples) / fs

        if 'FP' in labels[ch]:  # Frontal - add beta (13-30 Hz)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            base_signal += beta

        elif 'F' in labels[ch]:  # Frontal - add alpha (8-13 Hz)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            base_signal += alpha

        elif 'C' in labels[ch]:  # Central - add mu rhythm (8-12 Hz)
            mu = 0.4 * np.sin(2 * np.pi * 11 * t + np.random.rand() * 2 * np.pi)
            base_signal += mu

        elif 'T' in labels[ch]:  # Temporal - add theta (4-8 Hz)
            theta = 0.6 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            base_signal += theta

        # Normalize to typical EEG amplitude range (microvolts)
        base_signal = base_signal * 50e-6  # ~50 µV amplitude
        data[ch, :] = base_signal

    # Add some events/artifacts
    print("Adding synthetic events...")

    # Add a spike at 5 seconds (affects temporal channels more)
    spike_time = 5 * fs
    spike_duration = int(0.07 * fs)  # 70ms spike
    for ch in range(n_channels):
        if 'T' in labels[ch]:  # Temporal channels
            spike_amp = 200e-6 * np.exp(-np.arange(spike_duration)/20)
            data[ch, spike_time:spike_time+spike_duration] += spike_amp
        elif 'C' in labels[ch]:  # Some spread to central
            spike_amp = 100e-6 * np.exp(-np.arange(spike_duration)/20)
            data[ch, spike_time:spike_time+spike_duration] += spike_amp

    # Add a slower wave at 12 seconds
    slow_wave_time = 12 * fs
    slow_wave_duration = int(0.5 * fs)  # 500ms
    for ch in range(n_channels):
        if 'F' in labels[ch] or 'C' in labels[ch]:
            t_wave = np.arange(slow_wave_duration) / fs
            slow_wave = 150e-6 * np.sin(2 * np.pi * 3 * t_wave) * np.exp(-t_wave*3)
            data[ch, slow_wave_time:slow_wave_time+slow_wave_duration] += slow_wave

    # Add movement artifact at 20 seconds (affects frontal channels)
    artifact_time = 20 * fs
    artifact_duration = int(2 * fs)  # 2 seconds
    for ch in range(n_channels):
        if 'FP' in labels[ch]:
            artifact = 300e-6 * np.random.randn(artifact_duration) * 0.5
            data[ch, artifact_time:artifact_time+artifact_duration] += artifact

    # Define special channels and annotations
    soz_channels = ['T1', 'T2', 'T3']  # Seizure onset zone
    prop_channels = ['C1', 'C2']  # Propagation channels
    bad_channels = ['FP5']  # Bad channel

    annotations = {
        'Spike': '0:0:5',
        'Slow Wave': '0:0:12',
        'Movement': '0:0:20',
        'Baseline': '0:0:25'
    }

    # Create custom color map by region
    color_map = {
        'FP': 'darkblue',
        'F': 'blue',
        'C': 'green',
        'T': 'red',
        'P': 'purple',
        'O': 'orange'
    }

    # Visualize the data
    print("\nVisualizing EEG data...")
    print(f"Output directory: ./example_eeg_output/")
    print(f"Window length: 10 seconds")
    print(f"Window step: 5 seconds")
    print(f"Special channels:")
    print(f"  - SOZ (red): {', '.join(soz_channels)}")
    print(f"  - Propagation (orange): {', '.join(prop_channels)}")
    print(f"  - Bad (excluded): {', '.join(bad_channels)}")
    print(f"Annotations: {list(annotations.keys())}")

    # Call the visualization function
    viz_eeg(
        data=data,
        labels=labels,
        fs=fs,
        WIN_LEN_SEC=10,  # 10-second windows
        WIN_STEP_SEC=5,   # 5-second step (50% overlap)
        scaling_factor=1.5e-4,  # Slightly more separation
        fig_out_dir="./example_eeg_output",
        fig_name="synthetic_eeg",
        plot_title="Synthetic 20-Channel EEG Example",
        start_time='0:0:0',
        detect_time='0:0:15',  # Mark detection at 15 seconds
        colorMaps=color_map,
        annotations=annotations,
        soz_channels=soz_channels,
        prop_channels=prop_channels,
        bad_channels=bad_channels
    )

    print("\nVisualization complete!")
    print("Check ./example_eeg_output/synthetic_eeg/ for output images.")
    print("\nTo view the images in sequence, you can use an image viewer")
    print("or create an animated GIF with tools like ImageMagick:")
    print("  convert -delay 50 -loop 0 ./example_eeg_output/synthetic_eeg/*.png animation.gif")