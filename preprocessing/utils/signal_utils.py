"""
Signal Processing Utilities Module

This module provides utility functions for signal processing and analysis,
including change point detection and information theory calculations.

Functions:
    _xlogx: Safe computation of x*log(x) for entropy calculations
    change_pnt: Detect change points in monotonically decreasing signals

Dependencies:
    numpy
    matplotlib (optional, for plotting in change_pnt)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def _xlogx(x, base=2):
    """
    Compute x * log_base(x) safely, handling zero and negative values.

    This function calculates x * log(x) while properly handling edge cases:
    - Returns 0 when x = 0 (following the limit convention)
    - Returns NaN for negative values
    - Uses specified logarithm base (default base 2 for information theory)

    This is particularly useful for entropy calculations where the power
    spectrum density may contain zero values.

    Parameters
    ----------
    x : array-like
        Input values. Can be scalar, list, or numpy array.
    base : int or float, optional
        Logarithm base. Default is 2 (for bits).
        Use np.e for natural logarithm (nats).

    Returns
    -------
    numpy.ndarray
        Array of x * log_base(x) values with same shape as input.
        - 0 where x == 0
        - NaN where x < 0
        - x * log_base(x) where x > 0

    Examples
    --------
    ### import numpy as np
    ### _xlogx([0, 1, 2, 4])
    array([0., 0., 2., 8.])

    ### _xlogx([0.5, 1, 2], base=np.e)  # Natural log
    array([-0.34657359, 0., 1.38629436])

    ### _xlogx([-1, 0, 1])  # Negative values return NaN
    array([nan, 0., 0.])

    Applications
    ------------
    Commonly used in:
    - Shannon entropy: H = -sum(p * log(p))
    - Kullback-Leibler divergence
    - Mutual information calculations
    - Any formula involving p*log(p) terms where p might be 0

    Mathematical Note
    -----------------
    The convention that 0 * log(0) = 0 comes from the limit:
    lim(x→0+) x * log(x) = 0

    This is standard in information theory and allows entropy
    calculations to handle probability distributions with zeros.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx


def change_pnt(sig, plot=False):
    """
    Detect change points in a monotonically decreasing signal.

    This function identifies significant change points in a signal by
    analyzing the derivative (differences between consecutive points).
    Points where the derivative exceeds median + std/4 are considered
    significant.

    Parameters
    ----------
    sig : array-like
        Input signal, should be monotonically decreasing.
        Example: [1110, 5000, 500, 200, 100, 10, 8, 7, 6, 5, 2, 1]
    plot : bool, optional
        If True, creates a plot showing the signal, its derivative,
        and the detected change point. Default is False.

    Returns
    -------
    numpy.ndarray
        Indices of significant change points, ordered from first to last.
        These are the indices where significant changes occur.

    Examples
    --------
    ### import numpy as np
    ### # Example: Rapid initial decrease then gradual decline
    ### signal = [1000, 500, 100, 50, 45, 40, 38, 36, 35, 34]
    ### change_points = change_pnt(signal)
    ### print(f"Change points at indices: {change_points}")

    ### # With plotting
    ### signal = [1110, 5000, 500, 200, 100, 10, 8, 7, 6, 5, 2, 1]
    ### change_points = change_pnt(signal, plot=True)
    ### # Shows plot with signal, derivative, and last change point marked

    Algorithm
    ---------
    1. Compute absolute differences between consecutive points
    2. Calculate threshold: median(diff) + std(diff)/4
    3. Find all points where difference exceeds threshold
    4. Return indices of these significant change points

    Use Cases
    ---------
    - Feature selection: Identifying where feature importance drops significantly
    - Signal segmentation: Finding transitions between different signal regimes
    - Quality metrics: Detecting where quality measures plateau
    - Elbow detection: Finding the "elbow" in scree plots or similar curves

    Notes
    -----
    - Designed for monotonically decreasing signals
    - The threshold (median + std/4) can be adjusted for different sensitivities
    - Returns ALL significant points, not just the most significant
    - The plot (if enabled) marks only the LAST significant change point

    Visualization
    -------------
    When plot=True, generates a figure with:
    - Blue line: Original signal
    - Orange line: Absolute derivative
    - Red vertical line: Last significant change point

    This helps visualize where the signal's rate of change becomes less dramatic.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        if plot:
            print("matplotlib not available. Skipping plot.")
            plot = False

    diff = np.abs(np.concatenate(([np.diff(sig)[0]], np.diff(sig))))
    significant_channels_idx = np.where(diff > (np.median(diff) + np.std(diff)/4))[0]

    if plot:
        plt.figure()
        plt.plot(sig, label='Signal')
        plt.plot(diff, label='|Derivative|')
        if len(significant_channels_idx) > 0:
            plt.axvline(significant_channels_idx[-1], color='red',
                       linestyle='--', label=f'Last change point (idx={significant_channels_idx[-1]})')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Change Point Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return significant_channels_idx


def analyze_correlations(X1, X2, labels, fs=2000, window_duration=1.0, overlap=0.5):
    """
    Analyze windowed correlations between two multichannel signals.

    Parameters
    ----------
    X1 : numpy.ndarray
        First signal (e.g., original) with shape (n_channels, n_samples)
    X2 : numpy.ndarray
        Second signal (e.g., reconstructed) with shape (n_channels, n_samples)
    labels : list
        Channel labels
    fs : int, optional
        Sampling frequency in Hz. Default is 2000.
    window_duration : float, optional
        Window duration in seconds. Default is 1.0.
    overlap : float, optional
        Window overlap fraction (0 to 1). Default is 0.5.

    Returns
    -------
    dict
        Dictionary containing:
        - 'df': DataFrame with all correlation values
        - 'summary': Summary statistics
        - 'correlation_array': 2D array of correlations
        - 'thresholds': Dictionary of thresholds per channel
    """

    # Calculate window parameters
    window_size = int(window_duration * fs)
    step_size = int(window_size * (1 - overlap))
    n_windows = (X1.shape[1] - window_size) // step_size + 1

    print(f"\n{'=' * 60}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Channels: {len(labels)}")
    print(f"Signal duration: {X1.shape[1] / fs:.1f} seconds")
    print(f"Window: {window_duration}s, Overlap: {overlap * 100}%, Windows: {n_windows}")
    print(f"{'=' * 60}\n")

    # Store correlations
    correlation_data = []
    correlation_array = np.zeros((X1.shape[0], n_windows))

    # Calculate correlations for each channel and window
    for i in tqdm(range(len(labels)), desc="Computing correlations"):
        channel = labels[i]
        signal_X1 = X1[i, :]
        signal_X2 = X2[i, :]

        for j, start in enumerate(range(0, len(signal_X1) - window_size + 1, step_size)):
            end = start + window_size

            # Calculate correlation
            try:
                corr = np.corrcoef(signal_X1[start:end], signal_X2[start:end])[0, 1]
                corr = np.abs(corr) if not np.isnan(corr) else 0
            except:
                corr = 0

            correlation_data.append({
                "Window": f"{start}-{end}",
                "Time_Start": start / fs,
                "Time_End": end / fs,
                "Correlation": corr,
                "Channel": channel
            })
            correlation_array[i, j] = corr

    # Create DataFrame
    df = pd.DataFrame(correlation_data)

    # Calculate thresholds and statistics for each channel
    print("\nCalculating statistics per channel...")
    thresholds = {}
    channel_stats = []

    for i, label in enumerate(labels):
        channel_corrs = df[df['Channel'] == label]['Correlation'].values

        Q1 = np.percentile(channel_corrs, 25)
        Q3 = np.percentile(channel_corrs, 75)
        IQR = Q3 - Q1
        threshold = Q3 - 2 * IQR

        thresholds[i] = max(0, threshold)

        channel_stats.append({
            'Channel': label,
            'Mean': np.mean(channel_corrs),
            'Median': np.median(channel_corrs),
            'Std': np.std(channel_corrs),
            'Min': np.min(channel_corrs),
            'Max': np.max(channel_corrs),
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Threshold': thresholds[i],
            'Below_Threshold_%': np.sum(channel_corrs < thresholds[i]) / len(channel_corrs) * 100
        })

    df_stats = pd.DataFrame(channel_stats)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    all_corrs = df['Correlation'].values
    print(f"\nOverall Statistics:")
    print(f"  Mean correlation: {np.mean(all_corrs):.3f}")
    print(f"  Median correlation: {np.median(all_corrs):.3f}")
    print(f"  Std deviation: {np.std(all_corrs):.3f}")
    print(f"  Min correlation: {np.min(all_corrs):.3f}")
    print(f"  Max correlation: {np.max(all_corrs):.3f}")

    print(f"\nPer-Channel Summary:")
    print("-" * 60)
    print(f"{'Channel':<10} {'Mean':>6} {'Median':>7} {'Min':>6} {'Max':>6} {'Below Threshold':>15}")
    print("-" * 60)
    for _, row in df_stats.iterrows():
        print(f"{row['Channel']:<10} {row['Mean']:>6.3f} {row['Median']:>7.3f} "
              f"{row['Min']:>6.3f} {row['Max']:>6.3f} {row['Below_Threshold_%']:>14.1f}%")

    # Identify best and worst channels
    best_channel = df_stats.loc[df_stats['Mean'].idxmax(), 'Channel']
    worst_channel = df_stats.loc[df_stats['Mean'].idxmin(), 'Channel']
    print(f"\nBest reconstruction: {best_channel} (mean corr = {df_stats['Mean'].max():.3f})")
    print(f"Worst reconstruction: {worst_channel} (mean corr = {df_stats['Mean'].min():.3f})")

    # Create visualization
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))

    # 1. Boxplot by channel
    ax1 = plt.subplot(2, 3, 1)
    sns.boxplot(x='Channel', y='Correlation', data=df, ax=ax1)
    ax1.set_title('Correlation Distribution by Channel', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add threshold lines
    for i, label in enumerate(labels):
        threshold = thresholds[i]
        x_pos = i
        ax1.plot([x_pos - 0.4, x_pos + 0.4], [threshold, threshold],
                 'r--', linewidth=1, alpha=0.5)

    # 2. Correlation heatmap
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(correlation_array, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Window Index')
    ax2.set_ylabel('Channel')
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Overall histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(all_corrs, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(np.mean(all_corrs), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(all_corrs):.3f}')
    ax3.axvline(np.median(all_corrs), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(all_corrs):.3f}')
    ax3.set_title('Overall Correlation Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Mean correlation by channel (bar plot)
    ax4 = plt.subplot(2, 3, 4)
    mean_corrs = [df_stats[df_stats['Channel'] == label]['Mean'].values[0] for label in labels]
    bars = ax4.bar(range(len(labels)), mean_corrs)

    # Color bars based on correlation strength
    for i, (bar, corr) in enumerate(zip(bars, mean_corrs)):
        if corr > 0.8:
            bar.set_color('green')
        elif corr > 0.6:
            bar.set_color('yellow')
        elif corr > 0.4:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax4.set_title('Mean Correlation by Channel', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Mean Correlation')
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=45, fontsize=8)
    ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Correlation over time (average across channels)
    ax5 = plt.subplot(2, 3, 5)
    mean_corr_over_time = np.mean(correlation_array, axis=0)
    time_points = np.arange(len(mean_corr_over_time)) * step_size / fs
    ax5.plot(time_points, mean_corr_over_time, linewidth=2, color='navy')
    ax5.fill_between(time_points, mean_corr_over_time, alpha=0.3)
    ax5.set_title('Average Correlation Over Time', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Mean Correlation')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1])

    # 6. Summary table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')

    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Windows', f'{n_windows}'],
        ['Mean Correlation', f'{np.mean(all_corrs):.3f}'],
        ['Median Correlation', f'{np.median(all_corrs):.3f}'],
        ['Std Deviation', f'{np.std(all_corrs):.3f}'],
        ['Min Correlation', f'{np.min(all_corrs):.3f}'],
        ['Max Correlation', f'{np.max(all_corrs):.3f}'],
        ['Best Channel', f'{best_channel}'],
        ['Worst Channel', f'{worst_channel}']
    ]

    table = ax6.table(cellText=summary_data, loc='center', cellLoc='left',
                      colWidths=[0.5, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(summary_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    plt.suptitle('Correlation Analysis Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Return results
    return {
        'df': df,
        'summary': df_stats,
        'correlation_array': correlation_array,
        'thresholds': thresholds
    }


def compute_spectrogram(input_vector, sample_rate, window_size=None, hop_size=None, n_fft=None, window_type="hann",
                        log_info='off'):
    """
    Computes the spectrogram using PyTorch on a GPU, considering the inputs similar to the scipy.signal.periodogram function,
    and also returns the frequency axis.

    Parameters:
    ----------
    input_vector : list or np.array
        The input signal or data sequence for which to compute the spectrogram.
    sample_rate : int
        The sample rate of the input vector, measured in Hertz (Hz).
    window_size : int, optional
        The size of the analysis window used in the FFT computation.
        This parameter determines the size of the analysis window used in the
        FFT computation. It specifies the number of samples in each window. A larger
        window size provides better frequency resolution but sacrifices time resolution.
        It is usually a power of 2 for efficient FFT computation.
    hop_size : int, optional
        The number of samples between the start of one window and the start
        of the next window in the spectrogram.
        A smaller hop size leads to a higher overlap, while a larger hop size
        reduces the overlap. Increasing the overlap can improve time resolution
        by reducing the spacing between consecutive time frames in the spectrogram.
        However,it also increases computational complexity. A smaller hop size
        increases the time resolution but may introduce more spectral leakage.
    n_fft : int, optional
        The number of FFT points or bins used in the FFT computation.
        The parameter specifies the number of FFT points or bins used in the FFT
        computation. It determines the frequency resolution of the resulting spectrogram.
        More FFT points yield finer frequency resolution but increase computational complexity.
        Typically, n_fft is also a power of 2 for efficient FFT computation.
    window_type : str, optional
        The type of window function to be used. Available options: "hann" (default), "hamming",
        "boxcar", "bartlett", "blackman", "kaiser".


    Returns:
    -------
    power_spectrogram : np.array
        The power spectrogram computed using the STFT.
    time_axis : np.array
        The time axis corresponding to the spectrogram. It provides the time points at which the spectrogram is computed.
    frequency_axis : np.array
        The frequency axis corresponding to the spectrogram. It provides the frequencies at which the power spectrum is
        calculated.

    Example usage I (General):
    --------------
    ### input_vector = [...]  # Your input vector
    ### sample_rate = 44100  # Sample rate of the input vector
    ### window_size = 1024  # Size of the analysis window (in samples)
    ### hop_size = 512  # Hop size (in samples)
    ### n_fft = 1024  # Number of FFT bins
    ### time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector, sample_rate, window_size, hop_size, n_fft)

    Example usage II (Synthetic signal):
    --------------
    ###  # Generate a sample signal
    ###  fs = 2  # Sample rate (Hz)
    ###  t = np.arange(0, 180, 1/fs)  # Time vector
    ###  f1 = 0.8  # Frequency of the signal
    ###  sig = np.sin(2 * np.pi * f1 * t)
    ###  time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector=sig, sample_rate=2, window_type="hann")
    ###
    ###  # Plot the spectrogram
    ###  plt.plot(frequency_axis, power_spectrogram)
    ###  plt.xlabel('Frequency')
    ###  plt.ylabel('Power Spectrum')
    ###  plt.show()

    Example usage III (Synthetic signal):
    --------------
    ### fs = 10e3
    ### N = 1e5
    ### amp = 2 * np.sqrt(2)
    ### noise_power = 0.01 * fs / 2
    ### time = np.arange(N) / float(fs)
    ### mod = 500*np.cos(2*np.pi*0.25*time)
    ### carrier = amp * np.sin(2*np.pi*3e3*time + mod)
    ### rng = np.random.default_rng()
    ### noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    ### noise *= np.exp(-time/5)
    ### x = carrier + noise
    ###
    ### # frequency_axis, power_spectrogram = compute_spectrogram(input_vector=sig, sample_rate=2, window_type="hann")
    ### time_axis, frequency_axis, power_spectrogram = compute_spectrogram(input_vector=x, sample_rate=fs, window_size=3*60, hop_size=2*60, n_fft=3*60, window_type="hann")
    ###
    ### # Plot the computed spectrogram
    ### fig, ax = plt.subplots(figsize=(10,6))
    ### plt.pcolormesh(time_axis, frequency_axis, power_spectrogram, shading='gouraud', cmap="turbo")
    ### ax.tick_params(axis='x', rotation=45, labelsize=13)
    ### ax.tick_params(axis='y', rotation=45, labelsize=13)
    ### ax.set_xlabel("Time [sec]", fontsize=15)
    ### ax.set_ylabel("Frequency [Hz]", fontsize=15)
    ### ax.grid(False)
    ### plt.show()
    """

    import torch

    def boxcar_window(window_size):
        # Create a tensor of ones with the desired window size
        window = torch.ones(window_size)
        return window

    # Handle input variables
    if window_size == None:
        # If None the length of x will be used.
        window_size = len(input_vector)
    if hop_size == None:
        # If None the length of x will be used.
        hop_size = len(input_vector)
    if n_fft is None:
        # If None the length of x will be used.
        n_fft =  len(input_vector)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert input vector to a PyTorch tensor and move it to the GPU
    input_tensor = torch.tensor(input_vector, dtype=torch.float64).to(device)

    # Print the device and tensor contents
    if log_info == 'on':
        if device.type == "cuda":
            print("INFO: GPU available. Tensor moved to GPU.")
        else:
            print("INFO: No GPU available. Tensor remains on CPU.")

    # Compute the STFT (Short-Time Fourier Transform)
    if window_type == "boxcar":
        window = boxcar_window(window_size)
    else:
        window = getattr(torch, f"{window_type}_window")(window_size)
    spectrogram = torch.stft(input_tensor, n_fft=n_fft, hop_length=hop_size, win_length=window_size,
                             window=window.cuda(), center=False, return_complex=True, onesided=True,
                             normalized=True)

    # Convert complex spectrogram to magnitude spectrogram
    magnitude_spectrogram = torch.abs(spectrogram)

    # Compute the power spectrogram
    # power_spectrogram = magnitude_spectrogram.pow(2)
    power_spectrogram = magnitude_spectrogram ** 2

    # Scale the spectrogram by the window energy
    # power_spectrogram *= window_size / float(hop_size * sample_rate)

    # Compute the time axis
    total_time = len(input_vector) / sample_rate
    num_segments = power_spectrogram.shape[1]
    if num_segments == 1:
        time_axis = torch.zeros(1)
    else:
        time_step = total_time / (num_segments - 1)
        time_axis = torch.arange(0, total_time + time_step, time_step)

    # Compute the frequency axis
    frequency_axis = torch.linspace(0, sample_rate/2, n_fft//2 + 1)

    # Convert output vector to a Numpy array and move it to the CPU
    if device.type == "cuda":
        power_spectrogram = power_spectrogram.cpu().detach().numpy()
    time_axis = time_axis.detach().numpy()
    frequency_axis = frequency_axis.detach().numpy()

    return time_axis, frequency_axis, power_spectrogram


def compute_signal_energy(data, fs, window_size_sec=1.0, overlap=0.5):
    """
    Compute windowed energy of EEG signal.

    Parameters:
    -----------
    data : np.ndarray
        EEG data matrix (n_channels x n_samples)
    fs : float
        Sampling frequency in Hz
    window_size_sec : float
        Window size in seconds for energy computation
    overlap : float
        Overlap fraction between windows (0-1)

    Returns:
    --------
    energy : np.ndarray
        Energy matrix (n_channels x n_windows)
    energy_normalized : np.ndarray
        Row-normalized energy matrix (n_channels x n_windows)
    time_axis : np.ndarray
        Time axis for energy windows (in seconds)
    """
    n_channels, n_samples = data.shape
    window_size = int(window_size_sec * fs)
    step_size = int(window_size * (1 - overlap))

    # Calculate number of windows
    n_windows = int((n_samples - window_size) / step_size) + 1

    # Initialize energy matrix
    energy = np.zeros((n_channels, n_windows))

    # Compute energy for each window
    for win_idx in range(n_windows):
        start_idx = win_idx * step_size
        end_idx = start_idx + window_size

        if end_idx > n_samples:
            end_idx = n_samples

        # Energy = sum of squared amplitudes
        window_data = data[:, start_idx:end_idx]
        energy[:, win_idx] = np.sum(window_data ** 2, axis=1)

    # Normalize by dividing by window size (energy per sample)
    energy = energy / window_size

    # Row-wise normalization (normalize each channel by its maximum)
    energy_normalized = np.zeros_like(energy)
    for ch in range(n_channels):
        max_energy = np.max(energy[ch, :])
        if max_energy > 0:
            energy_normalized[ch, :] = energy[ch, :] / max_energy
        else:
            energy_normalized[ch, :] = energy[ch, :]

    # Create time axis
    time_axis = np.arange(n_windows) * (window_size_sec * (1 - overlap))

    return energy, energy_normalized, time_axis
