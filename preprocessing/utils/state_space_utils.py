"""
State-Space Modeling Utilities Module

This module provides utility functions for state-space modeling, autoregressive
model estimation, signal reconstruction, and network analysis of multichannel
time series data (e.g., EEG/iEEG).

Functions:
    estimateA: Estimate state transition matrix using least squares
    estimateA_subject: Estimate state transition matrices for windowed data
    reconstruct_signal: Reconstruct signal using estimated state-space model
    identifySS: Identify sources and sinks in network connectivity

Dependencies:
    numpy
    tqdm
"""

import numpy as np
from tqdm import tqdm


def estimateA(X):
    """
    Estimate state transition matrix A using least squares method.

    This function implements Jeff Craley's method for estimating the state
    transition matrix A in the autoregressive model: X(t) = A * X(t-1) + noise

    Parameters
    ----------
    X : numpy.ndarray
        2D array of time series data with shape (n_channels, n_samples)
        where each row is a channel and columns are time points.

    Returns
    -------
    numpy.ndarray
        State transition matrix A with shape (n_channels, n_channels)
        where A[i,j] represents the influence of channel j on channel i.

    Theory
    ------
    The AR(1) model assumes: X(t) = A * X(t-1) + ε(t)

    Using least squares: A_hat = Y @ Z^+ where:
    - Y = X[:, 1:] (future states)
    - Z = X[:, 0:-1] (current states)
    - Z^+ is the pseudoinverse of Z

    Examples
    --------
    ### import numpy as np
    ### # Create sample data: 3 channels, 100 time points
    ### X = np.random.randn(3, 100)
    ### A_hat = estimateA(X)
    ### print(f"Transition matrix shape: {A_hat.shape}")
    Transition matrix shape: (3, 3)

    Notes
    -----
    - Uses Moore-Penrose pseudoinverse for robust estimation
    - Assumes stationarity within the data segment
    - Best suited for short time windows where stationarity holds
    """
    # Jeff Craley's Method using definition of least squares
    Y = X[:, 1:]      # Future states
    Z = X[:, 0:-1]    # Current states
    A_hat = Y @ np.linalg.pinv(Z)  # Least squares solution
    return A_hat


def estimateA_subject(data, fs=2000, winsize=0.5):
    """
    Estimate state transition matrices for windowed segments of multichannel data.

    This function divides the data into overlapping or non-overlapping windows
    and estimates a state transition matrix for each window, allowing for
    time-varying connectivity analysis.

    Parameters
    ----------
    data : numpy.ndarray
        2D array of time series data with shape (n_channels, n_samples)
    fs : int, optional
        Sampling frequency in Hz. Default is 2000.
    winsize : float, optional
        Window size in seconds. Default is 0.5 seconds.

    Returns
    -------
    numpy.ndarray
        3D array of state transition matrices with shape
        (n_channels, n_channels, n_windows) where A_hat[:,:,i]
        is the transition matrix for window i.

    Examples
    --------
    ### # Analyze 10 seconds of 64-channel data
    ### data = np.random.randn(64, 20000)  # 64 channels, 10 seconds at 2000 Hz
    ### A_matrices = estimateA_subject(data, fs=2000, winsize=0.5)
    ### print(f"Number of windows: {A_matrices.shape[2]}")
    Number of windows: 20

    ### # Analyze connectivity in specific window
    ### window_5_connectivity = A_matrices[:, :, 5]
    ### print(f"Max connectivity in window 5: {np.max(np.abs(window_5_connectivity)):.3f}")

    Applications
    ------------
    - Time-varying connectivity analysis
    - Seizure propagation studies
    - Dynamic network analysis
    - State-space model identification

    Notes
    -----
    - Each window is processed independently
    - Windows are non-overlapping by default
    - Progress bar shows processing status
    - Suitable for identifying time-varying dynamics
    """
    window = int(np.floor(winsize * fs))
    time = data.shape[1]
    n_chs = data.shape[0]
    n_wins = int(np.round(time / window))

    # Initialize output array
    A_hat = np.zeros((n_chs, n_chs, n_wins))

    # Process each window
    for win in tqdm(range(0, n_wins), desc="Estimating A matrices"):
        if win * window < data.shape[1]:
            data_win = data[:, win*window:(win+1)*window]
            A_hat[:, :, win] = estimateA(data_win)

    return A_hat


def reconstruct_signal(x, A_hat, fs=2000, win_size=0.5):
    """
    Reconstruct signal using estimated state-space model matrices.

    This function uses the estimated state transition matrices to reconstruct
    the signal forward in time, resetting to the true value at the start of
    each window for stability.

    Parameters
    ----------
    x : numpy.ndarray
        Original signal with shape (n_channels, n_samples)
    A_hat : numpy.ndarray
        State transition matrices with shape (n_channels, n_channels, n_windows)
        as returned by estimateA_subject()
    fs : int, optional
        Sampling frequency in Hz. Default is 2000.
    win_size : float, optional
        Window size in seconds (must match the window size used in estimateA_subject).
        Default is 0.5.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal with same shape as input x

    Algorithm
    ---------
    1. Initialize with true initial condition
    2. For each window:
       - Use the corresponding A matrix
       - Propagate forward: x_hat(t) = A * x_hat(t-1)
       - Reset to true value at window boundaries

    Examples
    --------
    ### # Generate test signal
    ### signal = np.random.randn(10, 10000)  # 10 channels, 5 seconds
    ###
    ### # Estimate model
    ### A_matrices = estimateA_subject(signal, fs=2000, winsize=0.5)
    ###
    ### # Reconstruct
    ### signal_reconstructed = reconstruct_signal(signal, A_matrices, fs=2000, win_size=0.5)
    ###
    ### # Compare reconstruction error
    ### error = np.mean((signal - signal_reconstructed)**2)
    ### print(f"MSE: {error:.6f}")

    Use Cases
    ---------
    - Model validation (compare reconstruction to original)
    - Predictive modeling
    - Anomaly detection (large reconstruction errors)
    - Signal denoising

    Notes
    -----
    - Resets to true signal at each window boundary to prevent drift
    - Reconstruction quality depends on model accuracy
    - Large reconstruction errors may indicate nonlinear dynamics
    """
    nWin = A_hat.shape[2]
    nCH = A_hat.shape[0]
    signal_length = x.shape[1]

    # Initialize the reconstructed signal
    xhat = np.zeros((nCH, signal_length))
    x_initial = x[:, 0]
    xhat[:, 0] = x_initial

    # Process each window
    progress_bar = tqdm(range(0, nWin), desc="Reconstructing signal")

    for i in progress_bar:
        progress_bar.set_description(f'Reconstructing window {i+1}/{nWin}')
        A = A_hat[:, :, i]

        # Propagate forward within the window
        for j in range(1, int(win_size*fs)):
            sample_idx = int(i*fs*win_size) + j
            if sample_idx < signal_length:
                # Reconstruct using state-space model
                xhat[:, sample_idx] = A @ xhat[:, sample_idx - 1]

        # Reset to true value at the start of next window
        if i < nWin - 1:
            next_window_start = int((i+1)*fs*win_size)
            if next_window_start < signal_length:
                xhat[:, next_window_start] = x[:, next_window_start]

    return xhat


def identifySS(A):
    """
    Identify sources and sinks in network connectivity from state transition matrix.

    This function analyzes the connectivity pattern in matrix A to identify
    which nodes act as sources (net information outflow) and which act as
    sinks (net information inflow).

    Parameters
    ----------
    A : numpy.ndarray
        State transition matrix with shape (n_channels, n_channels)
        where A[i,j] represents influence from channel j to channel i.

    Returns
    -------
    tuple
        Contains four arrays:
        - sink_ : numpy.ndarray
            Sink index for each channel (higher values = stronger sink)
        - source_ : numpy.ndarray
            Source index for each channel (higher values = stronger source)
        - row_ranks : numpy.ndarray
            Normalized row sum ranks (0 to 1)
        - col_ranks_ : numpy.ndarray
            Normalized column sum ranks (0 to 1)

    Algorithm
    ---------
    1. Compute row sums (total input to each channel)
    2. Compute column sums (total output from each channel)
    3. Rank channels by row and column sums
    4. Combine ranks to identify sources and sinks:
       - Sinks: High row sum (receive) + Low column sum (send little)
       - Sources: Low row sum (receive little) + High column sum (send)

    Examples
    --------
    ### # Create a simple connectivity pattern
    ### A = np.array([[0, 0.8, 0.1],
    ...               [0.2, 0, 0.1],
    ...               [0.7, 0.2, 0]])
    ###
    ### sink_idx, source_idx, row_ranks, col_ranks = identifySS(A)
    ###
    ### # Identify strongest sink and source
    ### strongest_sink = np.argmax(sink_idx)
    ### strongest_source = np.argmax(source_idx)
    ### print(f"Strongest sink: Channel {strongest_sink}")
    ### print(f"Strongest source: Channel {strongest_source}")

    ### # Visualize results
    ### import matplotlib.pyplot as plt
    ### channels = np.arange(len(sink_idx))
    ### plt.figure(figsize=(10, 4))
    ### plt.subplot(1, 2, 1)
    ### plt.bar(channels, sink_idx)
    ### plt.title('Sink Index by Channel')
    ### plt.subplot(1, 2, 2)
    ### plt.bar(channels, source_idx)
    ### plt.title('Source Index by Channel')
    ### plt.show()

    Interpretation
    --------------
    - Sink index near √2: Strong sink (receives information)
    - Source index near √2: Strong source (sends information)
    - Values near 0: Neutral nodes

    Applications
    ------------
    - Epilepsy: Identify seizure onset zones (sources) and propagation zones (sinks)
    - Brain networks: Find hub regions
    - Information flow: Track directional connectivity

    Notes
    -----
    - Diagonal elements are ignored (self-connections)
    - Based on geometric combination of row and column ranks
    - Higher values indicate stronger source/sink characteristics
    """
    nCh = A.shape[0]
    A_abs = np.abs(A)

    # Set diagonals to zero (ignore self-connections)
    A_abs[np.diag_indices_from(A_abs)] = 0

    # Compute row and column sums
    sum_A_r = np.sum(A_abs, axis=1)  # Total input to each channel
    sum_A_c = np.sum(A_abs, axis=0)  # Total output from each channel

    # Rank channels by row sum (ascending: low input gets low rank)
    sort_ch_r = np.argsort(sum_A_r)  # ascending
    row_ranks = np.argsort(sort_ch_r)  # rearrange back to original order
    row_ranks = row_ranks / nCh  # normalize to [0, 1]

    # Rank channels by column sum (ascending: low output gets low rank)
    sort_ch_c = np.argsort(sum_A_c)  # ascending
    col_ranks_ = np.argsort(sort_ch_c)  # rearrange back to original order
    col_ranks_ = col_ranks_ / nCh  # normalize to [0, 1]

    # Calculate sink and source indices using geometric distance
    # Sink: high row rank (receives a lot) + low column rank (sends little)
    sink_ = np.sqrt(2) - np.sqrt((1-row_ranks)**2 + (col_ranks_)**2)

    # Source: low row rank (receives little) + high column rank (sends a lot)
    source_ = np.sqrt(2) - np.sqrt((1/nCh - row_ranks)**2 + (1-col_ranks_)**2)

    return sink_, source_, row_ranks, col_ranks_


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Complete state-space analysis pipeline on synthetic data.
    
    This example demonstrates:
    1. Generating synthetic multichannel data with known dynamics
    2. Estimating state-space models
    3. Reconstructing signals
    4. Identifying sources and sinks
    """

    import matplotlib.pyplot as plt

    print("State-Space Modeling Example")
    print("=" * 50)

    # Generate synthetic data with known connectivity
    np.random.seed(42)
    n_channels = 100  # Can be changed to any number
    n_samples = 10000
    fs = 500
    duration = n_samples / fs

    print(f"\n1. Generating synthetic {n_channels}-channel signal")
    print(f"   Duration: {duration} seconds at {fs} Hz")

    # Create random connectivity pattern with some structure
    print(f"\n   Creating random connectivity matrix for {n_channels} channels...")

    # Generate a random A matrix with some constraints for stability
    A_true = np.random.randn(n_channels, n_channels) * 0.9  # Random connections

    # Add some structure to make it more realistic:
    # 1. Make it more sparse (some connections are zero)
    sparsity = 0.5  # Proportion of connections to keep
    mask = np.random.random((n_channels, n_channels)) < sparsity
    A_true = A_true * mask

    # 2. Add stronger diagonal elements (self-connections) for stability
    np.fill_diagonal(A_true, np.random.uniform(0.3, 0.7, n_channels))

    # 3. Create some directionality (lower triangular bias for forward propagation)
    # Make lower triangular elements slightly stronger
    for i in range(n_channels):
        for j in range(i):
            if mask[i, j]:
                A_true[i, j] *= 1.5  # Strengthen forward connections

    # 4. Ensure stability (largest eigenvalue < 1)
    eigenvalues = np.linalg.eigvals(A_true)
    max_eigenvalue = np.max(np.abs(eigenvalues))
    if max_eigenvalue >= 0.95:
        A_true = A_true * (0.9 / max_eigenvalue)  # Scale to ensure stability

    print(f"   Max eigenvalue of A: {np.max(np.abs(np.linalg.eigvals(A_true))):.3f}")
    print(f"   Sparsity: {np.sum(A_true == 0) / (n_channels**2) * 100:.1f}% zeros")

    # Identify expected sources and sinks from true matrix
    true_sink, true_source, _, _ = identifySS(A_true)
    print(f"   Designed strongest source: Channel {np.argmax(true_source)}")
    print(f"   Designed strongest sink: Channel {np.argmax(true_sink)}")

    # Generate data using the true model
    data = np.zeros((n_channels, n_samples))
    data[:, 0] = np.random.randn(n_channels)

    for t in range(1, n_samples):
        data[:, t] = A_true @ data[:, t-1] + 0.1 * np.random.randn(n_channels)

    # Add some amplitude scaling for visualization
    data = data * 50e-6  # Scale to microvolts

    print("\n2. Estimating state-space models")
    winsize = 0.5  # 500ms windows
    A_hat = estimateA_subject(data, fs=fs, winsize=winsize)
    print(f"   Estimated {A_hat.shape[2]} transition matrices")

    # Analyze the average connectivity
    A_mean = np.mean(A_hat, axis=2)
    print(f"   Mean absolute error from true A: {np.mean(np.abs(A_mean - A_true)):.4f}")

    print("\n3. Reconstructing signal")
    data_recon = reconstruct_signal(data, A_hat, fs=fs, win_size=winsize)

    # Calculate reconstruction error
    mse = np.mean((data - data_recon)**2)
    correlation = np.corrcoef(data.flatten(), data_recon.flatten())[0, 1]
    print(f"   MSE: {mse:.2e}")
    print(f"   Correlation: {correlation:.4f}")

    print("\n4. Identifying sources and sinks")
    # Use the mean connectivity matrix
    sink_idx, source_idx, row_ranks, col_ranks = identifySS(A_mean)

    strongest_sink = np.argmax(sink_idx)
    strongest_source = np.argmax(source_idx)
    print(f"   Strongest sink: Channel {strongest_sink} (index: {sink_idx[strongest_sink]:.3f})")
    print(f"   Strongest source: Channel {strongest_source} (index: {source_idx[strongest_source]:.3f})")

    # Visualization
    print("\n5. Creating visualizations...")

    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Original vs Reconstructed Signal (multiple channels)
    ax1 = fig.add_subplot(gs[0, :])
    time_plot = np.arange(2000) / fs  # First second
    n_plot_channels = min(3, n_channels)  # Plot up to 3 channels
    for ch in range(n_plot_channels):
        offset = ch * 100  # Offset for visualization
        ax1.plot(time_plot, data[ch, :2000]*1e6 + offset, 'b-',
                label=f'Ch{ch} Original' if ch == 0 else '', alpha=0.7, linewidth=1)
        ax1.plot(time_plot, data_recon[ch, :2000]*1e6 + offset, 'r--',
                label=f'Ch{ch} Reconstructed' if ch == 0 else '', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (μV) + offset')
    ax1.set_title(f'Original vs Reconstructed Signals (first {n_plot_channels} channels)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: True Connectivity Matrix
    ax2 = fig.add_subplot(gs[1, 0])
    im1 = ax2.imshow(A_true, cmap='RdBu_r', vmin=-np.max(np.abs(A_true)),
                     vmax=np.max(np.abs(A_true)))
    ax2.set_title('True Connectivity Matrix A')
    ax2.set_xlabel('From Channel')
    ax2.set_ylabel('To Channel')
    plt.colorbar(im1, ax=ax2, fraction=0.046, pad=0.04)

    # Plot 3: Estimated Mean Connectivity Matrix
    ax3 = fig.add_subplot(gs[1, 1])
    im2 = ax3.imshow(A_mean, cmap='RdBu_r', vmin=-np.max(np.abs(A_true)),
                     vmax=np.max(np.abs(A_true)))
    ax3.set_title('Estimated Mean Connectivity')
    ax3.set_xlabel('From Channel')
    ax3.set_ylabel('To Channel')
    plt.colorbar(im2, ax=ax3, fraction=0.046, pad=0.04)

    # Plot 4: Estimation Error Matrix
    ax4 = fig.add_subplot(gs[1, 2])
    error_matrix = A_mean - A_true
    im3 = ax4.imshow(error_matrix, cmap='RdBu_r',
                     vmin=-np.max(np.abs(error_matrix)),
                     vmax=np.max(np.abs(error_matrix)))
    ax4.set_title('Estimation Error (Est - True)')
    ax4.set_xlabel('From Channel')
    ax4.set_ylabel('To Channel')
    plt.colorbar(im3, ax=ax4, fraction=0.046, pad=0.04)

    # Plot 5: Reconstruction Error Over Time
    ax5 = fig.add_subplot(gs[2, 0])
    error = np.mean((data - data_recon)**2, axis=0)
    time_full = np.arange(len(error)) / fs
    ax5.plot(time_full, error*1e12, linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('MSE (pV²)')
    ax5.set_title('Reconstruction Error Over Time')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Sink Index
    ax6 = fig.add_subplot(gs[2, 1])
    channels = np.arange(n_channels)
    bars1 = ax6.bar(channels, sink_idx, color='blue', alpha=0.7)
    ax6.set_xlabel('Channel')
    ax6.set_ylabel('Sink Index')
    ax6.set_title('Sink Index by Channel')
    ax6.set_xticks(channels)
    ax6.grid(True, alpha=0.3, axis='y')

    # Highlight strongest sink
    bars1[strongest_sink].set_color('darkblue')
    bars1[strongest_sink].set_edgecolor('black')
    bars1[strongest_sink].set_linewidth(2)

    # Plot 7: Source Index
    ax7 = fig.add_subplot(gs[2, 2])
    bars2 = ax7.bar(channels, source_idx, color='red', alpha=0.7)
    ax7.set_xlabel('Channel')
    ax7.set_ylabel('Source Index')
    ax7.set_title('Source Index by Channel')
    ax7.set_xticks(channels)
    ax7.grid(True, alpha=0.3, axis='y')

    # Highlight strongest source
    bars2[strongest_source].set_color('darkred')
    bars2[strongest_source].set_edgecolor('black')
    bars2[strongest_source].set_linewidth(2)

    plt.suptitle(f'State-Space Modeling Analysis ({n_channels} Channels)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\nAnalysis complete!")
    print("\nConnectivity Matrix Properties:")
    print(f"- Randomly generated for {n_channels} channels")
    print(f"- Sparsity level: {sparsity*100:.0f}% connections active")
    print(f"- Stable dynamics (max eigenvalue < 1)")
    print(f"- Forward propagation bias (lower triangular emphasis)")
    print(f"\nEstimation Performance:")
    print(f"- Correlation between true and estimated: {np.corrcoef(A_true.flatten(), A_mean.flatten())[0,1]:.3f}")
    print(f"- Signal reconstruction correlation: {correlation:.3f}")