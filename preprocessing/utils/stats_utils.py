"""
Statistical Utilities Module

This module provides robust statistical functions for data analysis,
including robust standard deviation estimation and kernel-based mode estimation.

Functions:
    robustSTD: Compute robust standard deviation using MAD
    kernelModeEstimate: Estimate mode using kernel density estimation

Dependencies:
    numpy
    scipy.stats (for kernelModeEstimate)
"""

import numpy as np


def robustSTD(data, median=None):
    """
    Compute robust standard deviation using Median Absolute Deviation (MAD).

    This function calculates a robust estimate of standard deviation that is
    less sensitive to outliers than the traditional standard deviation.
    It uses the Median Absolute Deviation scaled by a constant factor.

    Parameters
    ----------
    data : array-like
        Input data. Can be 1D or 2D array. If 2D, computes robust STD
        for each column independently.
    median : array-like, optional
        Pre-computed median values. If None, medians are computed from data.
        Must match the number of columns in data if provided.

    Returns
    -------
    numpy.ndarray
        Robust standard deviation estimate(s). Returns scalar for 1D input,
        array of values for 2D input (one per column).

    Examples
    --------
    ### import numpy as np
    ### # 1D array with outliers
    ### data = np.array([1, 2, 3, 4, 5, 100])  # 100 is an outlier
    ### robust_std = robustSTD(data)
    ### regular_std = np.std(data)
    ### print(f"Robust STD: {robust_std:.2f}, Regular STD: {regular_std:.2f}")

    ### # 2D array - compute for each column
    ### data = np.random.randn(100, 3)
    ### robust_stds = robustSTD(data)  # Returns 3 values

    Theory
    ------
    MAD = median(|X - median(X)|)

    For normally distributed data:
    robust_std = MAD / 0.6745

    The constant 0.6745 is the inverse cumulative normal evaluated at 0.75.
    This scaling makes the robust STD comparable to the regular STD for
    normal distributions.

    References
    ----------
    Huber, P.J. (1981). Robust Statistics. Wiley. QA 276.H785

    Notes
    -----
    - Performs well when:
      1. Data near mean approximately follows normal distribution
      2. Outliers are outside the 25%-75% percentiles
    - Uses nanmedian to handle NaN values gracefully
    - Empty input returns empty array with warning
    """
    # handle empty case
    data = np.array(data)
    data = np.squeeze(data)

    if data.size == 0:
        print("WARNING: Input array is empty.\n The function returns an empty array.")
        rstd = np.array([])
        return rstd

    # if one dimensional, make sure it is a column
    if np.ndim(data) == 1:  # is a vector
        data = data.reshape([len(data), 1])

    if median == None:
        # identify median of each column, and subtract it from each entry in the column
        median_cols = np.nanmedian(data, axis=0)[np.newaxis]
    else:
        median_cols = median

    data = data - median_cols

    # rest of computation
    rstd = np.nanmedian(np.abs(data), axis=0) / 0.6741891400433162

    return rstd


def kernelModeEstimate(data):
    """
    Estimate the mode of data using kernel density estimation.

    This function finds the approximate mode (most frequent value) of the data
    by fitting a kernel density estimate and finding its maximum. Works for
    1D, 2D, and 3D arrays.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array. Can be:
        - 1D: Single dataset
        - 2D: Multiple datasets (mode computed for each column)
        - 3D: Matrix of datasets (mode computed for each [i,j,:] slice)

    Returns
    -------
    numpy.ndarray or float
        - For 1D input: scalar mode estimate
        - For 2D input: 1D array of mode estimates (one per column)
        - For 3D input: 2D array of mode estimates

    Examples
    --------
    ### import numpy as np
    ### import matplotlib.pyplot as plt
    ###
    ### # Example with F-distribution
    ### dfnum = 10  # between group degrees of freedom
    ### dfden = 48  # within groups degrees of freedom
    ### data = np.random.f(dfnum, dfden, 1000)
    ###
    ### # Estimate mode
    ### mode = kernelModeEstimate(data)
    ### std = robustSTD(data)
    ### std2 = np.std(data)
    ###
    ### # Plot time series with mode and std bands
    ### plt.figure(figsize=(10,5))
    ### plt.plot(data)
    ### plt.axhline(mode, c='r', label='Mode')
    ### plt.axhline(mode+std, c='black', label='Robust STD')
    ### plt.axhline(mode-std, c='black')
    ### plt.axhline(mode+std2, c='gray', label='Regular STD')
    ### plt.axhline(mode-std2, c='gray')
    ### plt.legend()
    ### plt.show()
    ###
    ### # Plot histogram with mode
    ### plt.figure(figsize=(10,5))
    ### plt.hist(data, bins=100, density=True, alpha=0.7)
    ### plt.axvline(mode, c='r', label=f'Mode={mode:.2f}')
    ### plt.legend()
    ### plt.show()

    Algorithm
    ---------
    1. Fit a Gaussian kernel density estimate to the data
    2. Evaluate the density on a fine grid (10,000 points)
    3. Find the point with maximum density

    Notes
    -----
    - More robust than simple histogram-based mode estimation
    - Computational cost increases with data size and dimensionality
    - For 3D data, prints progress dots while processing
    - Uses scipy's gaussian_kde with default bandwidth selection

    Credits
    -------
    Original MATLAB version: Adam Charles (2022)
    Python translation: Amir Hossein Daraie (2022)
    """
    from scipy.stats import gaussian_kde

    modeEst = np.array([])
    data = np.squeeze(data)  # for (1,N) or (N,1) cases

    if np.ndim(data) == 1:
        kde = gaussian_kde(data)
        xden = np.linspace(np.min(data), np.max(data), int(10e4))
        fden = kde.evaluate(xden)
        modeEst = xden[np.argmax(fden)]
    elif np.ndim(data) == 2:
        modeEst = np.zeros(data.shape[1])
        for ll in range(data.shape[1]):
            kde = gaussian_kde(data[:, ll])
            xden = np.linspace(np.min(data[:, ll]), np.max(data[:, ll]), int(10e4))
            fden = kde.evaluate(xden)
            modeEst[ll] = xden[np.argmax(fden)]
    elif np.ndim(data) == 3:
        modeEst = np.zeros((data.shape[0], data.shape[1]))
        for ll in range(data.shape[0]):
            for kk in range(data.shape[1]):
                kde = gaussian_kde(data[ll, kk, :])
                xden = np.linspace(np.min(data[ll, kk, :]), np.max(data[ll, kk, :]), int(10e4))
                fden = kde.evaluate(xden)
                modeEst[ll, kk] = xden[np.argmax(fden)]
            print('.', end='')
        print()

    return modeEst


def tukeys_method(data, q1_percentile=25, q3_percentile=75, inner_fence_multiplier=1.5, outer_fence_multiplier=3.0):
    """
    Apply Tukey's method for outlier detection.

    Parameters:
    -----------
    data : array-like
        Data to analyze for outliers
    q1_percentile : float
        Lower quartile percentile (default 25)
    q3_percentile : float
        Upper quartile percentile (default 75)
    inner_fence_multiplier : float
        Multiplier for inner fence (default 1.5)
    outer_fence_multiplier : float
        Multiplier for outer fence (default 3.0)

    Returns:
    --------
    probable_outliers : array
        Indices of probable outliers (beyond outer fence)
    possible_outliers : array
        Indices of possible outliers (between inner and outer fence)
    """
    q1 = np.percentile(data, q1_percentile)
    q3 = np.percentile(data, q3_percentile)
    iqr = q3 - q1

    # Calculate fences
    inner_fence_lower = q1 - inner_fence_multiplier * iqr
    inner_fence_upper = q3 + inner_fence_multiplier * iqr
    outer_fence_lower = q1 - outer_fence_multiplier * iqr
    outer_fence_upper = q3 + outer_fence_multiplier * iqr

    # Identify outliers
    probable_outliers = np.where((data < outer_fence_lower) | (data > outer_fence_upper))[0]
    possible_outliers = np.where(
        ((data < inner_fence_lower) & (data >= outer_fence_lower)) |
        ((data > inner_fence_upper) & (data <= outer_fence_upper))
    )[0]

    return probable_outliers, possible_outliers