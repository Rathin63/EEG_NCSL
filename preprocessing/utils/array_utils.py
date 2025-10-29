"""
Array Processing Utilities Module

This module provides utility functions for array manipulation and processing,
particularly for creating overlapping chunks from arrays.

Functions:
    overlap: Generate overlapping chunks from an array

Dependencies:
    numpy
"""

import numpy as np


def overlap(array, len_chunk, len_sep=1):
    """
    Returns a matrix of all full overlapping chunks of the input array.

    This function creates a 2D matrix where each row is a chunk of the input
    array. The chunks have a specified length and separation (stride) between
    consecutive chunks. Only full chunks are returned (no partial chunks at
    the end of the array).

    Parameters
    ----------
    array : numpy.ndarray
        The input array to be chunked. Should be 1-dimensional.
    len_chunk : int
        The length of each chunk (number of elements in each chunk).
    len_sep : int, optional
        The separation length between consecutive chunks (stride).
        Default is 1, meaning chunks overlap by (len_chunk - 1) elements.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row is a chunk of the original array.
        Shape is (n_chunks, len_chunk) where n_chunks depends on array size,
        chunk length, and separation.

    Examples
    --------
    ### import numpy as np
    ### arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ### overlap(arr, len_chunk=4, len_sep=2)
    array([[1, 2, 3, 4],
           [3, 4, 5, 6],
           [5, 6, 7, 8],
           [7, 8, 9, 10]])

    ### # With default separation of 1 (maximum overlap)
    ### overlap(arr, len_chunk=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5],
           [4, 5, 6],
           [5, 6, 7],
           [6, 7, 8],
           [7, 8, 9],
           [8, 9, 10]])

    Notes
    -----
    - Only returns complete chunks; partial chunks at the end are ignored
    - Uses advanced NumPy indexing for efficient chunk extraction
    - Useful for sliding window operations in signal processing

    Algorithm
    ---------
    1. Calculate number of full chunks that fit in the array
    2. Create a tiled matrix of the repeated input array
    3. Use fancy indexing to extract the appropriate chunks
    """
    n_arrays = int(np.ceil((array.size - len_chunk + 1) / len_sep))

    array_matrix = np.tile(array, n_arrays).reshape(n_arrays, -1)

    columns = np.array(((len_sep * np.arange(0, n_arrays)).reshape(n_arrays, -1) + np.tile(
        np.arange(0, len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    rows = np.array((np.arange(n_arrays).reshape(n_arrays, -1) + np.tile(
        np.zeros(len_chunk), n_arrays).reshape(n_arrays, -1)), dtype=np.intp)

    return array_matrix[rows, columns]