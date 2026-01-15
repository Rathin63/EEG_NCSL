import numpy as np
from scipy.stats import rankdata

def compute_sink_source_scores(A):

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

    # # ---------- Normalize row sums to [0,1] ----------
    # r_min = np.min(sum_A_r)
    # r_max = np.max(sum_A_r)
    # row_ranks = (sum_A_r - r_min) / (r_max - r_min)
    #
    # # ---------- Normalize column sums to [0,1] ----------
    # c_min = np.min(sum_A_c)
    # c_max = np.max(sum_A_c)
    # col_ranks_ = (sum_A_c - c_min) / (c_max - c_min)

    # ---------- Source / Sink strengths ----------
    # Sink: strong receiver + weak sender
    #sink_ = row_ranks * (1.0 - col_ranks_)

    # Source: strong sender + weak receiver
    #source_ = col_ranks_ * (1.0 - row_ranks)

    # Calculate sink and source indices using geometric distance
    # Sink: high row rank (receives a lot) + low column rank (sends little)
    sink_ = np.sqrt(2) - np.sqrt((1-row_ranks)**2 + (col_ranks_)**2)

    # Source: low row rank (receives little) + high column rank (sends a lot)
    source_ = np.sqrt(2) - np.sqrt((row_ranks)**2 + (1-col_ranks_)**2)

    # ---------- Optional: normalize to [0,1] across channels ----------
    #sink_   = sink_   / (np.max(sink_))
    #source_ = source_ / (np.max(source_))

    return sink_, source_, row_ranks, col_ranks_

