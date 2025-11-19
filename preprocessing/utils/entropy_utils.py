import numpy as np

def spatial_entropy(values, bins=32):
    """Compute Shannon entropy of a 1D array."""
    v = np.array(values, dtype=float)

    # Normalize to 0–1
    v = v - np.nanmin(v)
    vmax = np.nanmax(v)
    if vmax > 0:
        v = v / vmax
    else:
        return 0.0  # flat map → zero entropy

    # Histogram → probability distribution
    hist, _ = np.histogram(v, bins=bins, density=True)
    p = hist / np.sum(hist)

    p = p[p > 0]  # remove zero entries
    return float(-np.sum(p * np.log(p)))

